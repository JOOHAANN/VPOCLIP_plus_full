#!/usr/bin/env python
"""Build 2s/1s windowed CLIPGCN tensor, skeleton, and joint-xy caches.

The output is intentionally compatible with the existing X3D and CTR-GCN
feature extraction scripts:

* X3D reads ``<split>_float16.npy`` and ``<split>_labels.npy``.
* CTR-GCN reads ``windowed_skeleton_uniform13.npz``.
* CLIPGCN alignment uses ``<split>_joint_xy.npy`` and sample names.
"""

import argparse
import csv
import json
import math
import os
import re
import signal
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "quiet")

import cv2
import numpy as np
import torch
import torch.nn.functional as F

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1, 1)


@dataclass(frozen=True)
class WindowSpec:
    source_video: str
    source_csv: str
    sample_name: str
    skeleton_sample_name: str
    label: int
    action_id: str
    subject_id: str
    window_index: int
    start_frame: int
    end_frame_exclusive: int
    sample_frames: tuple[int, ...]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-root", default="/workspace/CLIPGCN/data")
    parser.add_argument("--reference-metadata", default="/workspace/X3D/data/clipgcn_tensor_cs_70_10_20/metadata.json")
    parser.add_argument("--out-dir", default="/workspace/CLIPGCN/data/windowed_2s1s_13f")
    parser.add_argument("--window-seconds", type=float, default=2.0)
    parser.add_argument("--stride-seconds", type=float, default=1.0)
    parser.add_argument("--frames", type=int, default=13)
    parser.add_argument("--size", type=int, default=160)
    parser.add_argument("--depth-width", type=float, default=512.0)
    parser.add_argument("--depth-height", type=float, default=424.0)
    parser.add_argument("--cache-dtype", choices=("float16", "uint8"), default="float16")
    parser.add_argument("--joint-xy-dtype", choices=("float32", "float16"), default="float32")
    parser.add_argument("--resize-device", default="cuda:0")
    parser.add_argument("--gpu-batch-size", type=int, default=32)
    parser.add_argument(
        "--decode-workers",
        type=int,
        default=1,
        help="Number of source videos to decode in parallel while building windows.",
    )
    parser.add_argument("--video-timeout", type=int, default=30)
    parser.add_argument("--exclude-action", action="append", default=None)
    parser.add_argument(
        "--include-action",
        action="append",
        default=None,
        help="Only include these actions, e.g. A35 or A035. Can be passed multiple times.",
    )
    parser.add_argument(
        "--anchor-first-frame-actions",
        action="append",
        default=None,
        help=(
            "For these actions, force the first sampled frame of every window to be "
            "the first frame of the full source video."
        ),
    )
    parser.add_argument("--limit-videos", type=int, default=None)
    parser.add_argument("--plan-only", action="store_true", help="Only write metadata/window specs; do not decode videos.")
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "val", "test"),
        default=("train", "val", "test"),
        help="Dataset splits to build. Use '--splits test' to resume a failed test build without rewriting train/val.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


class VideoTimeoutError(TimeoutError):
    pass


def raise_video_timeout(_signum, _frame):
    raise VideoTimeoutError("video decode timed out")


def progress(items, desc):
    return tqdm(items, desc=desc) if tqdm is not None else items


def action_id(path):
    match = re.match(r"(A\d+)_", Path(path).name)
    if match is None:
        raise ValueError(f"Cannot parse action id from {path}")
    return match.group(1)


def normalize_action_id(value):
    match = re.fullmatch(r"A0*(\d+)", str(value).strip().upper())
    if match is None:
        raise ValueError(f"Invalid action id: {value}")
    return f"A{int(match.group(1)):03d}"


def subject_id(path, src_root):
    return Path(path).relative_to(src_root).parts[0]


def csv_name_from_video(rel_video):
    return str(Path(rel_video).with_suffix(".csv"))


def windowed_name(rel_path, window_index, start_frame, end_frame_exclusive, suffix):
    path = Path(rel_path)
    stem = f"{path.stem}__w{window_index:04d}_f{start_frame:06d}-{end_frame_exclusive:06d}"
    return str(path.with_name(stem).with_suffix(suffix))


def load_reference_metadata(path):
    metadata = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        "class_to_idx": metadata["class_to_idx"],
        "classes": metadata["classes"],
        "excluded_actions": set(metadata.get("excluded_actions") or []),
        "split_subjects": metadata["split_subjects"],
        "split_rule": metadata.get("split_rule"),
        "split_mode": metadata.get("split_mode"),
        "split_seed": metadata.get("split_seed"),
    }


def joint_columns(header, joints=25):
    index = {name: idx for idx, name in enumerate(header)}
    xyz_columns = [
        (index[f"joint{joint}_3dX"], index[f"joint{joint}_3dY"], index[f"joint{joint}_3dZ"])
        for joint in range(1, joints + 1)
    ]
    depth_columns = [
        (index[f"joint{joint}_depthX"], index[f"joint{joint}_depthY"])
        for joint in range(1, joints + 1)
    ]
    state_columns = [index.get(f"joint{joint}_trackingState") for joint in range(1, joints + 1)]
    return index["frameNum"], index.get("trackingID"), index.get("bodyindexID"), xyz_columns, depth_columns, state_columns


def read_main_track(csv_path, joints=25):
    tracks = defaultdict(dict)
    tracked_score = Counter()
    with open(csv_path, "r", newline="", errors="replace") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        frame_col, tracking_col, body_col, xyz_cols, depth_cols, state_cols = joint_columns(header, joints)

        for row in reader:
            if len(row) < len(header):
                continue
            try:
                frame = int(float(row[frame_col])) - 1
            except ValueError:
                continue
            if frame < 0:
                continue

            if tracking_col is not None and row[tracking_col]:
                track_id = row[tracking_col]
            elif body_col is not None and row[body_col]:
                track_id = f"body_{row[body_col]}"
            else:
                track_id = "single"

            xyz = np.zeros((joints, 3), dtype=np.float32)
            depth_xy = np.zeros((joints, 2), dtype=np.float32)
            valid_value = False
            for joint_idx, (xyz_col, depth_col) in enumerate(zip(xyz_cols, depth_cols)):
                try:
                    xyz[joint_idx] = (float(row[xyz_col[0]]), float(row[xyz_col[1]]), float(row[xyz_col[2]]))
                    depth_xy[joint_idx] = (float(row[depth_col[0]]), float(row[depth_col[1]]))
                except ValueError:
                    pass
                valid_value = valid_value or bool(np.any(xyz[joint_idx]) or np.any(depth_xy[joint_idx]))
                state_col = state_cols[joint_idx]
                if state_col is not None:
                    try:
                        tracked_score[track_id] += int(float(row[state_col]))
                    except ValueError:
                        pass
            if valid_value:
                tracks[track_id][frame] = (xyz, depth_xy)

    if not tracks:
        raise ValueError(f"No valid skeleton track in {csv_path}")

    def rank_track(item):
        track_id, frames = item
        return len(frames), tracked_score[track_id], str(track_id)

    _track_id, frame_to_values = max(tracks.items(), key=rank_track)
    return frame_to_values


def nearest_frame(frame_to_values, frame_idx):
    if frame_idx in frame_to_values:
        return frame_to_values[frame_idx]
    nearest = min(frame_to_values, key=lambda item: abs(item - frame_idx))
    return frame_to_values[nearest]


def normalize_depth_xy(depth_xy, width, height):
    out = np.nan_to_num(depth_xy.copy(), nan=0.0, posinf=width - 1.0, neginf=0.0)
    out[..., 0] = (out[..., 0] / (width - 1.0)) * 2.0 - 1.0
    out[..., 1] = (out[..., 1] / (height - 1.0)) * 2.0 - 1.0
    return np.clip(out, -1.0, 1.0)


def video_info(path):
    cap = cv2.VideoCapture(str(path))
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if total_frames <= 0 or fps <= 0:
            raise ValueError(f"Cannot read frame count/fps from {path}")
        return total_frames, fps
    finally:
        cap.release()


def make_window_specs(video_path, src_root, class_to_idx, args):
    rel_video = str(video_path.relative_to(src_root))
    rel_csv = csv_name_from_video(rel_video)
    csv_path = src_root / rel_csv
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing skeleton CSV for {rel_video}")

    total_frames, fps = video_info(video_path)
    frame_to_values = read_main_track(csv_path)
    valid_start = max(0, min(frame_to_values))
    valid_end_exclusive = min(total_frames, max(frame_to_values) + 1)
    window_frames = max(1, int(round(args.window_seconds * fps)))
    stride_frames = max(1, int(round(args.stride_seconds * fps)))
    if valid_end_exclusive - valid_start < window_frames:
        starts = [valid_start]
    else:
        last_start = valid_end_exclusive - window_frames
        starts = list(range(valid_start, last_start + 1, stride_frames))
        if starts and starts[-1] != last_start:
            starts.append(last_start)

    specs = []
    video_action = action_id(video_path)
    anchor_first_frame = video_action in set(args.anchor_first_frame_actions or [])
    label = int(class_to_idx[video_action])
    subj = subject_id(video_path, src_root)
    for window_index, start_frame in enumerate(starts):
        end_frame = min(total_frames, start_frame + window_frames)
        if anchor_first_frame and args.frames > 1:
            tail = np.linspace(start_frame, end_frame - 1, args.frames - 1).round().astype(np.int64)
            sample_frames = np.concatenate([np.array([0], dtype=np.int64), tail])
        else:
            sample_frames = np.linspace(start_frame, end_frame - 1, args.frames).round().astype(np.int64)
        sample_frames = np.clip(sample_frames, 0, total_frames - 1)
        specs.append(
            WindowSpec(
                source_video=rel_video,
                source_csv=rel_csv,
                sample_name=windowed_name(rel_video, window_index, start_frame, end_frame, ".mp4"),
                skeleton_sample_name=windowed_name(rel_csv, window_index, start_frame, end_frame, ".csv"),
                label=label,
                action_id=video_action,
                subject_id=subj,
                window_index=window_index,
                start_frame=int(start_frame),
                end_frame_exclusive=int(end_frame),
                sample_frames=tuple(int(x) for x in sample_frames.tolist()),
            )
        )
    return specs


def collect_split_specs(args, reference):
    src_root = Path(args.src_root)
    excluded_actions = set(args.exclude_action or reference["excluded_actions"])
    included_actions = set(args.include_action or [])
    videos = sorted(
        path for path in src_root.glob("P*/**/*.mp4")
        if action_id(path) not in excluded_actions and (not included_actions or action_id(path) in included_actions)
    )
    if args.limit_videos is not None:
        videos = videos[: args.limit_videos]

    split_subjects = {split: set(subjects) for split, subjects in reference["split_subjects"].items()}
    split_specs = {"train": [], "val": [], "test": []}
    failures = []
    for video in progress(videos, "planning windows"):
        subj = subject_id(video, src_root)
        split = next((name for name, subjects in split_subjects.items() if subj in subjects), None)
        if split is None:
            failures.append({"video": str(video.relative_to(src_root)), "error": f"subject {subj} is not in reference split"})
            continue
        try:
            split_specs[split].extend(make_window_specs(video, src_root, reference["class_to_idx"], args))
        except Exception as exc:
            failures.append({"video": str(video.relative_to(src_root)), "error": str(exc)})
    return split_specs, failures


def read_video_frames(video_path, sample_frames, timeout):
    old_handler = None
    cap = cv2.VideoCapture(str(video_path))
    try:
        if timeout and timeout > 0:
            old_handler = signal.signal(signal.SIGALRM, raise_video_timeout)
            signal.alarm(timeout)
        frames = []
        last_frame = None
        for frame_idx in sample_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ok, frame = cap.read()
            if not ok:
                if last_frame is None:
                    raise ValueError(f"Failed to read frame {frame_idx}")
                frame = last_frame
            last_frame = frame
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return np.stack(frames, axis=0)
    finally:
        cap.release()
        if timeout and timeout > 0:
            signal.alarm(0)
            if old_handler is not None:
                signal.signal(signal.SIGALRM, old_handler)


def read_resized_frames_once(video_path, requested_frames, size, timeout):
    """Sequentially decode one video and cache only frames used by its windows.

    Random seeking once per window is extremely slow with long-GOP mp4 files.
    This function walks the video once, resizes requested frames immediately,
    and lets all windows from the same source video reuse those cached frames.
    """

    requested = sorted({int(frame) for frame in requested_frames})
    if not requested:
        return {}

    old_handler = None
    cap = cv2.VideoCapture(str(video_path))
    frame_cache = {}
    last_frame = None
    try:
        if timeout and timeout > 0:
            old_handler = signal.signal(signal.SIGALRM, raise_video_timeout)
            signal.alarm(timeout)

        wanted_pos = 0
        max_requested = requested[-1]
        frame_idx = 0
        while frame_idx <= max_requested:
            ok, frame = cap.read()
            if not ok:
                break
            last_frame = frame
            if wanted_pos < len(requested) and frame_idx == requested[wanted_pos]:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_cache[frame_idx] = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
                wanted_pos += 1
                while wanted_pos < len(requested) and requested[wanted_pos] == frame_idx:
                    wanted_pos += 1
            frame_idx += 1

        if not frame_cache:
            raise ValueError(f"Failed to read requested frames from {video_path}")
        if len(frame_cache) != len(requested):
            fallback = next(reversed(frame_cache.values()))
            if last_frame is not None:
                rgb = cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB)
                fallback = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
            for frame in requested:
                frame_cache.setdefault(frame, fallback)
        return frame_cache
    finally:
        cap.release()
        if timeout and timeout > 0:
            signal.alarm(0)
            if old_handler is not None:
                signal.signal(signal.SIGALRM, old_handler)


def preprocess_resized_batch(clips, cache_dtype):
    batch = np.stack(clips, axis=0).transpose(0, 4, 1, 2, 3)
    if cache_dtype == "uint8":
        return batch.astype(np.uint8, copy=False)
    return (((batch.astype(np.float32) / 255.0) - IMAGENET_MEAN) / IMAGENET_STD).astype(np.float16)


def preprocess_batch(clips, size, resize_device, cache_dtype):
    if resize_device != "cpu" and not torch.cuda.is_available():
        resize_device = "cpu"
    if resize_device == "cpu":
        resized = []
        for clip in clips:
            resized.append(np.stack([cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA) for frame in clip], axis=0))
        batch = np.stack(resized, axis=0).transpose(0, 4, 1, 2, 3)
        if cache_dtype == "uint8":
            return batch.astype(np.uint8, copy=False)
        return (((batch.astype(np.float32) / 255.0) - IMAGENET_MEAN) / IMAGENET_STD).astype(np.float16)

    with torch.no_grad():
        batch = torch.from_numpy(np.stack(clips, axis=0)).to(resize_device, non_blocking=True)
        bsz, timesteps, height, width, channels = batch.shape
        tensor = batch.view(bsz * timesteps, height, width, channels).permute(0, 3, 1, 2).float()
        tensor = F.interpolate(tensor, size=(size, size), mode="bilinear", align_corners=False)
        if cache_dtype == "uint8":
            tensor = tensor.clamp_(0, 255).to(torch.uint8)
            return tensor.view(bsz, timesteps, channels, size, size).permute(0, 2, 1, 3, 4).cpu().numpy()
        tensor = tensor.view(bsz, timesteps, channels, size, size).permute(0, 2, 1, 3, 4).div_(255.0)
        mean = torch.tensor([0.485, 0.456, 0.406], device=resize_device).view(1, 3, 1, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=resize_device).view(1, 3, 1, 1, 1)
        return ((tensor - mean) / std).to(torch.float16).cpu().numpy()


def process_video_windows(job):
    (
        rel_video,
        video_specs,
        src_root,
        size,
        video_timeout,
        frames,
        depth_width,
        depth_height,
    ) = job
    src_root = Path(src_root)
    failures = []
    try:
        csv_values = read_main_track(src_root / csv_name_from_video(rel_video))
        requested_frames = [
            frame_idx
            for _idx, spec in video_specs
            for frame_idx in spec.sample_frames
        ]
        frame_cache = read_resized_frames_once(
            src_root / rel_video,
            requested_frames,
            size,
            video_timeout,
        )
    except Exception as exc:
        return {
            "indices": np.empty((0,), dtype=np.int64),
            "labels": np.empty((0,), dtype=np.int64),
            "clips": np.empty((0, frames, size, size, 3), dtype=np.uint8),
            "skeleton_x": np.empty((0, frames, 150), dtype=np.float32),
            "joint_xy": np.empty((0, frames, 25, 2), dtype=np.float32),
            "manifest_rows": [],
            "failures": [
                {
                    "sample": spec.sample_name,
                    "error": f"source video/skeleton read failed: {exc}",
                }
                for _idx, spec in video_specs
            ],
        }

    indices = []
    labels = []
    clips = []
    skeleton_rows = []
    joint_rows = []
    manifest_rows = []
    for idx, spec in video_specs:
        try:
            clip = np.stack([frame_cache[int(frame_idx)] for frame_idx in spec.sample_frames], axis=0)
            xyz_frames = []
            xy_frames = []
            for frame_idx in spec.sample_frames:
                xyz, depth_xy = nearest_frame(csv_values, frame_idx)
                xyz_frames.append(xyz)
                xy_frames.append(depth_xy)

            skeleton_sample = np.zeros((frames, 2, 25, 3), dtype=np.float32)
            skeleton_sample[:, 0, :, :] = np.stack(xyz_frames, axis=0)
            indices.append(idx)
            labels.append(spec.label)
            clips.append(clip)
            skeleton_rows.append(skeleton_sample.reshape(frames, 150))
            joint_rows.append(normalize_depth_xy(np.stack(xy_frames, axis=0), depth_width, depth_height))
            manifest_rows.append(f"{spec.sample_name} {spec.label}\n")
        except Exception as exc:
            failures.append({"sample": spec.sample_name, "error": str(exc)})

    if clips:
        clips_array = np.stack(clips, axis=0).astype(np.uint8, copy=False)
        skeleton_array = np.stack(skeleton_rows, axis=0).astype(np.float32, copy=False)
        joint_array = np.stack(joint_rows, axis=0).astype(np.float32, copy=False)
    else:
        clips_array = np.empty((0, frames, size, size, 3), dtype=np.uint8)
        skeleton_array = np.empty((0, frames, 150), dtype=np.float32)
        joint_array = np.empty((0, frames, 25, 2), dtype=np.float32)

    return {
        "indices": np.asarray(indices, dtype=np.int64),
        "labels": np.asarray(labels, dtype=np.int64),
        "clips": clips_array,
        "skeleton_x": skeleton_array,
        "joint_xy": joint_array,
        "manifest_rows": manifest_rows,
        "failures": failures,
    }


def flush_result(result, labels, joint_xy, skeleton_x, skeleton_y, pending, manifest, args, joint_dtype):
    result_indices = result["indices"]
    if result_indices.size == 0:
        return
    labels[result_indices] = result["labels"]
    for local_idx, sample_idx in enumerate(result_indices.tolist()):
        skeleton_y[sample_idx, int(result["labels"][local_idx])] = 1.0
        skeleton_x[sample_idx] = result["skeleton_x"][local_idx]
        joint_xy[sample_idx] = result["joint_xy"][local_idx].astype(joint_dtype, copy=False)
        pending.append((sample_idx, result["clips"][local_idx]))
    manifest.writelines(result["manifest_rows"])
    if len(pending) >= args.gpu_batch_size:
        flush_tensor_batch(pending, args._tensor_memmap, args)
        pending.clear()


def write_split(split, specs, args, out_dir):
    suffix = "float16" if args.cache_dtype == "float16" else "uint8"
    tensor_dtype = np.float16 if args.cache_dtype == "float16" else np.uint8
    joint_dtype = np.float32 if args.joint_xy_dtype == "float32" else np.float16
    tensor = np.lib.format.open_memmap(out_dir / f"{split}_{suffix}.npy", mode="w+", dtype=tensor_dtype, shape=(len(specs), 3, args.frames, args.size, args.size))
    labels = np.lib.format.open_memmap(out_dir / f"{split}_labels.npy", mode="w+", dtype=np.int64, shape=(len(specs),))
    joint_xy = np.lib.format.open_memmap(out_dir / f"{split}_joint_xy.npy", mode="w+", dtype=joint_dtype, shape=(len(specs), args.frames, 25, 2))
    skeleton_x = np.lib.format.open_memmap(out_dir / f"{split}_skeleton_x.npy", mode="w+", dtype=np.float32, shape=(len(specs), args.frames, 150))
    skeleton_y = np.zeros((len(specs), len(json.loads(Path(args.reference_metadata).read_text(encoding="utf-8"))["classes"])), dtype=np.float32)

    src_root = Path(args.src_root)
    specs_by_video = defaultdict(list)
    for idx, spec in enumerate(specs):
        specs_by_video[spec.source_video].append((idx, spec))

    failures = []
    pending = []
    with open(out_dir / f"{split}_manifest.txt", "w", encoding="utf-8", buffering=1) as manifest:
        args._tensor_memmap = tensor
        jobs = [
            (
                rel_video,
                video_specs,
                str(src_root),
                args.size,
                args.video_timeout,
                args.frames,
                args.depth_width,
                args.depth_height,
            )
            for rel_video, video_specs in specs_by_video.items()
        ]
        if args.decode_workers <= 1:
            iterator = progress(jobs, f"building {split}")
            for job in iterator:
                result = process_video_windows(job)
                failures.extend(result["failures"])
                flush_result(result, labels, joint_xy, skeleton_x, skeleton_y, pending, manifest, args, joint_dtype)
        else:
            progress_bar = tqdm(total=len(jobs), desc=f"building {split}") if tqdm is not None else None
            with ProcessPoolExecutor(max_workers=args.decode_workers) as executor:
                futures = [executor.submit(process_video_windows, job) for job in jobs]
                for future in as_completed(futures):
                    result = future.result()
                    failures.extend(result["failures"])
                    flush_result(result, labels, joint_xy, skeleton_x, skeleton_y, pending, manifest, args, joint_dtype)
                    if progress_bar is not None:
                        progress_bar.update(1)
            if progress_bar is not None:
                progress_bar.close()
        flush_tensor_batch(pending, tensor, args)
        pending.clear()
        del args._tensor_memmap

    np.save(out_dir / f"{split}_sample_names.npy", np.array([spec.sample_name for spec in specs], dtype=object))
    np.save(out_dir / f"{split}_skeleton_sample_names.npy", np.array([spec.skeleton_sample_name for spec in specs], dtype=object))
    np.save(out_dir / f"{split}_skeleton_y.npy", skeleton_y)
    with open(out_dir / f"{split}_windows.jsonl", "w", encoding="utf-8") as handle:
        for spec in specs:
            payload = asdict(spec)
            payload["sample_frames"] = list(spec.sample_frames)
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    if failures:
        (out_dir / f"{split}_failures.json").write_text(json.dumps(failures, indent=2, ensure_ascii=False), encoding="utf-8")
    del tensor, labels, joint_xy, skeleton_x
    return failures


def flush_tensor_batch(pending, tensor, args):
    if not pending:
        return
    clips = [item[1] for item in pending]
    already_resized = all(clip.shape[1:3] == (args.size, args.size) for clip in clips)
    if already_resized:
        batch = preprocess_resized_batch(clips, args.cache_dtype)
    else:
        batch = preprocess_batch(clips, args.size, args.resize_device, args.cache_dtype)
    for offset, (idx, _clip) in enumerate(pending):
        tensor[idx] = batch[offset]


def write_skeleton_npz(out_dir, splits):
    payload = {}
    for split, specs in splits.items():
        payload[f"x_{split}"] = np.load(out_dir / f"{split}_skeleton_x.npy", mmap_mode="r")
        payload[f"y_{split}"] = np.load(out_dir / f"{split}_skeleton_y.npy", mmap_mode="r")
        payload[f"{split}_sample_name"] = np.load(out_dir / f"{split}_skeleton_sample_names.npy", allow_pickle=True)
        payload[f"{split}_subject"] = np.array([spec.subject_id for spec in specs], dtype=object)
        payload[f"{split}_action"] = np.array([spec.action_id for spec in specs], dtype=object)
        payload[f"{split}_frames"] = np.full(len(specs), 13, dtype=np.int64)
    np.savez_compressed(out_dir / "windowed_skeleton_uniform13.npz", **payload)


def main():
    args = parse_args()
    args.exclude_action = [normalize_action_id(item) for item in (args.exclude_action or [])]
    args.include_action = [normalize_action_id(item) for item in (args.include_action or [])]
    args.anchor_first_frame_actions = [normalize_action_id(item) for item in (args.anchor_first_frame_actions or [])]
    out_dir = Path(args.out_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{out_dir} is not empty. Pass --overwrite to replace files.")
    out_dir.mkdir(parents=True, exist_ok=True)

    reference = load_reference_metadata(args.reference_metadata)
    split_specs, planning_failures = collect_split_specs(args, reference)
    if args.plan_only:
        bytes_per_value = 2 if args.cache_dtype == "float16" else 1
        tensor_values_per_window = 3 * args.frames * args.size * args.size
        estimated_tensor_bytes = {
            split: int(len(specs) * tensor_values_per_window * bytes_per_value)
            for split, specs in split_specs.items()
        }
        metadata = {
            "name": "windowed_2s1s_13f_plan",
            "source_root": str(Path(args.src_root)),
            "reference_metadata": str(Path(args.reference_metadata)),
            "window_seconds": args.window_seconds,
            "stride_seconds": args.stride_seconds,
            "frames_per_window": args.frames,
            "shape": {split: [len(specs), 3, args.frames, args.size, args.size] for split, specs in split_specs.items()},
            "estimated_tensor_bytes": estimated_tensor_bytes,
            "estimated_tensor_total_bytes": int(sum(estimated_tensor_bytes.values())),
            "classes": reference["classes"],
            "class_to_idx": reference["class_to_idx"],
            "excluded_actions": sorted(set(args.exclude_action or reference["excluded_actions"])),
            "included_actions": sorted(set(args.include_action or [])),
            "anchor_first_frame_actions": sorted(set(args.anchor_first_frame_actions or [])),
            "split_subjects": reference["split_subjects"],
            "failures": {"planning": len(planning_failures)},
        }
        (out_dir / "plan_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        for split, specs in split_specs.items():
            with open(out_dir / f"{split}_windows.plan.jsonl", "w", encoding="utf-8") as handle:
                for spec in specs:
                    payload = asdict(spec)
                    payload["sample_frames"] = list(spec.sample_frames)
                    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        return

    requested_splits = list(dict.fromkeys(args.splits))
    all_failures = {"planning": planning_failures}
    for split, specs in split_specs.items():
        if split in requested_splits:
            all_failures[split] = write_split(split, specs, args, out_dir)
    write_skeleton_npz(out_dir, split_specs)

    metadata = {
        "name": "windowed_2s1s_13f",
        "source_root": str(Path(args.src_root)),
        "reference_metadata": str(Path(args.reference_metadata)),
        "window_seconds": args.window_seconds,
        "stride_seconds": args.stride_seconds,
        "frames_per_window": args.frames,
        "shape": {split: [len(specs), 3, args.frames, args.size, args.size] for split, specs in split_specs.items()},
        "classes": reference["classes"],
        "class_to_idx": reference["class_to_idx"],
        "excluded_actions": sorted(set(args.exclude_action or reference["excluded_actions"])),
        "included_actions": sorted(set(args.include_action or [])),
        "anchor_first_frame_actions": sorted(set(args.anchor_first_frame_actions or [])),
        "split_rule": reference["split_rule"],
        "split_mode": reference["split_mode"],
        "split_seed": reference["split_seed"],
        "split_subjects": reference["split_subjects"],
        "cache_dtype": args.cache_dtype,
        "joint_xy_dtype": args.joint_xy_dtype,
        "files": {
            "skeleton_npz": "windowed_skeleton_uniform13.npz",
            "per_split": {
                split: {
                    "tensor": f"{split}_{args.cache_dtype}.npy" if args.cache_dtype == "float16" else f"{split}_uint8.npy",
                    "labels": f"{split}_labels.npy",
                    "sample_names": f"{split}_sample_names.npy",
                    "joint_xy": f"{split}_joint_xy.npy",
                    "manifest": f"{split}_manifest.txt",
                    "windows": f"{split}_windows.jsonl",
                }
                for split in split_specs
            },
        },
        "failures": {key: len(value) for key, value in all_failures.items()},
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "failures.json").write_text(json.dumps(all_failures, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
