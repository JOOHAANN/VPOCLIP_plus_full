"""Build causal per-frame object-position tracks for the active-view cache.

The ordinary RL cache stores one YOLO RS map from the middle frame of each
view.  That is useful as a static object sensor but it is not an object
trajectory.  This helper runs the same YOLO model on the 13 frames used by the
frozen VPOCLIP cache and writes, for every physical view, one row per sampled
frame and object class:

    [presence, x, y, confidence]

Coordinates are normalized to [-1, 1], with the same image convention as the
cached person keypoints (x right, y down).  If several detections of one class
occur in a frame, their confidence-weighted center is stored.  The cache is
sensor data only; it contains no VPOCLIP embedding, logits, or labels beyond
the ordering metadata needed to join it to the RL cache.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch


NUM_VIEWS = 4
NUM_FRAMES = 13
NUM_OBJECT_CLASSES = 50


def read_requested_frames(
    path: str | Path,
    requested: list[int],
    width: int = 640,
    height: int = 480,
) -> np.ndarray:
    """Decode exactly the requested window, resized to the deployment shape."""

    cap = cv2.VideoCapture(str(path))
    unique = sorted(set(max(0, int(x)) for x in requested))
    captured: dict[int, np.ndarray] = {}
    target = 0
    frame_index = 0
    while target < len(unique):
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index == unique[target]:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            captured[frame_index] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            target += 1
        frame_index += 1
    cap.release()
    if not captured:
        raise RuntimeError(f"could not decode any requested frame from {path}")
    # A few recordings report one more frame in metadata than OpenCV can
    # decode.  Match the existing VPOCLIP window reader: use the last decoded
    # frame for missing tail indices instead of failing the whole split.
    decoded_indices = sorted(captured)
    last_decoded = decoded_indices[-1]
    for index in unique:
        if index in captured:
            continue
        earlier = [item for item in decoded_indices if item <= index]
        source = earlier[-1] if earlier else decoded_indices[0]
        captured[index] = captured[source]
    return np.stack([captured[int(x)] for x in requested], axis=0)


def detect_track(
    model: Any,
    frames: list[np.ndarray],
    device: torch.device,
    frame_batch: int,
    include_box_size: bool = False,
) -> np.ndarray:
    """Run YOLO and return [T,50,C] object tracks.

    The historical four-channel format is preserved by default:
    ``[presence, center_x, center_y, confidence]``.  The optional six-channel
    format adds normalized bounding-box width and height:
    ``[presence, center_x, center_y, width, height, confidence]``.
    """

    channels = 6 if include_box_size else 4
    output = np.zeros((len(frames), NUM_OBJECT_CLASSES, channels), dtype=np.float32)
    class_to_slot = {index: index for index in range(NUM_OBJECT_CLASSES)}
    for begin in range(0, len(frames), frame_batch):
        chunk = frames[begin : begin + frame_batch]
        results = model(chunk, size=640)
        for local_index, detections in enumerate(results.xyxy):
            frame = chunk[local_index]
            height, width = frame.shape[:2]
            accum: dict[int, list[float]] = {}
            for detection in detections:
                class_id = int(detection[5].item())
                slot = class_to_slot.get(class_id)
                if slot is None:
                    continue
                x1, y1, x2, y2, confidence = (
                    float(value) for value in detection[:5].tolist()
                )
                weight = max(confidence, 1e-6)
                x = 2.0 * ((x1 + x2) * 0.5) / max(width, 1) - 1.0
                y = 2.0 * ((y1 + y2) * 0.5) / max(height, 1) - 1.0
                if slot not in accum:
                    accum[slot] = [0.0, 0.0, 0.0, 0.0, 0.0]
                accum[slot][0] += weight
                accum[slot][1] += weight * x
                accum[slot][2] += weight * y
                if include_box_size:
                    accum[slot][3] += weight * max(x2 - x1, 0.0) / max(width, 1)
                    accum[slot][4] += weight * max(y2 - y1, 0.0) / max(height, 1)
            for slot, (weight, x_sum, y_sum, width_sum, height_sum) in accum.items():
                if include_box_size:
                    output[begin + local_index, slot] = (
                        1.0,
                        x_sum / weight,
                        y_sum / weight,
                        width_sum / weight,
                        height_sum / weight,
                        min(weight, 1.0),
                    )
                else:
                    output[begin + local_index, slot] = (
                        1.0,
                        x_sum / weight,
                        y_sum / weight,
                        min(weight, 1.0),
                    )
    return output


def load_yolo(repo: Path, weights: Path, device: torch.device, half: bool) -> Any:
    model = torch.hub.load(
        str(repo), "custom", path=str(weights), source="local", verbose=False
    )
    model.to(device).eval()
    model.conf = 0.25
    model.iou = 0.45
    names = getattr(model, "names", None)
    if names is not None and len(names) != NUM_OBJECT_CLASSES:
        raise RuntimeError(
            f"expected the 50-class YOLO checkpoint, got {len(names)} classes"
        )
    if half and device.type == "cuda":
        model.half()
    return model


def build_split(
    split: str,
    cache_root: Path,
    output_root: Path,
    model: Any,
    device: torch.device,
    decode_workers: int,
    video_batch: int,
    frame_batch: int,
    max_episodes: int | None,
    include_box_size: bool = False,
) -> dict[str, Any]:
    metadata_path = cache_root / split / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    episodes = metadata["episodes"]
    if max_episodes is not None:
        episodes = episodes[: max(0, int(max_episodes))]
    output_root.mkdir(parents=True, exist_ok=True)
    final_path = output_root / "object_position_track.npy"
    done_path = output_root / "complete.json"
    progress_path = output_root / "progress.json"
    if done_path.exists() and final_path.exists():
        return json.loads(done_path.read_text(encoding="utf-8"))

    n = len(episodes)
    # open_memmap requires the temporary filename itself to end in .npy.
    temp_path = output_root / "object_position_track.tmp.npy"
    channels = 6 if include_box_size else 4
    shape = (n, NUM_VIEWS, NUM_FRAMES, NUM_OBJECT_CLASSES, channels)
    if temp_path.exists():
        existing = np.load(temp_path, mmap_mode="r", allow_pickle=False)
        if tuple(existing.shape) != shape:
            raise ValueError(f"existing partial track shape {existing.shape} != {shape}")
        tracks = np.lib.format.open_memmap(temp_path, mode="r+")
        if progress_path.exists():
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            resume_from = int(progress.get("episodes_done", 0))
        else:
            processed = np.any(existing, axis=(1, 2, 3, 4))
            resume_from = 0
            while resume_from < n and bool(processed[resume_from]):
                resume_from += 1
    else:
        tracks = np.lib.format.open_memmap(
            temp_path, mode="w+", dtype=np.float16, shape=shape
        )
        tracks[:] = 0
        resume_from = 0
    tracks.flush()
    started = time.monotonic()

    def decode_task(task: tuple[int, str, list[int]]) -> tuple[int, np.ndarray]:
        view_id, video, frames = task
        return view_id, read_requested_frames(video, frames)

    for start in range(resume_from, n, video_batch):
        batch = episodes[start : start + video_batch]
        tasks: list[tuple[int, str, list[int]]] = []
        for episode in batch:
            sample_frames = [int(x) for x in episode["window"]["sample_frames"]]
            for view_id, view in enumerate(episode.get("views", [])[:NUM_VIEWS]):
                tasks.append((view_id, str(view["video"]), sample_frames))
        decoded: dict[tuple[int, str], np.ndarray] = {}
        if decode_workers > 1:
            with ThreadPoolExecutor(max_workers=decode_workers) as pool:
                futures = [pool.submit(decode_task, task) for task in tasks]
                for task, future in zip(tasks, futures):
                    view_id, frames = future.result()
                    decoded[(view_id, task[1])] = frames
        else:
            for task in tasks:
                view_id, frames = decode_task(task)
                decoded[(view_id, task[1])] = frames

        # Run all decoded views in this batch through YOLO together.  The
        # policy remains causal; this only improves offline extraction
        # throughput and keeps the 13-frame sampling identical for each view.
        flat_frames: list[np.ndarray] = []
        task_offsets: list[tuple[int, int, int]] = []
        cursor = 0
        for task in tasks:
            frames = decoded[(task[0], task[1])]
            flat_frames.extend(list(frames))
            task_offsets.append((task[0], cursor, cursor + len(frames)))
            cursor += len(frames)
        flat_track = detect_track(
            model, flat_frames, device, frame_batch, include_box_size=include_box_size
        )
        task_track = {
            (view_id, tasks[index][1]): flat_track[begin:end]
            for index, (view_id, begin, end) in enumerate(task_offsets)
        }
        for offset, episode in enumerate(batch):
            for view_id, view in enumerate(episode.get("views", [])[:NUM_VIEWS]):
                track = task_track[(view_id, str(view["video"]))]
                tracks[start + offset, view_id] = track.astype(np.float16)
        tracks.flush()
        progress_path.write_text(
            json.dumps({"split": split, "episodes_done": start + len(batch)}),
            encoding="utf-8",
        )
        elapsed = max(time.monotonic() - started, 1e-6)
        done = start + len(batch)
        print(
            json.dumps(
                {
                    "split": split,
                    "episodes_done": done,
                    "episodes_total": n,
                    "episodes_per_second": done / elapsed,
                    "seconds": elapsed,
                }
            ),
            flush=True,
        )

    del tracks
    os.replace(temp_path, final_path)
    if progress_path.exists():
        progress_path.unlink()
    result = {
        "split": split,
        "episodes": n,
        "shape": [n, NUM_VIEWS, NUM_FRAMES, NUM_OBJECT_CLASSES, channels],
        "dtype": "float16",
        "frame_shape": [640, 480],
        "sample_frames_source": "cache metadata window.sample_frames",
        "fields": (
            ["presence", "x", "y", "width", "height", "confidence"]
            if include_box_size
            else ["presence", "x", "y", "confidence"]
        ),
        "box_size_normalization": (
            "width/frame_width, height/frame_height in [0,1]"
            if include_box_size
            else None
        ),
        "coordinate_convention": "normalized image x/y in [-1,1], y down",
        "detector": "YOLO custom50, conf=0.25, iou=0.45",
        "seconds": time.monotonic() - started,
        "output": str(final_path),
    }
    done_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, default=Path("data/rl_candidate_rank_v6_new50_5/cache"))
    parser.add_argument("--output-root", type=Path, default=Path("data/angle_object_trajectory_v1"))
    parser.add_argument("--splits", nargs="+", default=["dqn_train", "val", "seen_test", "test"])
    parser.add_argument("--yolo-repo", type=Path, default=Path("/home/youhan/ws/yolov5_full"))
    parser.add_argument("--yolo-weights", type=Path, default=Path("/home/youhan/ws/yolov5_full/best.pt"))
    parser.add_argument("--decode-workers", type=int, default=8)
    parser.add_argument("--video-batch", type=int, default=8)
    parser.add_argument("--frame-batch", type=int, default=64)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument(
        "--include-box-size",
        action="store_true",
        help="Append normalized YOLO bbox width/height; changes track shape from C=4 to C=6.",
    )
    parser.add_argument("--no-half", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for object trajectory extraction")
    device = torch.device("cuda:0")
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model = load_yolo(args.yolo_repo, args.yolo_weights, device, not args.no_half)
    for split in args.splits:
        result = build_split(
            split,
            args.cache_root,
            args.output_root / split,
            model,
            device,
            args.decode_workers,
            args.video_batch,
            args.frame_batch,
            args.max_episodes,
            args.include_box_size,
        )
        print("OBJECT_TRACK_COMPLETE", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
