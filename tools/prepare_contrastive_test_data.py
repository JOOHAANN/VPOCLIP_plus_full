#!/usr/bin/env python
"""Prepare an aligned held-out test split for CLIPGCN trimodal data.

The existing contrastive_train_data directory is aligned after several filters:
X3D valid video rows, skeleton availability, and matching labels. This script
applies the same idea to the X3D subject-held-out split and writes a new
CLIPGCN data directory that can be used as the stable basis for test features.
"""

import argparse
import csv
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--data-root", default="/workspace/CLIPGCN/data")
    parser.add_argument("--x3d-data-dir", default="/workspace/X3D/data/clipgcn_tensor_cs_70_10_20")
    parser.add_argument("--skeleton-npz", default="/workspace/CTR-GCN/data/etri/ETRI_P1_P230_CS_raw_uniform13.npz")
    parser.add_argument("--output-dir", default="/workspace/CLIPGCN/data/contrastive_test_data")
    parser.add_argument("--prefix", default="trimodal_test")
    parser.add_argument("--frames", type=int, default=13)
    parser.add_argument("--joints", type=int, default=25)
    parser.add_argument("--depth-width", type=float, default=512.0)
    parser.add_argument("--depth-height", type=float, default=424.0)
    parser.add_argument("--joint-xy-dtype", choices=("float16", "float32"), default="float32")
    parser.add_argument("--skip-joint-xy", action="store_true")
    parser.add_argument(
        "--video-feature",
        default=None,
        help="Optional extracted X3D feature file for this split. Rows must match X3D valid rows.",
    )
    parser.add_argument(
        "--object-file",
        default=None,
        help="Optional YOLO object file for this split. Rows must match original X3D split rows.",
    )
    parser.add_argument(
        "--pose-feature",
        default=None,
        help="Optional CTR-GCN pose feature file aligned by --pose-sample-names.",
    )
    parser.add_argument("--pose-labels", default=None)
    parser.add_argument("--pose-sample-names", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def action_id(path):
    match = re.match(r"(A\d+)_", Path(path).name)
    if match is None:
        raise ValueError(f"Cannot parse action id from {path}")
    return match.group(1)


def video_name_from_csv(rel_csv):
    return str(Path(str(rel_csv)).with_suffix(".mp4"))


def csv_name_from_video(rel_video):
    return str(Path(str(rel_video)).with_suffix(".csv"))


def reconstruct_split_videos(metadata_path, split):
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    src_root = Path(metadata["source_root"])
    subjects = set(metadata["split_subjects"][split])
    excluded_actions = set(metadata.get("excluded_actions") or [])
    class_to_idx = metadata["class_to_idx"]

    videos = sorted(
        path
        for path in src_root.glob("P*/**/*.mp4")
        if action_id(path) not in excluded_actions and path.relative_to(src_root).parts[0] in subjects
    )
    rel_videos = np.array([str(path.relative_to(src_root)) for path in videos], dtype=object)
    expected_labels = np.array([class_to_idx[action_id(path)] for path in videos], dtype=np.int64)
    return metadata, rel_videos, expected_labels


def load_skeleton_lookup(npz_path):
    npz = np.load(npz_path, allow_pickle=True)
    names = np.concatenate([npz["train_sample_name"], npz["test_sample_name"]], axis=0)
    x = np.concatenate([npz["x_train"], npz["x_test"]], axis=0)
    y = np.concatenate([npz["y_train"], npz["y_test"]], axis=0)
    subjects = np.concatenate([npz["train_subject"], npz["test_subject"]], axis=0)
    actions = np.concatenate([npz["train_action"], npz["test_action"]], axis=0)
    frames = np.concatenate([npz["train_frames"], npz["test_frames"]], axis=0)
    by_video = {video_name_from_csv(name): idx for idx, name in enumerate(names)}
    return by_video, {
        "names": names,
        "x": x,
        "y": y,
        "subjects": subjects,
        "actions": actions,
        "frames": frames,
    }


def joint_columns(header, joints):
    index = {name: idx for idx, name in enumerate(header)}
    columns = []
    for joint in range(1, joints + 1):
        columns.append((index[f"joint{joint}_depthX"], index[f"joint{joint}_depthY"]))
    frame_col = index["frameNum"]
    tracking_col = index.get("trackingID")
    body_col = index.get("bodyindexID")
    state_cols = [index.get(f"joint{joint}_trackingState") for joint in range(1, joints + 1)]
    return frame_col, tracking_col, body_col, columns, state_cols


def read_main_track_depth_xy(csv_path, joints):
    tracks = defaultdict(dict)
    tracked_score = Counter()

    with open(csv_path, "r", newline="", errors="replace") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        frame_col, tracking_col, body_col, columns, state_cols = joint_columns(header, joints)

        for row in reader:
            if len(row) < len(header):
                continue
            try:
                frame = int(float(row[frame_col]))
            except ValueError:
                continue

            if tracking_col is not None and row[tracking_col] != "":
                track_id = row[tracking_col]
            elif body_col is not None and row[body_col] != "":
                track_id = f"body_{row[body_col]}"
            else:
                track_id = "single"

            xy = np.zeros((joints, 2), dtype=np.float32)
            valid_value = False
            for joint_idx, (x_col, y_col) in enumerate(columns):
                try:
                    x = float(row[x_col])
                    y = float(row[y_col])
                except ValueError:
                    x = y = 0.0
                xy[joint_idx] = (x, y)
                valid_value = valid_value or x != 0.0 or y != 0.0

                state_col = state_cols[joint_idx]
                if state_col is not None:
                    try:
                        tracked_score[track_id] += int(float(row[state_col]))
                    except ValueError:
                        pass

            if valid_value:
                tracks[track_id][frame] = xy

    if not tracks:
        raise ValueError(f"No valid skeleton track in {csv_path}")

    def rank_track(item):
        track_id, frames = item
        return (len(frames), tracked_score[track_id], str(track_id))

    _track_id, frame_to_xy = max(tracks.items(), key=rank_track)
    frames = sorted(frame_to_xy)
    return np.stack([frame_to_xy[frame] for frame in frames], axis=0).astype(np.float32)


def uniform_resample(sequence, frames):
    if sequence.shape[0] <= 0:
        raise ValueError("Cannot resample empty sequence")
    indices = np.linspace(0, sequence.shape[0] - 1, frames).round().astype(np.int64)
    return sequence[indices]


def normalize_depth_xy(xy, width, height):
    out = np.nan_to_num(xy.copy(), nan=0.0, posinf=width - 1.0, neginf=0.0)
    out[..., 0] = (out[..., 0] / (width - 1.0)) * 2.0 - 1.0
    out[..., 1] = (out[..., 1] / (height - 1.0)) * 2.0 - 1.0
    return np.clip(out, -1.0, 1.0)


def write_joint_xy(sample_names, args, output_path, metadata_path):
    output_dtype = np.float16 if args.joint_xy_dtype == "float16" else np.float32
    data_root = Path(args.data_root)
    joint_xy = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=output_dtype,
        shape=(len(sample_names), args.frames, args.joints, 2),
    )

    raw_min = np.array([np.inf, np.inf], dtype=np.float64)
    raw_max = np.array([-np.inf, -np.inf], dtype=np.float64)
    norm_min = np.array([np.inf, np.inf], dtype=np.float64)
    norm_max = np.array([-np.inf, -np.inf], dtype=np.float64)
    nonfinite_values = 0
    clipped_values = 0
    total_values = 0
    failures = []

    for idx, name in enumerate(sample_names):
        csv_path = data_root / csv_name_from_video(name)
        try:
            xy = read_main_track_depth_xy(csv_path, args.joints)
            xy = uniform_resample(xy, args.frames)
            flat_xy = xy.reshape(-1, 2)
            finite_mask = np.isfinite(flat_xy)
            nonfinite_values += int((~finite_mask).sum())
            finite_xy = flat_xy[np.all(finite_mask, axis=1)]
            if finite_xy.size:
                raw_min = np.minimum(raw_min, finite_xy.min(axis=0))
                raw_max = np.maximum(raw_max, finite_xy.max(axis=0))
            normalized = normalize_depth_xy(xy, args.depth_width, args.depth_height)
            clipped_values += int(((normalized <= -1.0) | (normalized >= 1.0)).sum())
            total_values += int(normalized.size)
            norm_min = np.minimum(norm_min, normalized.reshape(-1, 2).min(axis=0))
            norm_max = np.maximum(norm_max, normalized.reshape(-1, 2).max(axis=0))
            joint_xy[idx] = normalized.astype(output_dtype, copy=False)
        except Exception as exc:  # Keep alignment metadata useful if a CSV is bad.
            failures.append({"index": int(idx), "sample_name": str(name), "error": str(exc)})
            joint_xy[idx] = 0
        if (idx + 1) % 500 == 0:
            print(f"joint_xy saved {idx + 1}/{len(sample_names)}", flush=True)

    del joint_xy
    metadata = {
        "output": str(output_path),
        "shape": [int(len(sample_names)), args.frames, args.joints, 2],
        "dtype": args.joint_xy_dtype,
        "source": "ETRI skeleton CSV joint*_depthX/depthY",
        "data_root": str(data_root),
        "coordinate_order": ["x", "y"],
        "normalization": {
            "range": [-1.0, 1.0],
            "depth_width": args.depth_width,
            "depth_height": args.depth_height,
            "formula": "x_norm = 2*x/(width-1)-1; y_norm = 2*y/(height-1)-1; clipped to [-1,1]",
        },
        "raw_depth_xy_min": raw_min.tolist(),
        "raw_depth_xy_max": raw_max.tolist(),
        "normalized_xy_min": norm_min.tolist(),
        "normalized_xy_max": norm_max.tolist(),
        "nonfinite_raw_values": nonfinite_values,
        "clipped_normalized_values": clipped_values,
        "total_coordinate_values": total_values,
        "clipped_value_fraction": clipped_values / total_values if total_values else 0.0,
        "failed_rows": failures,
        "alignment_note": "Rows match labels/sample_names/alignment outputs from prepare_contrastive_test_data.py.",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return metadata


def copy_feature_rows(src_path, dst_path, rows):
    src = np.load(src_path, mmap_mode="r", allow_pickle=False)
    dst = np.lib.format.open_memmap(dst_path, mode="w+", dtype=src.dtype, shape=(len(rows),) + src.shape[1:])
    chunk = 128
    for start in range(0, len(rows), chunk):
        end = min(start + chunk, len(rows))
        dst[start:end] = src[rows[start:end]]
    del dst
    return {"shape": [int(len(rows)), *map(int, src.shape[1:])], "dtype": str(src.dtype)}


def copy_object_rows(src_path, dst_path, rows, sample_names, labels):
    value = np.load(src_path, allow_pickle=True)
    if isinstance(value, np.ndarray) and value.ndim == 4:
        dst_meta = copy_feature_rows(src_path, dst_path, rows)
        return {"format": "array", **dst_meta}

    payload = value.item()
    out = {}
    row_count = None
    for key, item in payload.items():
        if isinstance(item, np.ndarray) and row_count is None and item.shape[:1] == (len(rows),):
            row_count = len(rows)
    for key, item in payload.items():
        if isinstance(item, np.ndarray) and item.shape[:1] == (max(rows) + 1,):
            out[key] = item[rows]
        else:
            out[key] = item
    out["manifest"] = np.array([[name, int(label)] for name, label in zip(sample_names, labels)], dtype=object)
    out["aligned_sample_names"] = sample_names.astype(object)
    out["alignment_note"] = "Filtered to rows that passed X3D label and skeleton alignment."
    np.save(dst_path, out, allow_pickle=True)
    return {"format": "dict", "keys": sorted(out.keys())}


def copy_pose_by_names(feature_path, labels_path, names_path, dst_path, aligned_names, labels):
    features = np.load(feature_path, mmap_mode="r", allow_pickle=False)
    pose_labels = np.load(labels_path, mmap_mode="r", allow_pickle=False)
    pose_names = np.load(names_path, allow_pickle=True)
    lookup = {video_name_from_csv(name): idx for idx, name in enumerate(pose_names)}

    rows = []
    missing = []
    for name in aligned_names:
        idx = lookup.get(str(name))
        if idx is None:
            missing.append(str(name))
        else:
            rows.append(idx)
    if missing:
        raise RuntimeError(f"Pose feature file is missing {len(missing)} aligned samples; first={missing[:5]}")

    rows = np.asarray(rows, dtype=np.int64)
    if not np.array_equal(pose_labels[rows].astype(np.int64), labels.astype(np.int64)):
        raise RuntimeError("Pose labels do not match aligned labels")
    if feature_path.resolve() == dst_path.resolve():
        return {"shape": list(features.shape), "dtype": str(features.dtype)}
    return copy_feature_rows(feature_path, dst_path, rows)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{output_dir} is not empty. Pass --overwrite to replace files.")
    output_dir.mkdir(parents=True, exist_ok=True)

    x3d_data_dir = Path(args.x3d_data_dir)
    metadata, rel_videos, expected_labels = reconstruct_split_videos(x3d_data_dir / "metadata.json", args.split)
    labels = np.load(x3d_data_dir / f"{args.split}_labels.npy", mmap_mode="r")
    if len(rel_videos) != len(labels):
        raise RuntimeError(f"Reconstructed {len(rel_videos)} videos but labels have {len(labels)} rows")
    valid_indices = np.flatnonzero(labels >= 0)
    if not np.array_equal(expected_labels[valid_indices], labels[valid_indices]):
        raise RuntimeError("Reconstructed X3D video order does not match labels")

    skel_by_video, skel = load_skeleton_lookup(args.skeleton_npz)
    keep_feature_rows = []
    keep_original_rows = []
    keep_skeleton_rows = []
    missing_skeleton = []
    missing_csv = []
    data_root = Path(args.data_root)

    for x3d_feature_row, original_row in enumerate(valid_indices):
        rel_video = str(rel_videos[original_row])
        skel_row = skel_by_video.get(rel_video)
        if skel_row is None:
            missing_skeleton.append(rel_video)
            continue
        if not (data_root / csv_name_from_video(rel_video)).exists():
            missing_csv.append(rel_video)
            continue
        keep_feature_rows.append(int(x3d_feature_row))
        keep_original_rows.append(int(original_row))
        keep_skeleton_rows.append(int(skel_row))

    keep_feature_rows = np.asarray(keep_feature_rows, dtype=np.int64)
    keep_original_rows = np.asarray(keep_original_rows, dtype=np.int64)
    keep_skeleton_rows = np.asarray(keep_skeleton_rows, dtype=np.int64)
    aligned_names = rel_videos[keep_original_rows].astype(object)
    aligned_labels = labels[keep_original_rows].astype(np.int64)
    skeleton_labels = np.where(skel["y"][keep_skeleton_rows] > 0)[1]
    if not np.array_equal(aligned_labels, skeleton_labels):
        mismatch = np.flatnonzero(aligned_labels != skeleton_labels)[:10]
        raise RuntimeError(f"Skeleton labels do not match X3D labels at aligned rows {mismatch.tolist()}")

    prefix = args.prefix
    labels_path = output_dir / f"{prefix}_labels.npy"
    names_path = output_dir / f"{prefix}_sample_names.npy"
    alignment_path = output_dir / f"{prefix}_alignment.npz"
    skeleton_path = output_dir / f"{prefix}_skeleton_aligned.npz"
    np.save(labels_path, aligned_labels)
    np.save(names_path, aligned_names)
    np.savez(
        alignment_path,
        x3d_feature_rows=keep_feature_rows,
        x3d_original_rows=keep_original_rows,
        skeleton_rows=keep_skeleton_rows,
        sample_names=aligned_names,
        labels=aligned_labels,
        missing_skeleton=np.array(missing_skeleton, dtype=object),
        missing_csv=np.array(missing_csv, dtype=object),
    )
    np.savez_compressed(
        skeleton_path,
        x_train=skel["x"][keep_skeleton_rows],
        y_train=skel["y"][keep_skeleton_rows],
        train_sample_name=np.array([csv_name_from_video(name) for name in aligned_names], dtype=object),
        train_subject=skel["subjects"][keep_skeleton_rows],
        train_action=skel["actions"][keep_skeleton_rows],
        train_frames=skel["frames"][keep_skeleton_rows],
    )

    files = {
        "labels": labels_path.name,
        "sample_names": names_path.name,
        "alignment": alignment_path.name,
        "aligned_skeleton_npz": skeleton_path.name,
    }
    shapes = {
        "labels": list(aligned_labels.shape),
        "sample_names": list(aligned_names.shape),
        "aligned_skeleton_x_train": list(skel["x"][keep_skeleton_rows].shape),
    }
    dtypes = {
        "labels": str(aligned_labels.dtype),
        "sample_names": str(aligned_names.dtype),
        "aligned_skeleton_x_train": str(skel["x"].dtype),
    }

    if not args.skip_joint_xy:
        joint_path = output_dir / f"{prefix}_joint_xy.npy"
        joint_meta_path = output_dir / f"{prefix}_joint_xy.metadata.json"
        joint_meta = write_joint_xy(aligned_names, args, joint_path, joint_meta_path)
        files["joint_xy"] = joint_path.name
        files["joint_xy_metadata"] = joint_meta_path.name
        shapes["joint_xy"] = joint_meta["shape"]
        dtypes["joint_xy"] = joint_meta["dtype"]

    if args.video_feature:
        video_path = output_dir / f"{prefix}_video.npy"
        video_meta = copy_feature_rows(Path(args.video_feature), video_path, keep_feature_rows)
        files["video"] = video_path.name
        shapes["video"] = video_meta["shape"]
        dtypes["video"] = video_meta["dtype"]

    if args.object_file:
        object_path = output_dir / f"{prefix}_object.npy"
        object_meta = copy_object_rows(Path(args.object_file), object_path, keep_original_rows, aligned_names, aligned_labels)
        files["object"] = object_path.name
        files["object_metadata"] = object_meta

    if args.pose_feature:
        if not args.pose_labels or not args.pose_sample_names:
            raise ValueError("--pose-feature requires --pose-labels and --pose-sample-names")
        pose_path = output_dir / f"{prefix}_pose.npy"
        pose_meta = copy_pose_by_names(
            Path(args.pose_feature),
            Path(args.pose_labels),
            Path(args.pose_sample_names),
            pose_path,
            aligned_names,
            aligned_labels,
        )
        files["pose"] = pose_path.name
        shapes["pose"] = pose_meta["shape"]
        dtypes["pose"] = pose_meta["dtype"]

    manifest_dst = output_dir / f"{prefix}_manifest.txt"
    with open(manifest_dst, "w", encoding="utf-8") as handle:
        for name, label in zip(aligned_names, aligned_labels):
            handle.write(f"{name} {int(label)}\n")
    files["manifest"] = manifest_dst.name

    readme = output_dir / "README.md"
    readme.write_text(
        "# CLIPGCN held-out test data\n\n"
        "This directory is generated by `tools/prepare_contrastive_test_data.py`.\n"
        "Rows are filtered so labels, sample names, skeleton rows, and joint xy are aligned.\n"
        "Video/pose/object feature arrays are added only when their source feature files are supplied.\n",
        encoding="utf-8",
    )

    summary = {
        "name": prefix,
        "description": "Subject-held-out CLIPGCN test split aligned across available modalities.",
        "split": args.split,
        "num_samples": int(len(aligned_labels)),
        "class_ids": sorted(int(x) for x in np.unique(aligned_labels).tolist()),
        "files": files,
        "shapes": shapes,
        "dtypes": dtypes,
        "source": {
            "x3d_data_dir": str(x3d_data_dir),
            "x3d_metadata": str(x3d_data_dir / "metadata.json"),
            "skeleton_npz": str(Path(args.skeleton_npz)),
            "data_root": str(data_root),
            "split_rule": metadata.get("split_rule"),
            "split_mode": metadata.get("split_mode"),
            "split_seed": metadata.get("split_seed"),
            "split_subjects": metadata.get("split_subjects", {}).get(args.split),
        },
        "alignment": {
            "x3d_total_rows": int(len(rel_videos)),
            "x3d_valid_rows": int(len(valid_indices)),
            "aligned_rows": int(len(aligned_labels)),
            "filtered_bad_label_rows": int(len(rel_videos) - len(valid_indices)),
            "missing_skeleton_rows": int(len(missing_skeleton)),
            "missing_csv_rows": int(len(missing_csv)),
            "missing_skeleton": missing_skeleton,
            "missing_csv": missing_csv,
        },
        "feature_status": {
            "video": "present" if "video" in files else "missing_source_feature",
            "pose": "present" if "pose" in files else "missing_source_feature",
            "object": "present" if "object" in files else "missing_source_feature",
            "joint_xy": "present" if "joint_xy" in files else "skipped",
        },
        "checks": {
            "x3d_reconstructed_labels_match": True,
            "skeleton_labels_match": True,
            "row_count_equal_for_written_core_files": True,
        },
    }
    metadata_path = output_dir / f"{prefix}_metadata.json"
    metadata_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    shutil.copy2(metadata_path, output_dir / "metadata.json")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
