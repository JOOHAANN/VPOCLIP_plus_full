#!/usr/bin/env python
"""Extract aligned 2D joint coordinates for the trimodal train split.

The source ETRI skeleton CSVs contain both 3D joint coordinates and depth-image
2D coordinates. This script exports joint*_depthX/depthY for the already aligned
trimodal sample order and normalizes them to [-1, 1].
"""

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Export normalized [B,13,25,2] joint xy coordinates.")
    parser.add_argument("--data-root", default="/workspace/CLIPGCN/data")
    parser.add_argument("--sample-names", default="/workspace/CLIPGCN/data/contrastive_train_data/trimodal_train_sample_names.npy")
    parser.add_argument("--labels", default="/workspace/CLIPGCN/data/contrastive_train_data/trimodal_train_labels.npy")
    parser.add_argument("--output", default="/workspace/CLIPGCN/data/contrastive_train_data/trimodal_train_joint_xy.npy")
    parser.add_argument("--metadata-output", default="/workspace/CLIPGCN/data/contrastive_train_data/trimodal_train_joint_xy.metadata.json")
    parser.add_argument("--frames", type=int, default=13)
    parser.add_argument("--joints", type=int, default=25)
    parser.add_argument("--depth-width", type=float, default=512.0)
    parser.add_argument("--depth-height", type=float, default=424.0)
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float32")
    return parser.parse_args()


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


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    sample_names = np.load(args.sample_names, allow_pickle=True)
    labels = np.load(args.labels, mmap_mode="r")
    output_path = Path(args.output)
    output_dtype = np.float16 if args.dtype == "float16" else np.float32

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

    for idx, name in enumerate(sample_names):
        csv_rel = str(Path(str(name)).with_suffix(".csv"))
        csv_path = data_root / csv_rel
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
        if (idx + 1) % 500 == 0:
            print(f"saved {idx + 1}/{len(sample_names)}", flush=True)

    del joint_xy

    metadata = {
        "output": str(output_path),
        "shape": [int(len(sample_names)), args.frames, args.joints, 2],
        "dtype": args.dtype,
        "source": "ETRI skeleton CSV joint*_depthX/depthY",
        "data_root": str(data_root),
        "sample_names": str(Path(args.sample_names)),
        "labels": str(Path(args.labels)),
        "labels_shape": list(labels.shape),
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
        "alignment_note": "Rows match trimodal_train_pose/video/object/labels/sample_names.",
    }
    Path(args.metadata_output).write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved {output_path}")
    print(f"saved {args.metadata_output}")


if __name__ == "__main__":
    main()
