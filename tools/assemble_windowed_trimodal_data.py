#!/usr/bin/env python
"""Assemble windowed X3D, CTR-GCN, object RS, and joint-xy features for CLIPGCN."""

import argparse
import json
import re
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window-dir", default="/workspace/CLIPGCN/data/windowed_2s1s_13f")
    parser.add_argument("--x3d-output-dir", default="/workspace/X3D/outputs/x3d-s_clipgcn_windowed_2s1s_13f")
    parser.add_argument("--pose-output-dir", default="/workspace/CTR-GCN/data/etri/windowed_2s1s_13f_features")
    parser.add_argument("--object-dir", default="/workspace/X3D/data/clipgcn_windowed_2s1s_13f")
    parser.add_argument("--output-dir", default="/workspace/CLIPGCN/data/contrastive_windowed_2s1s_13f")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--prefix-template", default="trimodal_{split}")
    parser.add_argument("--x3d-template", default="{split}_res5_model_007000.npy")
    parser.add_argument("--pose-template", default="{split}_l4_raw_NMCTV.npy")
    parser.add_argument("--object-template", default="{split}_frame7_yolov5m_objects.npy")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_array(path, mmap=True, allow_pickle=False):
    if allow_pickle:
        mmap = False
    mmap_mode = "r" if mmap else None
    return np.load(path, mmap_mode=mmap_mode, allow_pickle=allow_pickle)


def copy_rows(src, rows, dst_path):
    dst = np.lib.format.open_memmap(dst_path, mode="w+", dtype=src.dtype, shape=(len(rows),) + src.shape[1:])
    chunk = 256
    for start in range(0, len(rows), chunk):
        end = min(start + chunk, len(rows))
        dst[start:end] = src[rows[start:end]]
    del dst


def load_x3d_valid_rows(feature_path):
    labels_path = feature_path.with_suffix(".labels.npy")
    valid_path = feature_path.with_suffix(".valid_indices.npy")
    if not labels_path.exists() or not valid_path.exists():
        raise FileNotFoundError(
            f"Missing X3D sidecars for {feature_path}. Re-run extract_x3d_features.py with --save-sidecars."
        )
    return load_array(labels_path), load_array(valid_path)


def validate_labels(name, expected, actual):
    if not np.array_equal(expected.astype(np.int64), actual.astype(np.int64)):
        mismatch = np.flatnonzero(expected.astype(np.int64) != actual.astype(np.int64))[:10]
        raise RuntimeError(f"{name} labels do not match at rows {mismatch.tolist()}")


def infer_labels_from_sample_names(sample_names):
    labels = np.empty(len(sample_names), dtype=np.int64)
    for index, sample_name in enumerate(sample_names):
        match = re.search(r"A0*(\d+)", str(sample_name))
        if match is None:
            raise ValueError(f"Cannot infer action label from sample name: {sample_name}")
        labels[index] = int(match.group(1)) - 1
    return labels


def assemble_split(split, args, output_dir):
    window_dir = Path(args.window_dir)
    prefix = args.prefix_template.format(split=split)
    feature_path = Path(args.x3d_output_dir) / args.x3d_template.format(split=split)
    pose_path = Path(args.pose_output_dir) / args.pose_template.format(split=split)
    object_path = Path(args.object_dir) / args.object_template.format(split=split)

    source_labels = load_array(window_dir / f"{split}_labels.npy")
    source_names = load_array(window_dir / f"{split}_sample_names.npy", allow_pickle=True)
    inferred_labels = infer_labels_from_sample_names(source_names)
    valid_source_rows = source_labels.astype(np.int64) == inferred_labels
    joint_xy = load_array(window_dir / f"{split}_joint_xy.npy")
    video = load_array(feature_path)
    pose = load_array(pose_path)
    obj = load_array(object_path)
    x3d_labels, valid_rows = load_x3d_valid_rows(feature_path)
    keep_x3d_rows = valid_source_rows[valid_rows]
    if not np.all(keep_x3d_rows):
        dropped = int(len(keep_x3d_rows) - np.count_nonzero(keep_x3d_rows))
        print(f"Dropping {dropped} invalid {split} source rows whose labels do not match sample names.")
    video_rows = np.flatnonzero(keep_x3d_rows).astype(np.int64)
    valid_rows = valid_rows[keep_x3d_rows]
    x3d_labels = x3d_labels[keep_x3d_rows]

    if obj.ndim != 4:
        raise ValueError(f"{object_path} must already be converted to RS maps [N,50,6,6]; got shape {obj.shape}")
    validate_labels("X3D", inferred_labels[valid_rows], x3d_labels)
    pose_labels_path = pose_path.with_name(pose_path.stem + "_labels.npy")
    if pose_labels_path.exists():
        pose_labels = load_array(pose_labels_path)
        validate_labels("pose", inferred_labels[valid_rows], pose_labels[valid_rows])

    expected_feature_rows = len(video_rows)
    if len(video_rows) > len(video):
        raise RuntimeError(f"Filtered X3D row index exceeds feature rows: max={video_rows.max()}, rows={len(video)}")
    if len(pose) != len(source_labels) or len(obj) != len(source_labels) or len(joint_xy) != len(source_labels):
        raise RuntimeError(
            f"Source modality counts are not aligned for {split}: "
            f"pose={len(pose)}, object={len(obj)}, joint_xy={len(joint_xy)}, labels={len(source_labels)}"
        )

    copy_rows(video, video_rows, output_dir / f"{prefix}_video.npy")
    copy_rows(pose, valid_rows, output_dir / f"{prefix}_pose.npy")
    copy_rows(obj, valid_rows, output_dir / f"{prefix}_object.npy")
    copy_rows(joint_xy, valid_rows, output_dir / f"{prefix}_joint_xy.npy")
    np.save(output_dir / f"{prefix}_labels.npy", inferred_labels[valid_rows].astype(np.int64))
    np.save(output_dir / f"{prefix}_sample_names.npy", source_names[valid_rows].astype(object))
    np.savez_compressed(
        output_dir / f"{prefix}_alignment.npz",
        source_rows=valid_rows,
        x3d_rows=video_rows,
        sample_names=source_names[valid_rows].astype(object),
        labels=inferred_labels[valid_rows].astype(np.int64),
    )

    return {
        "num_samples": int(len(valid_rows)),
        "dropped_invalid_source_rows": int(len(keep_x3d_rows) - np.count_nonzero(keep_x3d_rows)),
        "class_ids": sorted(int(x) for x in np.unique(inferred_labels[valid_rows]).tolist()),
        "files": {
            "video": f"{prefix}_video.npy",
            "pose": f"{prefix}_pose.npy",
            "object": f"{prefix}_object.npy",
            "joint_xy": f"{prefix}_joint_xy.npy",
            "labels": f"{prefix}_labels.npy",
            "sample_names": f"{prefix}_sample_names.npy",
            "alignment": f"{prefix}_alignment.npz",
        },
        "source": {
            "window_dir": str(window_dir),
            "x3d_features": str(feature_path),
            "pose_features": str(pose_path),
            "object_maps": str(object_path),
        },
    }


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{output_dir} is not empty. Pass --overwrite to replace files.")
    output_dir.mkdir(parents=True, exist_ok=True)

    window_metadata = json.loads((Path(args.window_dir) / "metadata.json").read_text(encoding="utf-8"))
    splits = {}
    for split in args.splits:
        splits[split] = assemble_split(split, args, output_dir)

    metadata = {
        "name": "contrastive_windowed_2s1s_13f",
        "description": "Windowed trimodal features aligned by source window row.",
        "windowing": {
            "window_seconds": window_metadata.get("window_seconds"),
            "stride_seconds": window_metadata.get("stride_seconds"),
            "frames_per_window": window_metadata.get("frames_per_window"),
        },
        "source_split": {
            "split_rule": window_metadata.get("split_rule"),
            "split_mode": window_metadata.get("split_mode"),
            "split_seed": window_metadata.get("split_seed"),
            "split_subjects": window_metadata.get("split_subjects"),
        },
        "splits": splits,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
