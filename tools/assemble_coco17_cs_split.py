#!/usr/bin/env python
"""Assemble X3D and object features using the exact RTMPose sample order."""

import argparse
import json
from pathlib import Path

import numpy as np


RTMPOSE_NPZ = Path("/workspace/CTR-GCN_17/data/etri_coco17/ETRI_55_CS_rtmpose_coco17_13.npz")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tensor-dir", required=True)
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--object-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--full-tensor-dir", default="/workspace/X3D/data/clipgcn_tensor_cs_70_10_20")
    parser.add_argument("--chunk-size", type=int, default=256)
    return parser.parse_args()


def copy_rows(source_path, output_path, rows, chunk_size):
    source = np.load(source_path, mmap_mode="r")
    output = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=source.dtype,
        shape=(len(rows), *source.shape[1:]),
    )
    for start in range(0, len(rows), chunk_size):
        stop = min(start + chunk_size, len(rows))
        output[start:stop] = source[rows[start:stop]]
    del output


def align_source_to_rtmpose(full_labels, rtmpose_labels):
    """Map original tensor rows to RTMPose rows; RTMPose omits five bad clips."""
    valid_rows = np.flatnonzero(full_labels >= 0)
    source_to_pose = {}
    pose_row = 0
    skipped = []
    for source_row in valid_rows:
        if pose_row < len(rtmpose_labels) and full_labels[source_row] == rtmpose_labels[pose_row]:
            source_to_pose[int(source_row)] = pose_row
            pose_row += 1
        else:
            skipped.append(int(source_row))
    if pose_row != len(rtmpose_labels):
        raise ValueError(
            f"RTMPose label sequence is not a subsequence of X3D labels: matched {pose_row}/{len(rtmpose_labels)}"
        )
    return source_to_pose, skipped


def main():
    args = parse_args()
    tensor_dir = Path(args.tensor_dir)
    feature_dir = Path(args.feature_dir)
    object_dir = Path(args.object_dir)
    output_dir = Path(args.output_dir)
    full_tensor_dir = Path(args.full_tensor_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pose_cache = np.load(RTMPOSE_NPZ, allow_pickle=True)
    prefixes = {
        "train": "trimodal_train",
        "val": "trimodal_val",
        "test": "trimodal_test",
        "test_seen": "trimodal_test_seen",
    }
    split_results = {}
    for split, prefix in prefixes.items():
        source_split = "test" if split.startswith("test") else split
        source_indices = np.load(tensor_dir / f"{split}_source_indices.npy")
        target_labels = np.load(tensor_dir / f"{split}_labels.npy")
        full_labels = np.load(full_tensor_dir / f"{source_split}_labels.npy")
        pose_labels = np.asarray(pose_cache[f"y_{source_split}"], dtype=np.int64)
        pose_names = np.asarray(pose_cache[f"{source_split}_sample_name"], dtype=object)
        source_to_pose, skipped_source_rows = align_source_to_rtmpose(full_labels, pose_labels)

        keep_target_rows = np.asarray(
            [row for row, source_row in enumerate(source_indices) if int(source_row) in source_to_pose],
            dtype=np.int64,
        )
        pose_rows = np.asarray(
            [source_to_pose[int(source_indices[row])] for row in keep_target_rows], dtype=np.int64
        )
        labels = target_labels[keep_target_rows].astype(np.int64)
        if not np.array_equal(labels, pose_labels[pose_rows]):
            raise ValueError(f"{split}: X3D and RTMPose labels do not align")

        feature_path = feature_dir / f"{split}_res5_model_final.npy"
        feature_labels = np.load(feature_path.with_suffix(".labels.npy"))
        if not np.array_equal(feature_labels.astype(np.int64), target_labels.astype(np.int64)):
            raise ValueError(f"{split}: extracted X3D labels differ from tensor labels")

        copy_rows(feature_path, output_dir / f"{prefix}_video.npy", keep_target_rows, args.chunk_size)
        copy_rows(
            object_dir / f"{split}_frame7_yolov5m_objects.npy",
            output_dir / f"{prefix}_object.npy",
            keep_target_rows,
            args.chunk_size,
        )
        np.save(output_dir / f"{prefix}_labels.npy", labels)
        np.save(output_dir / f"{prefix}_sample_names.npy", pose_names[pose_rows].astype(object))
        np.savez_compressed(
            output_dir / f"{prefix}_alignment.npz",
            target_rows=keep_target_rows,
            source_rows=source_indices[keep_target_rows],
            pose_rows=pose_rows,
            labels=labels,
            sample_names=pose_names[pose_rows].astype(object),
        )
        split_results[split] = {
            "num_samples": int(len(labels)),
            "dropped_missing_rtmpose": int(len(target_labels) - len(labels)),
            "class_ids": sorted(np.unique(labels).astype(int).tolist()),
            "source_rows_without_rtmpose": skipped_source_rows,
        }
        print(
            f"{prefix}: {len(labels)} samples; dropped {len(target_labels) - len(labels)} without RTMPose",
            flush=True,
        )

    tensor_metadata = json.loads((tensor_dir / "metadata.json").read_text(encoding="utf-8"))
    metadata = {
        "name": output_dir.name,
        "description": "Current X3D-S, YOLOv5m RS and RTMPose COCO-17/CTR-GCN feature pipeline.",
        "subject_protocol": tensor_metadata.get("split_subjects"),
        "split_seed": tensor_metadata.get("split_seed"),
        "seen_classes": tensor_metadata["seen_classes"],
        "unseen_classes": tensor_metadata["unseen_classes"],
        "label_ids_are_original": True,
        "splits": split_results,
        "sources": {
            "tensor_dir": str(tensor_dir.resolve()),
            "x3d_features": str(feature_dir.resolve()),
            "object_maps": str(object_dir.resolve()),
            "rtmpose": str(RTMPOSE_NPZ),
        },
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
