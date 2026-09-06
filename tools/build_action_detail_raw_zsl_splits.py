#!/usr/bin/env python
"""Create seen/unseen raw X3D and CTR-GCN datasets from action-detail raw data."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


X3D_MODALITIES = ("float16", "labels", "sample_names")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-metadata", default="/workspace/CLIPGCN/data/contrastive_zsl_splits/50_5/metadata.json")
    parser.add_argument("--x3d-input-dir", default="/workspace/X3D/data/clipgcn_action_detail_raw_13f")
    parser.add_argument("--ctrgcn-input", default="/workspace/CTR-GCN/data/etri/ETRI_P1_P230_action_detail_raw_13f.npz")
    parser.add_argument("--x3d-output-dir", default="/workspace/X3D/data/clipgcn_action_detail_raw_13f_zsl_50_5")
    parser.add_argument("--ctrgcn-output", default="/workspace/CTR-GCN/data/etri/ETRI_P1_P230_action_detail_raw_13f_zsl_50_5.npz")
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_names(path: Path):
    return np.load(path, allow_pickle=True)


def copy_npy_rows(src_path: Path, dst_path: Path, rows: np.ndarray, chunk_size: int):
    allow_pickle = src_path.name.endswith("_sample_names.npy")
    if allow_pickle:
        data = np.load(src_path, allow_pickle=True)
        np.save(dst_path, data[rows].astype(object))
        return

    src = np.load(src_path, mmap_mode="r")
    dst = np.lib.format.open_memmap(
        dst_path,
        mode="w+",
        dtype=src.dtype,
        shape=(len(rows),) + src.shape[1:],
    )
    for start in range(0, len(rows), chunk_size):
        end = min(start + chunk_size, len(rows))
        dst[start:end] = src[rows[start:end]]
    del dst


def rows_by_labels(labels: np.ndarray, class_ids: list[int]) -> np.ndarray:
    return np.flatnonzero(np.isin(labels, np.asarray(class_ids, dtype=np.int64))).astype(np.int64)


def copy_x3d_split(input_dir: Path, output_dir: Path, source_split: str, output_split: str, class_ids, chunk_size):
    labels = np.load(input_dir / f"{source_split}_labels.npy", mmap_mode="r")
    rows = rows_by_labels(labels, class_ids)
    copy_npy_rows(input_dir / f"{source_split}_float16.npy", output_dir / f"{output_split}_float16.npy", rows, chunk_size)
    copy_npy_rows(input_dir / f"{source_split}_labels.npy", output_dir / f"{output_split}_labels.npy", rows, chunk_size)
    copy_npy_rows(input_dir / f"{source_split}_sample_names.npy", output_dir / f"{output_split}_sample_names.npy", rows, chunk_size)

    names = load_names(output_dir / f"{output_split}_sample_names.npy")
    out_labels = np.load(output_dir / f"{output_split}_labels.npy", mmap_mode="r")
    with (output_dir / f"{output_split}_manifest.txt").open("w", encoding="utf-8") as handle:
        for name, label in zip(names, out_labels):
            handle.write(f"{name} {int(label)}\n")
    return {
        "source_split": source_split,
        "num_samples": int(len(rows)),
        "class_ids": sorted(int(x) for x in np.unique(labels[rows]).tolist()),
    }


def subset_npz_arrays(npz, source_split: str, rows: np.ndarray):
    payload = {
        f"x_{source_split}": npz[f"x_{source_split}"][rows],
        f"y_{source_split}": npz[f"y_{source_split}"][rows],
        f"{source_split}_sample_name": npz[f"{source_split}_sample_name"][rows],
        f"{source_split}_subject": npz[f"{source_split}_subject"][rows],
        f"{source_split}_action": npz[f"{source_split}_action"][rows],
        f"{source_split}_frames": npz[f"{source_split}_frames"][rows],
    }
    return payload


def one_hot_labels(npz, split: str):
    labels = npz[f"y_{split}"]
    return labels.argmax(axis=1).astype(np.int64)


def main():
    args = parse_args()
    split_metadata = json.loads(Path(args.split_metadata).read_text(encoding="utf-8"))
    seen_classes = [int(x) for x in split_metadata["seen_classes"]]
    unseen_classes = [int(x) for x in split_metadata["unseen_classes"]]

    x3d_input_dir = Path(args.x3d_input_dir)
    x3d_output_dir = Path(args.x3d_output_dir)
    if x3d_output_dir.exists() and any(x3d_output_dir.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"{x3d_output_dir} is not empty. Pass --overwrite to replace it.")
        shutil.rmtree(x3d_output_dir)
    x3d_output_dir.mkdir(parents=True, exist_ok=True)

    x3d_subsets = {
        "train": copy_x3d_split(x3d_input_dir, x3d_output_dir, "train", "train", seen_classes, args.chunk_size),
        "val": copy_x3d_split(x3d_input_dir, x3d_output_dir, "val", "val", seen_classes, args.chunk_size),
        "test": copy_x3d_split(x3d_input_dir, x3d_output_dir, "test", "test", unseen_classes, args.chunk_size),
        "test_seen": copy_x3d_split(x3d_input_dir, x3d_output_dir, "test", "test_seen", seen_classes, args.chunk_size),
    }

    source_metadata = json.loads((x3d_input_dir / "metadata.json").read_text(encoding="utf-8"))
    metadata = {
        **source_metadata,
        "name": "clipgcn_action_detail_raw_13f_zsl_50_5",
        "description": "Seen/unseen raw split for training X3D/CTR-GCN on seen classes and testing on unseen classes.",
        "source_x3d_dir": str(x3d_input_dir),
        "source_ctrgcn_npz": str(Path(args.ctrgcn_input)),
        "split_metadata": str(Path(args.split_metadata)),
        "seen_classes": seen_classes,
        "unseen_classes": unseen_classes,
        "label_ids_are_original": True,
        "shape": {
            split: [x3d_subsets[split]["num_samples"], 3, 13, 160, 160]
            for split in ("train", "val", "test", "test_seen")
        },
        "subsets": x3d_subsets,
    }
    (x3d_output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    ctrgcn_input = Path(args.ctrgcn_input)
    ctrgcn_output = Path(args.ctrgcn_output)
    ctrgcn_output.parent.mkdir(parents=True, exist_ok=True)
    npz = np.load(ctrgcn_input, allow_pickle=True)

    train_rows = rows_by_labels(one_hot_labels(npz, "train"), seen_classes)
    val_rows = rows_by_labels(one_hot_labels(npz, "val"), seen_classes)
    test_unseen_rows = rows_by_labels(one_hot_labels(npz, "test"), unseen_classes)
    test_seen_rows = rows_by_labels(one_hot_labels(npz, "test"), seen_classes)

    train_payload = subset_npz_arrays(npz, "train", train_rows)
    val_payload = subset_npz_arrays(npz, "val", val_rows)
    test_payload = subset_npz_arrays(npz, "test", test_unseen_rows)
    test_seen_payload = subset_npz_arrays(npz, "test", test_seen_rows)

    np.savez_compressed(
        ctrgcn_output,
        x_train=train_payload["x_train"],
        y_train=train_payload["y_train"],
        train_sample_name=train_payload["train_sample_name"],
        train_subject=train_payload["train_subject"],
        train_action=train_payload["train_action"],
        train_frames=train_payload["train_frames"],
        x_val=val_payload["x_val"],
        y_val=val_payload["y_val"],
        val_sample_name=val_payload["val_sample_name"],
        val_subject=val_payload["val_subject"],
        val_action=val_payload["val_action"],
        val_frames=val_payload["val_frames"],
        x_test=test_payload["x_test"],
        y_test=test_payload["y_test"],
        test_sample_name=test_payload["test_sample_name"],
        test_subject=test_payload["test_subject"],
        test_action=test_payload["test_action"],
        test_frames=test_payload["test_frames"],
        x_test_seen=test_seen_payload["x_test"],
        y_test_seen=test_seen_payload["y_test"],
        test_seen_sample_name=test_seen_payload["test_sample_name"],
        test_seen_subject=test_seen_payload["test_subject"],
        test_seen_action=test_seen_payload["test_action"],
        test_seen_frames=test_seen_payload["test_frames"],
    )

    ctrgcn_metadata = {
        **metadata,
        "ctrgcn_npz": str(ctrgcn_output),
        "ctrgcn_subsets": {
            "train": {"num_samples": int(len(train_rows)), "class_ids": sorted(np.unique(one_hot_labels(npz, "train")[train_rows]).astype(int).tolist())},
            "val": {"num_samples": int(len(val_rows)), "class_ids": sorted(np.unique(one_hot_labels(npz, "val")[val_rows]).astype(int).tolist())},
            "test": {"num_samples": int(len(test_unseen_rows)), "class_ids": sorted(np.unique(one_hot_labels(npz, "test")[test_unseen_rows]).astype(int).tolist())},
            "test_seen": {"num_samples": int(len(test_seen_rows)), "class_ids": sorted(np.unique(one_hot_labels(npz, "test")[test_seen_rows]).astype(int).tolist())},
        },
    }
    Path(ctrgcn_output.with_suffix("").as_posix() + "_metadata.json").write_text(
        json.dumps(ctrgcn_metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(ctrgcn_metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
