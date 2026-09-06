#!/usr/bin/env python
"""Build zero-shot contrastive splits from aligned CLIPGCN trimodal data."""

import argparse
import json
from pathlib import Path

import numpy as np


MODALITY_FILES = {
    "video": "trimodal_train_video.npy",
    "pose": "trimodal_train_pose.npy",
    "joint_xy": "trimodal_train_joint_xy.npy",
    "object": "trimodal_train_object.npy",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        default="/workspace/CLIPGCN/data/contrastive_train_data",
        help="Directory containing trimodal_train_* files.",
    )
    parser.add_argument(
        "--output-dir",
        default="/workspace/CLIPGCN/data/contrastive_zsl_splits",
        help="Output directory for 50_5, 45_10, 40_15, and 35_20 splits.",
    )
    parser.add_argument("--seed", type=int, default=20260615)
    parser.add_argument("--unseen-counts", type=int, nargs="+", default=[5, 10, 15, 20])
    parser.add_argument("--dtype-video", default=None, help="Optional dtype override for saved video arrays.")
    parser.add_argument("--dtype-pose", default=None, help="Optional dtype override for saved pose arrays.")
    return parser.parse_args()


def load_object(path):
    value = np.load(path, allow_pickle=True)
    if value.ndim == 4:
        return value
    if value.shape != ():
        raise ValueError(f"Expected object npy containing a dict, got shape {value.shape}")
    obj = value.item()
    if not isinstance(obj, dict):
        raise ValueError(f"Expected object dict, got {type(obj)}")
    return obj


def subset_object(obj, indices, labels, sample_names, split_name):
    if isinstance(obj, np.ndarray):
        return obj[indices]

    n = len(labels)
    out = {}
    for key, value in obj.items():
        arr = np.asarray(value)
        if arr.shape[:1] == (n,):
            out[key] = arr[indices]
        else:
            out[key] = value
    out["split"] = split_name
    out["aligned_sample_names"] = sample_names[indices]
    out["manifest"] = np.stack([sample_names[indices], labels[indices]], axis=1)
    out["alignment_note"] = f"Subset generated for {split_name}; rows match video/pose/joint_xy/labels/sample_names."
    return out


def subset_alignment(npz_path, indices, split_dir, prefix, n):
    if not npz_path.exists():
        return None
    alignment = np.load(npz_path, allow_pickle=True)
    payload = {}
    for key in alignment.files:
        arr = alignment[key]
        payload[key] = arr[indices] if arr.shape[:1] == (n,) else arr
    out_path = split_dir / f"{prefix}_alignment.npz"
    np.savez_compressed(out_path, **payload)
    return out_path.name


def save_subset_array(src_path, indices, out_path, dtype_override=None):
    src = np.load(src_path, mmap_mode="r", allow_pickle=False)
    dtype = np.dtype(dtype_override) if dtype_override else src.dtype
    dst = np.lib.format.open_memmap(out_path, mode="w+", dtype=dtype, shape=(len(indices),) + src.shape[1:])
    chunk = 256
    for start in range(0, len(indices), chunk):
        end = min(start + chunk, len(indices))
        dst[start:end] = src[indices[start:end]].astype(dtype, copy=False)
    del dst
    return list(src.shape), str(dtype)


def write_subset(input_dir, split_dir, prefix, indices, labels, sample_names, obj, alignment_path, dtype_video, dtype_pose):
    split_labels = labels[indices]
    split_names = sample_names[indices]

    for modality, filename in MODALITY_FILES.items():
        if modality == "object" and not isinstance(obj, np.ndarray):
            continue
        dtype_override = dtype_video if modality == "video" else dtype_pose if modality == "pose" else None
        save_subset_array(
            input_dir / filename,
            indices,
            split_dir / f"{prefix}_{modality}.npy",
            dtype_override=dtype_override,
        )

    np.save(split_dir / f"{prefix}_labels.npy", split_labels)
    np.save(split_dir / f"{prefix}_sample_names.npy", split_names)
    if not isinstance(obj, np.ndarray):
        np.save(
            split_dir / f"{prefix}_object.npy",
            subset_object(obj, indices, labels, sample_names, prefix),
            allow_pickle=True,
        )
    alignment_name = subset_alignment(alignment_path, indices, split_dir, prefix, len(labels))
    return {
        "num_samples": int(len(indices)),
        "class_ids": sorted(int(x) for x in np.unique(split_labels).tolist()),
        "files": {
            "video": f"{prefix}_video.npy",
            "pose": f"{prefix}_pose.npy",
            "joint_xy": f"{prefix}_joint_xy.npy",
            "object": f"{prefix}_object.npy",
            "labels": f"{prefix}_labels.npy",
            "sample_names": f"{prefix}_sample_names.npy",
            "alignment": alignment_name,
        },
    }


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = np.load(input_dir / "trimodal_train_labels.npy")
    sample_names = np.load(input_dir / "trimodal_train_sample_names.npy", allow_pickle=True)
    obj = load_object(input_dir / "trimodal_train_object.npy")
    alignment_path = input_dir / "trimodal_train_alignment.npz"

    classes = np.array(sorted(np.unique(labels).tolist()), dtype=np.int64)
    rng = np.random.default_rng(args.seed)
    global_summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "seed": args.seed,
        "num_samples": int(len(labels)),
        "all_classes": classes.astype(int).tolist(),
        "splits": {},
        "note": "Each split independently samples unseen classes with the same RNG seed stream.",
    }

    for unseen_count in args.unseen_counts:
        if unseen_count <= 0 or unseen_count >= len(classes):
            raise ValueError(f"unseen_count must be in [1, {len(classes) - 1}], got {unseen_count}")
        unseen_classes = np.sort(rng.choice(classes, size=unseen_count, replace=False))
        seen_classes = np.setdiff1d(classes, unseen_classes)
        seen_mask = np.isin(labels, seen_classes)
        unseen_mask = np.isin(labels, unseen_classes)
        seen_indices = np.flatnonzero(seen_mask)
        unseen_indices = np.flatnonzero(unseen_mask)

        split_name = f"{len(seen_classes)}_{len(unseen_classes)}"
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        seen_meta = write_subset(
            input_dir,
            split_dir,
            "seen",
            seen_indices,
            labels,
            sample_names,
            obj,
            alignment_path,
            args.dtype_video,
            args.dtype_pose,
        )
        unseen_meta = write_subset(
            input_dir,
            split_dir,
            "unseen",
            unseen_indices,
            labels,
            sample_names,
            obj,
            alignment_path,
            args.dtype_video,
            args.dtype_pose,
        )

        metadata = {
            "split": split_name,
            "seed": args.seed,
            "seen_class_count": int(len(seen_classes)),
            "unseen_class_count": int(len(unseen_classes)),
            "seen_classes": seen_classes.astype(int).tolist(),
            "unseen_classes": unseen_classes.astype(int).tolist(),
            "seen": seen_meta,
            "unseen": unseen_meta,
            "label_ids_are_original": True,
        }
        with open(split_dir / "metadata.json", "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, ensure_ascii=False)
        global_summary["splits"][split_name] = metadata
        print(f"{split_name}: unseen={metadata['unseen_classes']} seen_samples={seen_meta['num_samples']} unseen_samples={unseen_meta['num_samples']}")

    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(global_summary, handle, indent=2, ensure_ascii=False)
    print(f"Saved summary: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
