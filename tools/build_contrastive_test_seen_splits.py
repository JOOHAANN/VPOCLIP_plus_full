#!/usr/bin/env python
"""Build seen-only held-out test splits from contrastive_test_data."""

import argparse
import json
from pathlib import Path

import numpy as np


MODALITY_FILES = {
    "video": "trimodal_test_video.npy",
    "pose": "trimodal_test_pose.npy",
    "object": "trimodal_test_object.npy",
    "joint_xy": "trimodal_test_joint_xy.npy",
    "labels": "trimodal_test_labels.npy",
    "sample_names": "trimodal_test_sample_names.npy",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="/workspace/CLIPGCN/data/contrastive_test_data")
    parser.add_argument("--zsl-dir", default="/workspace/CLIPGCN/data/contrastive_zsl_splits")
    parser.add_argument("--output-dir", default="/workspace/CLIPGCN/data/contrastive_test_seen_splits")
    parser.add_argument("--splits", nargs="+", default=["50_5", "45_10", "40_15", "35_20"])
    parser.add_argument("--prefix", default="seen")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def save_subset_array(src_path, dst_path, indices, allow_pickle=False):
    if allow_pickle:
        src = np.load(src_path, allow_pickle=True)
        np.save(dst_path, src[indices], allow_pickle=True)
        return list(src[indices].shape), str(src.dtype)

    src = np.load(src_path, mmap_mode="r", allow_pickle=False)
    dst = np.lib.format.open_memmap(dst_path, mode="w+", dtype=src.dtype, shape=(len(indices),) + src.shape[1:])
    chunk = 256
    for start in range(0, len(indices), chunk):
        end = min(start + chunk, len(indices))
        dst[start:end] = src[indices[start:end]]
    del dst
    return [int(len(indices)), *map(int, src.shape[1:])], str(src.dtype)


def write_manifest(path, sample_names, labels):
    with open(path, "w", encoding="utf-8") as handle:
        for name, label in zip(sample_names, labels):
            handle.write(f"{name} {int(label)}\n")


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    zsl_dir = Path(args.zsl_dir)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    labels = np.load(input_dir / MODALITY_FILES["labels"], mmap_mode="r", allow_pickle=False)
    sample_names = np.load(input_dir / MODALITY_FILES["sample_names"], allow_pickle=True)
    input_metadata_path = input_dir / "metadata.json"
    input_metadata = json.loads(input_metadata_path.read_text(encoding="utf-8")) if input_metadata_path.exists() else {}

    summary = {
        "input_dir": str(input_dir),
        "zsl_dir": str(zsl_dir),
        "output_dir": str(output_root),
        "splits": {},
        "note": "Each split keeps only held-out test samples whose label belongs to that split's seen_classes.",
    }

    for split_name in args.splits:
        split_metadata_path = zsl_dir / split_name / "metadata.json"
        if not split_metadata_path.exists():
            raise FileNotFoundError(f"Missing ZSL metadata: {split_metadata_path}")
        split_metadata = json.loads(split_metadata_path.read_text(encoding="utf-8"))
        seen_classes = [int(x) for x in split_metadata["seen_classes"]]
        unseen_classes = [int(x) for x in split_metadata["unseen_classes"]]
        keep = np.isin(labels, np.asarray(seen_classes, dtype=labels.dtype))
        indices = np.flatnonzero(keep)
        out_dir = output_root / split_name
        if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
            raise FileExistsError(f"{out_dir} is not empty. Pass --overwrite to replace files.")
        out_dir.mkdir(parents=True, exist_ok=True)

        shapes = {}
        dtypes = {}
        files = {}
        for modality, filename in MODALITY_FILES.items():
            out_name = f"{args.prefix}_{modality}.npy"
            allow_pickle = modality == "sample_names"
            shape, dtype = save_subset_array(input_dir / filename, out_dir / out_name, indices, allow_pickle=allow_pickle)
            shapes[modality] = shape
            dtypes[modality] = dtype
            files[modality] = out_name

        subset_labels = labels[indices].astype(np.int64)
        subset_names = sample_names[indices].astype(object)
        manifest_name = f"{args.prefix}_manifest.txt"
        write_manifest(out_dir / manifest_name, subset_names, subset_labels)
        files["manifest"] = manifest_name

        alignment_name = f"{args.prefix}_alignment.npz"
        np.savez(
            out_dir / alignment_name,
            source_indices=indices.astype(np.int64),
            sample_names=subset_names,
            labels=subset_labels,
            seen_classes=np.asarray(seen_classes, dtype=np.int64),
            removed_unseen_classes=np.asarray(unseen_classes, dtype=np.int64),
            removed_unseen_count=np.asarray([int((~keep).sum())], dtype=np.int64),
        )
        files["alignment"] = alignment_name

        class_ids, class_counts = np.unique(subset_labels, return_counts=True)
        removed_labels = labels[~keep]
        removed_class_ids, removed_class_counts = np.unique(removed_labels, return_counts=True)
        metadata = {
            "split": split_name,
            "source_dataset": str(input_dir),
            "source_metadata": str(input_metadata_path),
            "zsl_metadata": str(split_metadata_path),
            "prefix": args.prefix,
            "num_samples": int(len(indices)),
            "source_num_samples": int(len(labels)),
            "removed_unseen_samples": int((~keep).sum()),
            "seen_class_count": int(len(seen_classes)),
            "unseen_class_count": int(len(unseen_classes)),
            "seen_classes": seen_classes,
            "unseen_classes": unseen_classes,
            "all_classes": seen_classes + unseen_classes,
            "class_ids": class_ids.astype(int).tolist(),
            "class_counts": {str(int(k)): int(v) for k, v in zip(class_ids, class_counts)},
            "removed_unseen_class_counts": {
                str(int(k)): int(v) for k, v in zip(removed_class_ids, removed_class_counts)
            },
            "files": files,
            "shapes": shapes,
            "dtypes": dtypes,
            "checks": {
                "all_lengths_equal": len({shape[0] for shape in shapes.values()}) == 1,
                "labels_subset_of_seen_classes": bool(set(np.unique(subset_labels).astype(int)).issubset(seen_classes)),
                "removed_labels_subset_of_unseen_classes": bool(set(np.unique(removed_labels).astype(int)).issubset(unseen_classes)),
                "source_core_checks": input_metadata.get("checks", {}),
            },
        }
        (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        summary["splits"][split_name] = {
            "path": str(out_dir),
            "num_samples": metadata["num_samples"],
            "removed_unseen_samples": metadata["removed_unseen_samples"],
            "seen_classes": seen_classes,
            "unseen_classes": unseen_classes,
            "checks": metadata["checks"],
        }
        print(
            f"{split_name}: kept {metadata['num_samples']} seen samples; "
            f"removed {metadata['removed_unseen_samples']} unseen samples {unseen_classes}"
        )

    (output_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved summary: {output_root / 'summary.json'}")


if __name__ == "__main__":
    main()
