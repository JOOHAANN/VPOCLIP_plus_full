#!/usr/bin/env python
"""Build zero-shot seen/unseen splits from windowed trimodal CLIPGCN data."""

import argparse
import json
from pathlib import Path

import numpy as np


MODALITIES = ("video", "pose", "object", "joint_xy")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="/workspace/CLIPGCN/data/contrastive_windowed_2s1s_13f")
    parser.add_argument("--output-root", default="/workspace/CLIPGCN/data/contrastive_windowed_zsl_splits")
    parser.add_argument("--seed", type=int, default=20260616)
    parser.add_argument("--unseen-count", type=int, default=5)
    parser.add_argument(
        "--seen-count",
        type=int,
        default=None,
        help="Optional exact seen class count. Use 55 only when the source has at least 60 valid classes.",
    )
    parser.add_argument("--split-name", default=None, help="Optional output directory name, e.g. 55_5.")
    parser.add_argument("--train-prefix", default="trimodal_train")
    parser.add_argument("--val-prefix", default="trimodal_val")
    parser.add_argument("--test-prefix", default="trimodal_test")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_labels(input_dir, prefix):
    return np.load(input_dir / f"{prefix}_labels.npy", mmap_mode="r")


def copy_subset(input_dir, output_dir, source_prefix, output_prefix, indices):
    metadata = {"num_samples": int(len(indices)), "files": {}}
    for modality in MODALITIES:
        src = np.load(input_dir / f"{source_prefix}_{modality}.npy", mmap_mode="r")
        dst = np.lib.format.open_memmap(
            output_dir / f"{output_prefix}_{modality}.npy",
            mode="w+",
            dtype=src.dtype,
            shape=(len(indices),) + src.shape[1:],
        )
        chunk = 256
        for start in range(0, len(indices), chunk):
            end = min(start + chunk, len(indices))
            dst[start:end] = src[indices[start:end]]
        del dst
        metadata["files"][modality] = f"{output_prefix}_{modality}.npy"

    labels = np.load(input_dir / f"{source_prefix}_labels.npy", mmap_mode="r")[indices]
    names = np.load(input_dir / f"{source_prefix}_sample_names.npy", allow_pickle=True)[indices]
    np.save(output_dir / f"{output_prefix}_labels.npy", labels)
    np.save(output_dir / f"{output_prefix}_sample_names.npy", names.astype(object))
    np.savez_compressed(
        output_dir / f"{output_prefix}_alignment.npz",
        source_prefix=source_prefix,
        source_rows=indices,
        labels=labels,
        sample_names=names.astype(object),
    )
    metadata["class_ids"] = sorted(int(x) for x in np.unique(labels).tolist())
    metadata["files"].update(
        {
            "labels": f"{output_prefix}_labels.npy",
            "sample_names": f"{output_prefix}_sample_names.npy",
            "alignment": f"{output_prefix}_alignment.npz",
        }
    )
    return metadata


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_root = Path(args.output_root)

    train_labels = load_labels(input_dir, args.train_prefix)
    val_labels = load_labels(input_dir, args.val_prefix)
    test_labels = load_labels(input_dir, args.test_prefix)
    classes = np.array(sorted(np.unique(np.concatenate([train_labels, val_labels, test_labels])).tolist()), dtype=np.int64)

    if args.seen_count is not None and args.seen_count + args.unseen_count > len(classes):
        raise ValueError(
            f"Requested {args.seen_count}/{args.unseen_count}, but only {len(classes)} valid classes are present. "
            "For this CLIPGCN/ETRI source, A000 is excluded and the valid class count is 55, so 5 unseen implies 50 seen."
        )
    if args.unseen_count <= 0 or args.unseen_count >= len(classes):
        raise ValueError(f"--unseen-count must be in [1, {len(classes) - 1}], got {args.unseen_count}")

    rng = np.random.default_rng(args.seed)
    unseen_classes = np.sort(rng.choice(classes, size=args.unseen_count, replace=False))
    seen_classes = np.setdiff1d(classes, unseen_classes)
    if args.seen_count is not None and len(seen_classes) != args.seen_count:
        raise RuntimeError(f"Internal split mismatch: expected {args.seen_count} seen classes, got {len(seen_classes)}")

    split_name = args.split_name or f"{len(seen_classes)}_{len(unseen_classes)}"
    output_dir = output_root / split_name
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"{output_dir} is not empty. Pass --overwrite to replace files.")
    output_dir.mkdir(parents=True, exist_ok=True)

    subsets = {
        "seen": (args.train_prefix, np.flatnonzero(np.isin(train_labels, seen_classes))),
        "val_seen": (args.val_prefix, np.flatnonzero(np.isin(val_labels, seen_classes))),
        "test_seen": (args.test_prefix, np.flatnonzero(np.isin(test_labels, seen_classes))),
        "unseen": (args.test_prefix, np.flatnonzero(np.isin(test_labels, unseen_classes))),
    }
    subset_meta = {
        output_prefix: copy_subset(input_dir, output_dir, source_prefix, output_prefix, indices)
        for output_prefix, (source_prefix, indices) in subsets.items()
    }

    metadata = {
        "split": split_name,
        "seed": args.seed,
        "input_dir": str(input_dir),
        "class_count": int(len(classes)),
        "seen_class_count": int(len(seen_classes)),
        "unseen_class_count": int(len(unseen_classes)),
        "seen_classes": seen_classes.astype(int).tolist(),
        "unseen_classes": unseen_classes.astype(int).tolist(),
        "label_ids_are_original": True,
        "subsets": subset_meta,
        "note": "seen=train subjects/seen classes; val_seen=val subjects/seen classes; unseen=test subjects/unseen classes.",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    summary_path = output_root / "summary.json"
    summary = {}
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary[split_name] = metadata
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
