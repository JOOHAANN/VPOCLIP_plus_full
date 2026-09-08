#!/usr/bin/env python3
"""Assemble aligned 50/5 ZSL and 55/0 VPOCLIP datasets by subject split."""

import argparse
import json
import os
from pathlib import Path

import numpy as np


UNSEEN = np.asarray([9, 10, 11, 17, 49], dtype=np.int64)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--x3d-dir", type=Path, required=True)
    p.add_argument("--pose-dir", type=Path, required=True)
    p.add_argument("--object-dir", type=Path, required=True)
    p.add_argument("--zsl-dir", type=Path, required=True)
    p.add_argument("--closed-dir", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--unseen-classes", type=int, nargs=5, default=UNSEEN.tolist())
    p.add_argument("--zsl-only", action="store_true")
    p.add_argument("--unseen-recordings-manifest", type=Path)
    return p.parse_args()


def subset(src, dst, rows, chunk=256):
    src_arr = np.load(src, mmap_mode="r", allow_pickle=False)
    out = np.lib.format.open_memmap(dst, mode="w+", dtype=src_arr.dtype,
                                   shape=(len(rows), *src_arr.shape[1:]))
    for start in range(0, len(rows), chunk):
        stop = min(start + chunk, len(rows))
        out[start:stop] = src_arr[rows[start:stop]]
    out.flush()


def link(src, dst):
    if dst.exists():
        dst.unlink()
    os.link(src, dst)


def sources(args, split):
    return {
        "video": args.x3d_dir / f"{split}_video.npy",
        "pose_coco17": args.pose_dir / f"{split}_pose_coco17.npy",
        "joint_xy_coco17": args.pose_dir / f"{split}_joint_xy_coco17.npy",
        "object": args.object_dir / f"{split}_object.npy",
    }


def validate(args, split):
    labels = np.load(args.cache_dir / f"{split}_labels.npy")
    names = np.load(args.cache_dir / f"{split}_sample_names.npy", allow_pickle=True).astype(str)
    xlabels = np.load(args.x3d_dir / f"{split}_video.labels.npy")
    plabels = np.load(args.pose_dir / f"{split}_labels.npy")
    pnames = np.load(args.pose_dir / f"{split}_sample_names.npy", allow_pickle=True).astype(str)
    if not np.array_equal(labels, xlabels) or not np.array_equal(labels, plabels):
        raise ValueError(f"{split}: labels are not aligned across X3D/CTR-GCN/cache")
    if not np.array_equal(names, pnames):
        raise ValueError(f"{split}: names are not aligned across RGB/RTMPose")
    for name, path in sources(args, split).items():
        arr = np.load(path, mmap_mode="r", allow_pickle=False)
        if len(arr) != len(labels):
            raise ValueError(f"{split}/{name}: {len(arr)} != {len(labels)}")
    return labels, names


def write(args, out_dir, prefix, split, rows, labels, names, hardlink=False):
    out_dir.mkdir(parents=True, exist_ok=True)
    for modality, src in sources(args, split).items():
        dst = out_dir / f"{prefix}_{modality}.npy"
        if hardlink and len(rows) == len(labels) and np.array_equal(rows, np.arange(len(labels))):
            link(src, dst)
        else:
            subset(src, dst, rows)
    np.save(out_dir / f"{prefix}_labels.npy", labels[rows])
    np.save(out_dir / f"{prefix}_sample_names.npy", names[rows])
    np.save(out_dir / f"{prefix}_source_indices.npy", rows)


def main():
    global UNSEEN
    args = parse_args()
    UNSEEN = np.asarray(sorted(args.unseen_classes), dtype=np.int64)
    if len(set(UNSEEN.tolist())) != 5 or UNSEEN.min() < 0 or UNSEEN.max() >= 55:
        raise ValueError("Expected five unique zero-based class IDs in [0,54]")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    complete_recordings = None
    if args.unseen_recordings_manifest:
        selection = json.loads(args.unseen_recordings_manifest.read_text())
        if selection['unseen_classes_zero_based'] != UNSEEN.tolist():
            raise ValueError('Class selection and complete-recording manifest disagree')
        complete_recordings = {name for row in selection['selected']
                               for name in row['complete_recording_ids']}
    all_rows = {}
    for split in ("train", "val", "test"):
        labels, names = validate(args, split)
        all_rows[split] = (labels, names, np.arange(len(labels), dtype=np.int64))

    # Strict ZSL: train/validation only contain the 50 seen actions. Test is
    # split into unseen and seen portions for ZSL and GZSL evaluation.
    zsl_specs = [
        ("trimodal_train", "train", False),
        ("trimodal_val", "val", False),
        ("trimodal_test", "test", True),
        ("trimodal_test_seen", "test", False),
    ]
    zsl_counts = {}
    for prefix, split, want_unseen in zsl_specs:
        labels, names, _ = all_rows[split]
        mask = np.isin(labels, UNSEEN)
        selected_mask = mask if want_unseen else ~mask
        if want_unseen and complete_recordings is not None:
            selected_mask = selected_mask & np.asarray([
                Path(n).stem.rsplit('_C', 1)[0] in complete_recordings for n in names])
        rows = np.flatnonzero(selected_mask)
        write(args, args.zsl_dir, prefix, split, rows, labels, names)
        zsl_counts[prefix] = len(rows)
        print(f"{prefix}: {len(rows)}", flush=True)

    # Closed set uses all 55 actions but exactly the same 45/10/20 subject sets.
    closed_counts = {}
    for prefix, split in (("seen", "train"), ("val", "val"), ("test", "test")):
        if args.zsl_only:
            break
        labels, names, rows = all_rows[split]
        write(args, args.closed_dir, prefix, split, rows, labels, names, hardlink=True)
        closed_counts[prefix] = len(rows)
        print(f"closed/{prefix}: {len(rows)}", flush=True)

    common = {
        "canonical_subject_manifest": str(args.manifest.resolve()),
        "subject_split": manifest,
        "unseen_classes_zero_based": UNSEEN.tolist(),
        # test.py consumes these canonical candidate-bank fields.
        "seen_classes": np.setdiff1d(np.arange(55, dtype=np.int64), UNSEEN).tolist(),
        "unseen_classes": UNSEEN.tolist(),
        "unseen_actions": [f"A{x + 1:03d}" for x in UNSEEN],
        "unseen_recordings_manifest": str(args.unseen_recordings_manifest) if args.unseen_recordings_manifest else None,
        "sources": {
            "x3d": str(args.x3d_dir.resolve()),
            "pose": str(args.pose_dir.resolve()),
            "object": str(args.object_dir.resolve()),
        },
    }
    (args.zsl_dir / "metadata.json").write_text(
        json.dumps({**common, "protocol": "strict_zsl_50_5", "counts": zsl_counts}, indent=2),
        encoding="utf-8",
    )
    if args.zsl_only:
        return
    (args.closed_dir / "metadata.json").write_text(
        json.dumps({**common, "protocol": "closed_set_55_0", "counts": closed_counts}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
