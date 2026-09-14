#!/usr/bin/env python3
"""Build class-independent object maps aligned to the all-view tensor cache.

The two completed 50/5 VPOCLIP datasets contain the same detector output for
their respective class-filtered rows.  Their union covers every valid RGB row.
This utility reconstructs one full, sample-name-aligned object feature array
for the final 50/5 refit without using labels to change the detector output.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


VPO = Path("/home/youhan/ws/VPOCLIP_plus_full")
RAW = Path("/home/youhan/ws/X3D_full/data/etri_rgb_allviews_cs_45_25_10_20")
GROUPS = [VPO / "data/new50_5_group1_zsl", VPO / "data/new50_5_group2_zsl"]
OUT = VPO / "data/final50_5_pseudo_refit_object"


def source_files(split: str) -> list[tuple[Path, Path]]:
    if split == "test":
        prefixes = ["trimodal_test", "trimodal_test_seen"]
    else:
        prefixes = [f"trimodal_{split}"]
    return [
        (group / f"{prefix}_sample_names.npy", group / f"{prefix}_object.npy")
        for group in GROUPS
        for prefix in prefixes
    ]


def build(split: str) -> dict[str, object]:
    names = np.load(RAW / f"{split}_sample_names.npy", allow_pickle=True).astype(str)
    labels = np.load(RAW / f"{split}_labels.npy", allow_pickle=False)
    index: dict[str, tuple[Path, int]] = {}
    shape = None
    dtype = None
    for names_path, object_path in source_files(split):
        source_names = np.load(names_path, allow_pickle=True).astype(str)
        source = np.load(object_path, mmap_mode="r", allow_pickle=False)
        if len(source_names) != len(source):
            raise RuntimeError(f"misaligned object source: {names_path}")
        if shape is None:
            shape, dtype = source.shape[1:], source.dtype
        elif source.shape[1:] != shape or source.dtype != dtype:
            raise RuntimeError(f"inconsistent object source shape/dtype: {object_path}")
        for row, sample_name in enumerate(source_names):
            index.setdefault(sample_name, (object_path, row))

    missing = [name for name in names if name not in index]
    if missing:
        raise RuntimeError(f"{split}: {len(missing)} raw rows lack object features; first={missing[:3]}")

    OUT.mkdir(parents=True, exist_ok=True)
    output = OUT / f"{split}_object.npy"
    mem = np.lib.format.open_memmap(output, mode="w+", dtype=dtype, shape=(len(names), *shape))
    handles: dict[Path, np.ndarray] = {}
    for start in range(0, len(names), 512):
        stop = min(start + 512, len(names))
        for row, sample_name in enumerate(names[start:stop], start):
            path, source_row = index[sample_name]
            if path not in handles:
                handles[path] = np.load(path, mmap_mode="r", allow_pickle=False)
            mem[row] = handles[path][source_row]
    mem.flush()
    np.save(OUT / f"{split}_sample_names.npy", names)
    np.save(OUT / f"{split}_labels.npy", labels)
    return {"rows": len(names), "shape": list(mem.shape), "dtype": str(dtype), "missing": len(missing)}


def main() -> None:
    summary = {split: build(split) for split in ("train", "val", "test")}
    (OUT / "metadata.json").write_text(
        json.dumps(
            {
                "protocol": "final_50_5_pseudo_refit",
                "source": [str(path) for path in GROUPS],
                "alignment_key": "RGB sample name",
                "detector_features_are_class_independent": True,
                "splits": summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
