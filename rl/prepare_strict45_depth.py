#!/usr/bin/env python3
"""Complete the first-frame object-depth cache for the strict 45/10 cache.

The old depth extraction covers most of the strict-cache videos, but not all
of them because the subject split is different.  This script reuses those
rows and extracts only missing sample names with the same YOLO/MiDaS
procedure.  It writes a new cache and never changes the old 50/5 depth data.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from .extract_first_frame_object_depth import (
    FRAME_HEIGHT,
    FRAME_WIDTH,
    load_midas,
    load_yolo,
    process_batch,
    read_first_frame,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_ROOT = ROOT / "work_dir/frame0_body_angle_rank_strict45_10/cache"
DEFAULT_SOURCE_DEPTH = ROOT / "data/new50_5_group1_zsl_depth_first_frame_fp32"
DEFAULT_OUTPUT_ROOT = ROOT / "data/frame0_body_angle_rank_strict45_10_depth"
NUM_OBJECTS = 50


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def old_depth_lookup(source_root: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    lookup: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for directory in sorted(source_root.iterdir()):
        if not directory.is_dir():
            continue
        names_path = directory / "sample_names.npy"
        depth_path = directory / "depth_rel_fp32.npy"
        valid_path = directory / "depth_valid_uint8.npy"
        if not (names_path.exists() and depth_path.exists() and valid_path.exists()):
            continue
        names = np.load(names_path, allow_pickle=False).astype(str)
        depth = np.load(depth_path, mmap_mode="r", allow_pickle=False)
        valid = np.load(valid_path, mmap_mode="r", allow_pickle=False)
        if depth.shape != (len(names), NUM_OBJECTS) or valid.shape != depth.shape:
            raise ValueError(f"invalid source depth arrays in {directory}")
        for index, name in enumerate(names.tolist()):
            lookup.setdefault(str(name), (np.asarray(depth[index]), np.asarray(valid[index])))
    if not lookup:
        raise RuntimeError(f"no source depth rows found under {source_root}")
    return lookup


def strict_records(cache_root: Path, split: str) -> list[tuple[str, Path]]:
    metadata = read_json(cache_root / split / "metadata.json")
    records: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for episode in metadata["episodes"]:
        for view in episode.get("views", []):
            name = str(view["sample_name"])
            if name in seen:
                continue
            seen.add(name)
            records.append((name, Path(str(view["video"]))))
    return records


def write_split(
    output_root: Path,
    split: str,
    records: list[tuple[str, Path]],
    source_lookup: dict[str, tuple[np.ndarray, np.ndarray]],
    yolo: object | None,
    midas: object | None,
    transform: object | None,
    device: torch.device,
    batch_size: int,
    decode_workers: int,
) -> dict[str, object]:
    output = output_root / split
    output.mkdir(parents=True, exist_ok=True)
    names = [name for name, _ in records]
    depth = np.zeros((len(records), NUM_OBJECTS), dtype=np.float32)
    valid = np.zeros((len(records), NUM_OBJECTS), dtype=np.uint8)
    missing: list[tuple[int, str, Path]] = []
    reused = 0
    for index, (name, path) in enumerate(records):
        row = source_lookup.get(name)
        if row is None:
            missing.append((index, name, path))
        else:
            depth[index] = row[0]
            valid[index] = row[1]
            reused += 1

    errors: list[dict[str, object]] = []
    if missing:
        if yolo is None or midas is None or transform is None:
            raise RuntimeError("models must be loaded when missing rows exist")
        started = time.monotonic()
        for begin in range(0, len(missing), batch_size):
            batch = missing[begin : begin + batch_size]
            frames: list[np.ndarray] = []
            with ThreadPoolExecutor(max_workers=decode_workers) as pool:
                futures = [pool.submit(read_first_frame, path) for _, _, path in batch]
                for (index, name, path), future in zip(batch, futures):
                    try:
                        frame = future.result()
                    except Exception as exc:  # pragma: no cover - defensive path
                        frame = None
                        errors.append({"index": index, "sample_name": name, "video": str(path), "error": repr(exc)})
                    if frame is None:
                        errors.append({"index": index, "sample_name": name, "video": str(path), "error": "decode_failed"})
                        frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
                    frames.append(frame)
            values, masks = process_batch(frames, yolo, midas, transform, device, FRAME_WIDTH)
            for offset, (index, _name, _path) in enumerate(batch):
                depth[index] = values[offset]
                valid[index] = masks[offset]
            done = min(begin + len(batch), len(missing))
            print(
                json.dumps(
                    {
                        "split": split,
                        "missing_done": done,
                        "missing_total": len(missing),
                        "seconds": time.monotonic() - started,
                    }
                ),
                flush=True,
            )

    np.save(output / "sample_names.npy", np.asarray(names, dtype="U"))
    np.save(output / "depth_rel_fp32.npy", depth)
    np.save(output / "depth_valid_uint8.npy", valid)
    (output / "errors.json").write_text(json.dumps(errors, indent=2), encoding="utf-8")
    result = {
        "split": split,
        "samples": len(records),
        "reused_source_rows": reused,
        "newly_extracted_rows": len(missing),
        "valid_rows": int((valid.sum(axis=1) > 0).sum()),
        "errors": len(errors),
    }
    (output / "complete.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("STRICT45_DEPTH_SPLIT", json.dumps(result), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--source-depth-root", type=Path, default=DEFAULT_SOURCE_DEPTH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--decode-workers", type=int, default=16)
    parser.add_argument("--splits", nargs="+", default=["dqn_train", "val", "test"])
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for strict depth completion")
    device = torch.device("cuda:0")
    args.output_root.mkdir(parents=True, exist_ok=True)
    source_lookup = old_depth_lookup(args.source_depth_root)
    records = {split: strict_records(args.cache_root, split) for split in args.splits}
    missing_total = sum(name not in source_lookup for rows in records.values() for name, _ in rows)
    print(
        json.dumps(
            {
                "protocol": "strict_zsl_45_10",
                "splits": {key: len(value) for key, value in records.items()},
                "source_rows": len(source_lookup),
                "missing_rows": missing_total,
            }
        ),
        flush=True,
    )

    yolo = midas = transform = None
    if missing_total:
        class Args:
            yolo_repo = Path("/home/youhan/ws/yolov5_full")
            yolo_weights = yolo_repo / "best.pt"
            yolo_conf = 0.25
            yolo_iou = 0.45
            no_half_yolo = False

        yolo = load_yolo(Args(), device)
        midas, transform = load_midas(device)
    results = {
        split: write_split(
            args.output_root,
            split,
            rows,
            source_lookup,
            yolo,
            midas,
            transform,
            device,
            args.batch_size,
            args.decode_workers,
        )
        for split, rows in records.items()
    }
    (args.output_root / "metadata.json").write_text(
        json.dumps(
            {
                "protocol": "strict_zsl_45_10_depth",
                "source_cache": str(args.cache_root),
                "source_depth_cache": str(args.source_depth_root),
                "midas": "MiDaS_small relative inverse depth",
                "normalization": "per-frame 5th-95th percentile to [0,1]",
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("STRICT45_DEPTH_COMPLETE", json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
