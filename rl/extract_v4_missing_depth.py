#!/usr/bin/env python3
"""Fill the first-frame depth cache for v4's DQN-train views.

The strict VPOCLIP trimodal export contains val/test and VPO-train rows, but
the trajectory v4 cache also has a separate DQN-train subject split.  This
small companion job extracts exactly those missing views and writes them into
the existing first-frame depth cache using the same [N,50] FP32 contract.
"""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch

from .extract_first_frame_object_depth import (
    DEFAULT_OUTPUT_ROOT,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    NUM_OBJECTS,
    load_midas,
    load_yolo,
    process_batch,
)


TRAJECTORY_CACHE = Path(
    "/home/youhan/ws/VPOCLIP_plus_full/data/trajectory_latest_vpoclip_new50_5/cache_compat/dqn_train/metadata.json"
)
OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT / "dqn_train"


def read_first_frame(path: Path) -> np.ndarray | None:
    capture = cv2.VideoCapture(str(path))
    ok, frame = capture.read()
    capture.release()
    if not ok or frame is None:
        return None
    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def main() -> None:
    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    metadata = json.loads(TRAJECTORY_CACHE.read_text(encoding="utf-8"))
    records = [
        view
        for episode in metadata["episodes"]
        for view in episode.get("views", [])[:4]
    ]
    names = [str(view["sample_name"]) for view in records]
    paths = [Path(view["video"]) for view in records]
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    final_depth = OUTPUT_ROOT / "depth_rel_fp32.npy"
    final_valid = OUTPUT_ROOT / "depth_valid_uint8.npy"
    names_path = OUTPUT_ROOT / "sample_names.npy"
    complete_path = OUTPUT_ROOT / "complete.json"
    progress_path = OUTPUT_ROOT / "progress.json"
    temp_depth = OUTPUT_ROOT / "depth_rel_fp32.tmp.npy"
    temp_valid = OUTPUT_ROOT / "depth_valid_uint8.tmp.npy"
    if complete_path.exists() and final_depth.exists() and final_valid.exists():
        print(complete_path.read_text(encoding="utf-8"), flush=True)
        return

    np.save(names_path, np.asarray(names, dtype="U"))
    n = len(paths)
    if temp_depth.exists() and temp_valid.exists() and progress_path.exists():
        depth_store = np.lib.format.open_memmap(temp_depth, mode="r+")
        valid_store = np.lib.format.open_memmap(temp_valid, mode="r+")
        start = int(json.loads(progress_path.read_text(encoding="utf-8")).get("samples_done", 0))
    else:
        depth_store = np.lib.format.open_memmap(
            temp_depth, mode="w+", dtype=np.float32, shape=(n, NUM_OBJECTS)
        )
        valid_store = np.lib.format.open_memmap(
            temp_valid, mode="w+", dtype=np.uint8, shape=(n, NUM_OBJECTS)
        )
        depth_store[:] = 0.0
        valid_store[:] = 0
        depth_store.flush()
        valid_store.flush()
        start = 0

    class Args:
        yolo_repo = Path("/home/youhan/ws/yolov5_full")
        yolo_weights = Path("/home/youhan/ws/yolov5_full/best.pt")
        yolo_conf = 0.25
        yolo_iou = 0.45
        no_half_yolo = False

    args = Args()
    yolo = load_yolo(args, device)
    midas, transform = load_midas(device)
    errors_path = OUTPUT_ROOT / "errors.jsonl"
    started = time.monotonic()
    with errors_path.open("a", encoding="utf-8") as errors:
        for begin in range(start, n, 128):
            end = min(begin + 128, n)
            frames: list[np.ndarray] = []
            with ThreadPoolExecutor(max_workers=16) as pool:
                futures = [pool.submit(read_first_frame, path) for path in paths[begin:end]]
                for index, (path, future) in enumerate(zip(paths[begin:end], futures)):
                    try:
                        frame = future.result()
                    except Exception as exc:  # pragma: no cover
                        frame = None
                        errors.write(json.dumps({"index": begin + index, "path": str(path), "error": repr(exc)}) + "\n")
                    if frame is None:
                        errors.write(json.dumps({"index": begin + index, "path": str(path), "error": "decode_failed"}) + "\n")
                        frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
                    frames.append(frame)
            depth, valid = process_batch(frames, yolo, midas, transform, device, 640)
            depth_store[begin:end] = depth
            valid_store[begin:end] = valid
            depth_store.flush()
            valid_store.flush()
            done = end
            progress_path.write_text(
                json.dumps({"split": "dqn_train", "samples_done": done, "samples_total": n}),
                encoding="utf-8",
            )
            elapsed = max(time.monotonic() - started, 1e-6)
            rate = done / elapsed
            print(json.dumps({
                "split": "dqn_train",
                "samples_done": done,
                "samples_total": n,
                "samples_per_second": rate,
                "eta_minutes": (n - done) / max(rate, 1e-6) / 60.0,
            }), flush=True)

    del depth_store
    del valid_store
    os.replace(temp_depth, final_depth)
    os.replace(temp_valid, final_valid)
    if progress_path.exists():
        progress_path.unlink()
    elapsed = time.monotonic() - started
    result = {
        "split": "dqn_train",
        "samples": n,
        "shape": [n, NUM_OBJECTS],
        "dtype": "float32",
        "valid_dtype": "uint8",
        "seconds": elapsed,
        "samples_per_second": n / max(elapsed, 1e-6),
        "source_metadata": str(TRAJECTORY_CACHE),
        "output_depth": str(final_depth),
        "output_valid": str(final_valid),
        "errors": str(errors_path),
    }
    complete_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print("DEPTH_DQN_TRAIN_COMPLETE", json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
