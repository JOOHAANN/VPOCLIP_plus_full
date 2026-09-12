#!/usr/bin/env python3
"""Extract one relative-depth value for each of the 50 YOLO object slots.

The input is the strict new50/5 trimodal sample manifest.  Every manifest row
is one camera video (the ``Cxxx.mp4`` suffix is therefore part of the sample
identity).  For each row this script:

1. decodes the first frame and resizes it to the deployment shape 640x480;
2. runs the same custom-50 YOLO checkpoint used by the VPOCLIP object cache;
3. runs MiDaS_small on that exact frame;
4. takes the median depth inside the central part of the highest-confidence
   detection for each class slot; and
5. saves a 50-value float32 vector plus a validity mask.

The saved depth is a per-frame percentile-normalized MiDaS relative inverse
depth in [0, 1].  It is not metric depth.  The depth map itself is kept in
memory and is not saved, so storage stays small.

The script is resumable at split/chunk granularity.  It writes only to a new
output directory and never modifies the existing VPOCLIP caches.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F


NUM_OBJECTS = 50
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
DEFAULT_DATA_ROOT = Path("/home/youhan/ws/VPOCLIP_plus_full/data/new50_5_group1_zsl")
DEFAULT_VIDEO_ROOT = Path("/home/youhan/ws/ETRI-Activity3D-RGB")
DEFAULT_OUTPUT_ROOT = Path(
    "/home/youhan/ws/VPOCLIP_plus_full/data/new50_5_group1_zsl_depth_first_frame_fp32"
)
DEFAULT_YOLO_REPO = Path("/home/youhan/ws/yolov5_full")
DEFAULT_YOLO_WEIGHTS = DEFAULT_YOLO_REPO / "best.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--yolo-repo", type=Path, default=DEFAULT_YOLO_REPO)
    parser.add_argument("--yolo-weights", type=Path, default=DEFAULT_YOLO_WEIGHTS)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test_seen", "test"],
        choices=["train", "val", "test_seen", "test"],
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--decode-workers", type=int, default=16)
    parser.add_argument("--yolo-size", type=int, default=640)
    parser.add_argument("--yolo-conf", type=float, default=0.25)
    parser.add_argument("--yolo-iou", type=float, default=0.45)
    parser.add_argument("--midas-device", default="cuda:0")
    parser.add_argument("--no-half-yolo", action="store_true")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Only process this many rows per split; intended for smoke tests.",
    )
    return parser.parse_args()


def read_first_frame(path: Path) -> np.ndarray | None:
    """Decode and resize one frame using the deployment's 640x480 shape."""

    capture = cv2.VideoCapture(str(path))
    ok, frame = capture.read()
    capture.release()
    if not ok or frame is None:
        return None
    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def load_yolo(args: argparse.Namespace, device: torch.device) -> Any:
    model = torch.hub.load(
        str(args.yolo_repo),
        "custom",
        path=str(args.yolo_weights),
        source="local",
        verbose=False,
    )
    model.to(device).eval()
    model.conf = args.yolo_conf
    model.iou = args.yolo_iou
    names = getattr(model, "names", None)
    if names is not None and len(names) != NUM_OBJECTS:
        raise RuntimeError(f"expected 50 YOLO classes, got {len(names)}")
    if not args.no_half_yolo and device.type == "cuda":
        model.half()
    return model


def load_midas(device: torch.device) -> tuple[Any, Any]:
    model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", pretrained=True)
    transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = transforms.small_transform
    model.to(device).eval()
    return model, transform


def transform_batch(frames: list[np.ndarray], transform: Any, device: torch.device) -> torch.Tensor:
    tensors = []
    for frame in frames:
        value = transform(frame)
        if isinstance(value, np.ndarray):
            value = torch.from_numpy(value)
        if value.ndim == 4:
            value = value[0]
        tensors.append(value)
    return torch.stack(tensors, dim=0).to(device, non_blocking=True)


def normalize_relative_depth(prediction: torch.Tensor) -> torch.Tensor:
    """Normalize each frame independently to robust [0, 1] relative depth."""

    flat = prediction.flatten(1)
    low = torch.quantile(flat, 0.05, dim=1).view(-1, 1, 1)
    high = torch.quantile(flat, 0.95, dim=1).view(-1, 1, 1)
    scale = (high - low).clamp_min(1e-6)
    return ((prediction - low) / scale).clamp(0.0, 1.0)


def sample_box_depth(depth: torch.Tensor, box: list[float]) -> float:
    """Median depth from the central 50% of one detection box."""

    height, width = depth.shape[-2:]
    x1, y1, x2, y2 = box
    # Use the inner half of the box to reduce boundary/background leakage.
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    half_w = max((x2 - x1) * 0.25, 1.0)
    half_h = max((y2 - y1) * 0.25, 1.0)
    left = max(0, min(width - 1, int(round(cx - half_w))))
    right = max(left + 1, min(width, int(round(cx + half_w))))
    top = max(0, min(height - 1, int(round(cy - half_h))))
    bottom = max(top + 1, min(height, int(round(cy + half_h))))
    crop = depth[top:bottom, left:right]
    return float(torch.median(crop).item())


def process_batch(
    frames: list[np.ndarray],
    yolo: Any,
    midas: Any,
    transform: Any,
    device: torch.device,
    yolo_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``depth[N,50]`` and ``valid[N,50]`` for a frame batch."""

    n = len(frames)
    depth_values = np.zeros((n, NUM_OBJECTS), dtype=np.float32)
    valid_values = np.zeros((n, NUM_OBJECTS), dtype=np.uint8)
    if n == 0:
        return depth_values, valid_values

    with torch.inference_mode():
        detections = yolo(frames, size=yolo_size).xyxy
        midas_input = transform_batch(frames, transform, device)
        prediction = midas(midas_input)
        if prediction.ndim == 4:
            prediction = prediction[:, 0]
        prediction = normalize_relative_depth(prediction)

    # The YOLO output coordinates are in the 640x480 frame coordinates.  The
    # normalized depth map keeps the same aspect ratio, so map by scale.
    depth_h, depth_w = prediction.shape[-2:]
    scale_x = depth_w / float(FRAME_WIDTH)
    scale_y = depth_h / float(FRAME_HEIGHT)
    for index, det in enumerate(detections):
        best_by_class: dict[int, tuple[float, list[float]]] = {}
        for row in det:
            class_id = int(row[5].item())
            if not 0 <= class_id < NUM_OBJECTS:
                continue
            confidence = float(row[4].item())
            if confidence <= best_by_class.get(class_id, (-1.0, []))[0]:
                continue
            coordinates = [float(value.item()) for value in row[:4]]
            coordinates = [
                coordinates[0] * scale_x,
                coordinates[1] * scale_y,
                coordinates[2] * scale_x,
                coordinates[3] * scale_y,
            ]
            best_by_class[class_id] = (confidence, coordinates)
        for class_id, (_, box) in best_by_class.items():
            depth_values[index, class_id] = sample_box_depth(prediction[index], box)
            valid_values[index, class_id] = 1
    return depth_values, valid_values


def split_paths(data_root: Path, video_root: Path, split: str) -> list[Path]:
    names_path = data_root / f"trimodal_{split}_sample_names.npy"
    names = [str(value) for value in np.load(names_path, allow_pickle=True)]
    return [video_root / name for name in names]


def write_metadata(output_root: Path, args: argparse.Namespace) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    metadata_path = output_root / "metadata.json"
    if metadata_path.exists():
        return
    metadata = {
        "protocol": "strict_zsl_50_5",
        "source_data_root": str(args.data_root),
        "source_video_root": str(args.video_root),
        "yolo_weights": str(args.yolo_weights),
        "yolo_detector": "custom50, conf=0.25, iou=0.45; highest-confidence detection per class slot",
        "frame": "first decoded frame, resized to 640x480",
        "midas": "MiDaS_small relative inverse depth",
        "depth_value": "median over central 50% of each detected bounding box",
        "normalization": "per-frame 5th-95th percentile to [0,1]",
        "absent_slot": 0.0,
        "dtype": "float32",
        "arrays": {
            "depth_rel_fp32.npy": "[N,50]",
            "depth_valid_uint8.npy": "[N,50]",
        },
        "note": "This is relative depth, not metric XYZ depth.",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


def process_split(args: argparse.Namespace, split: str, yolo: Any, midas: Any, transform: Any, device: torch.device) -> dict[str, Any]:
    paths = split_paths(args.data_root, args.video_root, split)
    if args.max_samples is not None:
        paths = paths[: max(0, args.max_samples)]
    output_dir = args.output_root / split
    output_dir.mkdir(parents=True, exist_ok=True)
    final_depth = output_dir / "depth_rel_fp32.npy"
    final_valid = output_dir / "depth_valid_uint8.npy"
    names_path = output_dir / "sample_names.npy"
    complete_path = output_dir / "complete.json"
    progress_path = output_dir / "progress.json"
    temp_depth = output_dir / "depth_rel_fp32.tmp.npy"
    temp_valid = output_dir / "depth_valid_uint8.tmp.npy"

    if complete_path.exists() and final_depth.exists() and final_valid.exists():
        return json.loads(complete_path.read_text(encoding="utf-8"))

    n = len(paths)
    if not names_path.exists():
        names = np.load(args.data_root / f"trimodal_{split}_sample_names.npy", allow_pickle=True)
        if args.max_samples is not None:
            names = names[:n]
        np.save(names_path, names)

    if temp_depth.exists() and temp_valid.exists() and progress_path.exists():
        depth_store = np.lib.format.open_memmap(temp_depth, mode="r+")
        valid_store = np.lib.format.open_memmap(temp_valid, mode="r+")
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        start = int(progress.get("samples_done", 0))
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

    errors_path = output_dir / "errors.jsonl"
    started = time.monotonic()
    processed = start
    with errors_path.open("a", encoding="utf-8") as errors:
        for begin in range(start, n, args.batch_size):
            end = min(begin + args.batch_size, n)
            batch_paths = paths[begin:end]
            frames: list[np.ndarray] = [None] * len(batch_paths)  # type: ignore[list-item]
            with ThreadPoolExecutor(max_workers=args.decode_workers) as pool:
                futures = [pool.submit(read_first_frame, path) for path in batch_paths]
                for offset, (path, future) in enumerate(zip(batch_paths, futures)):
                    try:
                        frame = future.result()
                    except Exception as exc:  # pragma: no cover - defensive path
                        frame = None
                        errors.write(json.dumps({"index": begin + offset, "path": str(path), "error": repr(exc)}) + "\n")
                    if frame is None:
                        errors.write(json.dumps({"index": begin + offset, "path": str(path), "error": "decode_failed"}) + "\n")
                        frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
                    frames[offset] = frame

            batch_depth, batch_valid = process_batch(
                frames, yolo, midas, transform, device, args.yolo_size
            )
            depth_store[begin:end] = batch_depth
            valid_store[begin:end] = batch_valid
            depth_store.flush()
            valid_store.flush()
            processed = end
            progress_path.write_text(
                json.dumps({"split": split, "samples_done": processed, "samples_total": n}),
                encoding="utf-8",
            )
            elapsed = max(time.monotonic() - started, 1e-6)
            rate = processed / elapsed
            print(
                json.dumps(
                    {
                        "split": split,
                        "samples_done": processed,
                        "samples_total": n,
                        "samples_per_second": rate,
                        "eta_minutes": (n - processed) / max(rate, 1e-6) / 60.0,
                    }
                ),
                flush=True,
            )

    del depth_store
    del valid_store
    os.replace(temp_depth, final_depth)
    os.replace(temp_valid, final_valid)
    if progress_path.exists():
        progress_path.unlink()
    elapsed = time.monotonic() - started
    result = {
        "split": split,
        "samples": n,
        "shape": [n, NUM_OBJECTS],
        "dtype": "float32",
        "valid_dtype": "uint8",
        "seconds": elapsed,
        "samples_per_second": n / max(elapsed, 1e-6),
        "output_depth": str(final_depth),
        "output_valid": str(final_valid),
        "errors": str(errors_path),
    }
    complete_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print("DEPTH_SPLIT_COMPLETE", json.dumps(result, ensure_ascii=False), flush=True)
    return result


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for full extraction")
    device = torch.device(args.midas_device)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_metadata(args.output_root, args)
    print(json.dumps({"device": str(device), "gpu": torch.cuda.get_device_name(device), "output_root": str(args.output_root)}), flush=True)
    yolo = load_yolo(args, device)
    midas, transform = load_midas(device)
    for split in args.splits:
        process_split(args, split, yolo, midas, transform, device)


if __name__ == "__main__":
    main()
