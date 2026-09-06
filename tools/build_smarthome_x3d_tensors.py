"""Decode Toyota Smarthome clips into the tensor format X3D trains on.

Mirrors data/clipgcn_tensor_cs_70_10_20 from the ETRI pipeline: 13 uniformly
sampled frames per clip, resized to 160x160, ImageNet-normalised and stored as
float16 in [N, 3, 13, 160, 160]. Decoding is CPU-bound, so this can run while
the GPU is busy (or unavailable).

Usage:
  cd /workspace/VPOCLIP_plus
  python tools/build_smarthome_x3d_tensors.py --workers 16
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def parse_args():
    parser = argparse.ArgumentParser(description="Smarthome -> X3D tensor cache.")
    parser.add_argument("--video-dir", default="data/toyota_smarthome/mp4")
    parser.add_argument("--split-dir", default="data/smarthome_splits")
    parser.add_argument("--output-dir", default="/workspace/X3D/data/smarthome_tensor")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--frames", type=int, default=13)
    parser.add_argument("--size", type=int, default=160)
    parser.add_argument("--workers", type=int, default=16)
    return parser.parse_args()


def load_clip(video_path, frames, size):
    """13 uniform frames, resized and normalised, as [3, T, size, size]."""

    cv2.setNumThreads(1)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return None
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        capture.release()
        return None

    wanted = np.linspace(0, total - 1, frames).round().astype(np.int64)
    picked = []
    position = 0
    index = 0
    # One sequential pass: seeking per frame re-decodes from the previous
    # keyframe and is several times slower.
    while index < len(wanted):
        ok, frame = capture.read()
        if not ok:
            break
        while index < len(wanted) and wanted[index] == position:
            picked.append(frame)
            index += 1
        position += 1
    capture.release()
    if not picked:
        return None
    while len(picked) < frames:
        picked.append(picked[-1])

    clip = np.empty((frames, 3, size, size), dtype=np.float32)
    for t, frame in enumerate(picked):
        resized = cv2.resize(frame, (size, size), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).astype(np.float32) / 255.0
        clip[t] = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    return clip.transpose(1, 0, 2, 3)  # [3, T, H, W]


def main():
    args = parse_args()
    split_dir = Path(args.split_dir)
    video_dir = Path(args.video_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((split_dir / "metadata.json").read_text())

    shapes = {}
    for split in args.splits:
        target = output_dir / f"{split}_float16.npy"
        if target.exists():
            print(f"[{split}] already built, skipping")
            shapes[split] = list(np.load(target, mmap_mode="r").shape)
            continue

        names = [str(n) for n in np.load(split_dir / f"{split}_sample_names.npy", allow_pickle=True)]
        labels = np.load(split_dir / f"{split}_labels.npy")
        print(f"[{split}] {len(names)} clips", flush=True)

        # Written straight to disk: the full array is ~30 GB across splits.
        cache = np.lib.format.open_memmap(
            target, mode="w+", dtype=np.float16,
            shape=(len(names), 3, args.frames, args.size, args.size),
        )

        misses = 0
        start = time.time()
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {}
            for index, name in enumerate(names):
                futures[index] = pool.submit(load_clip, video_dir / name, args.frames, args.size)
                # Keep the queue bounded so memory stays flat.
                if len(futures) >= args.workers * 4 or index == len(names) - 1:
                    for done_index in sorted(futures):
                        clip = futures[done_index].result()
                        if clip is None:
                            misses += 1
                        else:
                            cache[done_index] = clip.astype(np.float16)
                    futures = {}
                    if (index + 1) % 500 == 0 or index == len(names) - 1:
                        rate = (index + 1) / (time.time() - start)
                        eta = (len(names) - index - 1) / rate / 60
                        print(f"  [{split} {index + 1}/{len(names)}] {rate:.1f} clip/s, eta {eta:.0f} min",
                              flush=True)

        cache.flush()
        np.save(output_dir / f"{split}_labels.npy", labels)
        np.save(output_dir / f"{split}_sample_names.npy", np.array(names, dtype=object))
        shapes[split] = list(cache.shape)
        if misses:
            print(f"  [{split}] unreadable clips: {misses}")

    (output_dir / "metadata.json").write_text(json.dumps({
        "source_root": str(video_dir),
        "dataset": "toyota_smarthome",
        "shape": shapes,
        "classes": metadata["actions"],
        "class_to_idx": metadata["action_to_label"],
        "split_rule": "cross-subject 70:10:20 over 18 subjects",
        "split_subjects": metadata["split_subjects"],
        "dtype": "float16",
        "normalisation": "imagenet",
    }, indent=2))
    print(f"\nwritten to {output_dir}")
    for split, shape in shapes.items():
        print(f"  {split}: {shape}")


if __name__ == "__main__":
    main()
