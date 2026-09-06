"""Build object RS maps for Toyota Smarthome with the custom50 YOLO detector.

Same construction as the ETRI pipeline: detections on the clip's middle frame
become a [50, 6, 6] map where each channel holds an inverse-distance field
around that category's centre. Channel 0 is "person", which the realtime
gating also reads.

Usage:
  cd /workspace/VPOCLIP_plus
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
    python tools/extract_smarthome_objects.py
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from test_raw_end_to_end import load_yolo_model, object_maps_from_yolo, read_uniform_rgb_frames

VIDEO_ROOT = Path("data/toyota_smarthome/mp4")
SPLIT_DIR = Path("data/smarthome_splits")


def parse_args():
    parser = argparse.ArgumentParser(description="Smarthome object RS maps.")
    parser.add_argument("--output-dir", default="data/smarthome_objects")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--frames", type=int, default=13)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--yolo-repo", default="/workspace/yolov5")
    parser.add_argument("--yolo-weights",
                        default="/workspace/yolov5/runs/train/coco_custom50_fixed2/weights/best.pt")
    parser.add_argument("--yolo-size", type=int, default=640)
    parser.add_argument("--yolo-conf", type=float, default=0.25)
    parser.add_argument("--yolo-iou", type=float, default=0.45)
    parser.add_argument("--yolo-half", action="store_true")
    parser.add_argument("--no-yolo", action="store_true")
    parser.add_argument("--object-grid-size", type=int, default=6)
    parser.add_argument("--object-value", choices=["presence", "confidence"], default="presence")
    parser.add_argument("--object-max-distance-weight", type=float, default=10.0)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    yolo = load_yolo_model(args, device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        target = output_dir / f"{split}_object.npy"
        if target.exists():
            print(f"[{split}] already built, skipping")
            continue

        names = [str(n) for n in np.load(SPLIT_DIR / f"{split}_sample_names.npy", allow_pickle=True)]
        print(f"[{split}] {len(names)} clips", flush=True)
        maps = np.zeros((len(names), 50, args.object_grid_size, args.object_grid_size), dtype=np.float32)

        start = time.time()
        for begin in range(0, len(names), args.batch_size):
            chunk = names[begin:begin + args.batch_size]
            middles = []
            keep = []
            for offset, name in enumerate(chunk):
                frames = read_uniform_rgb_frames(VIDEO_ROOT / name, args.frames)
                if frames is not None and len(frames) > 0:
                    middles.append(frames[len(frames) // 2])
                    keep.append(begin + offset)
            if middles:
                with torch.inference_mode():
                    batch = object_maps_from_yolo(yolo, middles, device, args)
                maps[keep] = batch.float().cpu().numpy()

            done = min(begin + args.batch_size, len(names))
            if done % (args.batch_size * 20) == 0 or done == len(names):
                rate = done / (time.time() - start)
                print(f"  [{split} {done}/{len(names)}] {rate:.1f} clip/s, "
                      f"eta {(len(names) - done) / rate / 60:.0f} min", flush=True)

        np.save(target, maps)
        person = (maps[:, 0].max(axis=(1, 2)) > 0).mean() * 100
        print(f"[{split}] saved {maps.shape}, person detected in {person:.1f}% of clips", flush=True)


if __name__ == "__main__":
    main()
