"""COCO-17 pose features for the cross-subject 50/5 split.

Every clip of the CS 50/5 prefixes is already covered by the RTMPose npz
(trimodal_train/val are subsets of the 55_0 seen/val clips, the two test
prefixes partition the 1601 test clips), so the skeletons are looked up by
sample name and only the CTR-GCN forward is new.

The 55-class backbone is used on purpose: the Kinect 50/5 baseline extracted
its pose stream with the 55-class Kinect backbone, so this keeps the
comparison to one variable (skeleton source). The 50-class backbone trained
for the leakage ablation can be swapped in via --ctrgcn-weights + --suffix.

Usage:
  cd /workspace/VPOCLIP_plus
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
    python tools/build_coco17_pose_features_cs50_5.py
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from build_coco17_pose_features import CTRGCN_17, RTMPOSE_NPZ, load_ctrgcn, resize_time

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SPLIT_DIR = ROOT / "data/contrastive_cs_70_10_20_zsl_50_5"


def parse_args():
    parser = argparse.ArgumentParser(description="COCO-17 pose features for the CS 50/5 split.")
    parser.add_argument("--ctrgcn-config", default=str(CTRGCN_17 / "config/etri-coco17/ctrgcn_joint_coco17_13.yaml"))
    parser.add_argument("--ctrgcn-weights", default=str(CTRGCN_17 / "work_dir/etri_coco17/ctrgcn_joint_coco17_13/runs-64-768.pt"))
    parser.add_argument("--split-dir", default=str(DEFAULT_SPLIT_DIR))
    parser.add_argument("--suffix", default="_coco17")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--window-size", type=int, default=64)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, captured = load_ctrgcn(args.ctrgcn_config, args.ctrgcn_weights, device)
    print(f"backbone: {args.ctrgcn_weights}")

    source = np.load(RTMPOSE_NPZ, allow_pickle=True)
    skeletons = {}
    for split in ("train", "val", "test"):
        for name, clip in zip(source[f"{split}_sample_name"], source[f"x_{split}"]):
            skeletons[str(name)] = clip

    target_dir = Path(args.split_dir)
    for prefix in ("trimodal_train", "trimodal_val", "trimodal_test", "trimodal_test_seen"):
        names = [str(n) for n in np.load(target_dir / f"{prefix}_sample_names.npy", allow_pickle=True)]
        missing = [n for n in names if n not in skeletons]
        if missing:
            raise ValueError(f"{prefix}: {len(missing)} clips missing from the RTMPose npz, e.g. {missing[0]}")
        clips = np.stack([skeletons[n] for n in names])

        features = []
        for start in range(0, len(clips), args.batch_size):
            chunk = clips[start:start + args.batch_size]
            batch = resize_time(chunk, args.window_size).to(device)
            with torch.inference_mode():
                model(batch)
            l4 = captured["l4"]
            samples, persons = chunk.shape[0], chunk.shape[-1]
            _, channels, frames, joints = l4.shape
            time_major = l4.permute(0, 1, 3, 2).reshape(-1, 1, frames)
            time_major = torch.nn.functional.interpolate(
                time_major, size=13, mode="linear", align_corners=False
            )
            pose_batch = time_major.reshape(samples * persons, channels, joints, 13)
            pose_batch = pose_batch.permute(0, 1, 3, 2).reshape(samples, persons, channels, 13, joints)
            features.append(pose_batch.float().cpu().numpy())

        pose = np.concatenate(features, axis=0).astype(np.float32)
        joint_xy = clips[:, :2, :, :, 0].transpose(0, 2, 3, 1).astype(np.float32)
        np.save(target_dir / f"{prefix}_pose{args.suffix}.npy", pose)
        np.save(target_dir / f"{prefix}_joint_xy{args.suffix}.npy", joint_xy)
        print(f"[{prefix}] pose {pose.shape}  joint_xy {joint_xy.shape}", flush=True)


if __name__ == "__main__":
    main()
