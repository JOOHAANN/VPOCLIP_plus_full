"""Split the Kinect -> RGB pose gap into its two suspected causes.

Two things change at once when the skeleton comes from MediaPipe instead of the
Kinect csv, and the per-class pattern alone cannot tell them apart:

  joint semantics - NTU-25 has hand tips, thumbs and a spine-shoulder joint that
                    MediaPipe has no equivalent for, so the demo's mapping fills
                    three of them with duplicates of other joints.
  coordinates     - Kinect Z is absolute distance from the camera (~1.7 m),
                    MediaPipe Z is hip-relative and centred on zero.

Each variant below degrades the *Kinect* skeleton in exactly one of those ways
and leaves everything else untouched, so the accuracy drop is attributable.

Usage:
  cd /workspace/VPOCLIP_plus
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
    python tools/decompose_pose_gap.py --max-samples 500
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from test_raw_end_to_end import (  # noqa: E402
    CTRGCN_ROOT,
    X3D_ROOT,
    ObjectMapRunner,
    ctrgcn_pose_from_model,
    ctrgcn_tensor_and_joint_xy_from_csv,
    load_clipgcn_model,
    load_ctrgcn_model,
    load_split_manifest,
    load_x3d_model,
    load_yolo_model,
    read_uniform_rgb_frames,
    x3d_features_from_model,
    x3d_tensor_from_frames,
)
from train import get_device, get_path_from_config, load_config  # noqa: E402

# 0-based NTU-25 indices. The demo's MediaPipe mapping fills these from joints
# that already exist, because MediaPipe has nothing to put there.
SPINE_SHOULDER, NECK = 20, 2
LEFT_HAND_TIP, LEFT_HAND = 21, 7
RIGHT_HAND_TIP, RIGHT_HAND = 23, 11
HIP_CENTER = 0


def degrade_hand_joints(skeleton):
    """Replace the joints MediaPipe cannot supply with duplicates, as the demo does."""

    out = skeleton.clone()
    out[:, :, SPINE_SHOULDER, :] = out[:, :, NECK, :]
    out[:, :, LEFT_HAND_TIP, :] = out[:, :, LEFT_HAND, :]
    out[:, :, RIGHT_HAND_TIP, :] = out[:, :, RIGHT_HAND, :]
    return out


def relative_z(skeleton):
    """Turn absolute camera distance into a hip-relative offset, as MediaPipe has."""

    out = skeleton.clone()
    out[2] = out[2] - out[2, :, HIP_CENTER : HIP_CENTER + 1, :]
    return out


VARIANTS = {
    "kinect": lambda s: s,
    "hands_duplicated": degrade_hand_joints,
    "z_hip_relative": relative_z,
    "both": lambda s: relative_z(degrade_hand_joints(s)),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Attribute the pose sensor gap to its causes.")
    parser.add_argument("--config", default="config_55_0_objconcat.yaml")
    parser.add_argument("--clipgcn-checkpoint",
                        default="./work_dir/cs_55_0_objconcat/run_20260726_115542/best_model.pth")
    parser.add_argument("--split-dir", default="./data/contrastive_test_data")
    parser.add_argument("--class-split-dir", default="./data/contrastive_zsl_splits/50_5")
    parser.add_argument("--prefix", default="trimodal_test")
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--frames", type=int, default=13)
    parser.add_argument("--output", default="./work_dir/pose_gap_decomposition.json")

    parser.add_argument("--x3d-root", default=str(X3D_ROOT))
    parser.add_argument("--x3d-config", default=str(X3D_ROOT / "configs" / "x3d-s_clipgcn_tensor_cross_subject_70_10_20_182.yaml"))
    parser.add_argument("--x3d-checkpoint", default=str(X3D_ROOT / "outputs" / "x3d-s_clipgcn_tensor_cs_70_10_20_182" / "model_007000.pth"))
    parser.add_argument("--x3d-layer", default="s5")
    parser.add_argument("--ctrgcn-root", default=str(CTRGCN_ROOT))
    parser.add_argument("--ctrgcn-config", default=str(CTRGCN_ROOT / "work_dir" / "etri_p1_p230_13frames" / "xsub" / "ctrgcn_joint_raw" / "config.yaml"))
    parser.add_argument("--ctrgcn-weights", default=str(CTRGCN_ROOT / "work_dir" / "etri_p1_p230_13frames" / "xsub" / "ctrgcn_joint_raw" / "runs-50-2700.pt"))
    parser.add_argument("--ctrgcn-hook-layer", default="l4")
    parser.add_argument("--yolo-repo", default="/workspace/yolov5")
    parser.add_argument("--yolo-weights", default="/workspace/yolov5/runs/train/coco_custom50_fixed2/weights/best.pt")
    parser.add_argument("--yolo-size", type=int, default=640)
    parser.add_argument("--yolo-conf", type=float, default=0.25)
    parser.add_argument("--yolo-iou", type=float, default=0.45)
    parser.add_argument("--yolo-half", action="store_true")
    parser.add_argument("--yolo-detect-every", type=int, default=1)
    parser.add_argument("--no-yolo", action="store_true")
    parser.add_argument("--object-grid-size", type=int, default=6)
    parser.add_argument("--object-value", choices=["presence", "confidence"], default="presence")
    parser.add_argument("--object-max-distance-weight", type=float, default=10.0)
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = str(ROOT / args.config)
    config = load_config(config_path)
    device = get_device(config["runtime"].get("device"))

    class_split_dir = get_path_from_config(config_path, args.class_split_dir)
    metadata = json.load(open(Path(class_split_dir) / "metadata.json"))
    candidate_labels = sorted(metadata["seen_classes"] + metadata["unseen_classes"])

    items = load_split_manifest(
        get_path_from_config(config_path, args.split_dir),
        args.prefix,
        get_path_from_config(config_path, args.data_root),
        max_samples=args.max_samples,
    )
    print(f"clips: {len(items)}   variants: {list(VARIANTS)}")

    x3d_model, x3d_captured, x3d_hook, x3d_cfg = load_x3d_model(args, device)
    ctrgcn_model, ctrgcn_captured, ctrgcn_hook = load_ctrgcn_model(args, device)
    object_runner = ObjectMapRunner(load_yolo_model(args, device), device, args)
    clipgcn_model, checkpoint_path, candidate_labels = load_clipgcn_model(
        args, config, config_path, device, candidate_labels
    )
    candidate_labels = [int(label) for label in candidate_labels]
    print(f"checkpoint: {checkpoint_path}")

    records = []
    start = time.time()
    for index, item in enumerate(items):
        frames = read_uniform_rgb_frames(item["video_path"], args.frames)
        x3d_clip = x3d_tensor_from_frames(
            frames,
            int(x3d_cfg.TRANSFORM.TEST.TENSOR_RESIZE_SIZE),
            x3d_cfg.TRANSFORM.MEAN,
            x3d_cfg.TRANSFORM.STD,
        ).unsqueeze(0).to(device)
        skeleton, joint_xy = ctrgcn_tensor_and_joint_xy_from_csv(item["csv_path"], args.frames)
        joint_xy = joint_xy.unsqueeze(0).to(device)

        with torch.inference_mode():
            object_maps = object_runner([frames[len(frames) // 2]])
            video_features = x3d_features_from_model(x3d_model, x3d_captured, x3d_clip, args)
            record = {"sample_name": item["sample_name"], "label": int(item["label"])}
            for name, transform in VARIANTS.items():
                pose_features = ctrgcn_pose_from_model(
                    ctrgcn_model, ctrgcn_captured, transform(skeleton).unsqueeze(0).to(device), args
                )
                logits = clipgcn_model(video_features, pose_features, object_maps, joint_xy)
                record[name] = int(candidate_labels[int(logits[0].argmax())])
        records.append(record)

        if (index + 1) % 50 == 0:
            accs = {n: np.mean([r[n] == r["label"] for r in records]) for n in VARIANTS}
            rate = (index + 1) / (time.time() - start)
            print(f"  [{index + 1}/{len(items)}] "
                  + "  ".join(f"{n}={a:.3f}" for n, a in accs.items())
                  + f"  ({rate:.1f} clip/s)", flush=True)

    x3d_hook.remove()
    ctrgcn_hook.remove()

    labels = np.array([r["label"] for r in records])
    summary = {n: float((np.array([r[n] for r in records]) == labels).mean()) for n in VARIANTS}
    output = get_path_from_config(config_path, args.output)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "records": records}, handle, indent=2)

    print("\n========== gap decomposition (Kinect skeleton, one factor at a time) ==========")
    base = summary["kinect"]
    for name, acc in summary.items():
        print(f"{name:20s} {acc * 100:6.2f}%   drop vs kinect: {(base - acc) * 100:+6.2f}")
    print(f"\nsaved: {output}")


if __name__ == "__main__":
    main()
