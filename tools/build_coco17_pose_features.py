"""Re-extract the pose stream from the COCO-17 CTR-GCN.

The cached *_pose.npy / *_joint_xy.npy hold Kinect-25 features from the old
backbone. This rebuilds both from the RTMPose skeletons so the pose stream
matches what the robot will actually produce:

  pose      [N, 2, 64, 13, 17]  <- CTR-GCN_17 l4 hook
  joint_xy  [N, 13, 17, 2]      <- RTMPose x/y, already in [-1, 1]

Video and object features are untouched; sample order is inherited from the
RTMPose npz, which was built from the same *_sample_names.npy files.

Usage:
  cd /workspace/VPOCLIP_plus
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
    python tools/build_coco17_pose_features.py
"""

import argparse
import importlib
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

CTRGCN_17 = Path("/workspace/CTR-GCN_17")
RTMPOSE_NPZ = CTRGCN_17 / "data/etri_coco17/ETRI_55_CS_rtmpose_coco17_13.npz"

# Which RTMPose split feeds which VPOCLIP prefix.
TARGETS = [
    ("train", "data/contrastive_zsl_splits/55_0", "seen"),
    ("val", "data/contrastive_zsl_splits/55_0", "val"),
    ("test", "data/contrastive_test_data", "trimodal_test"),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Build COCO-17 pose features for VPOCLIP.")
    parser.add_argument("--ctrgcn-config", default=str(CTRGCN_17 / "config/etri-coco17/ctrgcn_joint_coco17_13.yaml"))
    parser.add_argument("--ctrgcn-weights", default=str(CTRGCN_17 / "work_dir/etri_coco17/ctrgcn_joint_coco17_13/runs-64-768.pt"))
    parser.add_argument("--suffix", default="_coco17", help="Written next to the existing npy files.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--window-size", type=int, default=64)
    return parser.parse_args()


def load_ctrgcn(config_path, weights, device):
    sys.path.insert(0, str(CTRGCN_17))
    with open(config_path) as handle:
        config = yaml.safe_load(handle)
    module_name, class_name = config["model"].rsplit(".", 1)
    Model = getattr(importlib.import_module(module_name), class_name)
    model = Model(**config["model_args"]).to(device)

    state = torch.load(weights, map_location=device, weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()

    captured = {}
    model.l4.register_forward_hook(lambda _m, _i, output: captured.__setitem__("l4", output))

    # CTR-GCN ships its own `model` package; drop it so VPOCLIP's model.py wins.
    for name in list(sys.modules):
        if name == "model" or name.startswith("model."):
            del sys.modules[name]
    sys.path.remove(str(CTRGCN_17))
    return model, captured


def resize_time(clips, window_size):
    """Upsample the stored 13 frames to the length the backbone was trained on."""

    tensor = torch.from_numpy(clips).float()
    batch, channels, frames, joints, persons = tensor.shape
    tensor = tensor.permute(0, 1, 3, 4, 2).reshape(batch, channels * joints * persons, frames)
    tensor = torch.nn.functional.interpolate(tensor, size=window_size, mode="linear", align_corners=False)
    tensor = tensor.reshape(batch, channels, joints, persons, window_size)
    return tensor.permute(0, 1, 4, 2, 3).contiguous()


def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, captured = load_ctrgcn(args.ctrgcn_config, args.ctrgcn_weights, device)
    print(f"CTR-GCN_17 loaded from {args.ctrgcn_weights}")

    source = np.load(RTMPOSE_NPZ, allow_pickle=True)

    for split, target_dir, prefix in TARGETS:
        clips = source[f"x_{split}"]
        names = [str(name) for name in source[f"{split}_sample_name"]]

        target_dir = Path(target_dir)
        reference = np.load(target_dir / f"{prefix}_sample_names.npy", allow_pickle=True)
        if [str(name) for name in reference] != names:
            raise ValueError(f"{prefix}: sample order does not match {target_dir}")

        features = []
        for start in range(0, len(clips), args.batch_size):
            chunk = clips[start:start + args.batch_size]
            # Feed the 64-frame window the backbone was trained on.
            batch = resize_time(chunk, args.window_size).to(device)
            with torch.inference_mode():
                model(batch)

            l4 = captured["l4"]                       # [N*M, C, T', V]
            samples, persons = chunk.shape[0], chunk.shape[-1]
            _, channels, frames, joints = l4.shape
            # Resample the time axis back to the 13 steps the fusion head wants.
            time_major = l4.permute(0, 1, 3, 2).reshape(-1, 1, frames)
            time_major = torch.nn.functional.interpolate(
                time_major, size=13, mode="linear", align_corners=False
            )
            pose_batch = time_major.reshape(samples * persons, channels, joints, 13)
            pose_batch = pose_batch.permute(0, 1, 3, 2).reshape(
                samples, persons, channels, 13, joints
            )
            features.append(pose_batch.float().cpu().numpy())
            print(f"  [{prefix}] {min(start + args.batch_size, len(clips))}/{len(clips)}", flush=True)

        pose = np.concatenate(features, axis=0).astype(np.float32)
        # Person 0 only: joint_xy places the pose maps and the RS grid takes one
        # body. x/y are already normalised to [-1, 1] by the extractor.
        joint_xy = clips[:, :2, :, :, 0].transpose(0, 2, 3, 1).astype(np.float32)

        np.save(target_dir / f"{prefix}_pose{args.suffix}.npy", pose)
        np.save(target_dir / f"{prefix}_joint_xy{args.suffix}.npy", joint_xy)
        print(f"[{prefix}] pose {pose.shape}  joint_xy {joint_xy.shape}", flush=True)


if __name__ == "__main__":
    main()
