#!/usr/bin/env python3
"""Extract sample-aligned CTR-GCN l4 features from the current RTMPose cache."""

import argparse
import importlib
import json
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import yaml


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ctrgcn-root", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--npz", type=Path, required=True)
    p.add_argument("--split", choices=("train", "val", "test", "dqn"), required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--window-size", type=int, default=64)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--max-samples", type=int)
    return p.parse_args()


def resize_time(clips, window_size):
    x = torch.from_numpy(clips).float()
    n, c, t, v, m = x.shape
    x = x.permute(0, 1, 3, 4, 2).reshape(n, c * v * m, t)
    x = torch.nn.functional.interpolate(x, size=window_size, mode="linear", align_corners=False)
    return x.reshape(n, c, v, m, window_size).permute(0, 1, 4, 2, 3).contiguous()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.ctrgcn_root.resolve()))
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    module_name, class_name = config["model"].rsplit(".", 1)
    model_class = getattr(importlib.import_module(module_name), class_name)
    device = torch.device(args.device)
    model = model_class(**config["model_args"]).to(device)
    state = torch.load(args.weights, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = OrderedDict((k.removeprefix("module."), v) for k, v in state.items())
    model.load_state_dict(state, strict=True)
    model.eval()
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    captured = {}
    hook = model.l4.register_forward_hook(lambda _m, _i, out: captured.__setitem__("l4", out.detach()))
    source = np.load(args.npz, allow_pickle=True)
    clips = source[f"x_{args.split}"]
    labels = np.asarray(source[f"y_{args.split}"], dtype=np.int64)
    names = np.asarray(source[f"{args.split}_sample_name"]).astype(str)
    if args.max_samples is not None:
        clips, labels, names = clips[: args.max_samples], labels[: args.max_samples], names[: args.max_samples]

    pose_path = args.output_dir / f"{args.split}_pose_coco17.npy"
    pose = None
    for start in range(0, len(clips), args.batch_size):
        stop = min(start + args.batch_size, len(clips))
        batch = resize_time(clips[start:stop], args.window_size).to(device, non_blocking=True)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            model(batch)
        feature = captured["l4"].float()
        n, persons = stop - start, clips.shape[-1]
        _nm, channels, frames, joints = feature.shape
        feature = feature.permute(0, 1, 3, 2).reshape(-1, 1, frames)
        feature = torch.nn.functional.interpolate(feature, size=13, mode="linear", align_corners=False)
        feature = feature.reshape(n * persons, channels, joints, 13)
        feature = feature.permute(0, 1, 3, 2).reshape(n, persons, channels, 13, joints)
        feature = feature.cpu().numpy().astype(np.float32, copy=False)
        if pose is None:
            pose = np.lib.format.open_memmap(
                pose_path, mode="w+", dtype=np.float32,
                shape=(len(clips), persons, channels, 13, joints),
            )
            print(f"pose contract: {pose.shape} {pose.dtype}", flush=True)
        pose[start:stop] = feature
        print(f"[{args.split}] {stop}/{len(clips)}", flush=True)
    hook.remove()
    if pose is not None:
        pose.flush()
    joint_xy = clips[:, :2, :, :, 0].transpose(0, 2, 3, 1).astype(np.float32)
    np.save(args.output_dir / f"{args.split}_joint_xy_coco17.npy", joint_xy)
    np.save(args.output_dir / f"{args.split}_labels.npy", labels)
    np.save(args.output_dir / f"{args.split}_sample_names.npy", names)
    metadata = {
        "split": args.split, "samples": len(clips), "pose_shape": list(pose.shape),
        "joint_xy_shape": list(joint_xy.shape), "weights": str(args.weights.resolve()),
        "rtmpose_npz": str(args.npz.resolve()), "hook": "l4", "window_size": args.window_size,
    }
    (args.output_dir / f"{args.split}_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
