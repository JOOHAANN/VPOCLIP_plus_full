"""Assemble the tri-modal feature arrays for Toyota Smarthome.

Runs the two trained Smarthome backbones over the cached clips and writes the
four arrays the fusion head consumes, in the same layout as the ETRI splits:

  video     [N, 13, 192, 6, 6]   X3D s5 hook
  pose      [N, 2, 64, 13, 17]   CTR-GCN_17 l4 hook
  object    [N, 50, 6, 6]        already produced by the YOLO pass
  joint_xy  [N, 13, 17, 2]       RTMPose x/y, already in [-1, 1]

The ZSL views (zsl_train / zsl_test_seen / zsl_test_unseen) are then sliced out
of the per-split arrays by sample name, so every view shares one extraction.

Usage:
  cd /workspace/VPOCLIP_plus
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
    python tools/build_smarthome_trimodal.py
"""

import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

X3D_ROOT = Path("/workspace/X3D")
CTRGCN_ROOT = Path("/workspace/CTR-GCN_17")
SPLIT_DIR = ROOT / "data/smarthome_splits"
TENSOR_DIR = X3D_ROOT / "data/smarthome_tensor"
POSE_NPZ = CTRGCN_ROOT / "data/smarthome_coco17/SMARTHOME_31_CS_rtmpose_coco17_13.npz"
OBJECT_DIR = ROOT / "data/smarthome_objects"


def parse_args():
    parser = argparse.ArgumentParser(description="Smarthome tri-modal assembly.")
    parser.add_argument("--x3d-config", default=str(X3D_ROOT / "configs/x3d-s_smarthome_182.yaml"))
    parser.add_argument("--x3d-checkpoint", default=str(X3D_ROOT / "outputs/x3d-s_smarthome_182/model_final.pth"))
    parser.add_argument("--ctrgcn-config", default=str(CTRGCN_ROOT / "config/etri-coco17/ctrgcn_smarthome_coco17_13.yaml"))
    parser.add_argument("--ctrgcn-weights", default=str(CTRGCN_ROOT / "work_dir/smarthome_coco17/ctrgcn_joint_coco17_13/runs-58-4930.pt"))
    parser.add_argument("--output-dir", default="data/smarthome_trimodal")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--window-size", type=int, default=64)
    return parser.parse_args()


def load_x3d(config_path, checkpoint, device):
    sys.path.insert(0, str(X3D_ROOT))
    cwd = os.getcwd()
    os.chdir(X3D_ROOT)  # vendored yacs config uses relative paths
    from tsn.config import get_cfg_defaults
    from tsn.model.recognizers.build import build_recognizer

    cfg = get_cfg_defaults()
    cfg.merge_from_file(config_path)
    cfg.defrost()
    cfg.NUM_GPUS = 1
    cfg.MODEL.PRETRAINED = checkpoint
    cfg.freeze()
    model = build_recognizer(cfg, device=device)
    model.eval()
    os.chdir(cwd)

    captured = {}
    model.s5.register_forward_hook(lambda _m, _i, out: captured.__setitem__("s5", out))
    return model, captured, cfg


def load_ctrgcn(config_path, weights, device):
    sys.path.insert(0, str(CTRGCN_ROOT))
    with open(config_path) as handle:
        config = yaml.safe_load(handle)
    module_name, class_name = config["model"].rsplit(".", 1)
    Model = getattr(importlib.import_module(module_name), class_name)
    model = Model(**config["model_args"]).to(device)

    state = torch.load(weights, map_location=device, weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict({(k[7:] if k.startswith("module.") else k): v for k, v in state.items()})
    model.eval()

    captured = {}
    model.l4.register_forward_hook(lambda _m, _i, out: captured.__setitem__("l4", out))
    # CTR-GCN ships its own `model` package; drop it so VPOCLIP's model.py wins.
    for name in list(sys.modules):
        if name == "model" or name.startswith("model."):
            del sys.modules[name]
    sys.path.remove(str(CTRGCN_ROOT))
    return model, captured


def resize_time(clips, window_size):
    tensor = torch.from_numpy(clips).float()
    batch, channels, frames, joints, persons = tensor.shape
    tensor = tensor.permute(0, 1, 3, 4, 2).reshape(batch, channels * joints * persons, frames)
    tensor = torch.nn.functional.interpolate(tensor, size=window_size, mode="linear", align_corners=False)
    tensor = tensor.reshape(batch, channels, joints, persons, window_size)
    return tensor.permute(0, 1, 4, 2, 3).contiguous()


def main():
    args = parse_args()
    device = torch.device("cuda:0")
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    x3d, x3d_captured, x3d_cfg = load_x3d(args.x3d_config, args.x3d_checkpoint, device)
    ctrgcn, ctrgcn_captured = load_ctrgcn(args.ctrgcn_config, args.ctrgcn_weights, device)
    resize_to = int(x3d_cfg.TRANSFORM.TEST.TENSOR_RESIZE_SIZE)
    print(f"backbones loaded, X3D input resized to {resize_to}")

    pose_source = np.load(POSE_NPZ, allow_pickle=True)

    for split in ("train", "val", "test"):
        video_out = output_dir / f"{split}_video.npy"
        if video_out.exists():
            print(f"[{split}] already built, skipping")
            continue

        clips = np.load(TENSOR_DIR / f"{split}_float16.npy", mmap_mode="r")
        names = [str(n) for n in np.load(SPLIT_DIR / f"{split}_sample_names.npy", allow_pickle=True)]
        pose_clips = pose_source[f"x_{split}"]
        pose_names = [str(n) for n in pose_source[f"{split}_sample_name"]]
        if pose_names != names:
            raise ValueError(f"{split}: pose sample order does not match the split manifest")
        print(f"[{split}] {len(names)} clips", flush=True)

        video_cache = np.lib.format.open_memmap(
            video_out, mode="w+", dtype=np.float32, shape=(len(names), 13, 192, 6, 6))
        pose_cache = np.lib.format.open_memmap(
            output_dir / f"{split}_pose_coco17.npy", mode="w+", dtype=np.float32,
            shape=(len(names), 2, 64, 13, 17))

        start = time.time()
        for begin in range(0, len(names), args.batch_size):
            end = min(begin + args.batch_size, len(names))

            batch = torch.from_numpy(np.array(clips[begin:end])).float().to(device)
            samples, channels, frames = batch.shape[:3]
            flat = batch.transpose(1, 2).reshape(samples * frames, channels, *batch.shape[3:])
            flat = torch.nn.functional.interpolate(
                flat, size=(resize_to, resize_to), mode="bilinear", align_corners=False)
            flat = flat.reshape(samples, frames, channels, resize_to, resize_to).transpose(1, 2).contiguous()
            with torch.inference_mode():
                x3d(flat)
            # s5 is [N, 192, T, 6, 6]; the fusion head wants time first.
            video_cache[begin:end] = x3d_captured["s5"].permute(0, 2, 1, 3, 4).float().cpu().numpy()

            skeletons = resize_time(pose_clips[begin:end], args.window_size).to(device)
            with torch.inference_mode():
                ctrgcn(skeletons)
            l4 = ctrgcn_captured["l4"]
            persons = skeletons.shape[-1]
            _, dims, steps, joints = l4.shape
            time_major = l4.permute(0, 1, 3, 2).reshape(-1, 1, steps)
            time_major = torch.nn.functional.interpolate(time_major, size=13, mode="linear", align_corners=False)
            pose_cache[begin:end] = (
                time_major.reshape(samples * persons, dims, joints, 13)
                .permute(0, 1, 3, 2)
                .reshape(samples, persons, dims, 13, joints)
                .float().cpu().numpy()
            )

            if end % (args.batch_size * 20) == 0 or end == len(names):
                rate = end / (time.time() - start)
                print(f"  [{split} {end}/{len(names)}] {rate:.1f} clip/s, "
                      f"eta {(len(names) - end) / rate / 60:.0f} min", flush=True)

        video_cache.flush()
        pose_cache.flush()
        # joint_xy comes straight from the RTMPose x/y of person 0.
        np.save(output_dir / f"{split}_joint_xy_coco17.npy",
                pose_clips[:, :2, :, :, 0].transpose(0, 2, 3, 1).astype(np.float32))
        np.save(output_dir / f"{split}_object.npy", np.load(OBJECT_DIR / f"{split}_object.npy"))
        np.save(output_dir / f"{split}_labels.npy", np.load(SPLIT_DIR / f"{split}_labels.npy"))
        np.save(output_dir / f"{split}_sample_names.npy", np.array(names, dtype=object))
        print(f"[{split}] video {video_cache.shape}  pose {pose_cache.shape}", flush=True)

    # ZSL views are row subsets of the split arrays, selected by sample name.
    metadata = json.loads((SPLIT_DIR / "metadata.json").read_text())
    print()
    for view, source in (("zsl_train", "train"), ("zsl_test_seen", "test"), ("zsl_test_unseen", "test")):
        wanted = [str(n) for n in np.load(SPLIT_DIR / f"{view}_sample_names.npy", allow_pickle=True)]
        available = {n: i for i, n in enumerate(
            str(x) for x in np.load(output_dir / f"{source}_sample_names.npy", allow_pickle=True))}
        rows = [available[n] for n in wanted]
        for stream, suffix in (("video", ""), ("pose", "_coco17"), ("object", ""),
                               ("joint_xy", "_coco17"), ("labels", "")):
            array = np.load(output_dir / f"{source}_{stream}{suffix}.npy", mmap_mode="r")
            np.save(output_dir / f"{view}_{stream}{suffix}.npy", np.array(array[rows]))
        np.save(output_dir / f"{view}_sample_names.npy", np.array(wanted, dtype=object))
        print(f"  {view:17s} {len(rows):6d} clips")

    (output_dir / "metadata.json").write_text(json.dumps({
        "dataset": "toyota_smarthome",
        "seen_classes": metadata["seen_classes"],
        "unseen_classes": metadata["unseen_classes"],
        "actions": metadata["actions"],
        "x3d_checkpoint": args.x3d_checkpoint,
        "ctrgcn_checkpoint": args.ctrgcn_weights,
    }, indent=2))
    print(f"\nwritten to {output_dir}")


if __name__ == "__main__":
    main()
