#!/usr/bin/env python
"""Raw end-to-end CLIPGCN zero-shot evaluation.

This script does not read pre-extracted feature arrays such as
``unseen_video.npy``, ``unseen_pose.npy``, ``unseen_object.npy``, or
``unseen_joint_xy.npy``. It uses the split's sample-name/label manifest to find
the original ``.mp4`` and ``.csv`` files, then runs:

    raw video -> X3D feature map
    raw video middle frame -> YOLOv5 object RS map
    raw skeleton CSV -> CTR-GCN pose feature map + joint xy
    tri-modal maps -> trained CLIPGCN -> unseen action prediction
"""

import argparse
import csv
import importlib.util
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from collections import Counter, defaultdict, OrderedDict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as Data
import yaml

from Object_RS import get_multiplicative_RS_map
from model import apply_action_text_bank, build_model_from_config, load_action_descriptions
from test import load_latest_run_info, load_split_classes
from train import get_device, get_path_from_config, load_config, print_device_info, progress_bar


CLIPGCN_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = CLIPGCN_ROOT.parent
X3D_ROOT = WORKSPACE_ROOT / "X3D"
CTRGCN_ROOT = WORKSPACE_ROOT / "CTR-GCN"


OBJECT_CLASS_IDS = [
    0,
    *range(31, 80),
]


def ensure_import_path(path):
    path = str(path)
    if path not in sys.path:
        sys.path.insert(0, path)


def resolve_path(path, base):
    path = Path(path)
    return path if path.is_absolute() else Path(base) / path


def load_split_manifest(split_dir, prefix, data_root, max_samples=None):
    split_dir = Path(split_dir)
    sample_names = np.load(split_dir / f"{prefix}_sample_names.npy", allow_pickle=True)
    labels = np.load(split_dir / f"{prefix}_labels.npy", allow_pickle=False).astype(np.int64)
    if max_samples is not None:
        sample_names = sample_names[:max_samples]
        labels = labels[:max_samples]

    items = []
    for sample_name, label in zip(sample_names, labels):
        sample_name = str(sample_name)
        video_path = Path(data_root) / sample_name
        csv_path = Path(data_root) / str(Path(sample_name).with_suffix(".csv"))
        items.append(
            {
                "sample_name": sample_name,
                "video_path": str(video_path),
                "csv_path": str(csv_path),
                "label": int(label),
            }
        )
    return items


class RawManifestDataset(Data.Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def collate_manifest(batch):
    return batch


def read_uniform_rgb_frames(video_path, num_frames):
    cv2.setNumThreads(1)
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise ValueError(f"Cannot read frames from {video_path}")

    indices = set(np.linspace(0, total_frames - 1, num_frames).round().astype(np.int64).tolist())
    frames = []
    last_frame = None
    frame_idx = 0
    while len(frames) < num_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx in indices:
            last_frame = frame
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_idx += 1

    cap.release()
    if last_frame is None:
        raise ValueError(f"Failed to decode frames from {video_path}")

    while len(frames) < num_frames:
        frames.append(cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB))
    return np.stack(frames, axis=0)


def x3d_tensor_from_frames(frames, size, mean, std):
    resized = [cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA) for frame in frames]
    clip = np.stack(resized, axis=0).astype(np.float32) / 255.0
    clip = np.transpose(clip, (3, 0, 1, 2))
    mean = np.asarray(mean, dtype=np.float32).reshape(3, 1, 1, 1)
    std = np.asarray(std, dtype=np.float32).reshape(3, 1, 1, 1)
    return torch.from_numpy((clip - mean) / std).float()


def skeleton_columns(header):
    index = {name: idx for idx, name in enumerate(header)}
    columns_3d = []
    columns_depth = []
    for joint in range(1, 26):
        columns_3d.append(
            (
                index[f"joint{joint}_3dX"],
                index[f"joint{joint}_3dY"],
                index[f"joint{joint}_3dZ"],
            )
        )
        columns_depth.append((index[f"joint{joint}_depthX"], index[f"joint{joint}_depthY"]))
    frame_col = index["frameNum"]
    tracking_col = index.get("trackingID")
    body_col = index.get("bodyindexID")
    state_cols = [index.get(f"joint{joint}_trackingState") for joint in range(1, 26)]
    return frame_col, tracking_col, body_col, columns_3d, columns_depth, state_cols


def read_main_track_skeleton(csv_path):
    tracks_3d = defaultdict(dict)
    tracks_xy = defaultdict(dict)
    tracked_score = Counter()

    with open(csv_path, "r", newline="", errors="replace") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        frame_col, tracking_col, body_col, columns_3d, columns_depth, state_cols = skeleton_columns(header)

        for row in reader:
            if len(row) < len(header):
                continue
            try:
                frame = int(float(row[frame_col]))
            except ValueError:
                continue
            if tracking_col is not None and row[tracking_col] != "":
                track_id = row[tracking_col]
            elif body_col is not None and row[body_col] != "":
                track_id = f"body_{row[body_col]}"
            else:
                track_id = "single"

            joints_3d = np.zeros((25, 3), dtype=np.float32)
            joints_xy = np.zeros((25, 2), dtype=np.float32)
            valid_value = False
            for joint_idx, (cols_3d, cols_depth) in enumerate(zip(columns_3d, columns_depth)):
                try:
                    x3d, y3d, z3d = (float(row[col]) for col in cols_3d)
                except ValueError:
                    x3d = y3d = z3d = 0.0
                try:
                    x2d, y2d = (float(row[col]) for col in cols_depth)
                except ValueError:
                    x2d = y2d = 0.0

                joints_3d[joint_idx] = (x3d, y3d, z3d)
                joints_xy[joint_idx] = (x2d, y2d)
                valid_value = valid_value or x3d != 0.0 or y3d != 0.0 or z3d != 0.0

                state_col = state_cols[joint_idx]
                if state_col is not None:
                    try:
                        tracked_score[track_id] += int(float(row[state_col]))
                    except ValueError:
                        pass

            if valid_value:
                tracks_3d[track_id][frame] = joints_3d
                tracks_xy[track_id][frame] = joints_xy

    if not tracks_3d:
        raise ValueError(f"No valid skeleton track in {csv_path}")

    def rank_track(item):
        track_id, frames = item
        return (len(frames), tracked_score[track_id], str(track_id))

    main_track_id, frame_to_joints = max(tracks_3d.items(), key=rank_track)
    frames = sorted(frame_to_joints)
    data_3d = np.stack([tracks_3d[main_track_id][frame] for frame in frames], axis=0).astype(np.float32)
    data_xy = np.stack([tracks_xy[main_track_id][frame] for frame in frames], axis=0).astype(np.float32)
    return data_3d, data_xy


def valid_crop_resize(data_numpy, valid_frame_num, window_size):
    channels, total_frames, num_joints, num_person = data_numpy.shape
    valid = max(1, min(int(valid_frame_num), total_frames))
    data = data_numpy[:, :valid, :, :]
    if data.shape[1] == window_size:
        return data

    data_tensor = torch.tensor(data, dtype=torch.float32)
    data_tensor = data_tensor.permute(0, 2, 3, 1).contiguous().view(channels * num_joints * num_person, valid)
    data_tensor = data_tensor[None, None, :, :]
    data_tensor = F.interpolate(
        data_tensor,
        size=(channels * num_joints * num_person, window_size),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0).squeeze(0)
    return (
        data_tensor.contiguous()
        .view(channels, num_joints, num_person, window_size)
        .permute(0, 3, 1, 2)
        .contiguous()
        .numpy()
    )


def uniform_resample(sequence, frames):
    indices = np.linspace(0, sequence.shape[0] - 1, frames).round().astype(np.int64)
    return sequence[indices]


def normalize_depth_xy(xy, width=512.0, height=424.0):
    out = np.nan_to_num(xy.copy(), nan=0.0, posinf=width - 1.0, neginf=0.0)
    out[..., 0] = (out[..., 0] / (width - 1.0)) * 2.0 - 1.0
    out[..., 1] = (out[..., 1] / (height - 1.0)) * 2.0 - 1.0
    return np.clip(out, -1.0, 1.0).astype(np.float32)


def ctrgcn_tensor_and_joint_xy_from_csv(csv_path, window_size):
    data_3d, data_xy = read_main_track_skeleton(csv_path)
    data_numpy = np.zeros((3, data_3d.shape[0], 25, 2), dtype=np.float32)
    data_numpy[:, :, :, 0] = data_3d.transpose(2, 0, 1)
    data_numpy = valid_crop_resize(data_numpy, data_3d.shape[0], window_size)
    joint_xy = normalize_depth_xy(uniform_resample(data_xy, window_size))
    return torch.from_numpy(data_numpy).float(), torch.from_numpy(joint_xy).float()


def load_x3d_model(args, device):
    x3d_root = Path(args.x3d_root)
    ensure_import_path(x3d_root)
    from tsn.config import get_cfg_defaults
    from tsn.model.recognizers.build import build_recognizer

    cfg = get_cfg_defaults()
    cfg.merge_from_file(str(args.x3d_config))
    cfg.defrost()
    cfg.NUM_GPUS = 0
    if not Path(args.x3d_checkpoint).exists():
        raise FileNotFoundError(
            f"X3D checkpoint not found: {args.x3d_checkpoint}. "
            "For result_0007000.txt, the matching checkpoint is usually model_007000.pth."
        )
    cfg.MODEL.PRETRAINED = str(args.x3d_checkpoint)
    cfg.freeze()

    model = build_recognizer(cfg, device=device)
    model.eval()
    captured = {}

    def hook_fn(_module, _inputs, output):
        captured["features"] = output.detach()

    module = model
    for part in args.x3d_layer.split("."):
        module = getattr(module, part)
    hook = module.register_forward_hook(hook_fn)
    return model, captured, hook, cfg


def load_ctrgcn_model(args, device):
    ctrgcn_root = Path(args.ctrgcn_root)
    ensure_import_path(ctrgcn_root)
    ensure_import_path(ctrgcn_root / "torchlight")

    with open(args.ctrgcn_config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    spec = importlib.util.spec_from_file_location("clipgcn_ctrgcn_model", ctrgcn_root / "model" / "ctrgcn.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.Model(**config["model_args"]).to(device)

    try:
        weights = torch.load(args.ctrgcn_weights, map_location=device, weights_only=False)
    except TypeError:
        weights = torch.load(args.ctrgcn_weights, map_location=device)
    if isinstance(weights, dict) and "state_dict" in weights:
        weights = weights["state_dict"]
    state_dict = OrderedDict((key.split("module.")[-1], value) for key, value in weights.items())
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    captured = {}

    def hook_fn(_module, _inputs, output):
        captured["features"] = output.detach()

    hook = getattr(model, args.ctrgcn_hook_layer).register_forward_hook(hook_fn)
    return model, captured, hook


def load_yolo_model(args, device):
    if args.no_yolo:
        return None
    ensure_import_path(Path(args.yolo_repo))
    model = torch.hub.load(
        args.yolo_repo,
        "custom",
        path=args.yolo_weights,
        source="local",
        verbose=False,
    )
    model.to(device)
    model.eval()
    model.conf = args.yolo_conf
    model.iou = args.yolo_iou
    # 类别槽位自适应: 官方80类权重用旧映射[0,31..79]; 自定义模型(如coco_custom50)
    # 的类别本身就是想要的集合, 槽位=类别id, 直接取全量。
    names = getattr(model, "names", None)
    nc = len(names) if names else 80
    if nc != 80:
        OBJECT_CLASS_IDS[:] = list(range(nc))
        print(f"[yolo] custom {nc}-class model detected; OBJECT_CLASS_IDS set to 0..{nc-1}")
    if args.yolo_half and device.type == "cuda":
        try:
            model.half()
        except AttributeError:
            if hasattr(model, "model"):
                model.model.half()
    return model


def load_clipgcn_model(args, config, config_path, device, candidate_labels):
    latest_run = load_latest_run_info(config, config_path)
    if args.clipgcn_checkpoint:
        checkpoint_path = get_path_from_config(config_path, args.clipgcn_checkpoint)
    elif latest_run and latest_run.get("best_model"):
        checkpoint_path = latest_run["best_model"]
    else:
        checkpoint_path = get_path_from_config(config_path, config["outputs"]["best_model"])

    model = build_model_from_config(
        config,
        device=device,
        download_root=get_path_from_config(config_path, config["model"]["text_encoder"].get("download_root")),
    )
    try:
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except Exception:
        state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)

    text_config = config["data"]["text"]
    labels, _texts = apply_action_text_bank(
        model,
        text_config,
        get_path_from_config(config_path, text_config["xlsx"]),
        candidate_labels,
    )
    model.to(device)
    model.eval()
    return model, checkpoint_path, labels


def x3d_features_from_model(model, captured, clips, args):
    captured.clear()
    _ = model(clips)
    if "features" not in captured:
        raise RuntimeError(f"X3D hook layer {args.x3d_layer} did not capture features")
    features = captured["features"]
    if features.ndim != 5:
        raise ValueError(f"Expected X3D features [B,C,T,H,W], got {tuple(features.shape)}")
    return features.permute(0, 2, 1, 3, 4).contiguous()


def ctrgcn_pose_from_model(model, captured, skeletons, args):
    captured.clear()
    _ = model(skeletons)
    if "features" not in captured:
        raise RuntimeError(f"CTR-GCN hook layer {args.ctrgcn_hook_layer} did not capture features")
    features = captured["features"]
    batch_size = skeletons.shape[0]
    num_person = skeletons.shape[-1]
    return features.view(batch_size, num_person, features.shape[1], features.shape[2], features.shape[3]).contiguous()


def object_maps_from_yolo(yolo_model, frames, device, args):
    if yolo_model is None:
        return torch.zeros(len(frames), len(OBJECT_CLASS_IDS), args.object_grid_size, args.object_grid_size, device=device)

    results = yolo_model(frames, size=args.yolo_size)
    class_to_slot = {class_id: slot for slot, class_id in enumerate(OBJECT_CLASS_IDS)}
    max_det = max((len(det) for det in results.xyxy), default=0)
    max_det = max(max_det, 1)
    class_slots = torch.full((len(frames), max_det), -1, dtype=torch.long, device=device)
    xy = torch.zeros((len(frames), max_det, 2), dtype=torch.float32, device=device)
    conf = torch.zeros((len(frames), max_det), dtype=torch.float32, device=device)
    valid = torch.zeros((len(frames), max_det), dtype=torch.bool, device=device)

    for sample_idx, (frame, detections) in enumerate(zip(frames, results.xyxy)):
        height, width = frame.shape[:2]
        det_idx_out = 0
        for detection in detections:
            class_id = int(detection[5].item())
            slot = class_to_slot.get(class_id)
            if slot is None:
                continue
            x1, y1, x2, y2 = (float(value) for value in detection[:4].tolist())
            x_center = ((x1 + x2) * 0.5) / max(width, 1)
            y_center = ((y1 + y2) * 0.5) / max(height, 1)
            class_slots[sample_idx, det_idx_out] = slot
            xy[sample_idx, det_idx_out, 0] = 2.0 * x_center - 1.0
            xy[sample_idx, det_idx_out, 1] = 1.0 - 2.0 * y_center
            conf[sample_idx, det_idx_out] = float(detection[4].item()) if args.object_value == "confidence" else 1.0
            valid[sample_idx, det_idx_out] = True
            det_idx_out += 1
            if det_idx_out >= max_det:
                break

    return get_multiplicative_RS_map(
        class_slots,
        xy,
        num_classes=len(OBJECT_CLASS_IDS),
        confidence=conf,
        valid=valid,
        grid_size=args.object_grid_size,
        max_distance_weight=args.object_max_distance_weight,
    )


class ObjectMapRunner:
    """Runs YOLO object mapping and can reuse maps to simulate slower detector rates."""

    def __init__(self, yolo_model, device, args):
        self.yolo_model = yolo_model
        self.device = device
        self.args = args
        self.detect_every = max(1, int(args.yolo_detect_every))
        self.step = 0
        self.last_map = None

    def _zero_map(self):
        return torch.zeros(
            len(OBJECT_CLASS_IDS),
            self.args.object_grid_size,
            self.args.object_grid_size,
            dtype=torch.float32,
            device=self.device,
        )

    def __call__(self, frames):
        if self.yolo_model is None or self.detect_every == 1:
            maps = object_maps_from_yolo(self.yolo_model, frames, self.device, self.args)
            if maps.shape[0] > 0:
                self.last_map = maps[-1].detach()
                self.step += maps.shape[0]
            return maps

        output_maps = [None] * len(frames)
        detect_indices = []
        detect_frames = []
        for idx, frame in enumerate(frames):
            should_detect = self.last_map is None or self.step % self.detect_every == 0
            if should_detect:
                detect_indices.append(idx)
                detect_frames.append(frame)
            else:
                output_maps[idx] = self.last_map
            self.step += 1

        if detect_frames:
            detected_maps = object_maps_from_yolo(self.yolo_model, detect_frames, self.device, self.args)
            for map_idx, sample_idx in enumerate(detect_indices):
                self.last_map = detected_maps[map_idx].detach()
                output_maps[sample_idx] = self.last_map

        for idx, object_map in enumerate(output_maps):
            if object_map is None:
                output_maps[idx] = self._zero_map()

        return torch.stack(output_maps, dim=0)


def preprocess_raw_item(item, args, x3d_cfg):
    timing = {}
    start = time.perf_counter()
    frames = read_uniform_rgb_frames(item["video_path"], args.frames)
    timing["pre_read_video_frames"] = time.perf_counter() - start

    start = time.perf_counter()
    x3d_clip = x3d_tensor_from_frames(
        frames,
        int(x3d_cfg.TRANSFORM.TEST.TENSOR_RESIZE_SIZE),
        x3d_cfg.TRANSFORM.MEAN,
        x3d_cfg.TRANSFORM.STD,
    )
    timing["pre_x3d_tensor"] = time.perf_counter() - start

    start = time.perf_counter()
    skeleton, joint_xy = ctrgcn_tensor_and_joint_xy_from_csv(item["csv_path"], args.frames)
    timing["pre_ctrgcn_csv_tensor"] = time.perf_counter() - start
    return {
        "x3d_clip": x3d_clip,
        "yolo_frame": frames[len(frames) // 2],
        "skeleton": skeleton,
        "joint_xy": joint_xy,
        "label": int(item["label"]),
        "sample_name": item["sample_name"],
        "timing": timing,
    }


def default_output_path(config, config_path, prefix):
    output_root = Path(get_path_from_config(config_path, config["outputs"]["work_dir"])) / "raw_end_to_end_tests"
    output_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_root / f"{prefix}_{stamp}.json"


def percentile(values, q):
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def latency_metrics(values, prefix):
    if not values:
        return {
            f"{prefix}_avg_ms_per_sample": None,
            f"{prefix}_p50_ms_per_sample": None,
            f"{prefix}_p90_ms_per_sample": None,
            f"{prefix}_p95_ms_per_sample": None,
        }
    ms = [value * 1000.0 for value in values]
    return {
        f"{prefix}_avg_ms_per_sample": float(np.mean(ms)),
        f"{prefix}_p50_ms_per_sample": percentile(ms, 50),
        f"{prefix}_p90_ms_per_sample": percentile(ms, 90),
        f"{prefix}_p95_ms_per_sample": percentile(ms, 95),
    }


def add_stage_latency_metrics(metrics, timed_stage_samples):
    for stage_name, values in sorted(timed_stage_samples.items()):
        metrics.update(latency_metrics(values, f"timed_{stage_name}_latency"))


def parse_args():
    parser = argparse.ArgumentParser(description="Raw end-to-end CLIPGCN zero-shot evaluation.")
    parser.add_argument("--config", default="config_50_5.yaml")
    parser.add_argument("--split-dir", default=None)
    parser.add_argument("--prefix", default="unseen")
    parser.add_argument("--candidate-scope", choices=["unseen", "seen", "all"], default="unseen")
    parser.add_argument("--data-root", default="/workspace/CLIPGCN/data")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--preprocess-workers",
        type=int,
        default=4,
        help="Threads used inside each batch for raw video decoding and CSV parsing.",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--allow-cpu", action="store_true", help="Allow CPU fallback when the config requests CUDA.")
    parser.add_argument(
        "--robot-sim",
        action="store_true",
        help="Measure robot-like online latency: force batch size 1 and one preprocessing worker.",
    )
    parser.add_argument(
        "--warmup-samples",
        type=int,
        default=0,
        help="Initial samples kept for accuracy but excluded from latency percentiles.",
    )
    parser.add_argument(
        "--cudnn-benchmark",
        action="store_true",
        help="Enable cudnn benchmark for fixed-size inference on CUDA.",
    )
    parser.add_argument(
        "--profile-stages",
        action="store_true",
        help="Print per-stage timing for preprocessing, YOLO, X3D, CTR-GCN, and CLIPGCN. Adds CUDA sync overhead.",
    )

    parser.add_argument("--clipgcn-checkpoint", default=None)

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
    parser.add_argument("--yolo-half", action="store_true", help="Run YOLO in FP16 on CUDA.")
    parser.add_argument(
        "--yolo-detect-every",
        type=int,
        default=1,
        help="Run YOLO once every N samples and reuse the previous object map in between.",
    )
    parser.add_argument("--no-yolo", action="store_true")
    parser.add_argument("--object-grid-size", type=int, default=6)
    parser.add_argument("--object-value", choices=["presence", "confidence"], default="presence")
    parser.add_argument("--object-max-distance-weight", type=float, default=10.0)

    parser.add_argument("--frames", type=int, default=13)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.robot_sim:
        if args.batch_size != 1:
            print(f"--robot-sim enabled: overriding batch_size {args.batch_size} -> 1")
            args.batch_size = 1
        if args.num_workers != 0:
            print(f"--robot-sim enabled: overriding num_workers {args.num_workers} -> 0")
            args.num_workers = 0
        if args.preprocess_workers != 1:
            print(f"--robot-sim enabled: overriding preprocess_workers {args.preprocess_workers} -> 1")
            args.preprocess_workers = 1
    args.yolo_detect_every = max(1, int(args.yolo_detect_every))

    config_path = os.path.abspath(args.config)
    config = load_config(config_path)
    requested_device = str(config["runtime"].get("device"))
    if requested_device.startswith("cuda") and not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError(
            "The config requests a CUDA device, but torch.cuda.is_available() is False. "
            "This raw end-to-end test would run on CPU and be extremely slow. "
            "Check nvidia-smi / container GPU access, or pass --allow-cpu intentionally."
        )
    device = get_device(config["runtime"].get("device"))
    print_device_info(device)
    if args.cudnn_benchmark and device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    split_dir = args.split_dir or get_path_from_config(config_path, config["data"]["train"]["data_dir"])
    split_dir = get_path_from_config(config_path, split_dir)
    candidate_labels = load_split_classes(split_dir, args.candidate_scope)
    items = load_split_manifest(split_dir, args.prefix, args.data_root, args.max_samples)
    print(f"Loaded raw manifest: {len(items)} samples from {split_dir}/{args.prefix}_sample_names.npy")

    dataset = RawManifestDataset(items)
    loader = Data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_manifest)

    x3d_model, x3d_captured, x3d_hook, x3d_cfg = load_x3d_model(args, device)
    ctrgcn_model, ctrgcn_captured, ctrgcn_hook = load_ctrgcn_model(args, device)
    yolo_model = load_yolo_model(args, device)
    object_map_runner = ObjectMapRunner(yolo_model, device, args)
    clipgcn_model, clipgcn_checkpoint, candidate_labels = load_clipgcn_model(args, config, config_path, device, candidate_labels)
    candidate_tensor = torch.as_tensor(candidate_labels, dtype=torch.long, device=device)

    use_amp = bool(config["runtime"].get("amp", False)) and device.type == "cuda"
    total = 0
    correct1 = 0
    correct5 = 0
    end_to_end_time = 0.0
    model_forward_time = 0.0
    preprocessing_time = 0.0
    seen_samples = 0
    timed_end_to_end_samples = []
    timed_model_forward_samples = []
    timed_preprocessing_samples = []
    stage_times = defaultdict(float)
    timed_stage_samples = defaultdict(list)
    predictions = []

    preprocess_executor = (
        ThreadPoolExecutor(max_workers=args.preprocess_workers)
        if args.preprocess_workers and args.preprocess_workers > 1
        else None
    )

    try:
        with torch.inference_mode():
            bar = progress_bar(loader, "Raw end-to-end test")
            for batch in bar:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                total_start = time.perf_counter()

                preprocess_start = time.perf_counter()
                if preprocess_executor is not None:
                    processed = list(
                        preprocess_executor.map(
                            lambda raw_item: preprocess_raw_item(raw_item, args, x3d_cfg),
                            batch,
                        )
                    )
                else:
                    processed = [preprocess_raw_item(item, args, x3d_cfg) for item in batch]

                x3d_clips = torch.stack([item["x3d_clip"] for item in processed], dim=0).to(device, non_blocking=True)
                skeletons = torch.stack([item["skeleton"] for item in processed], dim=0).to(device, non_blocking=True)
                joint_xys = torch.stack([item["joint_xy"] for item in processed], dim=0).to(device, non_blocking=True)
                yolo_frames = [item["yolo_frame"] for item in processed]
                labels = [item["label"] for item in processed]
                names = [item["sample_name"] for item in processed]
                labels_tensor = torch.as_tensor(labels, dtype=torch.long, device=device)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                preprocess_elapsed = time.perf_counter() - preprocess_start
                preprocessing_time += preprocess_elapsed
                stage_batch_times = {}
                if args.profile_stages:
                    for item in processed:
                        for name, elapsed in item.get("timing", {}).items():
                            stage_batch_times[name] = stage_batch_times.get(name, 0.0) + float(elapsed)

                model_start = time.perf_counter()
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    if args.profile_stages:
                        if device.type == "cuda":
                            torch.cuda.synchronize(device)
                        stage_start = time.perf_counter()
                        object_maps = object_map_runner(yolo_frames)
                        if device.type == "cuda":
                            torch.cuda.synchronize(device)
                        stage_batch_times["model_yolo_object_map"] = time.perf_counter() - stage_start

                        stage_start = time.perf_counter()
                        video_features = x3d_features_from_model(x3d_model, x3d_captured, x3d_clips, args)
                        if device.type == "cuda":
                            torch.cuda.synchronize(device)
                        stage_batch_times["model_x3d_feature"] = time.perf_counter() - stage_start

                        stage_start = time.perf_counter()
                        pose_features = ctrgcn_pose_from_model(ctrgcn_model, ctrgcn_captured, skeletons, args)
                        if device.type == "cuda":
                            torch.cuda.synchronize(device)
                        stage_batch_times["model_ctrgcn_feature"] = time.perf_counter() - stage_start

                        stage_start = time.perf_counter()
                        logits = clipgcn_model(video_features, pose_features, object_maps, joint_xys)
                        if device.type == "cuda":
                            torch.cuda.synchronize(device)
                        stage_batch_times["model_clipgcn_fusion"] = time.perf_counter() - stage_start
                    else:
                        object_maps = object_map_runner(yolo_frames)
                        video_features = x3d_features_from_model(x3d_model, x3d_captured, x3d_clips, args)
                        pose_features = ctrgcn_pose_from_model(ctrgcn_model, ctrgcn_captured, skeletons, args)
                        logits = clipgcn_model(video_features, pose_features, object_maps, joint_xys)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                model_elapsed = time.perf_counter() - model_start
                total_elapsed = time.perf_counter() - total_start
                model_forward_time += model_elapsed
                end_to_end_time += total_elapsed
                batch_samples = labels_tensor.shape[0]
                if args.profile_stages:
                    for stage_name, elapsed in stage_batch_times.items():
                        stage_times[stage_name] += elapsed
                for sample_offset in range(batch_samples):
                    global_sample_index = seen_samples + sample_offset
                    if global_sample_index >= args.warmup_samples:
                        timed_end_to_end_samples.append(total_elapsed / max(batch_samples, 1))
                        timed_model_forward_samples.append(model_elapsed / max(batch_samples, 1))
                        timed_preprocessing_samples.append(preprocess_elapsed / max(batch_samples, 1))
                        if args.profile_stages:
                            for stage_name, elapsed in stage_batch_times.items():
                                timed_stage_samples[stage_name].append(elapsed / max(batch_samples, 1))

                pred_indices = torch.argmax(logits, dim=1)
                pred_labels = candidate_tensor[pred_indices]
                correct1 += int((pred_labels == labels_tensor).sum().item())
                if logits.shape[1] >= 5:
                    top5 = candidate_tensor[torch.topk(logits, k=5, dim=1).indices]
                    correct5 += int((top5 == labels_tensor[:, None]).any(dim=1).sum().item())

                total += batch_samples
                seen_samples += batch_samples
                predictions.extend(
                    {
                        "sample_name": name,
                        "label": int(label),
                        "pred": int(pred),
                    }
                    for name, label, pred in zip(names, labels, pred_labels.detach().cpu().tolist())
                )

                if hasattr(bar, "set_postfix"):
                    bar.set_postfix(
                        pre=f"{preprocess_elapsed:.2f}s",
                        model=f"{model_elapsed:.2f}s",
                        total=f"{total_elapsed:.2f}s",
                    )
    finally:
        if preprocess_executor is not None:
            preprocess_executor.shutdown(wait=True)
        x3d_hook.remove()
        ctrgcn_hook.remove()

    timed_samples = len(timed_end_to_end_samples)
    timed_end_to_end_time = float(sum(timed_end_to_end_samples))
    timed_model_forward_time = float(sum(timed_model_forward_samples))
    timed_preprocessing_time = float(sum(timed_preprocessing_samples))
    metrics = {
        "num_samples": total,
        "timed_num_samples": timed_samples,
        "warmup_samples": int(args.warmup_samples),
        "top1_acc": correct1 / max(total, 1),
        "top5_acc": correct5 / max(total, 1) if len(candidate_labels) >= 5 else None,
        "candidate_labels": [int(label) for label in candidate_labels],
        "end_to_end_time_seconds": end_to_end_time,
        "avg_end_to_end_time_seconds_per_sample": end_to_end_time / max(total, 1),
        "avg_end_to_end_time_ms_per_sample": end_to_end_time / max(total, 1) * 1000.0,
        "model_forward_time_seconds": model_forward_time,
        "avg_model_forward_time_ms_per_sample": model_forward_time / max(total, 1) * 1000.0,
        "preprocessing_time_seconds": preprocessing_time,
        "avg_preprocessing_time_ms_per_sample": preprocessing_time / max(total, 1) * 1000.0,
        "end_to_end_samples_per_second": total / end_to_end_time if end_to_end_time > 0 else None,
        "model_forward_samples_per_second": total / model_forward_time if model_forward_time > 0 else None,
        "timed_end_to_end_time_seconds": timed_end_to_end_time,
        "timed_model_forward_time_seconds": timed_model_forward_time,
        "timed_preprocessing_time_seconds": timed_preprocessing_time,
        "timed_end_to_end_samples_per_second": timed_samples / timed_end_to_end_time if timed_end_to_end_time > 0 else None,
        "timed_model_forward_samples_per_second": timed_samples / timed_model_forward_time if timed_model_forward_time > 0 else None,
    }
    metrics.update(latency_metrics(timed_end_to_end_samples, "timed_end_to_end_latency"))
    metrics.update(latency_metrics(timed_model_forward_samples, "timed_model_forward_latency"))
    metrics.update(latency_metrics(timed_preprocessing_samples, "timed_preprocessing_latency"))
    if args.profile_stages:
        metrics["stage_times_seconds"] = {name: float(value) for name, value in sorted(stage_times.items())}
        metrics["avg_stage_times_ms_per_sample"] = {
            name: float(value) / max(total, 1) * 1000.0 for name, value in sorted(stage_times.items())
        }
        metrics["timed_avg_stage_times_ms_per_sample"] = {
            name: float(np.mean(values) * 1000.0)
            for name, values in sorted(timed_stage_samples.items())
            if values
        }
        add_stage_latency_metrics(metrics, timed_stage_samples)

    output_path = Path(args.output) if args.output else default_output_path(config, config_path, args.prefix)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metrics": metrics,
        "predictions": predictions,
        "paths": {
            "config": config_path,
            "split_dir": str(split_dir),
            "clipgcn_checkpoint": str(clipgcn_checkpoint),
            "x3d_checkpoint": str(args.x3d_checkpoint),
            "ctrgcn_weights": str(args.ctrgcn_weights),
            "yolo_weights": None if args.no_yolo else str(args.yolo_weights),
        },
        "runtime_options": {
            "robot_sim": bool(args.robot_sim),
            "batch_size": int(args.batch_size),
            "num_workers": int(args.num_workers),
            "preprocess_workers": int(args.preprocess_workers),
            "yolo_size": int(args.yolo_size),
            "yolo_half": bool(args.yolo_half),
            "yolo_detect_every": int(args.yolo_detect_every),
            "cudnn_benchmark": bool(args.cudnn_benchmark),
            "profile_stages": bool(args.profile_stages),
            "amp": bool(use_amp),
            "device": str(device),
        },
        "note": "Raw end-to-end evaluation: original mp4/csv -> YOLO/X3D/CTR-GCN -> CLIPGCN.",
    }
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    print("Raw end-to-end unseen action recognition results")
    print(f"  split_dir: {split_dir}")
    print(f"  prefix: {args.prefix}")
    print(f"  candidate_scope: {args.candidate_scope}")
    print(f"  candidate_labels: {metrics['candidate_labels']}")
    print(f"  num_samples: {metrics['num_samples']}")
    if metrics["warmup_samples"] > 0:
        print(f"  warmup_samples_excluded_from_latency: {metrics['warmup_samples']}")
    print(f"  top1_acc: {metrics['top1_acc']:.4f}")
    if metrics["top5_acc"] is not None:
        print(f"  top5_acc: {metrics['top5_acc']:.4f}")
    print(f"  end_to_end_time: {metrics['end_to_end_time_seconds']:.4f}s")
    print(f"  avg_end_to_end_time: {metrics['avg_end_to_end_time_ms_per_sample']:.4f} ms/sample")
    print(f"  model_forward_time: {metrics['model_forward_time_seconds']:.4f}s")
    print(f"  avg_model_forward_time: {metrics['avg_model_forward_time_ms_per_sample']:.4f} ms/sample")
    print(f"  preprocessing_time: {metrics['preprocessing_time_seconds']:.4f}s")
    print(f"  avg_preprocessing_time: {metrics['avg_preprocessing_time_ms_per_sample']:.4f} ms/sample")
    if metrics["end_to_end_samples_per_second"] is not None:
        print(f"  end_to_end_throughput: {metrics['end_to_end_samples_per_second']:.2f} samples/s")
    if metrics["timed_end_to_end_latency_avg_ms_per_sample"] is not None:
        print("  timed latency after warmup:")
        print(f"    end_to_end_avg: {metrics['timed_end_to_end_latency_avg_ms_per_sample']:.4f} ms/sample")
        print(f"    end_to_end_p50: {metrics['timed_end_to_end_latency_p50_ms_per_sample']:.4f} ms/sample")
        print(f"    end_to_end_p90: {metrics['timed_end_to_end_latency_p90_ms_per_sample']:.4f} ms/sample")
        print(f"    model_forward_avg: {metrics['timed_model_forward_latency_avg_ms_per_sample']:.4f} ms/sample")
        print(f"    preprocessing_avg: {metrics['timed_preprocessing_latency_avg_ms_per_sample']:.4f} ms/sample")
    if args.profile_stages and metrics.get("timed_avg_stage_times_ms_per_sample"):
        print("  stage timing after warmup:")
        for name, value in sorted(metrics["timed_avg_stage_times_ms_per_sample"].items(), key=lambda item: item[1], reverse=True):
            print(f"    {name}: {value:.4f} ms/sample")
    print(f"Saved raw end-to-end results to {output_path}")


if __name__ == "__main__":
    main()
