"""Train a small RTMPose-17 to 3-D body-orientation estimator.

The model deliberately follows the deployment graph proposed for the active
view policies:

    RTMPose-17 (x, y, score) -> normalized 51-D vector
        -> Linear(51, 128) -> ReLU
        -> Linear(128, 64) -> ReLU
        -> Linear(64, 8)

The 3-D skeleton is used only while constructing the supervision labels.  At
inference time the estimator needs one RTMPose frame and does not load a
3-D skeleton or a depth sensor.

The orientation target is the yaw of the anatomical body-forward vector in
the source camera coordinate frame.  The continuous angle is in [-180, 180)
degrees and the classifier has eight circular 45-degree bins.  A circular
auxiliary loss is used in addition to cross entropy so that neighbouring
orientation bins are treated as neighbours rather than unrelated classes.

Example:

  /home/youhan/.conda/envs/clipgcn/bin/python \
      rl/train_rtmpose_orientation_mlp.py \
      --output /home/youhan/ws/VPOCLIP_plus_full/work_dir/rtmpose_orientation_mlp

The default cache uses the existing strict 50/5 subject-disjoint splits.  The
raw RTMPose arrays are read from ``allviews_cs45/.rtmpose_parts`` because the
older RL cache stores only x/y and omits the RTMPose confidence channel.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path("/home/youhan/ws")
DEFAULT_CACHE_ROOT = (
    ROOT / "VPOCLIP_plus_full/data/rl_raw_new50_5_group1_g1_pseudo_selected_stage_b/cache"
)
DEFAULT_RAW_POSE_ROOT = ROOT / "CTR-GCN_17_full/data/etri_coco17/allviews_cs45/.rtmpose_parts"
DEFAULT_ANGLE_CSV = ROOT / "ETRI-Activity3D-RGB/low_view_camera_angles_split55.csv"
DEFAULT_OUTPUT = ROOT / "VPOCLIP_plus_full/work_dir/rtmpose_orientation_mlp"

SPLITS = ("dqn_train", "val", "test")
RAW_SPLIT = {"dqn_train": "dqn", "val": "val", "test": "test"}

# ETRI Kinect body layout used by the existing 3-D geometry code.
REQUIRED_JOINTS = (0, 1, 4, 8, 12, 16, 20)
LEFT_SHOULDER, RIGHT_SHOULDER = 4, 8
LEFT_HIP, RIGHT_HIP = 12, 16

NUM_CLASSES = 8
BIN_WIDTH_DEG = 360.0 / NUM_CLASSES
BIN_CENTERS_DEG = np.asarray(
    [-180.0 + (i + 0.5) * BIN_WIDTH_DEG for i in range(NUM_CLASSES)],
    dtype=np.float32,
)


def wrap_degrees(value: float | np.ndarray) -> float | np.ndarray:
    """Wrap an angle to [-180, 180)."""

    return (np.asarray(value) + 180.0) % 360.0 - 180.0


def circular_abs_error(pred_deg: np.ndarray, target_deg: np.ndarray) -> np.ndarray:
    return np.abs(wrap_degrees(np.asarray(pred_deg) - np.asarray(target_deg)))


def yaw_bin(angle_deg: float) -> int:
    """Map a continuous yaw to one of eight half-open 45-degree sectors."""

    return int(math.floor((float(wrap_degrees(angle_deg)) + 180.0) / BIN_WIDTH_DEG)) % NUM_CLASSES


def body_frame(points: np.ndarray, valid: np.ndarray) -> dict[str, Any] | None:
    """Compute the anatomical forward direction from one 25-joint skeleton."""

    if points.shape != (25, 3) or valid.shape != (25,):
        return None
    if not valid[list(REQUIRED_JOINTS)].all():
        return None

    right = 0.5 * (
        (points[RIGHT_SHOULDER] - points[LEFT_SHOULDER])
        + (points[RIGHT_HIP] - points[LEFT_HIP])
    )
    up = points[20] - points[0]
    right[1] = 0.0
    right_norm = float(np.linalg.norm(right))
    if right_norm < 1e-5:
        return None
    right = right / right_norm

    forward = np.cross(up, right)
    forward[1] = 0.0
    forward_norm = float(np.linalg.norm(forward))
    if forward_norm < 1e-5:
        return None
    forward = forward / forward_norm
    yaw = math.degrees(math.atan2(float(forward[0]), float(forward[2])))
    return {"forward_yaw_deg": float(wrap_degrees(yaw)), "forward": forward}


def row_to_skeleton(row: list[str]) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Parse one released ETRI CSV row and return joints, validity, x score."""

    if len(row) < 253:
        return None
    try:
        points = np.full((25, 3), np.nan, dtype=np.float64)
        valid = np.zeros(25, dtype=bool)
        image_x = []
        for joint in range(25):
            offset = 3 + joint * 10
            xyz = np.asarray(row[offset : offset + 3], dtype=np.float64)
            tracking = float(row[offset + 9])
            if xyz.shape == (3,) and np.isfinite(xyz).all() and tracking > 0.0:
                points[joint] = xyz
                valid[joint] = True
            if joint in (0, 1, 20):
                image_x.append(float(row[offset + 3]))
        score = float(np.mean(image_x))
    except (IndexError, TypeError, ValueError):
        return None
    if not np.isfinite(score):
        return None
    return points, valid, score


@dataclass(frozen=True)
class SkeletonLabel:
    yaw_deg: float
    frame_id: int
    frame_error: int


def skeleton_yaw_at_video_frame(path: Path, video_frame: int) -> SkeletonLabel | None:
    """Read the 3-D yaw corresponding to one RGB video frame.

    ETRI skeleton rows are normally one-based while RGB frame indices in the
    RTMPose cache are zero-based.  The offset is inferred from the first data
    row.  If the desired row is missing or invalid, the closest valid row is
    used and the distance is recorded for auditing.
    """

    target_video_frame = int(video_frame)
    target_skeleton_frame: int | None = None
    best: tuple[int, int, float, float] | None = None
    # tuple: (distance to target, frame id, yaw, image-space rightness)

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        first_frame: int | None = None
        for row in reader:
            try:
                frame_id = int(row[0])
            except (IndexError, TypeError, ValueError):
                continue
            if first_frame is None:
                first_frame = frame_id
                # The released files observed in this dataset start at row 1.
                # This also handles a zero-based file without a hard-coded path.
                target_skeleton_frame = target_video_frame + (1 if first_frame == 1 else 0)
            parsed = row_to_skeleton(row)
            if parsed is None:
                continue
            points, valid, image_score = parsed
            anatomical = body_frame(points, valid)
            if anatomical is None:
                continue
            distance = abs(frame_id - int(target_skeleton_frame))
            candidate = (
                distance,
                frame_id,
                float(anatomical["forward_yaw_deg"]),
                image_score,
            )
            # For equal frame distance, retain the rightmost body, matching the
            # existing frame-0 3-D geometry implementation.
            if best is None or distance < best[0] or (
                distance == best[0] and image_score > best[3]
            ):
                best = candidate

    if best is None or target_skeleton_frame is None:
        return None
    return SkeletonLabel(yaw_deg=best[2], frame_id=best[1], frame_error=best[0])


def normalize_rtmpose_frame(raw_frame: np.ndarray) -> np.ndarray | None:
    """Convert one RTMPose frame to the requested 51-D input.

    ``raw_frame`` has shape [3, 17], with normalized x, y, and confidence.
    Coordinates are centered at the hip midpoint and scaled by torso length;
    this removes image translation and approximate distance-to-camera while
    retaining left/right and front/back visual asymmetries.  Missing joints
    are set to zero and their confidence remains zero.
    """

    raw_frame = np.asarray(raw_frame, dtype=np.float32)
    if raw_frame.shape != (3, 17):
        raise ValueError(f"expected raw RTMPose frame [3,17], got {raw_frame.shape}")
    x = raw_frame[0].copy()
    y = raw_frame[1].copy()
    score = np.nan_to_num(raw_frame[2].copy(), nan=0.0, posinf=0.0, neginf=0.0)
    score = np.clip(score, 0.0, 1.0)
    valid = score > 1e-3
    if int(valid.sum()) < 5:
        return None

    xy = np.stack([x, y], axis=-1)
    hip_valid = valid[[LEFT_HIP, RIGHT_HIP]].all()
    shoulder_valid = valid[[LEFT_SHOULDER, RIGHT_SHOULDER]].all()
    if hip_valid:
        center = xy[[LEFT_HIP, RIGHT_HIP]].mean(axis=0)
    else:
        center = xy[valid].mean(axis=0)

    if hip_valid and shoulder_valid:
        torso = xy[[LEFT_SHOULDER, RIGHT_SHOULDER]].mean(axis=0) - center
        scale = float(np.linalg.norm(torso))
    else:
        visible = xy[valid]
        scale = float(np.linalg.norm(visible.max(axis=0) - visible.min(axis=0)))
    if not np.isfinite(scale) or scale < 1e-4:
        scale = 1.0

    xy = (xy - center[None, :]) / scale
    xy[~valid] = 0.0
    features = np.concatenate([xy, score[:, None]], axis=1).reshape(-1)
    return features.astype(np.float32, copy=False)


@torch.no_grad()
def predict_orientation(
    model: nn.Module,
    raw_frame: np.ndarray,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Run the deploy-time estimator on one RTMPose [3,17] frame.

    The returned ``yaw_deg`` is the circular expectation of the eight class
    centers.  ``class_yaw_deg`` is the center of the argmax bin and is useful
    when a discrete action policy is desired.
    """

    feature = normalize_rtmpose_frame(raw_frame)
    if feature is None:
        return {"valid": False, "yaw_deg": None, "class_yaw_deg": None, "confidence": 0.0}
    target_device = torch.device(device)
    model = model.to(target_device).eval()
    logits = model(torch.from_numpy(feature)[None].to(target_device))
    probability = torch.softmax(logits, dim=-1)[0]
    centers = torch.as_tensor(BIN_CENTERS_DEG, dtype=probability.dtype, device=target_device)
    vector = torch.stack((torch.cos(torch.deg2rad(centers)), torch.sin(torch.deg2rad(centers))), dim=-1)
    expected = probability @ vector
    yaw = torch.rad2deg(torch.atan2(expected[1], expected[0]))
    argmax = int(probability.argmax().item())
    return {
        "valid": True,
        "yaw_deg": float(wrap_degrees(float(yaw.item()))),
        "class_yaw_deg": float(BIN_CENTERS_DEG[argmax]),
        "class_index": argmax,
        "confidence": float(probability.max().item()),
        "probabilities": probability.detach().cpu().numpy().tolist(),
    }


class OrientationMLP(nn.Module):
    """Exactly the proposed 51 -> 128 -> 64 -> 8 estimator."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(51, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 8),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_angle_index(path: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            sample_id = str(row.get("sample_id", "")).strip()
            skeleton_path = Path(str(row.get("skeleton_path", "")).strip())
            if sample_id and skeleton_path:
                result[sample_id] = skeleton_path
    return result


def load_raw_pose(raw_root: Path, split: str) -> tuple[np.ndarray, dict[str, int]]:
    raw_split = RAW_SPLIT[split]
    names = np.load(raw_root / f"{raw_split}_sample_names.npy", allow_pickle=True).astype(str)
    values = np.load(raw_root / f"{raw_split}_x.npy", mmap_mode="r")
    if values.ndim != 5 or values.shape[1:] != (3, 13, 17, 2):
        raise ValueError(f"unexpected RTMPose array shape for {split}: {values.shape}")
    lookup = {name: i for i, name in enumerate(names)}
    if len(lookup) != len(names):
        raise ValueError(f"duplicate RTMPose sample names in {raw_split}")
    return values, lookup


def _cache_file(output: Path, split: str, frame_index: int) -> Path:
    return output / f"orientation_{split}_frame{frame_index}.npz"


def build_split(
    split: str,
    cache_root: Path,
    raw_root: Path,
    angle_index: dict[str, Path],
    output: Path,
    frame_index: int,
    rebuild: bool,
) -> dict[str, Any]:
    """Build one subject-disjoint split from paired 2-D and 3-D files."""

    destination = _cache_file(output, split, frame_index)
    audit_path = output / f"orientation_{split}_frame{frame_index}_audit.json"
    if destination.exists() and audit_path.exists() and not rebuild:
        with np.load(destination) as data:
            return {
                "split": split,
                "samples": int(len(data["x"])),
                "cached": True,
                "audit": json.loads(audit_path.read_text()),
            }

    raw_values, raw_lookup = load_raw_pose(raw_root, split)
    metadata = json.loads((cache_root / split / "metadata.json").read_text())
    view_valid = np.load(cache_root / split / "view_valid.npy")

    features: list[np.ndarray] = []
    labels: list[int] = []
    angles: list[float] = []
    frame_errors: list[int] = []
    subjects: list[str] = []
    actions: list[int] = []
    cameras: list[str] = []
    sample_names: list[str] = []
    missing_pose: list[str] = []
    missing_skeleton: list[str] = []
    bad_pose = 0
    bad_skeleton = 0
    label_cache: dict[tuple[str, int], SkeletonLabel | None] = {}
    started = time.time()

    for episode_index, episode in enumerate(metadata["episodes"]):
        sample_frames = episode.get("window", {}).get("sample_frames", [])
        if not sample_frames:
            raise ValueError(f"episode {episode_index} has no sample_frames")
        pose_frame_index = min(max(int(frame_index), 0), len(sample_frames) - 1)
        video_frame = int(sample_frames[pose_frame_index])
        for view_index, view in enumerate(episode.get("views", [])):
            if view_index >= view_valid.shape[1] or not bool(view_valid[episode_index, view_index]):
                continue
            sample_name = str(view.get("sample_name", ""))
            raw_index = raw_lookup.get(sample_name)
            if raw_index is None:
                missing_pose.append(sample_name)
                continue

            # RTMPose extraction stores [x/y/score, time, joint, person].
            # Person 0 is the highest-confidence body used by the existing
            # CTR-GCN and active-view caches.
            raw_frame = np.asarray(raw_values[raw_index, :, pose_frame_index, :, 0], dtype=np.float32)
            feature = normalize_rtmpose_frame(raw_frame)
            if feature is None:
                bad_pose += 1
                continue

            sample_id = Path(sample_name).stem
            skeleton_path = angle_index.get(sample_id)
            if skeleton_path is None or not skeleton_path.exists():
                missing_skeleton.append(sample_id)
                continue
            cache_key = (str(skeleton_path), video_frame)
            if cache_key not in label_cache:
                label_cache[cache_key] = skeleton_yaw_at_video_frame(skeleton_path, video_frame)
            label = label_cache[cache_key]
            if label is None:
                bad_skeleton += 1
                continue

            features.append(feature)
            labels.append(yaw_bin(label.yaw_deg))
            angles.append(label.yaw_deg)
            frame_errors.append(label.frame_error)
            subjects.append(str(episode.get("subject", "")))
            actions.append(int(episode.get("label", -1)))
            cameras.append(str(view.get("camera", "")))
            sample_names.append(sample_name)

        if (episode_index + 1) % 250 == 0 or episode_index + 1 == len(metadata["episodes"]):
            elapsed = max(time.time() - started, 1e-6)
            print(
                f"BUILD {split}: episodes={episode_index + 1}/{len(metadata['episodes'])} "
                f"samples={len(features)} labels_cached={len(label_cache)} "
                f"rate={(episode_index + 1) / elapsed:.1f} episodes/s",
                flush=True,
            )

    if not features:
        raise RuntimeError(f"no valid orientation samples built for {split}")
    x = np.stack(features).astype(np.float32)
    y = np.asarray(labels, dtype=np.int64)
    angle = np.asarray(angles, dtype=np.float32)
    frame_error = np.asarray(frame_errors, dtype=np.int16)
    np.savez_compressed(destination, x=x, y=y, angle_deg=angle, frame_error=frame_error)
    audit = {
        "split": split,
        "frame_index": int(frame_index),
        "samples": int(len(x)),
        "feature_dim": int(x.shape[1]),
        "episodes": int(len(metadata["episodes"])),
        "missing_pose": int(len(missing_pose)),
        "missing_skeleton": int(len(missing_skeleton)),
        "bad_pose": int(bad_pose),
        "bad_skeleton": int(bad_skeleton),
        "mean_skeleton_frame_error": float(frame_error.mean()) if len(frame_error) else None,
        "max_skeleton_frame_error": int(frame_error.max()) if len(frame_error) else None,
        "class_counts": {str(k): int(v) for k, v in sorted(Counter(y.tolist()).items())},
        "subjects": sorted(set(subjects)),
        "raw_pose_root": str(raw_root),
        "angle_label": "3-D anatomical body-forward yaw in source camera frame",
        "normalization": "hip midpoint center; shoulder-to-hip torso scale; x/y/score interleaved",
        "missing_pose_examples": missing_pose[:10],
        "missing_skeleton_examples": missing_skeleton[:10],
    }
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")
    return {"split": split, "samples": int(len(x)), "cached": False, "audit": audit}


def class_weights(y: np.ndarray) -> torch.Tensor:
    counts = np.bincount(y, minlength=NUM_CLASSES).astype(np.float32)
    weights = 1.0 / np.sqrt(np.maximum(counts, 1.0))
    weights *= float(NUM_CLASSES) / float(weights.sum())
    return torch.from_numpy(weights.astype(np.float32))


def angular_metrics(logits: torch.Tensor, y: torch.Tensor, angle_deg: torch.Tensor) -> dict[str, Any]:
    probabilities = torch.softmax(logits, dim=-1)
    centers = torch.as_tensor(BIN_CENTERS_DEG, dtype=logits.dtype, device=logits.device)
    radians = torch.deg2rad(centers)
    vector = torch.stack((torch.cos(radians), torch.sin(radians)), dim=-1)
    expected = probabilities @ vector
    soft_yaw = torch.rad2deg(torch.atan2(expected[:, 1], expected[:, 0]))
    class_yaw = centers[logits.argmax(-1)]
    target_yaw = angle_deg.to(logits.device)
    soft_error = torch.abs((soft_yaw - target_yaw + 180.0) % 360.0 - 180.0)
    class_error = torch.abs((class_yaw - target_yaw + 180.0) % 360.0 - 180.0)
    predicted_class = logits.argmax(-1)
    target_class = y.to(logits.device)
    result = {
        "top1": float(predicted_class.eq(target_class).float().mean().item()),
        "class_mae_deg": float(class_error.mean().item()),
        "soft_mae_deg": float(soft_error.mean().item()),
        "class_within_22_5_deg": float((class_error <= 22.5).float().mean().item()),
        "class_within_45_deg": float((class_error <= 45.0).float().mean().item()),
        "soft_within_22_5_deg": float((soft_error <= 22.5).float().mean().item()),
        "soft_within_45_deg": float((soft_error <= 45.0).float().mean().item()),
        "mean_confidence": float(probabilities.max(-1).values.mean().item()),
        "predicted_class_counts": {
            str(k): int(v)
            for k, v in sorted(Counter(predicted_class.detach().cpu().tolist()).items())
        },
    }
    per_class = {}
    for cls in range(NUM_CLASSES):
        mask = target_class.eq(cls)
        if bool(mask.any()):
            per_class[str(cls)] = {
                "n": int(mask.sum().item()),
                "top1": float(predicted_class[mask].eq(target_class[mask]).float().mean().item()),
                "class_mae_deg": float(class_error[mask].mean().item()),
            }
    result["per_class"] = per_class
    return result


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    ce_weight: torch.Tensor | None = None,
    circular_weight: float = 0.2,
) -> tuple[float, dict[str, Any]]:
    model.eval()
    all_logits: list[torch.Tensor] = []
    all_y: list[torch.Tensor] = []
    all_angle: list[torch.Tensor] = []
    losses = []
    centers = torch.as_tensor(BIN_CENTERS_DEG, dtype=torch.float32, device=device)
    unit = torch.stack((torch.cos(torch.deg2rad(centers)), torch.sin(torch.deg2rad(centers))), dim=-1)
    for x, y, angle in loader:
        x, y, angle = x.to(device), y.to(device), angle.to(device)
        logits = model(x)
        ce = nn.functional.cross_entropy(logits, y, weight=ce_weight)
        pred_vector = torch.softmax(logits, -1) @ unit
        target_vector = torch.stack((torch.cos(torch.deg2rad(angle)), torch.sin(torch.deg2rad(angle))), dim=-1)
        circular = (1.0 - nn.functional.cosine_similarity(pred_vector, target_vector, dim=-1)).mean()
        losses.append(float((ce + circular_weight * circular).item()) * len(x))
        all_logits.append(logits.cpu())
        all_y.append(y.cpu())
        all_angle.append(angle.cpu())
    logits = torch.cat(all_logits)
    y = torch.cat(all_y)
    angle = torch.cat(all_angle)
    metrics = angular_metrics(logits, y, angle)
    return float(sum(losses) / max(len(y), 1)), metrics


def train(args: argparse.Namespace, output: Path) -> dict[str, Any]:
    set_seed(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    with np.load(_cache_file(output, "dqn_train", args.frame_index)) as data:
        train_x, train_y, train_angle = data["x"], data["y"], data["angle_deg"]
    with np.load(_cache_file(output, "val", args.frame_index)) as data:
        val_x, val_y, val_angle = data["x"], data["y"], data["angle_deg"]
    with np.load(_cache_file(output, "test", args.frame_index)) as data:
        test_x, test_y, test_angle = data["x"], data["y"], data["angle_deg"]

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y), torch.from_numpy(train_angle)),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y), torch.from_numpy(val_angle)),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y), torch.from_numpy(test_angle)),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = OrientationMLP().to(device)
    weight = class_weights(train_y).to(device) if args.class_weight else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_val = -float("inf")
    best_epoch = -1
    stale = 0
    history: list[dict[str, Any]] = []
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        count = 0
        centers = torch.as_tensor(BIN_CENTERS_DEG, dtype=torch.float32, device=device)
        unit = torch.stack((torch.cos(torch.deg2rad(centers)), torch.sin(torch.deg2rad(centers))), dim=-1)
        for x, y, angle in train_loader:
            x, y, angle = x.to(device, non_blocking=True), y.to(device, non_blocking=True), angle.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            ce = nn.functional.cross_entropy(logits, y, weight=weight, label_smoothing=args.label_smoothing)
            pred_vector = torch.softmax(logits, -1) @ unit
            target_vector = torch.stack((torch.cos(torch.deg2rad(angle)), torch.sin(torch.deg2rad(angle))), dim=-1)
            circular = (1.0 - nn.functional.cosine_similarity(pred_vector, target_vector, dim=-1)).mean()
            loss = ce + args.circular_weight * circular
            loss.backward()
            optimizer.step()
            running += float(loss.item()) * len(x)
            count += len(x)
        scheduler.step()
        train_loss = running / max(count, 1)
        val_loss, val_metrics = evaluate(model, val_loader, device, weight, args.circular_weight)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_top1": val_metrics["top1"],
            "val_class_mae_deg": val_metrics["class_mae_deg"],
            "val_soft_mae_deg": val_metrics["soft_mae_deg"],
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        selection = val_metrics["top1"] - 0.001 * val_metrics["class_mae_deg"]
        if selection > best_val:
            best_val = selection
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "config": vars(args),
                    "bin_centers_deg": BIN_CENTERS_DEG.tolist(),
                    "val_metrics": val_metrics,
                },
                output / "best.pt",
            )
        else:
            stale += 1
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(
                f"EPOCH {epoch:03d} loss={train_loss:.4f} val_top1={val_metrics['top1']:.4f} "
                f"val_mae={val_metrics['class_mae_deg']:.2f} best={best_epoch:03d}",
                flush=True,
            )
        if stale >= args.patience:
            print(f"EARLY STOP epoch={epoch} patience={args.patience}", flush=True)
            break

    checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    _, val_metrics = evaluate(model, val_loader, device, weight, args.circular_weight)
    _, test_metrics = evaluate(model, test_loader, device, weight, args.circular_weight)
    result = {
        "model": "Linear(51,128)-ReLU-Linear(128,64)-ReLU-Linear(64,8)",
        "seed": args.seed,
        "device": str(device),
        "frame_index": args.frame_index,
        "train_samples": int(len(train_y)),
        "val_samples": int(len(val_y)),
        "test_samples": int(len(test_y)),
        "epochs_ran": len(history),
        "best_epoch": int(best_epoch),
        "elapsed_seconds": time.time() - started,
        "val": val_metrics,
        "test": test_metrics,
        "history": history,
        "class_centers_deg": BIN_CENTERS_DEG.tolist(),
    }
    (output / "training_result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--raw-pose-root", type=Path, default=DEFAULT_RAW_POSE_ROOT)
    parser.add_argument("--angle-csv", type=Path, default=DEFAULT_ANGLE_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--frame-index", type=int, default=6, help="one of the 13 RTMPose frames used as input")
    parser.add_argument("--rebuild", action="store_true", help="rebuild paired 2-D/3-D orientation caches")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.03)
    parser.add_argument("--circular-weight", type=float, default=0.2)
    parser.add_argument("--class-weight", action="store_true", help="use inverse-sqrt class weights")
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--device", default=None, help="cuda, cuda:0, or cpu; defaults to CUDA when available")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"OUTPUT {args.output}", flush=True)
    print(f"frame index {args.frame_index}; bins {BIN_CENTERS_DEG.tolist()}", flush=True)
    angle_index = load_angle_index(args.angle_csv)
    print(f"angle index rows={len(angle_index)}", flush=True)

    build_reports = []
    for split in SPLITS:
        report = build_split(
            split=split,
            cache_root=args.cache_root,
            raw_root=args.raw_pose_root,
            angle_index=angle_index,
            output=args.output,
            frame_index=args.frame_index,
            rebuild=args.rebuild,
        )
        build_reports.append(report)
        print(
            f"CACHE {split}: samples={report['samples']} cached={report['cached']} "
            f"audit={report['audit'].get('class_counts')}",
            flush=True,
        )
    (args.output / "build_summary.json").write_text(json.dumps(build_reports, indent=2) + "\n")
    if args.build_only:
        return
    result = train(args, args.output)
    print(json.dumps({"val": result["val"], "test": result["test"], "best_epoch": result["best_epoch"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
