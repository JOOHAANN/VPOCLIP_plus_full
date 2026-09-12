"""Train an unknown/seen gate from skeleton activity and VPO confidence.

The gate is calibrated without using group1's true unseen classes.  Pseudo
classes [8, 27, 28, 35, 50] and [3, 13, 30, 41, 52] are treated as unknown;
the real group1 unseen classes [14, 15, 25, 39, 46] are read only for the final
held-out report.  Features are mostly skeleton geometry/activity statistics,
with a small set of VPO cosine/entropy summaries to make the gate useful when
the action is visually confident but its motion pattern is novel.

The output bundle is a compact torch checkpoint containing the MLP, feature
normalisation, threshold, pseudo split and feature schema.  No per-sample
feature cache is written.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model import l2_normalize  # noqa: E402
from train import (  # noqa: E402
    TrimodalContrastiveDataset,
    get_device,
    get_path_from_config,
    load_config,
    move_batch_to_device,
    prepare_action_text_bank,
)
from model import build_model_from_config  # noqa: E402


TRUE_UNSEEN = [14, 15, 25, 39, 46]
PSEUDO_GROUPS = {
    "p3": [8, 27, 28, 35, 50],
    "p4": [3, 13, 30, 41, 52],
}
PSEUDO_UNSEEN = sorted({label for values in PSEUDO_GROUPS.values() for label in values})

# COCO-17 undirected bones.  The ranges and temporal changes of these lengths
# are robust to a subject's absolute position in the image.
COCO_BONES = (
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
)


def _stats_axis(array, axis):
    """Finite mean/std/min/max/range for a float32 array."""
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    mean = array.mean(axis=axis)
    std = array.std(axis=axis)
    low = array.min(axis=axis)
    high = array.max(axis=axis)
    return np.concatenate((mean, std, low, high, high - low), axis=-1)


def skeleton_features(pose, joint_xy):
    """Return per-clip motion/geometry features from COCO-17 arrays.

    ``joint_xy`` is [N,T,17,2] in [-1,1].  ``pose`` is the hooked CTR-GCN
    feature tensor [N,2,64,T,17]; it contributes only low-dimensional energy
    summaries, not the full 2x64 channel map.
    """
    xy = np.nan_to_num(np.asarray(joint_xy, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    n, frames, joints, _ = xy.shape
    present = np.isfinite(joint_xy).all(axis=-1).any(axis=1)

    # Hip-centred and scale-normalised coordinates.  Missing hips fall back to
    # the mean of the available joints; scale has a conservative floor.
    hip = (xy[:, :, 11] + xy[:, :, 12]) * 0.5
    shoulder = np.linalg.norm(xy[:, :, 5] - xy[:, :, 6], axis=-1)
    hip_span = np.linalg.norm(xy[:, :, 11] - xy[:, :, 12], axis=-1)
    scale = np.maximum(0.5 * (shoulder + hip_span), 0.025).mean(axis=1)
    scale = np.maximum(scale, 0.025)[:, None, None, None]
    rel = (xy - hip[:, :, None, :]) / scale
    rel = np.clip(rel, -20.0, 20.0)

    # Position, velocity and acceleration distributions per joint.
    position = _stats_axis(rel, axis=1).reshape(n, -1)
    velocity = np.diff(rel, axis=1)
    velocity_stats = _stats_axis(np.abs(velocity), axis=1).reshape(n, -1)
    acceleration = np.diff(velocity, axis=1)
    acceleration_stats = _stats_axis(np.abs(acceleration), axis=1).reshape(n, -1)

    # Bone lengths, changes in bone lengths, and frame-level movement/bbox.
    bone_values = np.stack(
        [np.linalg.norm(rel[:, :, a] - rel[:, :, b], axis=-1) for a, b in COCO_BONES], axis=-1
    )
    bones = _stats_axis(bone_values, axis=1).reshape(n, -1)
    bone_velocity = np.diff(bone_values, axis=1)
    bone_motion = _stats_axis(np.abs(bone_velocity), axis=1).reshape(n, -1)

    speed = np.linalg.norm(velocity, axis=-1)
    frame_bbox = np.stack(
        [rel[..., 0].max(axis=2) - rel[..., 0].min(axis=2),
         rel[..., 1].max(axis=2) - rel[..., 1].min(axis=2)], axis=-1
    )
    global_stats = np.concatenate(
        [
            speed.mean(axis=(1, 2), keepdims=False)[:, None],
            speed.std(axis=(1, 2), keepdims=False)[:, None],
            speed.max(axis=(1, 2), keepdims=False)[:, None],
            np.quantile(speed, 0.75, axis=(1, 2))[:, None],
            frame_bbox.mean(axis=1),
            frame_bbox.std(axis=1),
            frame_bbox.max(axis=1),
            scale[:, 0, 0, 0, None],
            np.asarray(present.mean(axis=1), dtype=np.float32)[:, None],
        ],
        axis=1,
    ).astype(np.float32, copy=False)

    # Low-dimensional summaries of the learned CTR-GCN feature map.  Grouping
    # channels prevents the gate from memorising a 128-dimensional activation
    # fingerprint while retaining activity energy and temporal variation.
    raw_pose = np.nan_to_num(np.asarray(pose, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    pose_person = np.concatenate(
        [
            raw_pose.mean(axis=(2, 3, 4)),
            raw_pose.std(axis=(2, 3, 4)),
            np.abs(raw_pose).mean(axis=(2, 3, 4)),
            np.abs(raw_pose).max(axis=(2, 3, 4)),
        ],
        axis=1,
    )
    pose_delta = np.diff(raw_pose, axis=3)
    pose_temporal = np.concatenate(
        [
            np.abs(pose_delta).mean(axis=(2, 3, 4)),
            np.abs(pose_delta).std(axis=(2, 3, 4)),
        ],
        axis=1,
    )
    channels = raw_pose.shape[2]
    groups = 8
    if channels % groups == 0:
        grouped = raw_pose.reshape(n, raw_pose.shape[1], groups, channels // groups, raw_pose.shape[3], raw_pose.shape[4])
        pose_groups = np.concatenate(
            [
                np.abs(grouped).mean(axis=(3, 4, 5)),
                np.abs(grouped).std(axis=(3, 4, 5)),
            ],
            axis=1,
        ).reshape(n, -1)
    else:
        pose_groups = np.empty((n, 0), dtype=np.float32)
    pose_summary = np.concatenate((pose_person, pose_temporal, pose_groups), axis=1)

    result = np.concatenate(
        (position, velocity_stats, acceleration_stats, bones, bone_motion, global_stats, pose_summary),
        axis=1,
    ).astype(np.float32, copy=False)
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


def extract_skeleton_features(data_dir, prefix, chunk_size=2048):
    pose = np.load(Path(data_dir) / f"{prefix}_pose_coco17.npy", mmap_mode="r")
    joint_xy = np.load(Path(data_dir) / f"{prefix}_joint_xy_coco17.npy", mmap_mode="r")
    labels = np.load(Path(data_dir) / f"{prefix}_labels.npy", mmap_mode="r")
    outputs = []
    for start in range(0, len(labels), chunk_size):
        stop = min(start + chunk_size, len(labels))
        outputs.append(skeleton_features(pose[start:stop], joint_xy[start:stop]))
    return np.concatenate(outputs, axis=0), np.asarray(labels, dtype=np.int64)


@torch.inference_mode()
def extract_vpoclip_features(model, config, config_path, data_dir, prefix, device, batch_size, workers):
    dataset = TrimodalContrastiveDataset(data_dir, prefix=prefix, mmap=True, pose_suffix="_coco17")
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": workers,
        "persistent_workers": workers > 0,
        "pin_memory": device.type == "cuda",
    }
    if workers > 0:
        loader_kwargs["prefetch_factor"] = 4
    loader = DataLoader(dataset, **loader_kwargs)
    use_amp = bool(config.get("runtime", {}).get("amp", False)) and device.type == "cuda"
    text = l2_normalize(model.text_features.float())
    cosines = []
    labels = []
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            embedding = l2_normalize(
                model.encode_visual(batch["video"], batch["pose"], batch["object"], batch["joint_xy"]).float()
            )
            cosine = embedding @ text.t()
        cosines.append(cosine.float().cpu())
        labels.append(batch["label"].cpu())
    cosine = torch.cat(cosines).numpy().astype(np.float32, copy=False)
    labels = torch.cat(labels).numpy().astype(np.int64, copy=False)
    # Only confidence summaries are retained; the gate remains skeleton-led.
    scale = float(model.logit_scale.detach().float().exp().clamp(max=100).cpu())
    logits = cosine / max(scale, 1e-6)
    probabilities = torch.softmax(torch.from_numpy(logits / 0.1), dim=1).numpy()
    order = np.sort(cosine, axis=1)
    all_entropy = -(probabilities * np.log(np.maximum(probabilities, 1e-8))).sum(axis=1)
    summary = np.stack(
        [
            cosine.max(axis=1),
            cosine.mean(axis=1),
            cosine.std(axis=1),
            order[:, -1] - order[:, -2],
            order[:, -1] - order[:, -5],
            all_entropy,
            probabilities.max(axis=1),
        ], axis=1,
    ).astype(np.float32)
    return summary, labels


class UnknownGateMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        hidden = max(128, min(512, input_dim * 2))
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(hidden, max(32, hidden // 2)),
            nn.GELU(),
            nn.Linear(max(32, hidden // 2), 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _auc(labels, scores):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    positives = labels == 1
    negatives = labels == 0
    if not positives.any() or not negatives.any():
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    pos_ranks = ranks[positives].sum()
    return float((pos_ranks - positives.sum() * (positives.sum() + 1) / 2) /
                 (positives.sum() * negatives.sum()))


def gate_metrics(labels, probabilities, threshold):
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predicted = probabilities >= threshold
    unknown = labels == 1
    known = labels == 0
    tpr = float((predicted[unknown]).mean()) if unknown.any() else float("nan")
    tnr = float((~predicted[known]).mean()) if known.any() else float("nan")
    precision = float(unknown[predicted].mean()) if predicted.any() else 0.0
    balanced = 0.5 * (tpr + tnr)
    harmonic = 2 * tpr * tnr / max(tpr + tnr, 1e-12)
    return {
        "threshold": float(threshold),
        "unknown_recall": tpr,
        "known_acceptance": tnr,
        "unknown_precision": precision,
        "balanced_accuracy": balanced,
        "harmonic_unknown_known": harmonic,
        "num_unknown": int(unknown.sum()),
        "num_known": int(known.sum()),
        "num_predicted_unknown": int(predicted.sum()),
        "auc": _auc(labels, probabilities),
    }


def choose_threshold(labels, probabilities):
    # Quantile grid is deterministic and avoids a hidden test-set selection.
    grid = np.unique(np.quantile(probabilities, np.linspace(0.001, 0.999, 999)))
    candidates = [gate_metrics(labels, probabilities, float(t)) for t in grid]
    return max(candidates, key=lambda row: (row["harmonic_unknown_known"], row["balanced_accuracy"]))


def _run_model(model, x, device, batch_size=8192):
    output = []
    loader = DataLoader(torch.from_numpy(x), batch_size=batch_size, shuffle=False, pin_memory=device.type == "cuda")
    model.eval()
    for batch in loader:
        output.append(torch.sigmoid(model(batch.to(device, non_blocking=True))).detach().cpu())
    return torch.cat(output).numpy()


def main():
    parser = argparse.ArgumentParser(description="Train a pseudo-unseen skeleton unknown gate.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--gate-batch-size", type=int, default=8192)
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = str(Path(args.config).resolve())
    config = load_config(config_path)
    device = get_device(config.get("runtime", {}).get("device", "cuda:0"))
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    print(f"gate device: {device}", flush=True)
    print(f"pseudo unknown classes: {PSEUDO_UNSEEN}; true unseen held out: {TRUE_UNSEEN}", flush=True)

    # Build the same text bank as training and load only the requested final
    # checkpoint.  VPO confidence is an auxiliary feature, not the gate target.
    model = build_model_from_config(config, device=device,
                                    download_root=get_path_from_config(config_path, config["model"]["text_encoder"].get("download_root")))
    prepare_action_text_bank(model, config, config_path)
    try:
        state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(args.checkpoint, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"checkpoint loaded: missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    model.eval()

    skeleton = {}
    vpoclip = {}
    labels = {}
    for prefix in ("trimodal_train", "trimodal_val", "trimodal_test_seen", "trimodal_test"):
        print(f"extracting skeleton features: {prefix}", flush=True)
        skeleton[prefix], labels[prefix] = extract_skeleton_features(args.data_dir, prefix)
        print(f"extracting VPO confidence: {prefix} ({len(labels[prefix])} clips)", flush=True)
        vpoclip[prefix], vpo_labels = extract_vpoclip_features(
            model, config, config_path, args.data_dir, prefix, device, args.batch_size, args.workers
        )
        if not np.array_equal(labels[prefix], vpo_labels):
            raise RuntimeError(f"label order mismatch for {prefix}")
    feature_names = [f"skeleton_{i:04d}" for i in range(skeleton["trimodal_train"].shape[1])]
    feature_names += ["vpo_max_cosine", "vpo_mean_cosine", "vpo_std_cosine",
                      "vpo_top1_margin", "vpo_top5_margin", "vpo_entropy", "vpo_max_probability"]

    def joined(prefix):
        return np.concatenate((skeleton[prefix], vpoclip[prefix]), axis=1).astype(np.float32)

    train_x = joined("trimodal_train")
    val_x = joined("trimodal_val")
    seen_test_x = joined("trimodal_test_seen")
    unseen_test_x = joined("trimodal_test")
    train_y = np.isin(labels["trimodal_train"], PSEUDO_UNSEEN).astype(np.float32)
    val_y = np.isin(labels["trimodal_val"], PSEUDO_UNSEEN).astype(np.float32)

    mean = train_x.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = train_x.std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(std, 1e-5)
    train_x = np.clip((train_x - mean) / std, -12.0, 12.0).astype(np.float32)
    val_x = np.clip((val_x - mean) / std, -12.0, 12.0).astype(np.float32)
    seen_test_x = np.clip((seen_test_x - mean) / std, -12.0, 12.0).astype(np.float32)
    unseen_test_x = np.clip((unseen_test_x - mean) / std, -12.0, 12.0).astype(np.float32)

    x_tensor = torch.from_numpy(train_x)
    y_tensor = torch.from_numpy(train_y)
    loader = DataLoader(
        TensorDataset(x_tensor, y_tensor),
        batch_size=args.gate_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    gate = UnknownGateMLP(train_x.shape[1]).to(device)
    positives = max(float(train_y.sum()), 1.0)
    negatives = max(float((1.0 - train_y).sum()), 1.0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([negatives / positives], device=device))
    optimizer = torch.optim.AdamW(gate.parameters(), lr=2e-3, weight_decay=1e-4)

    best_state = None
    best_val_auc = -math.inf
    stale = 0
    for epoch in range(args.epochs):
        gate.train()
        losses = []
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = gate(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val_prob = _run_model(gate, val_x, device, args.gate_batch_size)
        val_auc = _auc(val_y, val_prob)
        if epoch == 0 or epoch % 10 == 9:
            print(f"gate epoch {epoch + 1}/{args.epochs} loss={np.mean(losses):.5f} val_auc={val_auc:.4f}", flush=True)
        if val_auc > best_val_auc + 1e-5:
            best_val_auc = val_auc
            best_state = {key: value.detach().cpu().clone() for key, value in gate.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= 20:
                print(f"gate early stop at epoch {epoch + 1}", flush=True)
                break

    if best_state is None:
        raise RuntimeError("gate did not produce a checkpoint")
    gate.load_state_dict(best_state)
    val_prob = _run_model(gate, val_x, device, args.gate_batch_size)
    threshold_row = choose_threshold(val_y.astype(np.int64), val_prob)
    threshold = float(threshold_row["threshold"])

    seen_prob = _run_model(gate, seen_test_x, device, args.gate_batch_size)
    unseen_prob = _run_model(gate, unseen_test_x, device, args.gate_batch_size)
    test_labels = np.concatenate((np.zeros(len(seen_prob), dtype=np.int64), np.ones(len(unseen_prob), dtype=np.int64)))
    test_prob = np.concatenate((seen_prob, unseen_prob))
    test_metrics = gate_metrics(test_labels, test_prob, threshold)
    val_metrics = gate_metrics(val_y.astype(np.int64), val_prob, threshold)

    bundle = {
        "state_dict": gate.state_dict(),
        "input_dim": train_x.shape[1],
        "feature_names": feature_names,
        "mean": torch.from_numpy(mean),
        "std": torch.from_numpy(std),
        "threshold": threshold,
        "pseudo_unknown": PSEUDO_UNSEEN,
        "true_unseen_held_out": TRUE_UNSEEN,
        "architecture": "UnknownGateMLP",
    }
    bundle_path = output_dir / "skeleton_unknown_gate.pt"
    torch.save(bundle, bundle_path)
    result = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "data_dir": str(Path(args.data_dir).resolve()),
        "feature_dim": int(train_x.shape[1]),
        "feature_schema": feature_names,
        "pseudo_groups": PSEUDO_GROUPS,
        "true_unseen_held_out": TRUE_UNSEEN,
        "train_counts": {"known": int((train_y == 0).sum()), "pseudo_unknown": int((train_y == 1).sum())},
        "val_counts": {"known": int((val_y == 0).sum()), "pseudo_unknown": int((val_y == 1).sum())},
        "validation": val_metrics,
        "validation_threshold_selection": threshold_row,
        "true_test": {
            "seen_test_samples": int(len(seen_prob)),
            "unseen_test_samples": int(len(unseen_prob)),
            **test_metrics,
        },
        "bundle": str(bundle_path),
    }
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    (output_dir / "README.txt").write_text(
        "Unknown gate trained on skeleton activity ranges/velocity/acceleration/bone-length "
        "statistics plus VPO confidence summaries. Threshold calibrated on pseudo classes "
        f"{PSEUDO_UNSEEN}; true unseen {TRUE_UNSEEN} was held out until the final report.\n"
    )
    print(json.dumps({"validation": val_metrics, "true_test": test_metrics, "bundle": str(bundle_path)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
