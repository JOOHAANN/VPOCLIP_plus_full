"""Structured active-view experiment from ``Structured_RL_vs_Handcrafted...``.

This file is deliberately independent of the historical SC-NBV trainers.  It
uses the old VPOCLIP cache/checkpoint, fixed view 0, exactly three legal
destinations, and the strict 40/10/5 class split.  The learned selector is a
one-step contextual-bandit utility scorer: privileged future views are used
only to construct an offline structured reward, never as model inputs.

The cached pose is 2-D RTMPose x/y without per-joint confidence.  Visibility
therefore means pose-observation/edge-cropping proxy, and is recorded as such
in the output metadata rather than presented as true occlusion labels.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


NUM_VIEWS = 4
NUM_CLASSES = 55
CURRENT_VIEW = 0
TRAIN_CLASSES = [1, 3, 4, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20,
                 21, 23, 24, 27, 28, 29, 30, 31, 32, 33, 35, 36, 37,
                 39, 40, 41, 43, 46, 47, 48, 49, 52, 53, 54]
PSEUDO_CLASSES = [5, 6, 14, 22, 25, 38, 42, 44, 45, 51]
SEEN_CLASSES = sorted(TRAIN_CLASSES + PSEUDO_CLASSES)
UNSEEN_CLASSES = [0, 2, 26, 34, 50]
POSE_DIM = 442
BODY_VIS_DIM = 7
INTERACTION_DIM = 5
ORIENTATION_DIM = 6
CONTEXT_DIM = BODY_VIS_DIM + INTERACTION_DIM + ORIENTATION_DIM + 1 + 4
CANDIDATE_DIM = 8


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def wrap_radians(value: np.ndarray) -> np.ndarray:
    return (value + np.pi) % (2.0 * np.pi) - np.pi


def load_raw(root: Path) -> dict[str, Any]:
    root = Path(root)
    def load(name: str, dtype: Any | None = None) -> np.ndarray:
        path = root / f"{name}.npy"
        if not path.exists():
            raise FileNotFoundError(path)
        value = np.load(path, mmap_mode="r", allow_pickle=False)
        return np.asarray(value, dtype=dtype) if dtype is not None else value

    labels_path = root / "target_columns.npy"
    labels = load("target_columns" if labels_path.exists() else "labels", np.int64)
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    confidence = np.zeros((len(metadata["episodes"]), NUM_VIEWS), dtype=np.float32)
    geometry_source: dict[str, int] = {}
    for episode_id, episode in enumerate(metadata["episodes"]):
        for view_id, view in enumerate(episode.get("views", [])[:NUM_VIEWS]):
            value = float(view.get("geometry_confidence", 0.0) or 0.0)
            confidence[episode_id, view_id] = np.clip(value, 0.0, 1.0)
            source = str(view.get("geometry_source", "unknown"))
            geometry_source[source] = geometry_source.get(source, 0) + 1
    return {
        "root": str(root),
        "z": load("z", np.float32),
        "logits": load("logits", np.float32),
        "pose": load("pose", np.float32),
        "object_map": load("object_map", np.float32),
        "quality": load("image_quality", np.float32),
        "geometry": load("view_geometry", np.float32),
        "cost": load("move_cost", np.float32),
        "reachable": load("reachable").astype(bool),
        "valid": load("view_valid").astype(bool),
        "labels": labels,
        "prototypes": load("text_prototypes", np.float32),
        "metadata": metadata,
        "geometry_confidence": confidence,
        "geometry_source": geometry_source,
    }


def select_episodes(raw: Mapping[str, Any], classes: Sequence[int]) -> np.ndarray:
    valid = np.asarray(raw["valid"])
    labels = np.asarray(raw["labels"])
    keep = (
        np.isin(labels, np.asarray(list(classes), dtype=np.int64))
        & (valid.sum(axis=1) == NUM_VIEWS)
        & valid[:, CURRENT_VIEW]
    )
    return np.flatnonzero(keep).astype(np.int64)


def _object_summary(object_map: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return object visibility proxy and normalized object centroid.

    Object maps are [50,6,6] inverse-distance maps.  They do not preserve
    individual detector boxes, so this intentionally uses aggregate mass and
    is not called a precise interaction annotation.
    """

    n, v, d = object_map.shape
    if d != 1800:
        raise ValueError(f"expected object map width 1800, got {d}")
    maps = np.maximum(object_map.reshape(n, v, 50, 6, 6), 0.0)
    class_max = maps.max(axis=(3, 4))
    max_value = class_max.max(axis=2)
    visibility = np.clip(max_value / 10.0, 0.0, 1.0)
    mass = maps.sum(axis=2)
    grid_x = np.linspace(-1.0, 1.0, 6, dtype=np.float32)[None, None, None, :]
    grid_y = np.linspace(1.0, -1.0, 6, dtype=np.float32)[None, None, :, None]
    total = mass.sum(axis=(2, 3))
    center_x = (mass * grid_x).sum(axis=(2, 3)) / np.maximum(total, 1e-6)
    center_y = (mass * grid_y).sum(axis=(2, 3)) / np.maximum(total, 1e-6)
    center = np.stack((center_x, center_y), axis=-1)
    center[total <= 1e-6] = 0.0
    return visibility.astype(np.float32), center.astype(np.float32)


def structured_view_features(
    pose: np.ndarray,
    quality: np.ndarray,
    object_map: np.ndarray,
) -> dict[str, np.ndarray]:
    """Build current/future structured sensors for every cached view."""

    if pose.shape[-1] != POSE_DIM:
        raise ValueError(f"pose must end in {POSE_DIM}, got {pose.shape}")
    n, v = pose.shape[:2]
    points = np.nan_to_num(pose).reshape(n, v, 13, 17, 2)
    present = np.linalg.norm(points, axis=-1) > 0.02
    x, y = points[..., 0], points[..., 1]
    edge_distance = np.minimum.reduce((x + 1.0, 1.0 - x, y + 1.0, 1.0 - y))
    edge_factor = np.clip(edge_distance / 0.20, 0.0, 1.0)
    observed = present.astype(np.float32) * edge_factor.astype(np.float32)

    # COCO-17: nose/eyes/ears, elbows/wrists, shoulders/hips, knees/ankles.
    parts = ([0, 1, 2, 3, 4], [7, 9], [8, 10], [5, 6, 11, 12],
             [13, 15], [14, 16])
    vis_parts = np.stack(
        [observed[..., indices].mean(axis=(-1, -2)) for indices in parts], axis=-1
    )
    object_visibility, object_center = _object_summary(object_map)
    visibility = np.concatenate((vis_parts, object_visibility[..., None]), axis=-1)

    shoulder_mid = 0.5 * (points[..., 5, :] + points[..., 6, :])
    hip_mid = 0.5 * (points[..., 11, :] + points[..., 12, :])
    torso_center = 0.5 * (shoulder_mid + hip_mid)
    shoulder_vec = points[..., 6, :] - points[..., 5, :]
    hip_vec = points[..., 12, :] - points[..., 11, :]
    shoulder_angle = np.arctan2(shoulder_vec[..., 1], shoulder_vec[..., 0])
    hip_angle = np.arctan2(hip_vec[..., 1], hip_vec[..., 0])
    relative_angle = wrap_radians(shoulder_angle - hip_angle)
    orientation = np.stack(
        (np.sin(shoulder_angle).mean(-1), np.cos(shoulder_angle).mean(-1),
         np.sin(hip_angle).mean(-1), np.cos(hip_angle).mean(-1),
         np.sin(relative_angle).mean(-1), np.cos(relative_angle).mean(-1)), axis=-1
    ).astype(np.float32)

    scale = np.linalg.norm(shoulder_vec, axis=-1)
    # Keep one robust torso scale per episode/view.  The explicit [N,V]
    # shape is important: pairwise distances below are [N,V,T], and adding
    # an extra singleton here would accidentally broadcast to four axes.
    scale = np.maximum(np.median(scale, axis=-1), 0.10)

    def pair_evidence(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        good = (np.linalg.norm(a, axis=-1) > 0.02) & (np.linalg.norm(b, axis=-1) > 0.02)
        distance = np.linalg.norm(a - b, axis=-1) / scale[..., None]
        return (good.astype(np.float32) * np.exp(-np.clip(distance, 0.0, 10.0))).mean(-1)

    head = points[..., [0, 1, 2, 3, 4], :].mean(axis=-2)
    left_hand, right_hand = points[..., 9, :], points[..., 10, :]
    hands = 0.5 * (left_hand + right_hand)
    hand_face = np.maximum(pair_evidence(left_hand, head), pair_evidence(right_hand, head))
    hand_torso = np.maximum(pair_evidence(left_hand, torso_center), pair_evidence(right_hand, torso_center))
    hand_hand = pair_evidence(left_hand, right_hand)
    object_xy = object_center[..., None, :]
    object_xy = np.broadcast_to(object_xy, (n, v, 13, 2))
    object_evidence = np.maximum(
        pair_evidence(left_hand, object_xy), pair_evidence(right_hand, object_xy)
    ) * object_visibility
    ankles = 0.5 * (points[..., 15, :] + points[..., 16, :])
    ground_proximity = np.clip((-ankles[..., 1] + 0.25) / 1.25, 0.0, 1.0)
    ground_evidence = (present[..., 15].astype(np.float32) * ground_proximity).mean(-1)
    # Ambiguity is high when the relevant relation is unobserved.  This is a
    # causal 2-D proxy, not a future-view occlusion label.
    ambiguity = 1.0 - np.stack(
        (hand_face, object_evidence, hand_torso, hand_hand, ground_evidence), axis=-1
    )
    ambiguity = np.clip(ambiguity, 0.0, 1.0).astype(np.float32)
    uncertainty = np.zeros((n, v, 1), dtype=np.float32)
    return {
        "visibility": np.clip(visibility, 0.0, 1.0).astype(np.float32),
        "ambiguity": ambiguity,
        "orientation": orientation,
        "uncertainty": uncertainty,
        "quality": np.clip(quality, 0.0, 1.0).astype(np.float32),
    }


def entropy(logits: np.ndarray, temperature: float = 2.0) -> np.ndarray:
    values = logits / max(float(temperature), 1e-4)
    values = values - values.max(axis=-1, keepdims=True)
    probability = np.exp(np.clip(values, -80.0, 80.0))
    probability /= np.maximum(probability.sum(axis=-1, keepdims=True), 1e-8)
    return (-(probability * np.log(np.maximum(probability, 1e-8))).sum(axis=-1)
            / math.log(float(logits.shape[-1]))).astype(np.float32)


def candidate_inputs(raw: Mapping[str, Any], episodes: np.ndarray, features: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    geometry = np.asarray(raw["geometry"][episodes], dtype=np.float32)
    bearings = np.arctan2(geometry[..., 0], geometry[..., 1])
    source = bearings[:, CURRENT_VIEW]
    relative = wrap_radians(bearings - source[:, None])
    cost = np.asarray(raw["cost"][episodes, CURRENT_VIEW], dtype=np.float32)
    confidence = np.asarray(raw["geometry_confidence"][episodes], dtype=np.float32)
    valid = np.asarray(raw["valid"][episodes], dtype=bool)
    valid &= np.asarray(raw["reachable"][episodes, CURRENT_VIEW], dtype=bool)
    valid[:, CURRENT_VIEW] = False
    candidate = np.stack(
        (np.sin(relative), np.cos(relative), np.abs(relative) / np.pi,
         np.sin(bearings), np.cos(bearings), cost, confidence,
         valid.astype(np.float32)), axis=-1
    ).astype(np.float32)
    current = np.concatenate(
        (features["visibility"][:, CURRENT_VIEW],
         features["ambiguity"][:, CURRENT_VIEW],
         features["orientation"][:, CURRENT_VIEW],
         features["uncertainty"][:, CURRENT_VIEW],
         features["quality"][:, CURRENT_VIEW]), axis=-1
    ).astype(np.float32)
    return current, candidate, {
        "relative": relative.astype(np.float32),
        "bearings": bearings.astype(np.float32),
        "valid": valid,
        "cost": cost,
        "confidence": confidence,
    }


def margin_scores(logits: np.ndarray, labels: np.ndarray, classes: Sequence[int]) -> np.ndarray:
    """Target-vs-rival margin, broadcasting labels over view axes.

    ``logits`` may be [N,C] or [N,V,C].  Labels may be [N] or [N,1]; the
    latter is used when one label must be compared with every candidate view.
    """
    bank = np.asarray(sorted({int(x) for x in classes}), dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    local = np.searchsorted(bank, labels)
    safe_local = np.minimum(local, len(bank) - 1)
    if np.any(local >= len(bank)) or np.any(bank[safe_local] != labels):
        raise ValueError("labels are outside the requested class bank")
    scores = logits[..., bank]
    if scores.ndim != local.ndim + 1:
        raise ValueError(f"logits/labels rank mismatch: {scores.shape} vs {labels.shape}")
    # A [N,1] label index broadcasts over the [N,V] view dimensions.
    index = local[..., None]
    true = np.take_along_axis(scores, index, axis=-1)[..., 0]
    rival = scores.copy()
    np.put_along_axis(rival, index, -np.inf, axis=-1)
    return true - rival.max(axis=-1)


def component_rewards(
    raw: Mapping[str, Any],
    episodes: np.ndarray,
    features: Mapping[str, np.ndarray],
    classes: Sequence[int],
    har_weight: float,
    standardize_stats: Mapping[str, float] | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, float]]:
    current, candidate, info = candidate_inputs(raw, episodes, features)
    del current, candidate
    valid = info["valid"]
    # ``prepare_data`` already keeps features and the selected raw rows in the
    # same local episode order.  Do not index them a second time: that would
    # silently become wrong if a future caller supplies non-contiguous IDs.
    vis = np.asarray(features["visibility"], dtype=np.float32)
    amb = np.asarray(features["ambiguity"], dtype=np.float32)
    if vis.shape[0] != len(episodes) or amb.shape[0] != len(episodes):
        raise ValueError("feature/episode alignment error")
    current_vis = vis[:, CURRENT_VIEW]
    current_amb = amb[:, CURRENT_VIEW]
    destination_vis = vis
    destination_amb = amb
    missing = 1.0 - current_vis
    delta_vis = ((destination_vis - current_vis[:, None]) * missing[:, None]).mean(-1)
    delta_interaction = (current_amb[:, None] - destination_amb).mean(-1)
    relative = info["relative"]
    absolute_angle = np.abs(relative)
    complementarity = np.exp(-((absolute_angle - (np.pi / 2.0)) / (np.pi / 3.0)) ** 2)
    uncertainty_all = entropy(np.asarray(raw["logits"][episodes]), temperature=2.0)
    fused_logits = 0.5 * (np.asarray(raw["logits"][episodes, CURRENT_VIEW])[:, None] + np.asarray(raw["logits"][episodes]))
    current_uncertainty = uncertainty_all[:, CURRENT_VIEW]
    fused_uncertainty = entropy(fused_logits, temperature=2.0)
    delta_information = current_uncertainty[:, None] - fused_uncertainty
    current_margin = margin_scores(np.asarray(raw["logits"][episodes, CURRENT_VIEW]), np.asarray(raw["labels"][episodes]), classes)
    fused_margin = margin_scores(fused_logits, np.asarray(raw["labels"][episodes])[:, None], classes)
    delta_har = fused_margin - current_margin[:, None]
    raw_components = {
        "body_visibility": delta_vis.astype(np.float32),
        "interaction_visibility": delta_interaction.astype(np.float32),
        "complementarity": complementarity.astype(np.float32),
        "information": delta_information.astype(np.float32),
        "motion_cost": info["cost"].astype(np.float32),
        "har_gain": delta_har.astype(np.float32),
    }
    stats: dict[str, float] = {}
    standardized: dict[str, np.ndarray] = {}
    for name, value in raw_components.items():
        finite = value[valid]
        if standardize_stats is None:
            mean = float(finite.mean()) if len(finite) else 0.0
            std = max(float(finite.std()), 1e-5) if len(finite) else 1.0
            stats[f"{name}_mean"] = mean
            stats[f"{name}_std"] = std
        else:
            mean = float(standardize_stats.get(f"{name}_mean", 0.0))
            std = max(float(standardize_stats.get(f"{name}_std", 1.0)), 1e-5)
        standardized[name] = ((value - mean) / std).astype(np.float32)
    reward = (
        0.35 * standardized["body_visibility"]
        + 0.30 * standardized["interaction_visibility"]
        + 0.25 * standardized["complementarity"]
        + 0.10 * standardized["information"]
        - 0.10 * standardized["motion_cost"]
        + float(har_weight) * standardized["har_gain"]
    ).astype(np.float32)
    reward[~valid] = 0.0
    return reward, raw_components, stats


class StructuredScorer(nn.Module):
    def __init__(self, hidden: int = 128) -> None:
        super().__init__()
        self.context = nn.Sequential(nn.Linear(CONTEXT_DIM, hidden), nn.LayerNorm(hidden), nn.GELU(),
                                      nn.Linear(hidden, hidden), nn.GELU())
        self.candidate = nn.Sequential(nn.Linear(CANDIDATE_DIM, hidden), nn.LayerNorm(hidden), nn.GELU(),
                                       nn.Linear(hidden, hidden), nn.GELU())
        self.head = nn.Sequential(nn.Linear(hidden * 3, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, context: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
        context_h = self.context(context)[:, None, :]
        candidate_h = self.candidate(candidate)
        return self.head(torch.cat((context_h.expand_as(candidate_h), candidate_h,
                                    context_h * candidate_h), dim=-1)).squeeze(-1)


def pairwise_loss(prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, margin: float = 0.10) -> torch.Tensor:
    losses = []
    for left in range(NUM_VIEWS):
        for right in range(left + 1, NUM_VIEWS):
            ok = valid[:, left] & valid[:, right]
            delta = target[:, left] - target[:, right]
            informative = ok & (delta.abs() > 0.02)
            if informative.any():
                sign = delta[informative].sign()
                losses.append(F.relu(float(margin) - sign * (prediction[informative, left] - prediction[informative, right])))
    return torch.cat(losses).mean() if losses else prediction.new_zeros(())


def action_from_model(model: nn.Module, context: np.ndarray, candidate: np.ndarray, valid: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.inference_mode():
        scores = model(torch.from_numpy(context).to(device), torch.from_numpy(candidate).to(device))
        scores = scores.masked_fill(~torch.from_numpy(valid).to(device), -torch.inf)
        return scores.argmax(-1).cpu().numpy().astype(np.int64)


def train_model(cfg: Mapping[str, Any], train_data: dict[str, Any], pseudo_data: dict[str, Any], name: str, har_weight: float, hidden: int, seed: int, output: Path, train_stats: dict[str, float]) -> dict[str, Any]:
    seed_all(seed)
    device = torch.device(str(cfg["device"]))
    model = StructuredScorer(hidden=hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, int(cfg["epochs"]), eta_min=float(cfg["lr"]) * 0.05)
    train_context, train_candidate, train_info = candidate_inputs(train_data["raw"], train_data["episodes"], train_data["features"])
    pseudo_context, pseudo_candidate, pseudo_info = candidate_inputs(pseudo_data["raw"], pseudo_data["episodes"], pseudo_data["features"])
    train_reward, _, _ = component_rewards(
        train_data["raw"], train_data["episodes"], train_data["features"],
        TRAIN_CLASSES, har_weight, standardize_stats=train_stats,
    )
    # Recompute target normalization on train only and carry it explicitly in
    # output.  The input features themselves are unchanged by this operation.
    train_valid = train_info["valid"]
    x_context = torch.from_numpy(train_context).to(device)
    x_candidate = torch.from_numpy(train_candidate).to(device)
    x_target = torch.from_numpy(train_reward).to(device)
    x_valid = torch.from_numpy(train_valid).to(device)
    best_score = -float("inf")
    best_epoch = 0
    run_dir = output / name / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "train.log"
    for epoch in range(1, int(cfg["epochs"]) + 1):
        model.train()
        permutation = torch.randperm(len(x_context), device=device)
        losses = []
        started = time.monotonic()
        for batch_ids in permutation.split(int(cfg["batch_size"])):
            prediction = model(x_context[batch_ids], x_candidate[batch_ids])
            valid = x_valid[batch_ids]
            weight = valid.float()
            huber = (F.smooth_l1_loss(prediction, x_target[batch_ids], reduction="none") * weight).sum() / weight.sum().clamp_min(1.0)
            rank = pairwise_loss(prediction, x_target[batch_ids], valid)
            loss = huber + float(cfg["rank_lambda"]) * rank
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        pseudo_reward, _, _ = component_rewards(
            pseudo_data["raw"], pseudo_data["episodes"], pseudo_data["features"],
            PSEUDO_CLASSES, har_weight, standardize_stats=train_stats,
        )
        pseudo_action = action_from_model(model, pseudo_context, pseudo_candidate, pseudo_info["valid"], device)
        score = float(pseudo_reward[np.arange(len(pseudo_action)), pseudo_action].mean())
        row = {"epoch": epoch, "loss": float(np.mean(losses)), "pseudo_structured_reward": score,
               "seconds": time.monotonic() - started, "hidden": hidden, "har_weight": har_weight}
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        print(json.dumps({"variant": name, "seed": seed, **row}), flush=True)
        if score > best_score:
            best_score, best_epoch = score, epoch
            torch.save({"model": model.state_dict(), "epoch": epoch, "best_score": score,
                        "seed": seed, "variant": name, "hidden": hidden, "har_weight": har_weight,
                        "train_stats": train_stats}, run_dir / "best.pt")
        torch.save({"model": model.state_dict(), "epoch": epoch, "best_score": best_score,
                    "seed": seed, "variant": name, "hidden": hidden, "har_weight": har_weight,
                    "train_stats": train_stats}, run_dir / "last.pt")
    result = {"variant": name, "seed": seed, "best_epoch": best_epoch, "best_score": best_score,
              "hidden": hidden, "har_weight": har_weight,
              "path": str(run_dir / "best.pt"), "parameters": sum(p.numel() for p in model.parameters())}
    (run_dir / "complete.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def handcrafted_components(data: dict[str, Any]) -> np.ndarray:
    context, candidate, info = candidate_inputs(data["raw"], data["episodes"], data["features"])
    del context
    current_vis = data["features"]["visibility"][:, CURRENT_VIEW]
    current_amb = data["features"]["ambiguity"][:, CURRENT_VIEW]
    rel = info["relative"]
    absolute = np.abs(rel)
    side = np.sin(absolute).clip(0.0, 1.0)
    front = (1.0 + np.cos(rel)) * 0.5
    part_weights = np.asarray([0.35, 1.0, 1.0, 0.55, 0.85, 0.85, 0.30], dtype=np.float32)
    part_gain = 0.25 * front[..., None] + 0.75 * side[..., None] * part_weights
    vis_score = ((1.0 - current_vis)[:, None] * part_gain).mean(-1)
    interaction_gain = current_amb.mean(-1)[:, None] * (0.35 * front + 0.65 * side) * info["confidence"]
    comp_score = np.exp(-((absolute - np.pi / 2.0) / (np.pi / 3.0)) ** 2)
    components = np.stack((comp_score, vis_score, interaction_gain, info["cost"]), axis=-1).astype(np.float32)
    components[~info["valid"]] = 0.0
    return components


def handcrafted_actions(
    components: np.ndarray,
    weights: Sequence[float],
    valid: np.ndarray | None = None,
) -> np.ndarray:
    weights = np.asarray(list(weights), dtype=np.float32)
    scores = components @ np.asarray([weights[0], weights[1], weights[2], -weights[3]], dtype=np.float32)
    if valid is None:
        raise ValueError("handcrafted_actions requires the explicit legal-action mask")
    scores[~np.asarray(valid, dtype=bool)] = -np.inf
    return scores.argmax(-1).astype(np.int64)


def paired_ci(delta: np.ndarray, seed: int, samples: int = 10000) -> list[float]:
    if len(delta) == 0:
        return [0.0, 0.0]
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float32)
    chunk = 500
    for begin in range(0, samples, chunk):
        count = min(chunk, samples - begin)
        indices = rng.integers(0, len(delta), size=(count, len(delta)))
        means[begin:begin + count] = delta[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def evaluate_actions(data: dict[str, Any], classes: Sequence[int], actions: Mapping[str, np.ndarray], seed: int) -> dict[str, Any]:
    raw = data["raw"]
    episodes = data["episodes"]
    labels = np.asarray(raw["labels"])[episodes]
    logits = np.asarray(raw["logits"][episodes])
    current = logits[:, CURRENT_VIEW]
    fused = 0.5 * (current[:, None] + logits)
    bank = np.asarray(sorted({int(x) for x in classes}), dtype=np.int64)
    scores = fused[..., bank]
    local = np.searchsorted(bank, labels)
    correct = scores.argmax(-1) == local[:, None]
    _, _, info = candidate_inputs(raw, episodes, data["features"])
    valid = info["valid"]
    correct &= valid
    utility = margin_scores(fused, labels[:, None], classes) - margin_scores(current, labels, classes)[:, None]
    utility[~valid] = -np.inf
    oracle = utility.argmax(-1)
    sorted_utility = np.sort(np.where(valid, utility, -np.inf), axis=-1)
    best_second = sorted_utility[:, -1] - sorted_utility[:, -2]
    random_expected = (correct.astype(np.float32) * valid).sum(-1) / valid.sum(-1)
    destination_features = data["features"]
    result: dict[str, Any] = {"episodes": int(len(episodes)), "classes": [int(x) for x in classes], "methods": {}}
    for name, action in actions.items():
        action = np.asarray(action, dtype=np.int64)
        if action.shape != (len(episodes),):
            raise ValueError(f"{name} action shape must be {(len(episodes),)}, got {action.shape}")
        if not np.all(valid[np.arange(len(action)), action]):
            raise ValueError(f"{name} selected an illegal candidate")
        chosen_correct = correct[np.arange(len(action)), action]
        selected_utility = utility[np.arange(len(action)), action]
        distribution = np.bincount(action, minlength=NUM_VIEWS).astype(np.float64)
        distribution = distribution / max(distribution.sum(), 1.0)
        entropy_action = -sum(float(p) * math.log(float(p)) for p in distribution if p > 0) / math.log(3.0)
        current_vis = destination_features["visibility"][:, CURRENT_VIEW]
        selected_vis = destination_features["visibility"][np.arange(len(action)), action]
        current_amb = destination_features["ambiguity"][:, CURRENT_VIEW]
        selected_amb = destination_features["ambiguity"][np.arange(len(action)), action]
        current_unc = entropy(current) 
        selected_unc = entropy(logits[np.arange(len(action)), action])
        diff = chosen_correct.astype(np.float32) - random_expected
        if name == "random":
            # The primary random baseline is the exact conditional expectation
            # over the three legal destinations.  The sampled action remains
            # available for motion/action-distribution diagnostics.
            top1 = float(random_expected.mean())
            top1_delta = 0.0
            ci95 = [0.0, 0.0]
        else:
            top1 = float(chosen_correct.mean())
            top1_delta = float(diff.mean())
            name_seed = sum((index + 1) * ord(char) for index, char in enumerate(name))
            ci95 = paired_ci(diff, seed + name_seed)
        result["methods"][name] = {
            "top1": top1,
            "sampled_top1": float(chosen_correct.mean()),
            "top1_delta_vs_random_expected": top1_delta,
            "paired_bootstrap_ci95": ci95,
            "oracle_regret": float((utility.max(-1) - selected_utility).mean()),
            "oracle_hit_top1": float((action == oracle).mean()),
            "best_second_utility_margin": float(best_second.mean()),
            "action_entropy": float(entropy_action),
            "action_histogram": distribution.tolist(),
            "movement_cost": float(info["cost"][np.arange(len(action)), action].mean()),
            "delta_body_visibility": float((selected_vis - current_vis).mean()),
            "delta_interaction_visibility": float((current_amb - selected_amb).mean()),
            "delta_uncertainty": float((current_unc - selected_unc).mean()),
            "complementarity": float(np.exp(-((np.abs(info["relative"][np.arange(len(action)), action]) - np.pi / 2.0) / (np.pi / 3.0)) ** 2).mean()),
        }
    result["random_expected_top1"] = float(random_expected.mean())
    result["oracle_top1"] = float(correct[np.arange(len(oracle)), oracle].mean())
    result["oracle_utility_gap"] = float(utility.max(-1).mean())
    return result


def tune_handcrafted(pseudo_data: dict[str, Any]) -> tuple[list[float], dict[str, Any]]:
    components = handcrafted_components(pseudo_data)
    _, _, info = candidate_inputs(pseudo_data["raw"], pseudo_data["episodes"], pseudo_data["features"])
    raw = pseudo_data["raw"]
    episodes = pseudo_data["episodes"]
    labels = np.asarray(raw["labels"])[episodes]
    logits = np.asarray(raw["logits"][episodes])
    fused = 0.5 * (logits[:, CURRENT_VIEW, None] + logits)
    bank = np.asarray(PSEUDO_CLASSES, dtype=np.int64)
    scores = fused[..., bank]
    local = np.searchsorted(bank, labels)
    correct = scores.argmax(-1) == local[:, None]
    valid = info["valid"]
    correct &= valid
    expected = (correct * valid).sum(-1) / valid.sum(-1)
    margins = margin_scores(fused, labels[:, None], PSEUDO_CLASSES) - margin_scores(logits[:, CURRENT_VIEW], labels, PSEUDO_CLASSES)[:, None]
    margins[~valid] = -np.inf
    best = None
    grid = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
    for alpha in grid:
        for beta in grid:
            for gamma in grid:
                for lam in [0.0, 0.05, 0.10, 0.20, 0.40, 0.80]:
                    action = handcrafted_actions(components, [alpha, beta, gamma, lam], valid)
                    ok = correct[np.arange(len(action)), action]
                    gain = float((ok.astype(np.float32) - expected).mean())
                    utility = float(margins[np.arange(len(action)), action].mean())
                    key = (gain, utility, -lam)
                    if best is None or key > best[0]:
                        best = (key, [alpha, beta, gamma, lam])
    assert best is not None
    return best[1], {"pseudo_gain": best[0][0], "pseudo_margin": best[0][1], "grid": grid}


def prepare_data(raw_root: Path, split: str, classes: Sequence[int]) -> dict[str, Any]:
    raw = load_raw(raw_root / split)
    episodes = select_episodes(raw, classes)
    # Build sensors only on selected full episodes, keeping the candidate rows
    # in physical cache order while direction comes from per-recording bearing.
    pose = np.asarray(raw["pose"][episodes], dtype=np.float32)
    quality = np.asarray(raw["quality"][episodes], dtype=np.float32)
    objects = np.asarray(raw["object_map"][episodes], dtype=np.float32)
    features = structured_view_features(pose, quality, objects)
    features["uncertainty"] = entropy(np.asarray(raw["logits"][episodes]), temperature=2.0)[..., None]
    raw_selected = dict(raw)
    raw_selected["labels"] = np.asarray(raw["labels"])[episodes]
    for key in ("logits", "geometry", "cost", "reachable", "valid", "geometry_confidence"):
        raw_selected[key] = np.asarray(raw[key][episodes])
    return {"raw": raw_selected, "episodes": np.arange(len(episodes), dtype=np.int64), "features": features,
            "source_episode_ids": episodes, "split": split, "classes": list(classes)}


def load_old_actions(data: dict[str, Any], classes: Sequence[int], checkpoint: Path, device: torch.device) -> np.ndarray:
    """Run the preserved v6 legacy selector for the requested contexts.

    This is reported as a legacy baseline.  It uses the current-view
    VPOCLIP/HAR/logit representation and the historical network architecture;
    it is kept separate from the new class-agnostic structured selector.
    """

    from .causal_rank_v6 import CausalRanker, causal_state, class_mask
    from .train_rl_fast import GPUCache

    cache = GPUCache(Path(data["raw"]["root"]), device)
    episode_ids = torch.as_tensor(data["source_episode_ids"], device=device, dtype=torch.long)
    starts = torch.zeros(len(episode_ids), device=device, dtype=torch.long)
    bank = class_mask(classes, len(episode_ids), device)
    network = CausalRanker().to(device)
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    network.load_state_dict(saved["online"])
    network.eval()
    actions = []
    with torch.inference_mode():
        for begin in range(0, len(episode_ids), 1024):
            state = causal_state(cache, episode_ids[begin:begin + 1024], starts[begin:begin + 1024], bank[begin:begin + 1024])
            q = network(state).masked_fill(~state["mask"], -torch.inf)
            actions.append(q.argmax(-1).cpu().numpy())
    return np.concatenate(actions).astype(np.int64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=Path("data/rl_candidate_rank_v6_new50_5/cache"))
    parser.add_argument("--old-checkpoint", type=Path, default=Path("work_dir/active_view_candidate_rank_v6_new50_5/seed_20260911/best.pt"))
    parser.add_argument("--output", type=Path, default=Path("work_dir/structured_vs_handcrafted_v1"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--seed-list", default="20260909,20260910,20260911")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    raw_root = args.raw_root if args.raw_root.is_absolute() else root / args.raw_root
    output = args.output if args.output.is_absolute() else root / args.output
    output.mkdir(parents=True, exist_ok=True)
    cfg = {"device": args.device, "epochs": args.epochs, "batch_size": 512,
           "lr": 2e-4, "weight_decay": 0.02, "rank_lambda": 0.5}
    print(json.dumps({"STRUCTURED_EXPERIMENT_START": True, "raw_root": str(raw_root),
                      "checkpoint": str(args.old_checkpoint), "train_classes": TRAIN_CLASSES,
                      "pseudo_unseen_classes": PSEUDO_CLASSES, "true_unseen_classes": UNSEEN_CLASSES,
                      "protocol": "fixed view0, three legal candidates, one movement, terminal",
                      "visibility_definition": "2-D pose presence and image-edge proxy; no per-joint confidence in cache"}), flush=True)
    train_data = prepare_data(raw_root, "dqn_train", TRAIN_CLASSES)
    pseudo_data = prepare_data(raw_root, "val", PSEUDO_CLASSES)
    seen_data = prepare_data(raw_root, "seen_test", SEEN_CLASSES)
    unseen_data = prepare_data(raw_root, "test", UNSEEN_CLASSES)
    split_stats = {name: {"episodes": int(len(data["episodes"])), "source_episode_ids": int(len(data["source_episode_ids"]))}
                   for name, data in [("train", train_data), ("pseudo_unseen", pseudo_data), ("seen_test", seen_data), ("true_unseen", unseen_data)]}
    (output / "split_stats.json").write_text(json.dumps(split_stats, indent=2), encoding="utf-8")
    train_reward, _, reward_stats = component_rewards(train_data["raw"], train_data["episodes"], train_data["features"], TRAIN_CLASSES, 0.0)
    (output / "reward_stats.json").write_text(json.dumps(reward_stats, indent=2), encoding="utf-8")
    seeds = [int(x.strip()) for x in args.seed_list.split(",") if x.strip()]
    model_specs = [("structured_rl", 0.0, 128), ("structured_rl_har_aux", 0.10, 128),
                   ("structured_rl_large", 0.0, 256)]
    trained: dict[str, list[dict[str, Any]]] = {}
    for name, har_weight, hidden in model_specs:
        trained[name] = []
        for seed in seeds:
            trained[name].append(train_model(cfg, train_data, pseudo_data, name, har_weight, hidden, seed, output, reward_stats))
    tuned_weights, tune_info = tune_handcrafted(pseudo_data)
    (output / "handcrafted_tuning.json").write_text(json.dumps({"weights": tuned_weights, **tune_info}, indent=2), encoding="utf-8")
    device = torch.device(args.device)
    old_checkpoint = args.old_checkpoint if args.old_checkpoint.is_absolute() else root / args.old_checkpoint
    all_reports: dict[str, Any] = {"config": cfg, "checkpoint": str(old_checkpoint), "splits": split_stats,
                                    "visibility_definition": "2-D pose presence plus edge-cropping proxy; object aggregate proxy",
                                    "future_view_inputs_to_new_methods": False, "trained": trained,
                                    "handcrafted_tuned_weights": tuned_weights, "results": {}}
    for split_name, data, classes in [("seen_test", seen_data, SEEN_CLASSES), ("pseudo_unseen", pseudo_data, PSEUDO_CLASSES), ("true_unseen", unseen_data, UNSEEN_CLASSES)]:
        actions: dict[str, np.ndarray] = {}
        components = handcrafted_components(data)
        _, _, action_info = candidate_inputs(data["raw"], data["episodes"], data["features"])
        random_generator = np.random.default_rng(20260909 + len(split_name))
        actions["random"] = np.asarray(
            [random_generator.choice(np.flatnonzero(row)) for row in action_info["valid"]],
            dtype=np.int64,
        )
        actions["fixed"] = np.ones(len(data["episodes"]), dtype=np.int64)
        actions["handcrafted_fixed"] = handcrafted_actions(
            components, [1.0, 1.0, 1.0, 0.20], action_info["valid"]
        )
        actions["handcrafted_tuned"] = handcrafted_actions(
            components, tuned_weights, action_info["valid"]
        )
        actions["old_rl_legacy"] = load_old_actions(data, classes, old_checkpoint, device)
        # Random uses the exact expected accuracy in the primary metric; its
        # sampled action is retained for histogram and movement diagnostics.
        for name, runs in trained.items():
            run_actions = []
            for run in runs:
                saved = torch.load(run["path"], map_location=device, weights_only=False)
                model = StructuredScorer(hidden=int(saved.get("hidden", run.get("hidden", 128)))).to(device)
                model.load_state_dict(saved["model"])
                context, candidate, info = candidate_inputs(data["raw"], data["episodes"], data["features"])
                run_actions.append(action_from_model(model, context, candidate, info["valid"], device))
            # Report each seed separately and the seed mean under the variant.
            for index, action in enumerate(run_actions):
                actions[f"{name}_seed_{seeds[index]}"] = action
        all_reports["results"][split_name] = evaluate_actions(data, classes, actions, 1000 + len(split_name))
    # Replace the random action's histogram-only result with the exact random
    # expectation as the primary number; keep the sampled action for behavior.
    all_reports["method_notes"] = {
        "random": "top1 is exact mean over the three legal candidates; sampled action is only for histogram",
        "old_rl_legacy": "preserved v6 model using current-view VPOCLIP/HAR/logit state; reported separately from the new structured selector",
        "structured_rl": "supervised one-step utility scorer trained on class-agnostic structured reward; guide recommends this Phase-1 form before multi-step RL",
        "oracle": "not a policy input; computed from true-label fused margin as offline upper bound",
        "selection": "no true-unseen tuning; handcrafted weights selected on pseudo-unseen only; learned checkpoints selected by pseudo structured reward",
    }
    # Add oracle and exact baseline through the common evaluator without
    # allowing oracle to affect any model selection.
    for split_name, data, classes in [("seen_test", seen_data, SEEN_CLASSES), ("pseudo_unseen", pseudo_data, PSEUDO_CLASSES), ("true_unseen", unseen_data, UNSEEN_CLASSES)]:
        raw = data["raw"]
        episodes = data["episodes"]
        labels = np.asarray(raw["labels"])[episodes]
        logits = np.asarray(raw["logits"][episodes])
        fused = 0.5 * (logits[:, CURRENT_VIEW, None] + logits)
        utility = margin_scores(fused, labels[:, None], classes) - margin_scores(logits[:, CURRENT_VIEW], labels, classes)[:, None]
        _, _, info = candidate_inputs(raw, episodes, data["features"])
        utility[~info["valid"]] = -np.inf
        oracle = utility.argmax(-1)
        report = all_reports["results"][split_name]
        oracle_top1 = report["oracle_top1"]
        report["methods"]["oracle"] = {"top1": oracle_top1, "oracle_regret": 0.0,
                                         "action_histogram": np.bincount(oracle, minlength=NUM_VIEWS).astype(float).tolist(),
                                         "selection_is_upper_bound": True}
    (output / "summary.json").write_text(json.dumps(all_reports, indent=2), encoding="utf-8")
    print(json.dumps({"STRUCTURED_EXPERIMENT_COMPLETE": True, "output": str(output), "splits": split_stats,
                      "tuned_weights": tuned_weights}, indent=2), flush=True)


if __name__ == "__main__":
    main()
