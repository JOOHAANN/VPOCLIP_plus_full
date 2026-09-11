"""Sequential Explore-Then-Decide active OV-HAR experiment.

This is an independent successor to the one-step viewpoint selectors.  The
recognizer remains frozen and the experiment is run offline on *real cached
view logits*: view 0 is observed, a class-agnostic exploratory view is chosen,
its cached observation is revealed, and only then is STOP/CONTINUE predicted.

The implementation follows Sequential_Explore_Then_Decide_Active_OVHAR.md:

* compare three deterministic first moves;
* build STOP/CONTINUE labels from the best remaining real observation;
* compare Logistic, Random Forest, Gradient Boosting and MLP predictors;
* gate the remaining-view ranker using validation only;
* train a small sequential Q model only if both gates pass.

No future logits, future pose, object maps, or GT labels are model inputs.
Labels are used only to create training targets and to report offline metrics.
The old cache contains variable legal-view counts, so all decisions honor the
per-episode valid/reachable mask instead of assuming slots 1/2/3 are present.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import pickle
import random
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from torch import nn

from .richer_cache_temporal_crossmodal_voi import TinyGradientBoosting, TinyRandomForest
from .structured_sa_experiment import (
    CURRENT_VIEW,
    NUM_VIEWS,
    PSEUDO_CLASSES,
    SEEN_CLASSES,
    TRAIN_CLASSES,
    UNSEEN_CLASSES,
    load_raw,
    structured_view_features,
)
from .need_to_move_diagnostics import _stratified_group_folds


STOP_ACTION = 0
EPSILON_GAIN = 0.02
SEQUENTIAL_DQN_EPOCHS = 120


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.set_float32_matmul_precision("high")


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (Path,)):
        return str(value)
    return value


def wrap_angle(value: np.ndarray) -> np.ndarray:
    return (value + np.pi) % (2.0 * np.pi) - np.pi


def prepare_variable_data(raw_root: Path, split: str, classes: Sequence[int]) -> dict[str, Any]:
    """Load episodes with >=2 legal views, retaining 2/3-view anomalies."""

    raw = load_raw(raw_root / split)
    labels = np.asarray(raw["labels"], dtype=np.int64)
    valid = np.asarray(raw["valid"], dtype=bool)
    keep = np.isin(labels, np.asarray(list(classes), dtype=np.int64))
    keep &= valid[:, CURRENT_VIEW]
    keep &= valid.sum(axis=1) >= 2
    source_ids = np.flatnonzero(keep).astype(np.int64)
    selected: dict[str, Any] = dict(raw)
    for key in ("z", "logits", "pose", "object_map", "quality", "geometry",
                "cost", "reachable", "valid", "labels", "geometry_confidence"):
        if (key in raw and isinstance(raw[key], np.ndarray) and raw[key].ndim > 0
                and len(raw[key]) == len(labels)):
            selected[key] = np.asarray(raw[key][keep])
    selected["labels"] = labels[keep]
    pose = np.asarray(raw["pose"][keep], dtype=np.float32)
    quality = np.asarray(raw["quality"][keep], dtype=np.float32)
    objects = np.asarray(raw["object_map"][keep], dtype=np.float32)
    features = structured_view_features(pose, quality, objects)
    features["uncertainty"] = normalized_entropy(
        np.asarray(raw["logits"][keep], dtype=np.float32), temperature=2.0
    )[..., None]
    return {
        "raw": selected,
        "episodes": np.arange(len(source_ids), dtype=np.int64),
        "source_episode_ids": source_ids,
        "features": features,
        "classes": list(classes),
        "split": split,
    }


def normalized_entropy(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    values = logits / max(float(temperature), 1e-5)
    values = values - values.max(axis=-1, keepdims=True)
    p = np.exp(np.clip(values, -80.0, 80.0))
    p /= np.maximum(p.sum(axis=-1, keepdims=True), 1e-8)
    return (-(p * np.log(np.maximum(p, 1e-8))).sum(axis=-1)
            / math.log(float(logits.shape[-1]))).astype(np.float32)


def probabilities(logits: np.ndarray) -> np.ndarray:
    values = logits - logits.max(axis=-1, keepdims=True)
    p = np.exp(np.clip(values, -80.0, 80.0))
    return (p / np.maximum(p.sum(axis=-1, keepdims=True), 1e-8)).astype(np.float32)


def prediction_stats(logits: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = probabilities(logits)
    order = np.sort(p, axis=-1)[..., ::-1]
    h = normalized_entropy(logits)
    top1 = order[..., 0]
    margin = order[..., 0] - order[..., 1]
    return h.astype(np.float32), top1.astype(np.float32), margin.astype(np.float32)


def geometry_arrays(data: Mapping[str, Any]) -> dict[str, np.ndarray]:
    raw = data["raw"]
    geometry = np.asarray(raw["geometry"], dtype=np.float32)[..., :2]
    bearings = np.arctan2(geometry[..., 0], geometry[..., 1]).astype(np.float32)
    relative = wrap_angle(bearings - bearings[:, CURRENT_VIEW, None]).astype(np.float32)
    reachable = np.asarray(raw["reachable"], dtype=bool)
    valid = np.asarray(raw["valid"], dtype=bool)
    legal = valid & reachable[:, CURRENT_VIEW]
    legal[:, CURRENT_VIEW] = False
    cost = np.asarray(raw["cost"], dtype=np.float32)
    if cost.ndim == 3:
        move_cost = cost
    else:
        move_cost = np.zeros((len(valid), NUM_VIEWS, NUM_VIEWS), dtype=np.float32)
        move_cost[:, CURRENT_VIEW] = cost
    return {
        "bearings": bearings,
        "relative": relative,
        "valid": valid,
        "legal": legal,
        "cost": move_cost,
        "confidence": np.asarray(raw["geometry_confidence"], dtype=np.float32),
    }


def candidate_lists(data: Mapping[str, Any]) -> list[np.ndarray]:
    legal = geometry_arrays(data)["legal"]
    return [np.flatnonzero(row).astype(np.int64) for row in legal]


def ordered_choice(candidates: np.ndarray, score: np.ndarray) -> int:
    if len(candidates) == 0:
        return -1
    # Stable descending sort makes ties reproducible and independent of class.
    order = sorted((int(v) for v in candidates), key=lambda v: (-float(score[v]), v))
    return int(order[0])


def exploratory_actions(data: Mapping[str, Any], method: str) -> np.ndarray:
    info = geometry_arrays(data)
    relative = info["relative"]
    legal = info["legal"]
    cost = info["cost"]
    actions = np.full(len(legal), -1, dtype=np.int64)
    target = np.pi / 2.0
    for i in range(len(legal)):
        choices = np.flatnonzero(legal[i])
        angle = np.abs(relative[i])
        if method == "max_angular_diversity":
            score = angle
        elif method == "preferred_orthogonal":
            score = -np.abs(angle - target)
            score = score - 0.30 * (angle > np.deg2rad(150.0))
        elif method == "diversity_minus_cost":
            score = angle / np.pi - 0.30 * np.clip(cost[i, CURRENT_VIEW], 0.0, 1.0)
        else:
            raise ValueError(method)
        actions[i] = ordered_choice(choices, score)
    return actions


def next_diverse_actions(data: Mapping[str, Any], selected: np.ndarray,
                         last_view: np.ndarray) -> np.ndarray:
    """Class-agnostic next action: maximize separation from visited views."""

    info = geometry_arrays(data)
    n = len(selected)
    actions = np.full(n, -1, dtype=np.int64)
    for i in range(n):
        choices = np.flatnonzero(info["legal"][i] & ~selected[i])
        if len(choices) == 0:
            continue
        visited = np.flatnonzero(selected[i])
        scores = np.full(NUM_VIEWS, -np.inf, dtype=np.float32)
        for candidate in choices:
            separation = np.abs(wrap_angle(
                info["bearings"][i, candidate] - info["bearings"][i, visited]
            ))
            score = float(np.min(separation) / np.pi)
            score -= 0.30 * float(np.clip(info["cost"][i, last_view[i], candidate], 0.0, 1.0))
            scores[candidate] = score
        actions[i] = ordered_choice(choices, scores)
    return actions


def transition_cost(info: Mapping[str, np.ndarray], row: int, path: Sequence[int]) -> float:
    cost = info["cost"]
    return float(sum(float(cost[row, int(a), int(b)]) for a, b in zip(path[:-1], path[1:])))


def target_margin(logits: np.ndarray, labels: np.ndarray, classes: Sequence[int]) -> np.ndarray:
    bank = np.asarray(sorted(set(int(x) for x in classes)), dtype=np.int64)
    local = np.searchsorted(bank, labels)
    if np.any(local >= len(bank)) or np.any(bank[np.minimum(local, len(bank) - 1)] != labels):
        raise ValueError("label outside class bank")
    score = logits[..., bank]
    index = np.broadcast_to(local.reshape((-1,) + (1,) * (score.ndim - 2) + (1,)),
                            score.shape[:-1] + (1,))
    true = np.take_along_axis(score, index, axis=-1)[..., 0]
    rival = score.copy()
    np.put_along_axis(rival, index, -np.inf, axis=-1)
    return true - rival.max(axis=-1)


def classify(logits: np.ndarray, labels: np.ndarray, classes: Sequence[int]) -> np.ndarray:
    bank = np.asarray(sorted(set(int(x) for x in classes)), dtype=np.int64)
    local = np.searchsorted(bank, labels)
    return logits[..., bank].argmax(axis=-1) == local


def fused_for_path(logits: np.ndarray, path: Sequence[int]) -> np.ndarray:
    return np.asarray(logits[:, list(path)], dtype=np.float32).mean(axis=1)


def path_correct(data: Mapping[str, Any], paths: Sequence[Sequence[int]],
                 classes: Sequence[int]) -> np.ndarray:
    logits = np.asarray(data["raw"]["logits"], dtype=np.float32)
    labels = np.asarray(data["raw"]["labels"], dtype=np.int64)
    out = np.zeros(len(paths), dtype=bool)
    for i, path in enumerate(paths):
        out[i] = bool(classify(fused_for_path(logits[i:i + 1], path), labels[i:i + 1], classes)[0])
    return out


def target_for_path(data: Mapping[str, Any], paths: Sequence[Sequence[int]],
                    classes: Sequence[int]) -> np.ndarray:
    logits = np.asarray(data["raw"]["logits"], dtype=np.float32)
    labels = np.asarray(data["raw"]["labels"], dtype=np.int64)
    result = np.zeros(len(paths), dtype=np.float32)
    for i, path in enumerate(paths):
        result[i] = target_margin(fused_for_path(logits[i:i + 1], path), labels[i:i + 1], classes)[0]
    return result


def select_fixed_actions(data: Mapping[str, Any]) -> np.ndarray:
    legal = geometry_arrays(data)["legal"]
    result = np.full(len(legal), -1, dtype=np.int64)
    for i, row in enumerate(legal):
        choices = np.flatnonzero(row)
        result[i] = int(1 if row[1] else choices[0])
    return result


def make_paths(data: Mapping[str, Any], action: np.ndarray) -> list[list[int]]:
    info = geometry_arrays(data)
    paths: list[list[int]] = []
    for i, a in enumerate(action):
        if int(a) < 0:
            paths.append([CURRENT_VIEW])
        elif info["legal"][i, int(a)]:
            paths.append([CURRENT_VIEW, int(a)])
        else:
            paths.append([CURRENT_VIEW, int(np.flatnonzero(info["legal"][i])[0])])
    return paths


def evaluate_paths(data: Mapping[str, Any], classes: Sequence[int],
                   paths: Sequence[Sequence[int]], method: str) -> dict[str, Any]:
    labels = np.asarray(data["raw"]["labels"], dtype=np.int64)
    logits = np.asarray(data["raw"]["logits"], dtype=np.float32)
    info = geometry_arrays(data)
    correct = np.asarray([
        bool(classify(fused_for_path(logits[i:i + 1], p), labels[i:i + 1], classes)[0])
        for i, p in enumerate(paths)
    ], dtype=np.float32)
    costs = np.asarray([transition_cost(info, i, p) for i, p in enumerate(paths)], dtype=np.float32)
    return {
        "method": method,
        "episodes": int(len(paths)),
        "top1": float(correct.mean()),
        "average_views": float(np.mean([len(p) for p in paths])),
        "average_movement_cost": float(costs.mean()),
        "correct_count": int(correct.sum()),
        "action_histogram": [int(sum(len(p) > 1 and p[-1] == v for p in paths))
                             for v in range(NUM_VIEWS)],
    }


def expected_paths(data: Mapping[str, Any], classes: Sequence[int],
                   path_builder: Any, method: str) -> dict[str, Any]:
    """Evaluate a randomized policy by exact per-episode expectation."""

    logits = np.asarray(data["raw"]["logits"], dtype=np.float32)
    labels = np.asarray(data["raw"]["labels"], dtype=np.int64)
    info = geometry_arrays(data)
    values: list[float] = []
    costs: list[float] = []
    view_counts: list[float] = []
    for i in range(len(labels)):
        choices = path_builder(info, i)
        rows_correct = []
        rows_cost = []
        rows_views = []
        for path in choices:
            rows_correct.append(float(classify(
                fused_for_path(logits[i:i + 1], path), labels[i:i + 1], classes
            )[0]))
            rows_cost.append(transition_cost(info, i, path))
            rows_views.append(len(path))
        values.append(float(np.mean(rows_correct)))
        costs.append(float(np.mean(rows_cost)))
        view_counts.append(float(np.mean(rows_views)))
    return {
        "method": method,
        "episodes": int(len(labels)),
        "top1": float(np.mean(values)),
        "average_views": float(np.mean(view_counts)),
        "average_movement_cost": float(np.mean(costs)),
        "exact_random_expectation": True,
    }


def pair_paths_for_action(data: Mapping[str, Any], first: np.ndarray) -> list[list[int]]:
    info = geometry_arrays(data)
    result = []
    for i, a in enumerate(first):
        result.append([0, int(a)])
    return result


def three_view_actions(data: Mapping[str, Any], first: np.ndarray,
                       method: str) -> np.ndarray:
    info = geometry_arrays(data)
    selected = np.zeros((len(first), NUM_VIEWS), dtype=bool)
    selected[:, 0] = True
    selected[np.arange(len(first)), first] = True
    last = first.copy()
    actions = np.full(len(first), -1, dtype=np.int64)
    for i in range(len(first)):
        choices = np.flatnonzero(info["legal"][i] & ~selected[i])
        if len(choices) == 0:
            continue
        if method == "max_angular_diversity":
            score = np.full(NUM_VIEWS, -np.inf, dtype=np.float32)
            for c in choices:
                separation = np.abs(wrap_angle(info["bearings"][i, c] - info["bearings"][i, [0, first[i]]]))
                score[c] = float(np.min(separation))
        elif method == "preferred_orthogonal":
            angle = np.abs(wrap_angle(info["bearings"][i, choices] - info["bearings"][i, last[i]]))
            score = np.full(NUM_VIEWS, -np.inf, dtype=np.float32)
            score[choices] = -np.abs(angle - np.pi / 2.0)
        elif method == "diversity_minus_cost":
            score = np.full(NUM_VIEWS, -np.inf, dtype=np.float32)
            for c in choices:
                separation = np.abs(wrap_angle(info["bearings"][i, c] - info["bearings"][i, [0, first[i]]]))
                score[c] = float(np.min(separation) / np.pi) - 0.30 * float(
                    np.clip(info["cost"][i, last[i], c], 0.0, 1.0)
                )
        else:
            raise ValueError(method)
        actions[i] = ordered_choice(choices, score)
    return actions


def oracle_action(data: Mapping[str, Any], classes: Sequence[int],
                  first: np.ndarray, k: int) -> list[list[int]]:
    info = geometry_arrays(data)
    logits = np.asarray(data["raw"]["logits"], dtype=np.float32)
    labels = np.asarray(data["raw"]["labels"], dtype=np.int64)
    paths: list[list[int]] = []
    for i in range(len(labels)):
        legal = np.flatnonzero(info["legal"][i])
        if k == 2:
            candidates = [[0, int(a)] for a in legal]
        else:
            # Oracle-3 is the best *pair* of additional legal views, not a
            # fixed-first-view oracle.  The supplied ``first`` is retained as
            # a deterministic tie-breaker only.
            candidates = [[0, int(a), int(b)] for a, b in itertools.combinations(legal, 2)]
            if not candidates:
                candidates = [[0, int(first[i])]]
        utilities = [target_margin(fused_for_path(logits[i:i + 1], p), labels[i:i + 1], classes)[0]
                     for p in candidates]
        paths.append(candidates[int(np.argmax(utilities))])
    return paths


def baseline_report(data: Mapping[str, Any], classes: Sequence[int],
                    validation_first: str | None = None) -> dict[str, Any]:
    methods = {"view0_only": [[0] for _ in range(len(data["episodes"]))]}
    fixed = select_fixed_actions(data)
    methods["fixed_2"] = pair_paths_for_action(data, fixed)
    first_actions: dict[str, np.ndarray] = {}
    for name in ("max_angular_diversity", "preferred_orthogonal", "diversity_minus_cost"):
        first_actions[name] = exploratory_actions(data, name)
        methods[f"{name}_2"] = pair_paths_for_action(data, first_actions[name])
        second = three_view_actions(data, first_actions[name], name)
        methods[f"{name}_3"] = [
            ([0, int(first_actions[name][i]), int(second[i])]
             if int(second[i]) >= 0 else [0, int(first_actions[name][i])])
            for i in range(len(second))
        ]
    methods["oracle_2"] = oracle_action(data, classes, fixed, 2)
    methods["oracle_3"] = oracle_action(data, classes, fixed, 3)
    all_paths = []
    info = geometry_arrays(data)
    for i in range(len(data["episodes"])):
        all_paths.append([0] + np.flatnonzero(info["legal"][i]).astype(int).tolist())
    methods["all_valid_views"] = all_paths
    reports = {name: evaluate_paths(data, classes, paths, name)
               for name, paths in methods.items()}
    random2 = expected_paths(
        data, classes,
        lambda inf, i: [[0, int(a)] for a in np.flatnonzero(inf["legal"][i])],
        "random_2",
    )
    random3 = expected_paths(
        data, classes,
        lambda inf, i: [
            [0, int(a), int(b)]
            for a, b in itertools.combinations(np.flatnonzero(inf["legal"][i]), 2)
        ] or [[0, int(np.flatnonzero(inf["legal"][i])[0])]],
        "random_3",
    )
    reports["random_2"] = random2
    reports["random_3"] = random3
    return {"methods": reports, "first_actions": first_actions,
            "validation_selected_first_strategy": validation_first}


def js_divergence(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    m = 0.5 * (p + q)
    return (0.5 * (p * np.log(np.maximum(p, 1e-8) / np.maximum(m, 1e-8))).sum(-1)
            + 0.5 * (q * np.log(np.maximum(q, 1e-8) / np.maximum(m, 1e-8))).sum(-1)).astype(np.float32)


def topk_overlap(p: np.ndarray, q: np.ndarray, k: int = 5) -> np.ndarray:
    a = np.argpartition(-p, kth=k - 1, axis=-1)[..., :k]
    b = np.argpartition(-q, kth=k - 1, axis=-1)[..., :k]
    return np.asarray([(len(set(x.tolist()) & set(y.tolist())) / float(k)) for x, y in zip(a, b)], dtype=np.float32)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    return (1.0 - (a * b).sum(-1) / np.maximum(den, 1e-8)).astype(np.float32)


def _append(parts: list[np.ndarray], names: list[str], value: np.ndarray,
           feature_names: Sequence[str]) -> None:
    parts.append(np.asarray(value, dtype=np.float32).reshape(len(value), -1))
    names.extend(str(x) for x in feature_names)


def state_features(data: Mapping[str, Any], selected: np.ndarray,
                   last_view: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Build causal state features after the selected real views are known."""

    raw = data["raw"]
    logits = np.asarray(raw["logits"], dtype=np.float32)
    info = geometry_arrays(data)
    n = len(logits)
    count = selected.sum(axis=1).astype(np.float32)
    aggregate = (logits * selected[..., None]).sum(axis=1) / np.maximum(count[:, None], 1.0)
    first = logits[:, 0]
    latest = logits[np.arange(n), last_view]
    p_first, p_agg, p_latest = probabilities(first), probabilities(aggregate), probabilities(latest)
    h0, c0, m0 = prediction_stats(first)
    ha, ca, ma = prediction_stats(aggregate)
    hl, cl, ml = prediction_stats(latest)
    parts: list[np.ndarray] = []
    names: list[str] = []
    for value, prefix in ((h0, "view0_entropy"), (c0, "view0_top1_prob"), (m0, "view0_margin"),
                          (hl, "latest_entropy"), (cl, "latest_top1_prob"), (ml, "latest_margin"),
                          (ha, "fused_entropy"), (ca, "fused_top1_prob"), (ma, "fused_margin")):
        _append(parts, names, value, [prefix])
    disagreement = np.stack([
        js_divergence(p_first, p_latest),
        js_divergence(p_first, p_agg),
        cosine_distance(first, latest),
        (p_first.argmax(-1) != p_latest.argmax(-1)).astype(np.float32),
        1.0 - topk_overlap(p_first, p_latest),
    ], axis=1)
    _append(parts, names, disagreement,
           ["js_first_latest", "js_first_fused", "logit_cosine_first_latest",
            "top1_mismatch_first_latest", "top5_disagreement_first_latest"])
    _append(parts, names, np.stack([ha - h0, ca - c0, ma - m0], axis=1),
           ["entropy_gain_fused_minus_view0", "confidence_gain_fused_minus_view0",
            "margin_gain_fused_minus_view0"])

    features = data["features"]
    for key, width in (("visibility", 7), ("ambiguity", 5), ("orientation", 6), ("quality", 4)):
        value = np.asarray(features[key], dtype=np.float32)
        pooled = (value * selected[..., None]).sum(axis=1) / np.maximum(count[:, None], 1.0)
        delta = pooled - value[:, 0]
        _append(parts, names, delta, [f"delta_{key}_{i}" for i in range(width)])

    bearings = info["bearings"]
    relative_latest = wrap_angle(bearings[np.arange(n), last_view] - bearings[:, 0])
    coverage = np.zeros(n, dtype=np.float32)
    for i in range(n):
        visited = np.flatnonzero(selected[i])
        if len(visited) > 1:
            coverage[i] = max(float(np.abs(wrap_angle(bearings[i, a] - bearings[i, b])))
                              for a, b in itertools.combinations(visited, 2)) / np.pi
    remaining_angle: list[list[float]] = []
    remaining_cost: list[list[float]] = []
    remaining_count = np.zeros(n, dtype=np.float32)
    for i in range(n):
        choices = np.flatnonzero(info["legal"][i] & ~selected[i])
        remaining_count[i] = len(choices)
        remaining_angle.append([float(np.abs(wrap_angle(bearings[i, c] - bearings[i, last_view[i]])) / np.pi) for c in choices])
        remaining_cost.append([float(np.clip(info["cost"][i, last_view[i], c], 0.0, 1.0)) for c in choices])
    def list_stats(rows: list[list[float]]) -> np.ndarray:
        return np.asarray([[np.mean(v) if v else 0.0, np.min(v) if v else 0.0,
                            np.max(v) if v else 0.0] for v in rows], dtype=np.float32)
    _append(parts, names, np.stack([
        relative_latest / np.pi, np.abs(relative_latest) / np.pi, coverage,
        count, remaining_count,
    ], axis=1), ["latest_relative_angle", "latest_abs_angle", "angular_coverage",
                 "views_seen", "remaining_view_count"])
    _append(parts, names, list_stats(remaining_angle),
           ["remaining_angle_mean", "remaining_angle_min", "remaining_angle_max"])
    _append(parts, names, list_stats(remaining_cost),
           ["remaining_cost_mean", "remaining_cost_min", "remaining_cost_max"])
    last_cost = np.asarray([transition_cost(info, i, [0, int(last_view[i])]) for i in range(n)], dtype=np.float32)
    _append(parts, names, last_cost, ["observed_move_cost"])
    return np.concatenate(parts, axis=1).astype(np.float32), names


def first_state(data: Mapping[str, Any], first: np.ndarray) -> tuple[np.ndarray, list[str], np.ndarray]:
    selected = np.zeros((len(first), NUM_VIEWS), dtype=bool)
    selected[:, 0] = True
    selected[np.arange(len(first)), first] = True
    return (*state_features(data, selected, first), selected)


def sequential_targets(data: Mapping[str, Any], classes: Sequence[int], first: np.ndarray) -> dict[str, Any]:
    info = geometry_arrays(data)
    logits = np.asarray(data["raw"]["logits"], dtype=np.float32)
    labels = np.asarray(data["raw"]["labels"], dtype=np.int64)
    n = len(logits)
    selected = np.zeros((n, NUM_VIEWS), dtype=bool)
    selected[:, 0] = True
    selected[np.arange(n), first] = True
    current = np.asarray([target_margin(fused_for_path(logits[i:i + 1], [0, int(first[i])]), labels[i:i + 1], classes)[0]
                          for i in range(n)], dtype=np.float32)
    gains = np.full((n, NUM_VIEWS), np.nan, dtype=np.float32)
    for i in range(n):
        for a in np.flatnonzero(info["legal"][i] & ~selected[i]):
            final = target_margin(fused_for_path(logits[i:i + 1], [0, int(first[i]), int(a)]), labels[i:i + 1], classes)[0]
            gains[i, a] = float(final - current[i])
    finite = np.isfinite(gains)
    best = np.where(finite, gains, -np.inf).max(axis=1)
    has_remaining = finite.any(axis=1)
    best[~has_remaining] = 0.0
    y = (best > EPSILON_GAIN).astype(np.int64)
    return {"continue": y, "max_remaining_gain": best.astype(np.float32),
            "remaining_gain": gains, "current_margin": current,
            "has_remaining": has_remaining}


def auc_metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y, dtype=np.int64)
    p = np.asarray(probability, dtype=np.float64)
    finite = np.isfinite(p)
    y, p = y[finite], p[finite]
    pos = int(y.sum())
    neg = int(len(y) - pos)
    result = {"n": int(len(y)), "positive": pos, "negative": neg,
              "positive_prevalence": float(y.mean()) if len(y) else 0.0}
    if pos and neg:
        order = np.argsort(p, kind="mergesort")
        ys = y[order]
        ps = p[order]
        ranks = np.empty(len(p), dtype=np.float64)
        # Average ranks handle ties without granting a false advantage.  The
        # ranks must be assigned in ascending-score order because the
        # Mann-Whitney formula below interprets rank=1 as the lowest score.
        start = 0
        while start < len(ps):
            end = start + 1
            while end < len(ps) and ps[end] == ps[start]:
                end += 1
            ranks[order[start:end]] = 0.5 * (start + 1 + end)
            start = end
        result["auroc"] = float((ranks[y == 1].sum() - pos * (pos + 1) / 2.0) / (pos * neg))
        pr_order = np.argsort(-p, kind="mergesort")
        ys_desc = y[pr_order]
        cumulative = np.cumsum(ys_desc)
        result["auprc"] = float(
            (cumulative[ys_desc == 1] / (np.flatnonzero(ys_desc == 1) + 1)).mean()
        )
    else:
        result["auroc"] = None
        result["auprc"] = None
    result["probability_mean"] = float(p.mean()) if len(p) else 0.0
    result["probability_std"] = float(p.std()) if len(p) else 0.0
    return result


def threshold_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, Any]:
    y = np.asarray(y, dtype=np.int64)
    pred = np.asarray(p >= threshold, dtype=np.int64)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {"threshold": float(threshold), "accuracy": float((pred == y).mean()),
            "precision_continue": float(precision), "recall_continue": float(recall),
            "f1_continue": float(f1), "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def best_threshold(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    candidates = np.unique(np.concatenate([np.linspace(0.05, 0.95, 37), np.asarray(p)]))
    rows = [threshold_metrics(y, p, float(t)) for t in candidates]
    return max(rows, key=lambda x: (x["f1_continue"], x["recall_continue"], -x["threshold"]))


class TorchBinaryModel:
    def __init__(self, width: int, kind: str) -> None:
        if kind == "logistic":
            self.net = nn.Linear(width, 1)
        elif kind == "mlp":
            self.net = nn.Sequential(nn.Linear(width, 64), nn.LayerNorm(64), nn.GELU(),
                                     nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 1))
        else:
            raise ValueError(kind)
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        self.kind = kind

    def fit(self, x: np.ndarray, y: np.ndarray, device: torch.device, seed: int,
            epochs: int = 220) -> "TorchBinaryModel":
        seed_all(seed)
        self.mean = x.mean(0).astype(np.float32)
        self.std = np.maximum(x.std(0), 1e-5).astype(np.float32)
        xs = torch.from_numpy(((x - self.mean) / self.std).astype(np.float32)).to(device)
        ys = torch.from_numpy(y.astype(np.float32)).to(device)
        self.net = self.net.to(device)
        positive = max(float(y.sum()), 1.0)
        negative = max(float(len(y) - y.sum()), 1.0)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negative / positive, device=device))
        optimizer = torch.optim.AdamW(self.net.parameters(), lr=2e-3, weight_decay=1e-2)
        for _ in range(epochs):
            logits = self.net(xs).squeeze(-1)
            loss = loss_fn(logits, ys)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        return self

    def predict(self, x: np.ndarray, device: torch.device) -> np.ndarray:
        assert self.mean is not None and self.std is not None
        self.net.eval()
        with torch.inference_mode():
            value = self.net(torch.from_numpy(((x - self.mean) / self.std).astype(np.float32)).to(device)).squeeze(-1)
        return torch.sigmoid(value).cpu().numpy().astype(np.float32)


def fit_classifier(name: str, x: np.ndarray, y: np.ndarray,
                   device: torch.device, seed: int) -> Any:
    if name in ("logistic", "mlp"):
        return TorchBinaryModel(x.shape[1], name).fit(x, y, device, seed)
    if name == "random_forest":
        return TinyRandomForest(seed=seed, n_estimators=32, max_depth=7, min_leaf=20).fit(x, y.astype(np.float32))
    if name == "gradient_boosting":
        return TinyGradientBoosting(seed=seed, n_estimators=48, depth=2, min_leaf=20).fit(x, y.astype(np.float32))
    raise ValueError(name)


def predict_classifier(name: str, model: Any, x: np.ndarray,
                       device: torch.device) -> np.ndarray:
    if name in ("logistic", "mlp"):
        return model.predict(x, device)
    return np.clip(np.asarray(model.predict(x), dtype=np.float32), 0.0, 1.0)


def bootstrap_ci(values: np.ndarray, seed: int, repeats: int = 1500) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2:
        return [float(values.mean()) if len(values) else 0.0] * 2
    rng = np.random.default_rng(seed)
    means = np.asarray([values[rng.integers(0, len(values), len(values))].mean() for _ in range(repeats)])
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def fit_stop_models(train_x: np.ndarray, train_y: np.ndarray, train_groups: np.ndarray,
                    device: torch.device, seed: int) -> tuple[dict[str, Any], dict[str, Any], list[Any]]:
    model_names = ("logistic", "random_forest", "gradient_boosting", "mlp")
    folds = _stratified_group_folds(train_y, train_groups, n_splits=5, seed=seed)
    oof = {name: np.full(len(train_y), np.nan, dtype=np.float32) for name in model_names}
    fold_records = []
    for fold, (tr, va) in enumerate(folds):
        for j, name in enumerate(model_names):
            model = fit_classifier(name, train_x[tr], train_y[tr], device, seed + fold * 100 + j)
            oof[name][va] = predict_classifier(name, model, train_x[va], device)
        fold_records.append({"fold": fold, "train_episodes": int(len(tr)),
                             "validation_episodes": int(len(va)),
                             "validation_subjects": sorted(set(train_groups[va].tolist())),
                             "validation_positive": int(train_y[va].sum())})
    final = {name: fit_classifier(name, train_x, train_y, device, seed + 1000 + j)
             for j, name in enumerate(model_names)}
    reports = {"oof_subject_holdout": {name: auc_metrics(train_y, p) for name, p in oof.items()},
               "folds": fold_records}
    return final, reports, fold_records


def evaluate_stop_models(models: Mapping[str, Any], names: Sequence[str],
                         x: np.ndarray, y: np.ndarray, device: torch.device) -> dict[str, Any]:
    return {name: auc_metrics(y, predict_classifier(name, models[name], x, device)) for name in names}


def candidate_features(data: Mapping[str, Any], selected: np.ndarray,
                       last_view: np.ndarray) -> tuple[np.ndarray, list[str]]:
    info = geometry_arrays(data)
    n = len(selected)
    bearings = info["bearings"]
    context = np.zeros((n, NUM_VIEWS, 10), dtype=np.float32)
    names = ["candidate_rel_view0", "candidate_rel_last", "candidate_abs_view0",
             "candidate_abs_last", "candidate_seen_separation", "candidate_cost",
             "candidate_geometry_confidence", "candidate_is_legal", "candidate_is_unvisited",
             "candidate_coverage_after"]
    for i in range(n):
        visited = np.flatnonzero(selected[i])
        for c in range(1, NUM_VIEWS):
            rel0 = wrap_angle(bearings[i, c] - bearings[i, 0])
            rell = wrap_angle(bearings[i, c] - bearings[i, last_view[i]])
            separation = np.abs(wrap_angle(bearings[i, c] - bearings[i, visited]))
            current_coverage = 0.0
            if len(visited) > 1:
                current_coverage = max(float(np.abs(wrap_angle(bearings[i, a] - bearings[i, b])))
                                       for a, b in itertools.combinations(visited, 2))
            after = max(current_coverage, float(separation.max()) if len(separation) else 0.0) / np.pi
            context[i, c] = [float(rel0 / np.pi), float(rell / np.pi), float(abs(rel0) / np.pi),
                             float(abs(rell) / np.pi), float((separation.min() if len(separation) else 0.0) / np.pi),
                             float(np.clip(info["cost"][i, last_view[i], c], 0.0, 1.0)),
                             float(info["confidence"][i, c]), float(info["legal"][i, c]),
                             float(not selected[i, c]), after]
    return context, names


class CandidateRanker(nn.Module):
    def __init__(self, context_dim: int, candidate_dim: int = 10) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(context_dim + candidate_dim, 128), nn.LayerNorm(128),
                                 nn.GELU(), nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1))
        self.context_dim = context_dim
        self.candidate_dim = candidate_dim

    def forward(self, context: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
        n, v, _ = candidate.shape
        c = context[:, None, :].expand(n, v, context.shape[-1])
        return self.net(torch.cat([c, candidate], dim=-1)).squeeze(-1)


def train_candidate_ranker(data: Mapping[str, Any], classes: Sequence[int], first: np.ndarray,
                           device: torch.device, seed: int, output: Path) -> dict[str, Any]:
    context, _, selected = first_state(data, first)
    last = first.copy()
    cand, cand_names = candidate_features(data, selected, last)
    targets = sequential_targets(data, classes, first)["remaining_gain"]
    info = geometry_arrays(data)
    mask = np.isfinite(targets) & info["legal"] & ~selected
    keep = mask.sum(1) >= 2
    model = CandidateRanker(context.shape[1]).to(device)
    tx = torch.from_numpy(context[keep]).to(device)
    tc = torch.from_numpy(cand[keep]).to(device)
    ty = torch.from_numpy(np.nan_to_num(targets[keep], nan=0.0)).to(device)
    tm = torch.from_numpy(mask[keep]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)
    for _ in range(180):
        prediction = model(tx, tc)
        mse = ((prediction - ty) ** 2)[tm].mean()
        pair_losses = []
        for a, b in itertools.combinations(range(1, NUM_VIEWS), 2):
            use = tm[:, a] & tm[:, b] & ((ty[:, a] - ty[:, b]).abs() > EPSILON_GAIN)
            if use.any():
                delta = ty[use, a] - ty[use, b]
                pair_losses.append(F.relu(0.10 - delta.sign() * (prediction[use, a] - prediction[use, b])))
        pair = torch.cat(pair_losses).mean() if pair_losses else mse.new_zeros(())
        loss = mse + 0.5 * pair
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    checkpoint = output / "remaining_candidate_ranker.pt"
    torch.save({"model": model.state_dict(), "context_dim": context.shape[1],
                "candidate_dim": cand.shape[2], "future_inputs_to_model": False,
                "first_strategy": "selected_validation_strategy"}, checkpoint)
    return {"model": model, "context": context, "candidate": cand, "mask": mask,
            "targets": targets, "candidate_names": cand_names, "checkpoint": checkpoint,
            "train_episodes": int(keep.sum())}


def ranker_actions(ranker: Mapping[str, Any], data: Mapping[str, Any], first: np.ndarray,
                   device: torch.device) -> np.ndarray:
    context, _, selected = first_state(data, first)
    cand, _ = candidate_features(data, selected, first)
    info = geometry_arrays(data)
    with torch.inference_mode():
        scores = ranker["model"](torch.from_numpy(context).to(device), torch.from_numpy(cand).to(device))
    scores = scores.cpu().numpy()
    legal = info["legal"] & ~selected
    scores[~legal] = -np.inf
    return scores.argmax(-1).astype(np.int64)


def ranker_report(ranker: Mapping[str, Any], data: Mapping[str, Any], classes: Sequence[int],
                  first: np.ndarray, device: torch.device, seed: int) -> dict[str, Any]:
    action = ranker_actions(ranker, data, first, device)
    info = geometry_arrays(data)
    logits = np.asarray(data["raw"]["logits"], dtype=np.float32)
    labels = np.asarray(data["raw"]["labels"], dtype=np.int64)
    selected = np.zeros((len(first), NUM_VIEWS), dtype=bool)
    selected[:, 0] = True
    selected[np.arange(len(first)), first] = True
    rank_correct, random_correct, rank_gain, random_gain = [], [], [], []
    for i in range(len(first)):
        choices = np.flatnonzero(info["legal"][i] & ~selected[i])
        if len(choices) == 0:
            continue
        candidates = [[0, int(first[i]), int(c)] for c in choices]
        cor = np.asarray([float(classify(fused_for_path(logits[i:i + 1], p), labels[i:i + 1], classes)[0]) for p in candidates])
        util = np.asarray([target_margin(fused_for_path(logits[i:i + 1], p), labels[i:i + 1], classes)[0] for p in candidates])
        rank_idx = int(np.flatnonzero(choices == action[i])[0]) if action[i] in choices else 0
        rank_correct.append(cor[rank_idx]); random_correct.append(float(cor.mean()))
        rank_gain.append(util[rank_idx]); random_gain.append(float(util.mean()))
    delta = np.asarray(rank_correct, dtype=np.float32) - np.asarray(random_correct, dtype=np.float32)
    return {"episodes_with_remaining": int(len(delta)), "ranker_top1": float(np.mean(rank_correct)) if len(delta) else 0.0,
            "random_remaining_expected_top1": float(np.mean(random_correct)) if len(delta) else 0.0,
            "delta_vs_random": float(delta.mean()) if len(delta) else 0.0,
            "paired_ci95": bootstrap_ci(delta, seed),
            "ranker_utility": float(np.mean(rank_gain)) if rank_gain else 0.0,
            "random_utility": float(np.mean(random_gain)) if random_gain else 0.0,
            "ranker_action_histogram": np.bincount(action[action >= 0], minlength=NUM_VIEWS).tolist(),
            "gate_b_passed": bool(len(delta) > 0 and delta.mean() > 0.01 and bootstrap_ci(delta, seed)[0] > 0.0)}


def adaptive_paths(data: Mapping[str, Any], first: np.ndarray, probability: np.ndarray,
                   threshold: float, ranker: Mapping[str, Any] | None,
                   device: torch.device) -> tuple[list[list[int]], np.ndarray]:
    stop = probability < threshold
    selected = np.zeros((len(first), NUM_VIEWS), dtype=bool)
    selected[:, 0] = True
    selected[np.arange(len(first)), first] = True
    next_action = ranker_actions(ranker, data, first, device) if ranker is not None else next_diverse_actions(data, selected, first)
    paths = []
    for i in range(len(first)):
        if stop[i] or next_action[i] < 0:
            paths.append([0, int(first[i])])
        else:
            paths.append([0, int(first[i]), int(next_action[i])])
    return paths, stop


class SequentialQNet(nn.Module):
    def __init__(self, state_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, 128), nn.LayerNorm(128), nn.GELU(),
                                 nn.Linear(128, 64), nn.GELU(), nn.Linear(64, NUM_VIEWS + 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def terminal_reward(data: Mapping[str, Any], classes: Sequence[int], paths: Sequence[Sequence[int]]) -> np.ndarray:
    correct = path_correct(data, paths, classes).astype(np.float32)
    margin = target_for_path(data, paths, classes)
    return correct + 0.05 * margin


def train_sequential_dqn(data: Mapping[str, Any], classes: Sequence[int], first: np.ndarray,
                         device: torch.device, seed: int, output: Path) -> dict[str, Any]:
    """Small cached-transition Q learner used only after Gates A and B pass."""

    seed_all(seed)
    n = len(first)
    context1, names, selected1 = first_state(data, first)
    # Build every legal stage-2 state from the actual first observation.
    info = geometry_arrays(data)
    rows2: list[tuple[int, int]] = []
    context2: list[np.ndarray] = []
    selected2: list[np.ndarray] = []
    last2: list[int] = []
    for i in range(n):
        for b in np.flatnonzero(info["legal"][i] & ~selected1[i]):
            mask = selected1[i].copy(); mask[b] = True
            x, _ = state_features(data_subset_row(data, i), mask[None], np.asarray([b]))
            context2.append(x[0]); selected2.append(mask); last2.append(int(b)); rows2.append((i, int(b)))
    if not context2:
        raise RuntimeError("no stage-2 transitions available")
    state2 = np.asarray(context2, dtype=np.float32)
    model = SequentialQNet(context1.shape[1]).to(device)
    target = SequentialQNet(context1.shape[1]).to(device)
    target.load_state_dict(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    x1 = torch.from_numpy(context1).to(device)
    x2 = torch.from_numpy(state2).to(device)
    labels = np.asarray(data["raw"]["labels"], dtype=np.int64)
    logits = np.asarray(data["raw"]["logits"], dtype=np.float32)
    # Stage-2 supervised Bellman targets: STOP or one final remaining move.
    q2_target = np.zeros((len(rows2), NUM_VIEWS + 1), dtype=np.float32)
    q2_mask = np.zeros_like(q2_target, dtype=bool)
    for j, (i, b) in enumerate(rows2):
        p2 = [0, int(first[i]), b]
        q2_target[j, STOP_ACTION] = terminal_reward(data_subset_rows(data, [i]), classes, [p2])[0]
        q2_mask[j, STOP_ACTION] = True
        for c in np.flatnonzero(info["legal"][i] & ~selected2[j]):
            p3 = [0, int(first[i]), b, int(c)]
            q2_target[j, int(c) + 1] = terminal_reward(data_subset_rows(data, [i]), classes, [p3])[0]
            q2_mask[j, int(c) + 1] = True
    q2_t = torch.from_numpy(q2_target).to(device)
    q2_m = torch.from_numpy(q2_mask).to(device)
    # Stage-1 targets bootstrap through the stage-2 Q values.
    row2_by_source: dict[int, list[int]] = {}
    row2_by_source_action: dict[tuple[int, int], int] = {}
    for j, (i, _) in enumerate(rows2):
        row2_by_source.setdefault(i, []).append(j)
        row2_by_source_action[(i, rows2[j][1])] = j
    for epoch in range(SEQUENTIAL_DQN_EPOCHS):
        model.train(); target.eval()
        with torch.no_grad():
            q2_next = target(x2)
            q2_next = q2_next.masked_fill(~q2_m, -torch.inf)
            best2 = q2_next.max(-1).values.cpu().numpy()
        q1_target = np.zeros((n, NUM_VIEWS + 1), dtype=np.float32)
        q1_mask = np.zeros_like(q1_target, dtype=bool)
        for i in range(n):
            p1 = [0, int(first[i])]
            q1_target[i, STOP_ACTION] = terminal_reward(data_subset_rows(data, [i]), classes, [p1])[0]
            q1_mask[i, STOP_ACTION] = True
            for b in np.flatnonzero(info["legal"][i] & ~selected1[i]):
                q1_target[i, int(b) + 1] = -0.02 * float(info["cost"][i, first[i], b]) + best2[row2_by_source_action[(i, int(b))]]
                q1_mask[i, int(b) + 1] = True
        q1_t = torch.from_numpy(q1_target).to(device)
        q1_m = torch.from_numpy(q1_mask).to(device)
        q1 = model(x1); q2 = model(x2)
        loss = F.smooth_l1_loss(q1[q1_m], q1_t[q1_m]) + F.smooth_l1_loss(q2[q2_m], q2_t[q2_m])
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        if (epoch + 1) % 10 == 0:
            target.load_state_dict(model.state_dict())
    checkpoint = output / "sequential_dqn.pt"
    torch.save({"model": model.state_dict(), "state_dim": context1.shape[1],
                "actions": ["STOP", "VIEW0", "VIEW1", "VIEW2", "VIEW3"],
                "future_inputs_to_model": False}, checkpoint)
    return {"model": model, "checkpoint": checkpoint, "state_dim": context1.shape[1],
            "feature_names": names}


def data_subset_row(data: Mapping[str, Any], index: int) -> dict[str, Any]:
    return data_subset_rows(data, [index])


def data_subset_rows(data: Mapping[str, Any], indices: Sequence[int]) -> dict[str, Any]:
    indices = np.asarray(indices, dtype=np.int64)
    raw = dict(data["raw"])
    for key, value in list(raw.items()):
        if isinstance(value, np.ndarray) and value.ndim > 0 and len(value) >= int(indices.max(initial=-1)) + 1:
            raw[key] = np.asarray(value[indices])
    features = {key: np.asarray(value[indices]) for key, value in data["features"].items()}
    return {"raw": raw, "features": features, "episodes": np.arange(len(indices)),
            "source_episode_ids": np.asarray(data["source_episode_ids"])[indices],
            "classes": data["classes"], "split": data["split"]}


def evaluate_sequential_dqn(q_info: Mapping[str, Any], data: Mapping[str, Any], classes: Sequence[int],
                            first: np.ndarray, device: torch.device) -> dict[str, Any]:
    model = q_info["model"]; model.eval()
    context, _, selected = first_state(data, first)
    with torch.inference_mode():
        q1 = model(torch.from_numpy(context).to(device)).cpu().numpy()
    info = geometry_arrays(data)
    paths = []
    decisions = []
    for i in range(len(first)):
        mask = np.zeros(NUM_VIEWS + 1, dtype=bool); mask[0] = True
        mask[1:] = info["legal"][i] & ~selected[i]
        a = int(np.argmax(np.where(mask, q1[i], -np.inf)))
        if a == 0:
            paths.append([0, int(first[i])]); decisions.append("STOP")
        else:
            b = a - 1
            selected_b = selected[i].copy(); selected_b[b] = True
            x2, _, _ = state_features(data_subset_row(data, i), selected_b[None], np.asarray([b]))
            with torch.inference_mode():
                q2 = model(torch.from_numpy(x2).to(device)).cpu().numpy()[0]
            mask2 = np.zeros(NUM_VIEWS + 1, dtype=bool); mask2[0] = True
            mask2[1:] = info["legal"][i] & ~selected_b
            a2 = int(np.argmax(np.where(mask2, q2, -np.inf)))
            if a2 == 0:
                paths.append([0, int(first[i]), b]); decisions.append(f"MOVE_{b}_STOP")
            else:
                paths.append([0, int(first[i]), b, a2 - 1]); decisions.append(f"MOVE_{b}_MOVE_{a2 - 1}")
    report = evaluate_paths(data, classes, paths, "sequential_dqn")
    report["decision_histogram"] = {k: decisions.count(k) for k in sorted(set(decisions))}
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=Path("data/rl_candidate_rank_v6_new50_5/cache"))
    parser.add_argument("--output", type=Path, default=Path("work_dir/sequential_explore_then_decide_v1"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260910)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    raw_root = args.raw_root if args.raw_root.is_absolute() else root / args.raw_root
    output = args.output if args.output.is_absolute() else root / args.output
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    seed_all(args.seed)
    print(json.dumps({"SEQUENTIAL_EXPLORE_THEN_DECIDE_START": True,
                      "raw_root": str(raw_root), "output": str(output), "device": str(device),
                      "frozen_recognizer": True, "epsilon_gain": EPSILON_GAIN,
                      "future_inputs_to_policy": False,
                      "active_module_protocol": "40 train / 10 pseudo-unseen validation / 5 true-unseen test",
                      "recognizer_protocol_note": "existing VPOCLIP checkpoint was trained on the 50 seen classes; strict recognizer 40/10/5 is not claimed"}), flush=True)

    train = prepare_variable_data(raw_root, "dqn_train", TRAIN_CLASSES)
    pseudo = prepare_variable_data(raw_root, "val", PSEUDO_CLASSES)
    seen = prepare_variable_data(raw_root, "seen_test", SEEN_CLASSES)
    unseen = prepare_variable_data(raw_root, "test", UNSEEN_CLASSES)
    datasets = {"train_40": (train, TRAIN_CLASSES), "pseudo_unseen_10": (pseudo, PSEUDO_CLASSES),
                "seen_test_50": (seen, SEEN_CLASSES), "true_unseen_5": (unseen, UNSEEN_CLASSES)}
    split_report = {}
    for name, (data, classes) in datasets.items():
        info = geometry_arrays(data)
        split_report[name] = {"episodes": int(len(data["episodes"])), "classes": list(classes),
                              "subjects": int(len(set(data["raw"]["metadata"]["episodes"][int(x)].get("subject", "unknown")
                                                      for x in data["source_episode_ids"]))),
                              "valid_view_count_histogram": {str(k): int((info["valid"].sum(1) == k).sum()) for k in range(1, NUM_VIEWS + 1)}}
    (output / "split_report.json").write_text(json.dumps(split_report, indent=2), encoding="utf-8")

    strategy_names = ("max_angular_diversity", "preferred_orthogonal", "diversity_minus_cost")
    validation_strategy_metrics = {}
    for strategy in strategy_names:
        a = exploratory_actions(pseudo, strategy)
        validation_strategy_metrics[strategy] = evaluate_paths(pseudo, PSEUDO_CLASSES,
                                                                pair_paths_for_action(pseudo, a), strategy + "_validation")
    selected_strategy = max(validation_strategy_metrics,
                            key=lambda k: (validation_strategy_metrics[k]["top1"]
                                           - 0.01 * validation_strategy_metrics[k]["average_movement_cost"],
                                           validation_strategy_metrics[k]["top1"]))
    first_actions = {name: exploratory_actions(data, selected_strategy)
                     for name, (data, _) in datasets.items()}
    baselines = {name: baseline_report(data, classes, selected_strategy)
                 for name, (data, classes) in datasets.items()}
    print(json.dumps({"FIRST_EXPLORATION_VALIDATION": validation_strategy_metrics,
                      "SELECTED_FIRST_STRATEGY": selected_strategy,
                      "split_report": split_report}, indent=2), flush=True)

    train_first = first_actions["train_40"]
    pseudo_first = first_actions["pseudo_unseen_10"]
    unseen_first = first_actions["true_unseen_5"]
    train_x, feature_names, _ = first_state(train, train_first)
    pseudo_x, _, _ = first_state(pseudo, pseudo_first)
    seen_x, _, _ = first_state(seen, first_actions["seen_test_50"])
    unseen_x, _, _ = first_state(unseen, unseen_first)
    train_target = sequential_targets(train, TRAIN_CLASSES, train_first)
    pseudo_target = sequential_targets(pseudo, PSEUDO_CLASSES, pseudo_first)
    seen_target = sequential_targets(seen, SEEN_CLASSES, first_actions["seen_test_50"])
    unseen_target = sequential_targets(unseen, UNSEEN_CLASSES, unseen_first)
    groups = np.asarray([str(train["raw"]["metadata"]["episodes"][int(x)].get("subject", "unknown"))
                         for x in train["source_episode_ids"]])
    model_names = ("logistic", "random_forest", "gradient_boosting", "mlp")
    models, oof_report, folds = fit_stop_models(train_x, train_target["continue"], groups, device, args.seed)
    split_x = {"pseudo_unseen_10": pseudo_x, "seen_test_50": seen_x, "true_unseen_5": unseen_x}
    split_y = {"pseudo_unseen_10": pseudo_target["continue"], "seen_test_50": seen_target["continue"],
               "true_unseen_5": unseen_target["continue"]}
    predictor_report = {"feature_names": feature_names, "feature_dim": int(train_x.shape[1]),
                        "label_definition": "CONTINUE iff max remaining target-margin gain > epsilon; epsilon=0.02",
                        "train": {"episodes": int(len(train_x)), "positive": int(train_target["continue"].sum()),
                                  "prevalence": float(train_target["continue"].mean()), "oof": oof_report},
                        "splits": {}}
    predictions: dict[str, dict[str, np.ndarray]] = {}
    for split_name, x in split_x.items():
        predictions[split_name] = {}
        predictor_report["splits"][split_name] = {}
        for model_name in model_names:
            p = predict_classifier(model_name, models[model_name], x, device)
            predictions[split_name][model_name] = p
            predictor_report["splits"][split_name][model_name] = {
                **auc_metrics(split_y[split_name], p), "best_threshold": best_threshold(split_y[split_name], p)
            }
    selected_model = max(model_names,
                         key=lambda k: ((predictor_report["splits"]["pseudo_unseen_10"][k]["auroc"] or -1.0),
                                        (predictor_report["splits"]["pseudo_unseen_10"][k]["auprc"] or -1.0)))
    selected_threshold = float(predictor_report["splits"]["pseudo_unseen_10"][selected_model]["best_threshold"]["threshold"])
    selected_val = predictor_report["splits"]["pseudo_unseen_10"][selected_model]
    prevalence = float(split_y["pseudo_unseen_10"].mean())
    auprc_clear = bool(selected_val["auprc"] is not None and selected_val["auprc"] >= prevalence * 1.15
                       and selected_val["auprc"] >= prevalence + 0.03)
    gate_a = bool((selected_val["auroc"] is not None and selected_val["auroc"] >= 0.70) or auprc_clear)
    gate_report: dict[str, Any] = {
        "selected_model_by_validation": selected_model,
        "selected_threshold": selected_threshold,
        "validation_auroc": selected_val["auroc"],
        "validation_auprc": selected_val["auprc"],
        "validation_positive_prevalence": prevalence,
        "auroc_threshold": 0.70,
        "auprc_clear_rule": "AUPRC >= prevalence*1.15 and >= prevalence+0.03",
        "passed": gate_a,
    }
    ranker_report_all: dict[str, Any] = {"gate_a": gate_report, "variants": {}}
    adaptive_reports: dict[str, Any] = {}
    if gate_a:
        ranker = train_candidate_ranker(train, TRAIN_CLASSES, train_first, device, args.seed + 500, output)
        for split_name, (data, classes) in datasets.items():
            if split_name == "train_40":
                continue
            first = first_actions[split_name]
            ranker_report_all["variants"][split_name] = ranker_report(ranker, data, classes, first, device, args.seed + len(split_name))
        gate_b = bool(ranker_report_all["variants"].get("pseudo_unseen_10", {}).get("gate_b_passed", False))
        ranker_report_all["gate_b"] = {"passed": gate_b, "criterion": "validation delta > 0.01 and paired CI lower bound > 0"}
        for split_name, (data, classes) in datasets.items():
            p = predictions.get(split_name, {}).get(selected_model)
            if p is None:
                # The train split is intentionally not part of split_x above;
                # compute its full-fit probability only when adaptive behavior
                # is actually being reported.
                p = predict_classifier(selected_model, models[selected_model],
                                       first_state(data, first_actions[split_name])[0], device)
            paths, stop = adaptive_paths(data, first_actions[split_name], p, selected_threshold,
                                         ranker if gate_b else None, device)
            adaptive_reports[split_name] = evaluate_paths(data, classes, paths, "adaptive_explore_then_decide")
            adaptive_reports[split_name]["stop_rate"] = float(stop.mean())
            adaptive_reports[split_name]["stop_predictor_model"] = selected_model
            adaptive_reports[split_name]["ranker_used"] = gate_b
        if gate_b:
            q_info = train_sequential_dqn(train, TRAIN_CLASSES, train_first, device, args.seed + 700, output)
            for split_name, (data, classes) in datasets.items():
                adaptive_reports.setdefault(split_name, {})["sequential_dqn"] = evaluate_sequential_dqn(
                    q_info, data, classes, first_actions[split_name], device
                )
    else:
        ranker_report_all["skip_reason"] = "Gate A failed on pseudo-unseen validation; no remaining-view ranker or sequential RL was trained."
        adaptive_reports = {name: {"status": "not_run_gate_a_failed"} for name in datasets}
    predictor_report["selected_model"] = selected_model
    predictor_report["selected_threshold"] = selected_threshold
    (output / "stop_continue_predictor_metrics.json").write_text(json.dumps(json_safe(predictor_report), indent=2), encoding="utf-8")
    with (output / "stop_continue_predictor_models.pkl").open("wb") as handle:
        pickle.dump({"models": models, "feature_names": feature_names, "selected_model": selected_model,
                     "selected_threshold": selected_threshold}, handle)
    (output / "ranker_and_gates.json").write_text(json.dumps(json_safe(ranker_report_all), indent=2), encoding="utf-8")
    summary = {
        "version": "sequential_explore_then_decide_active_ovhar_v1",
        "protocol": {
            "first_view": "physical view slot 0",
            "first_exploration_candidates": list(strategy_names),
            "selected_first_strategy_by_pseudo_unseen": selected_strategy,
            "active_module_split": "40 train / 10 pseudo-unseen validation / 5 true-unseen test",
            "recognizer_warning": "The frozen VPOCLIP checkpoint was trained on all 50 seen classes; this is not strict recognizer 40/10/5 ZSL.",
            "future_logits_or_pose_to_policy": False,
            "variable_view_counts_preserved": True,
        },
        "split_report": split_report,
        "first_exploration_validation": validation_strategy_metrics,
        "baselines": baselines,
        "stop_continue_predictor": predictor_report,
        "gates": ranker_report_all,
        "adaptive_reports": adaptive_reports,
        "limitations": {
            "offline_cached_observation": "real cached per-view VPOCLIP logits stand in for the deployed second observation",
            "cross_modal": "source cache has fused logits and sensor proxies, not independent RGB/Skeleton/Object logits",
            "strict_zsl": "requires a separately trained 40-class VPOCLIP checkpoint to make the recognizer itself strict 40/10/5",
        },
    }
    (output / "summary.json").write_text(json.dumps(json_safe(summary), indent=2), encoding="utf-8")
    print(json.dumps({"SEQUENTIAL_EXPLORE_THEN_DECIDE_COMPLETE": True, "output": str(output),
                      "gate_a": gate_report, "gate_b": ranker_report_all.get("gate_b"),
                      "selected_model": selected_model, "selected_first_strategy": selected_strategy}, indent=2), flush=True)


if __name__ == "__main__":
    main()
