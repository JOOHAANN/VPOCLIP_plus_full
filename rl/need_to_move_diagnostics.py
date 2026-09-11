"""Need-to-Move + interaction-aware diagnostics.

This is the next-stage gate described in
``Next_Stage_NeedToMove_Interaction_ViewSelection.md``.  It does not train a
new viewpoint policy.  It performs:

1. recoverable-only proxy/utility correlations;
2. recoverable-only future-observation proxy ceilings; and
3. a group-held-out diagnostic classifier for the binary Need-to-Move label.

The label and future-observation scores are privileged diagnostics only.  No
future candidate RGB/features/logits are included in the Need-to-Move input.
All old checkpoints and experiment directories remain untouched.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from torch import nn

from .structured_sa_experiment import (
    CANDIDATE_DIM,
    CURRENT_VIEW,
    NUM_VIEWS,
    PSEUDO_CLASSES,
    SEEN_CLASSES,
    TRAIN_CLASSES,
    UNSEEN_CLASSES,
    candidate_inputs,
    handcrafted_components,
    margin_scores,
    prepare_data,
)
from .diagnose_structured_viewpoint import (
    _correlation,
    _paired_ci,
    make_eval_arrays,
    weighted_future_scores,
)


CANDIDATES = np.asarray([1, 2, 3], dtype=np.int64)
RECOVERABLE_TIE_THRESHOLD = 0.05
MOVE_POSITIVE_MIN = 1


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(v) for v in value]
    return value


def _labels_and_utility(data: Mapping[str, Any]) -> dict[str, np.ndarray]:
    arrays = make_eval_arrays(data, UNSEEN_CLASSES)
    candidate_correct = arrays["candidate_correct"][:, CANDIDATES]
    utility = arrays["utility"][:, CANDIDATES]
    utility_range = np.ptp(utility, axis=1)
    all_correct = candidate_correct.all(axis=1)
    all_wrong = ~candidate_correct.any(axis=1)
    view_insensitive = all_correct | (utility_range <= RECOVERABLE_TIE_THRESHOLD)
    recoverable = (~view_insensitive) & (~all_correct) & (~all_wrong)
    unrecoverable = (~view_insensitive) & all_wrong
    if not np.all(view_insensitive | recoverable | unrecoverable):
        raise AssertionError("Need-to-Move labels do not partition true unseen")
    return {
        "candidate_correct": candidate_correct,
        "utility": utility,
        "recoverable": recoverable,
        "view_insensitive": view_insensitive,
        "unrecoverable": unrecoverable,
        "arrays": arrays,
    }


def _recoverable_proxy_correlation(data: Mapping[str, Any]) -> dict[str, Any]:
    info = _labels_and_utility(data)
    recoverable = info["recoverable"]
    arrays = info["arrays"]
    _, _, action_info = candidate_inputs(data["raw"], data["episodes"], data["features"])
    components = handcrafted_components(dict(data))[:, CANDIDATES]
    utility = arrays["utility"][:, CANDIDATES]
    valid = action_info["valid"][:, CANDIDATES]
    # The candidate-independent current uncertainty is deliberately repeated;
    # it cannot rank the three actions, so its within-episode correlation is
    # reported as undefined rather than replaced by future logits.
    logits = np.asarray(data["raw"]["logits"][data["episodes"]], dtype=np.float32)
    current = logits[:, CURRENT_VIEW]
    current_uncertainty = _normalized_entropy(current)
    proxies = {
        "geometry": components[..., 0],
        "visibility": components[..., 1],
        "interaction": components[..., 2],
        "uncertainty": np.repeat(current_uncertainty[:, None], len(CANDIDATES), axis=1),
    }
    selected_rows: list[dict[str, Any]] = []
    sample_ids = _sample_ids(data)
    for i in np.flatnonzero(recoverable):
        for j, view in enumerate(CANDIDATES):
            selected_rows.append({
                "sample_id": sample_ids[int(i)],
                "episode_index": int(i),
                "candidate_view": int(view),
                "S_geometry": float(proxies["geometry"][i, j]),
                "S_visibility": float(proxies["visibility"][i, j]),
                "S_interaction": float(proxies["interaction"][i, j]),
                "S_uncertainty": float(proxies["uncertainty"][i, j]),
                "utility": float(utility[i, j]),
                "candidate_correct": int(info["candidate_correct"][i, j]),
                "valid": int(valid[i, j]),
            })
    report: dict[str, Any] = {
        "episodes": int(recoverable.sum()),
        "candidate_rows": int(len(selected_rows)),
        "utility_definition": "fused GT margin minus current-view GT margin",
        "tie_threshold": RECOVERABLE_TIE_THRESHOLD,
        "correlations": {},
    }
    for name, proxy in proxies.items():
        report["correlations"][name] = {
            "pooled": _correlation(
                proxy[(recoverable[:, None] & valid)],
                utility[(recoverable[:, None] & valid)],
            ),
            "within_episode_pooled_mean": _within_episode_mean(proxy[recoverable], utility[recoverable], valid[recoverable]),
        }
    return {"report": report, "rows": selected_rows, "labels": info}


def _within_episode_mean(proxy: np.ndarray, utility: np.ndarray, valid: np.ndarray) -> dict[str, Any]:
    values: list[float] = []
    for x, y, ok in zip(proxy, utility, valid):
        keep = np.asarray(ok, dtype=bool)
        if keep.sum() >= 2 and np.unique(x[keep]).size > 1 and np.unique(y[keep]).size > 1:
            values.append(float(spearmanr(x[keep], y[keep]).statistic))
    return {
        "episodes_with_variation": int(len(values)),
        "mean_spearman": float(np.mean(values)) if values else None,
    }


def _sample_ids(data: Mapping[str, Any]) -> list[str]:
    metadata = data["raw"]["metadata"]
    source_ids = np.asarray(data["source_episode_ids"], dtype=np.int64)
    return [
        str(metadata["episodes"][int(source_id)].get("base_sample", source_id))
        for source_id in source_ids
    ]


def _normalized_entropy(logits: np.ndarray) -> np.ndarray:
    x = logits - logits.max(axis=-1, keepdims=True)
    p = np.exp(np.clip(x, -80.0, 80.0))
    p /= np.maximum(p.sum(axis=-1, keepdims=True), 1e-8)
    return (
        -(p * np.log(np.maximum(p, 1e-8))).sum(axis=-1) / math.log(logits.shape[-1])
    ).astype(np.float32)


def _future_ceiling(data: Mapping[str, Any]) -> dict[str, Any]:
    info = _labels_and_utility(data)
    mask = info["recoverable"]
    arrays = info["arrays"]
    valid = arrays["valid"]
    body, interaction = weighted_future_scores(data["features"])
    body_gain = body - body[:, CURRENT_VIEW, None]
    interaction_gain = interaction - interaction[:, CURRENT_VIEW, None]
    body_gain = body_gain[:, CANDIDATES]
    interaction_gain = interaction_gain[:, CANDIDATES]
    candidate_correct = arrays["candidate_correct"][:, CANDIDATES]
    expected = arrays["random_expected"]
    expected = expected[mask]
    scores = {
        "random": np.zeros((mask.sum(),), dtype=np.int64),
        "future_visibility_oracle": body_gain[mask].argmax(axis=-1),
        "future_interaction_oracle": interaction_gain[mask].argmax(axis=-1),
        "future_visibility_interaction_fixed": (body_gain[mask] + interaction_gain[mask]).argmax(axis=-1),
    }
    metrics: dict[str, Any] = {}
    rng = np.random.default_rng(20260910)
    sampled_random = rng.integers(0, len(CANDIDATES), size=int(mask.sum()))
    for name, action in scores.items():
        if name == "random":
            action = sampled_random
        correct = candidate_correct[mask][np.arange(mask.sum()), action]
        delta = correct.astype(np.float32) - expected
        metrics[name] = {
            "top1": float(expected.mean() if name == "random" else correct.mean()),
            "sampled_top1": float(correct.mean()),
            "delta_vs_random": float(0.0 if name == "random" else delta.mean()),
            "paired_bootstrap_ci95": [0.0, 0.0] if name == "random" else _paired_ci(delta, 23000),
            "body_gain": float(body_gain[mask][np.arange(mask.sum()), action].mean()),
            "interaction_gain": float(interaction_gain[mask][np.arange(mask.sum()), action].mean()),
        }
    metrics["full_har_oracle"] = {
        "top1": 1.0,
        "delta_vs_random": float(1.0 - expected.mean()),
        "paired_bootstrap_ci95": _paired_ci(1.0 - expected, 23001),
    }
    return {
        "episodes": int(mask.sum()),
        "random_expected_top1": float(expected.mean()),
        "methods": metrics,
        "body_gain_definition": "future candidate body-part visibility minus current view 0",
        "interaction_gain_definition": "future candidate interaction clarity minus current view 0",
        "note": "future observation is privileged diagnostic input; never used by Need-to-Move classifier",
    }


def _current_state_features(data: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Build only current-view causal features for Stage 1."""

    raw = data["raw"]
    episodes = np.asarray(data["episodes"], dtype=np.int64)
    logits = np.asarray(raw["logits"][episodes, CURRENT_VIEW], dtype=np.float32)
    values = logits - logits.max(axis=-1, keepdims=True)
    probability = np.exp(np.clip(values, -80.0, 80.0))
    probability /= np.maximum(probability.sum(axis=-1, keepdims=True), 1e-8)
    top = np.sort(probability, axis=-1)[:, ::-1]
    entropy = _normalized_entropy(logits)
    energy = np.log(np.exp(np.clip(logits, -40.0, 40.0)).sum(axis=-1) + 1e-8)
    confidence_features = np.stack(
        [entropy, top[:, 0], top[:, 0] - top[:, 1], energy], axis=-1
    ).astype(np.float32)
    features = data["features"]
    current_structured = np.concatenate(
        [
            features["visibility"][:, CURRENT_VIEW],
            features["ambiguity"][:, CURRENT_VIEW],
            features["orientation"][:, CURRENT_VIEW],
            features["uncertainty"][:, CURRENT_VIEW],
            features["quality"][:, CURRENT_VIEW],
            confidence_features,
        ],
        axis=-1,
    ).astype(np.float32)
    geometry = np.asarray(raw["geometry"][episodes, CURRENT_VIEW, :2], dtype=np.float32)
    geometry_confidence = np.asarray(
        raw["geometry_confidence"][episodes, CURRENT_VIEW, None], dtype=np.float32
    )
    x = np.concatenate([current_structured, geometry, geometry_confidence], axis=-1)
    availability = {
        "causal_feature_groups": [
            "current_entropy",
            "current_top1",
            "current_top1_top2_margin",
            "current_energy",
            "body_visibility_proxy",
            "interaction_ambiguity_proxy",
            "torso_orientation_proxy",
            "image_quality",
            "current_body_relative_geometry",
        ],
        "unavailable_not_fabricated": [
            "RGB_vs_skeleton_modality_disagreement",
            "RGB_vs_object_modality_disagreement",
            "per_joint_skeleton_confidence",
            "per_frame_logit_instability",
        ],
        "feature_dim": int(x.shape[-1]),
        "future_candidate_inputs": False,
    }
    return x, availability


class MoveMLP(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(width, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _classification_metrics(y: np.ndarray, probability: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    prediction = (probability >= threshold).astype(np.int64)
    y = np.asarray(y, dtype=np.int64)
    tp = float(np.sum((y == 1) & (prediction == 1)))
    fp = float(np.sum((y == 0) & (prediction == 1)))
    fn = float(np.sum((y == 1) & (prediction == 0)))
    positive = float(np.sum(y == 1))
    negative = float(np.sum(y == 0))
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    # Rank-based AUROC, with average ranks for ties.
    order = np.argsort(np.asarray(probability, dtype=np.float64), kind="mergesort")
    sorted_probability = np.asarray(probability, dtype=np.float64)[order]
    ranks = np.empty(len(sorted_probability), dtype=np.float64)
    begin = 0
    while begin < len(ranks):
        end = begin + 1
        while end < len(ranks) and sorted_probability[end] == sorted_probability[begin]:
            end += 1
        ranks[begin:end] = 0.5 * (begin + 1 + end)
        begin = end
    original_ranks = np.empty_like(ranks)
    original_ranks[order] = ranks
    positive_rank_sum = float(original_ranks[y == 1].sum())
    auroc = (
        (positive_rank_sum - positive * (positive + 1.0) / 2.0) / max(positive * negative, 1.0)
        if positive > 0 and negative > 0 else float("nan")
    )
    # Average precision: mean precision at every positive-ranked example.
    descending = np.argsort(-np.asarray(probability, dtype=np.float64), kind="mergesort")
    sorted_y = y[descending]
    cumulative_positive = np.cumsum(sorted_y == 1)
    positive_positions = np.flatnonzero(sorted_y == 1)
    auprc = float(
        np.mean(cumulative_positive[positive_positions] / (positive_positions + 1.0))
        if positive_positions.size else 0.0
    )
    return {
        "auroc": float(auroc),
        "auprc": auprc,
        "positive_prevalence": float(y.mean()),
        "accuracy": float(np.mean(y == prediction)),
        "precision": precision,
        "recall_recoverable": recall,
        "f1": f1,
        "predicted_positive_rate": float(prediction.mean()),
    }


def _standardize_train_test(
    x_train: np.ndarray, x_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(x_train, dtype=np.float32).mean(axis=0)
    std = np.asarray(x_train, dtype=np.float32).std(axis=0)
    std = np.maximum(std, 1e-5)
    return ((x_train - mean) / std).astype(np.float32), ((x_test - mean) / std).astype(np.float32)


def _fit_linear_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    seed: int,
    device: torch.device,
) -> np.ndarray:
    seed_all(seed)
    x_train, x_test = _standardize_train_test(x_train, x_test)
    train = torch.from_numpy(x_train).to(device)
    test = torch.from_numpy(x_test).to(device)
    labels = torch.from_numpy(y_train.astype(np.float32)).to(device)
    positive = max(float(y_train.sum()), 1.0)
    negative = max(float(len(y_train) - y_train.sum()), 1.0)
    model = nn.Linear(x_train.shape[-1], 1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-2)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negative / positive, device=device))
    model.train()
    for _ in range(500):
        loss = loss_fn(model(train).squeeze(-1), labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.inference_mode():
        probability = torch.sigmoid(model(test).squeeze(-1)).cpu().numpy()
    return probability.astype(np.float32)


def _fit_mlp_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    seed: int,
    device: torch.device,
) -> np.ndarray:
    seed_all(seed)
    x_train, x_test = _standardize_train_test(x_train, x_test)
    train = torch.from_numpy(x_train).to(device)
    test = torch.from_numpy(x_test).to(device)
    labels = torch.from_numpy(y_train.astype(np.float32)).to(device)
    positive = max(float(y_train.sum()), 1.0)
    negative = max(float(len(y_train) - y_train.sum()), 1.0)
    model = MoveMLP(x_train.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negative / positive, device=device))
    model.train()
    for _ in range(300):
        logits = model(train)
        loss = loss_fn(logits, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.inference_mode():
        probability = torch.sigmoid(model(test)).cpu().numpy()
    return probability.astype(np.float32)


def _stratified_group_folds(
    y: np.ndarray, groups: np.ndarray, n_splits: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Small dependency-free approximation of StratifiedGroupKFold.

    Groups are assigned greedily to the fold with the lowest normalized
    positive/size load.  The true-unseen set has 20 subjects, so this keeps
    subjects disjoint while preserving positives in every fold.
    """

    rng = np.random.default_rng(seed)
    unique = np.asarray(np.unique(groups))
    rng.shuffle(unique)
    group_rows = []
    for group in unique:
        indices = np.flatnonzero(groups == group)
        group_rows.append((group, indices, int(y[indices].sum()), int(len(indices))))
    group_rows.sort(key=lambda row: (-row[2], -row[3]))
    fold_groups: list[list[Any]] = [[] for _ in range(n_splits)]
    fold_positive = np.zeros(n_splits, dtype=np.float64)
    fold_size = np.zeros(n_splits, dtype=np.float64)
    for group, indices, positive, size in group_rows:
        score = fold_positive / max(float(y.sum()), 1.0) + fold_size / max(float(len(y)), 1.0)
        fold = int(np.argmin(score))
        fold_groups[fold].append(group)
        fold_positive[fold] += positive
        fold_size[fold] += size
    result = []
    for fold in range(n_splits):
        test_mask = np.isin(groups, np.asarray(fold_groups[fold]))
        test_index = np.flatnonzero(test_mask)
        train_index = np.flatnonzero(~test_mask)
        if np.unique(y[train_index]).size < 2 or np.unique(y[test_index]).size < 2:
            raise ValueError(
                f"group fold {fold} lacks both classes: train={np.bincount(y[train_index], minlength=2).tolist()}, "
                f"test={np.bincount(y[test_index], minlength=2).tolist()}"
            )
        result.append((train_index, test_index))
    return result


def need_to_move_predictability(
    data: Mapping[str, Any], device: torch.device, seeds: Sequence[int]
) -> dict[str, Any]:
    labels_info = _labels_and_utility(data)
    y = labels_info["recoverable"].astype(np.int64)
    x, availability = _current_state_features(data)
    metadata = data["raw"]["metadata"]
    source_ids = np.asarray(data["source_episode_ids"], dtype=np.int64)
    groups = np.asarray(
        [str(metadata["episodes"][int(i)].get("subject", "unknown")) for i in source_ids]
    )
    if np.unique(groups).size < 5:
        raise ValueError("Need-to-Move diagnostic requires at least five subject groups")
    all_results: dict[str, list[dict[str, Any]]] = {"logistic_regression": [], "mlp_64_64": []}
    predictions: dict[str, list[np.ndarray]] = {"logistic_regression": [], "mlp_64_64": []}
    fold_records: list[dict[str, Any]] = []
    for seed in seeds:
        folds = _stratified_group_folds(y, groups, n_splits=5, seed=int(seed))
        for fold, (train_index, test_index) in enumerate(folds):
            logistic_probability = _fit_linear_predict(
                x[train_index], y[train_index], x[test_index], int(seed + fold), device
            )
            mlp_probability = _fit_mlp_predict(
                x[train_index], y[train_index], x[test_index], int(seed + fold), device
            )
            for name, probability in [
                ("logistic_regression", logistic_probability),
                ("mlp_64_64", mlp_probability),
            ]:
                metric = _classification_metrics(y[test_index], probability)
                metric.update({"seed": int(seed), "fold": int(fold), "test_episodes": int(len(test_index))})
                all_results[name].append(metric)
                predictions[name].append(
                    np.column_stack((test_index, y[test_index], probability))
                )
            fold_records.append({
                "seed": int(seed), "fold": int(fold),
                "train_episodes": int(len(train_index)), "test_episodes": int(len(test_index)),
                "train_positive": int(y[train_index].sum()), "test_positive": int(y[test_index].sum()),
                "test_subjects": sorted(set(groups[test_index].tolist())),
            })
    aggregate: dict[str, Any] = {}
    for name, folds in all_results.items():
        aggregate[name] = {
            "fold_mean": {
                key: float(np.mean([row[key] for row in folds]))
                for key in ["auroc", "auprc", "accuracy", "precision", "recall_recoverable", "f1"]
            },
            "fold_std": {
                key: float(np.std([row[key] for row in folds]))
                for key in ["auroc", "auprc", "accuracy", "precision", "recall_recoverable", "f1"]
            },
            "folds": folds,
        }
    return {
        "episodes": int(len(y)),
        "positive_recoverable": int(y.sum()),
        "negative_nonrecoverable": int((1 - y).sum()),
        "positive_prevalence": float(y.mean()),
        "group_definition": "subject; no subject appears in both train and test fold",
        "feature_availability": availability,
        "models": aggregate,
        "folds": fold_records,
        "stop_gate": {
            "auroc_threshold": 0.70,
            "auprc_baseline": float(y.mean()),
            "interpretation": "diagnostic only; true-unseen labels are never used to train a policy",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=Path("data/rl_candidate_rank_v6_new50_5/cache"))
    parser.add_argument("--output", type=Path, default=Path("work_dir/need_to_move_interaction_v1"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed-list", default="20260909,20260910,20260911")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    raw_root = args.raw_root if args.raw_root.is_absolute() else root / args.raw_root
    output = args.output if args.output.is_absolute() else root / args.output
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    seeds = [int(x.strip()) for x in args.seed_list.split(",") if x.strip()]
    print(json.dumps({
        "NEED_TO_MOVE_DIAGNOSTICS_START": True,
        "raw_root": str(raw_root), "output": str(output), "device": str(device),
        "no_new_viewpoint_policy_training": True,
        "protocol": "true unseen, fixed view0, candidates 1/2/3",
    }), flush=True)
    data = prepare_data(raw_root, "test", UNSEEN_CLASSES)
    proxy = _recoverable_proxy_correlation(data)
    (output / "recoverable_proxy_rows.json").write_text(
        json.dumps(_json_safe(proxy["rows"]), indent=2), encoding="utf-8"
    )
    (output / "recoverable_proxy_correlation.json").write_text(
        json.dumps(_json_safe(proxy["report"]), indent=2), encoding="utf-8"
    )
    future = _future_ceiling(data)
    (output / "recoverable_future_ceiling.json").write_text(
        json.dumps(_json_safe(future), indent=2), encoding="utf-8"
    )
    predictability = need_to_move_predictability(data, device, seeds)
    (output / "need_to_move_predictability.json").write_text(
        json.dumps(_json_safe(predictability), indent=2), encoding="utf-8"
    )
    interaction = future["methods"]["future_interaction_oracle"]["top1"]
    random_top1 = future["random_expected_top1"]
    best_auroc = max(
        value["fold_mean"]["auroc"] for value in predictability["models"].values()
    )
    best_auprc = max(
        value["fold_mean"]["auprc"] for value in predictability["models"].values()
    )
    summary = {
        "version": "need_to_move_interaction_v1",
        "protocol": "true unseen, fixed view0, candidates 1/2/3",
        "no_new_viewpoint_policy_training": True,
        "split": {"episodes": int(len(data["episodes"])), "classes": UNSEEN_CLASSES},
        "step1_recoverable_proxy_correlation": proxy["report"],
        "step2_recoverable_future_proxy_ceiling": future,
        "step3_need_to_move_predictability": predictability,
        "gates": {
            "future_interaction_oracle_top1": interaction,
            "future_interaction_random_delta": interaction - random_top1,
            "future_interaction_gate_60_percent": bool(interaction >= 0.60),
            "best_need_to_move_auroc": best_auroc,
            "need_to_move_auroc_gate_70_percent": bool(best_auroc > 0.70),
            "best_need_to_move_auprc": best_auprc,
            "need_to_move_auprc_baseline": predictability["positive_prevalence"],
            "both_gates_pass": bool(interaction >= 0.60 and best_auroc > 0.70),
        },
        "decision": (
            "proceed_to_interaction_forecasting"
            if interaction >= 0.60 and best_auroc > 0.70
            else "stop_before_new_policy_training"
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(_json_safe(summary), indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "NEED_TO_MOVE_DIAGNOSTICS_COMPLETE": True,
        "output": str(output),
        "future_interaction_oracle": interaction,
        "random": random_top1,
        "best_need_to_move_auroc": best_auroc,
        "best_need_to_move_auprc": best_auprc,
        "decision": summary["decision"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
