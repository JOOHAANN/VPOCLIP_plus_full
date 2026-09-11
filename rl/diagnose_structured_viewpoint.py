"""Four diagnostics for the structured active-view hypothesis.

This module deliberately does not train a new policy.  It answers four
questions before another training run is justified:

1. Do the causal geometry/visibility/interaction proxies correlate with the
   offline HAR utility?
2. How much could be obtained if the *future observation* were known?
3. Which true-unseen episodes are actually recoverable by changing view?
4. Is the current pseudo-unseen split a valid recognizer-ZSL validation set?

The future observation is used only for diagnostics B/C and is never passed
to a policy.  The experiment uses the old VPOCLIP cache, fixed view 0, and
physical candidate slots 1, 2, and 3.  The code is separate from all prior
training scripts and writes to a new output directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

from .structured_sa_experiment import (
    CURRENT_VIEW,
    NUM_VIEWS,
    PSEUDO_CLASSES,
    SEEN_CLASSES,
    StructuredScorer,
    TRAIN_CLASSES,
    UNSEEN_CLASSES,
    action_from_model,
    candidate_inputs,
    handcrafted_actions,
    handcrafted_components,
    load_old_actions,
    margin_scores,
    prepare_data,
)


CANDIDATE_VIEWS = np.asarray([1, 2, 3], dtype=np.int64)
VISIBILITY_WEIGHTS = np.asarray(
    [0.10, 0.20, 0.20, 0.20, 0.10, 0.10, 0.10], dtype=np.float32
)
INTERACTION_WEIGHTS = np.asarray(
    [0.30, 0.35, 0.15, 0.10, 0.10], dtype=np.float32
)
CORRELATION_FIELDS = [
    "S_geometry",
    "S_visibility",
    "S_interaction",
    "S_uncertainty",
    "S_motion",
]


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def weighted_future_scores(features: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Return per-view body visibility and interaction clarity in [0, 1]."""

    visibility = np.asarray(features["visibility"], dtype=np.float32)
    ambiguity = np.asarray(features["ambiguity"], dtype=np.float32)
    body = np.sum(visibility * VISIBILITY_WEIGHTS[None, None, :], axis=-1)
    clarity = np.sum(
        (1.0 - ambiguity) * INTERACTION_WEIGHTS[None, None, :], axis=-1
    )
    return body.astype(np.float32), clarity.astype(np.float32)


def _class_local_index(labels: np.ndarray, classes: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    bank = np.asarray(sorted({int(x) for x in classes}), dtype=np.int64)
    local = np.searchsorted(bank, labels)
    if np.any(local >= len(bank)) or np.any(bank[np.minimum(local, len(bank) - 1)] != labels):
        raise ValueError("labels are outside the requested class bank")
    return bank, local.astype(np.int64)


def _classification_arrays(
    data: Mapping[str, Any], classes: Sequence[int]
) -> dict[str, np.ndarray]:
    raw = data["raw"]
    episodes = np.asarray(data["episodes"], dtype=np.int64)
    labels = np.asarray(raw["labels"])[episodes]
    logits = np.asarray(raw["logits"][episodes], dtype=np.float32)
    current_logits = logits[:, CURRENT_VIEW]
    fused_logits = 0.5 * (current_logits[:, None, :] + logits)
    bank, local = _class_local_index(labels, classes)
    scores = fused_logits[..., bank]
    correct = scores.argmax(-1) == local[:, None]
    valid = np.zeros((len(episodes), NUM_VIEWS), dtype=bool)
    valid[:, CANDIDATE_VIEWS] = True
    candidate_correct = correct & valid
    current_scores = current_logits[..., bank]
    current_target = current_scores[np.arange(len(labels)), local]
    fused_target = scores[np.arange(len(labels))[:, None], :, local]
    # The advanced indexing above is intentionally rewritten below to avoid
    # NumPy's axis-order surprises when labels are per-episode.
    fused_target = np.take_along_axis(
        scores, np.broadcast_to(local[:, None, None], (len(labels), NUM_VIEWS, 1)), axis=-1
    )[..., 0]
    current_margin = margin_scores(current_logits, labels, classes)
    fused_margin = margin_scores(fused_logits, labels[:, None], classes)
    utility = fused_margin - current_margin[:, None]
    current_entropy = _normalized_entropy(current_logits)
    future_entropy = _normalized_entropy(logits)
    return {
        "labels": labels.astype(np.int64),
        "logits": logits,
        "current_logits": current_logits,
        "fused_logits": fused_logits,
        "correct": correct,
        "candidate_correct": candidate_correct,
        "valid": valid,
        "current_margin": current_margin.astype(np.float32),
        "fused_margin": fused_margin.astype(np.float32),
        "utility": utility.astype(np.float32),
        "current_target_logit": current_target.astype(np.float32),
        "fused_target_logit": fused_target.astype(np.float32),
        "gt_logit_gain": (fused_target - current_target[:, None]).astype(np.float32),
        "current_correct": correct[:, CURRENT_VIEW].astype(bool),
        "current_entropy": current_entropy,
        "future_entropy": future_entropy,
        "classes": bank,
    }


def _normalized_entropy(logits: np.ndarray) -> np.ndarray:
    values = logits - logits.max(axis=-1, keepdims=True)
    probability = np.exp(np.clip(values, -80.0, 80.0))
    probability /= np.maximum(probability.sum(axis=-1, keepdims=True), 1e-8)
    return (
        -(probability * np.log(np.maximum(probability, 1e-8))).sum(axis=-1)
        / math.log(float(logits.shape[-1]))
    ).astype(np.float32)


def _correlation(x: np.ndarray, y: np.ndarray) -> dict[str, float | int | None]:
    mask = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x[mask], dtype=np.float64)
    y = np.asarray(y[mask], dtype=np.float64)
    result: dict[str, float | int | None] = {"n": int(len(x))}
    if len(x) < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
        result.update({"spearman": None, "pearson": None})
        return result
    result["spearman"] = float(spearmanr(x, y).statistic)
    result["pearson"] = float(pearsonr(x, y).statistic)
    return result


def _within_episode_correlation(
    proxy: np.ndarray, utility: np.ndarray, valid: np.ndarray
) -> dict[str, float | int | None]:
    values: list[float] = []
    for row_proxy, row_utility, row_valid in zip(proxy, utility, valid):
        keep = np.asarray(row_valid, dtype=bool)
        x, y = row_proxy[keep], row_utility[keep]
        if len(x) >= 2 and np.unique(x).size >= 2 and np.unique(y).size >= 2:
            values.append(float(spearmanr(x, y).statistic))
    if not values:
        return {"episodes_with_variation": 0, "mean_spearman": None}
    return {
        "episodes_with_variation": int(len(values)),
        "mean_spearman": float(np.mean(values)),
    }


def _sample_ids(data: Mapping[str, Any]) -> list[str]:
    metadata = data["raw"]["metadata"]
    source_ids = np.asarray(data["source_episode_ids"], dtype=np.int64)
    ids: list[str] = []
    for source_id in source_ids:
        episode = metadata["episodes"][int(source_id)]
        ids.append(str(episode.get("base_sample", episode.get("episode_id", source_id))))
    return ids


def build_proxy_rows(
    data: Mapping[str, Any], classes: Sequence[int]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build Diagnostic A's per-candidate table and summary arrays."""

    raw = data["raw"]
    episodes = np.asarray(data["episodes"], dtype=np.int64)
    n = len(episodes)
    cls = _classification_arrays(data, classes)
    _, _, info = candidate_inputs(raw, episodes, data["features"])
    components = handcrafted_components(dict(data))
    body_future, interaction_future = weighted_future_scores(data["features"])
    current_body = body_future[:, CURRENT_VIEW]
    current_interaction = interaction_future[:, CURRENT_VIEW]
    current_entropy = cls["current_entropy"]
    future_uncertainty_reduction = current_entropy[:, None] - cls["future_entropy"]
    sample_ids = _sample_ids(data)

    # The causal uncertainty state is identical for all candidate actions.
    # It is retained explicitly so the diagnostic can prove when its rank
    # correlation is undefined, rather than silently replacing it with future
    # logits.  The future reduction is a separate, privileged diagnostic field.
    s_uncertainty = np.repeat(current_entropy[:, None], NUM_VIEWS, axis=1)
    rows: list[dict[str, Any]] = []
    camera_names: dict[tuple[int, int], str] = {}
    for episode_index, source_id in enumerate(data["source_episode_ids"]):
        episode = raw["metadata"]["episodes"][int(source_id)]
        for view_id in CANDIDATE_VIEWS:
            views = episode.get("views", [])
            camera_names[(episode_index, int(view_id))] = (
                str(views[int(view_id)].get("camera", f"view{view_id}"))
                if int(view_id) < len(views)
                else f"view{view_id}"
            )
    for episode_index in range(n):
        for view_id in CANDIDATE_VIEWS:
            view = int(view_id)
            rows.append(
                {
                    "sample_id": sample_ids[episode_index],
                    "source_episode_id": int(data["source_episode_ids"][episode_index]),
                    "current_view": CURRENT_VIEW,
                    "candidate_view": view,
                    "candidate_camera": camera_names[(episode_index, view)],
                    "true_label": int(cls["labels"][episode_index]),
                    "S_geometry": float(components[episode_index, view, 0]),
                    "S_visibility": float(components[episode_index, view, 1]),
                    "S_interaction": float(components[episode_index, view, 2]),
                    "S_uncertainty": float(s_uncertainty[episode_index, view]),
                    "S_uncertainty_future_reduction": float(
                        future_uncertainty_reduction[episode_index, view]
                    ),
                    "S_motion": float(components[episode_index, view, 3]),
                    "HAR_score_before": float(cls["current_margin"][episode_index]),
                    "HAR_score_after_candidate": float(
                        cls["fused_margin"][episode_index, view]
                    ),
                    "HAR_gain": float(cls["utility"][episode_index, view]),
                    "GT_margin_before": float(cls["current_margin"][episode_index]),
                    "GT_margin_after": float(cls["fused_margin"][episode_index, view]),
                    "GT_margin_gain": float(cls["utility"][episode_index, view]),
                    "GT_logit_gain": float(cls["gt_logit_gain"][episode_index, view]),
                    "top1_after": int(cls["correct"][episode_index, view]),
                    "top1_improvement": int(
                        cls["correct"][episode_index, view]
                        and not cls["current_correct"][episode_index]
                    ),
                    "future_body_visibility": float(body_future[episode_index, view]),
                    "future_interaction_clarity": float(
                        interaction_future[episode_index, view]
                    ),
                    "causal_valid": int(info["valid"][episode_index, view]),
                }
            )
    summary = {
        "utility_definition": "fused GT margin minus current-view GT margin",
        "candidate_views": [int(x) for x in CANDIDATE_VIEWS],
        "candidate_rows": int(len(rows)),
        "proxy_note": (
            "S_uncertainty is current-view uncertainty repeated across candidates; "
            "future uncertainty reduction is privileged and reported separately"
        ),
        "arrays": {
            "proxy": {name: components[:, CANDIDATE_VIEWS, index].astype(np.float32)
                       for index, name in enumerate(
                           ["S_geometry", "S_visibility", "S_interaction", "S_motion"]
                       )},
            "S_uncertainty": s_uncertainty[:, CANDIDATE_VIEWS].astype(np.float32),
            "S_uncertainty_future_reduction": future_uncertainty_reduction[:, CANDIDATE_VIEWS].astype(np.float32),
            "utility": cls["utility"][:, CANDIDATE_VIEWS].astype(np.float32),
            "gt_logit_gain": cls["gt_logit_gain"][:, CANDIDATE_VIEWS].astype(np.float32),
            "correct": cls["correct"][:, CANDIDATE_VIEWS].astype(bool),
            "valid": info["valid"][:, CANDIDATE_VIEWS].astype(bool),
            "body_future": body_future[:, CANDIDATE_VIEWS].astype(np.float32),
            "interaction_future": interaction_future[:, CANDIDATE_VIEWS].astype(np.float32),
        },
    }
    return rows, summary


def write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def proxy_correlation_report(proxy_summary: Mapping[str, Any]) -> dict[str, Any]:
    arrays = proxy_summary["arrays"]
    utility = np.asarray(arrays["utility"], dtype=np.float32)
    valid = np.asarray(arrays["valid"], dtype=bool)
    report: dict[str, Any] = {}
    for name in CORRELATION_FIELDS:
        proxy = np.asarray(
            arrays["S_uncertainty"] if name == "S_uncertainty" else arrays["proxy"][name],
            dtype=np.float32,
        )
        report[name] = {
            "pooled": _correlation(proxy[valid], utility[valid]),
            "within_episode": _within_episode_correlation(proxy, utility, valid),
        }
    future_unc = np.asarray(arrays["S_uncertainty_future_reduction"], dtype=np.float32)
    report["S_uncertainty_future_reduction"] = {
        "pooled": _correlation(future_unc[valid], utility[valid]),
        "within_episode": _within_episode_correlation(future_unc, utility, valid),
        "note": "privileged future-observation diagnostic, not a causal policy input",
    }
    report["utility_alternatives"] = {
        "gt_logit_gain_mean": float(np.asarray(arrays["gt_logit_gain"])[valid].mean()),
        "top1_improvement_rate": float(
            np.asarray(arrays["correct"])[valid].mean()
        ),
    }
    return report


def _random_action(valid: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.asarray(
        [rng.choice(np.flatnonzero(row)) for row in np.asarray(valid, dtype=bool)],
        dtype=np.int64,
    )


def _paired_ci(delta: np.ndarray, seed: int, samples: int = 10000) -> list[float]:
    delta = np.asarray(delta, dtype=np.float32)
    if not len(delta):
        return [0.0, 0.0]
    rng = np.random.default_rng(seed)
    means: list[np.ndarray] = []
    for start in range(0, samples, 500):
        count = min(500, samples - start)
        indices = rng.integers(0, len(delta), size=(count, len(delta)))
        means.append(delta[indices].mean(axis=1))
    values = np.concatenate(means)
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def action_metrics(
    data: Mapping[str, Any],
    classes: Sequence[int],
    arrays: Mapping[str, np.ndarray],
    action: np.ndarray,
    name: str,
    seed: int,
) -> dict[str, Any]:
    action = np.asarray(action, dtype=np.int64)
    valid = np.asarray(arrays["valid"], dtype=bool)
    candidate_correct = np.asarray(arrays["candidate_correct"], dtype=bool)
    utility = np.asarray(arrays["utility"], dtype=np.float32)
    if action.shape != (len(action),):
        raise ValueError(f"bad action shape for {name}: {action.shape}")
    rows = np.arange(len(action))
    if not np.all(valid[rows, action]):
        raise ValueError(f"{name} selected an invalid candidate")
    correct = candidate_correct[rows, action]
    expected = np.asarray(arrays["random_expected"], dtype=np.float32)
    delta = correct.astype(np.float32) - expected
    cost = np.asarray(arrays["cost"], dtype=np.float32)[rows, action]
    body = np.asarray(arrays["body_gain"], dtype=np.float32)[rows, action]
    interaction = np.asarray(arrays["interaction_gain"], dtype=np.float32)[rows, action]
    uncertainty = np.asarray(arrays["uncertainty_gain"], dtype=np.float32)[rows, action]
    complementarity = np.asarray(arrays["complementarity"], dtype=np.float32)[rows, action]
    distribution = np.bincount(action, minlength=NUM_VIEWS).astype(np.float64)
    distribution /= max(float(distribution.sum()), 1.0)
    action_entropy = -sum(
        float(p) * math.log(float(p)) for p in distribution if p > 0
    ) / math.log(3.0)
    name_seed = sum((index + 1) * ord(char) for index, char in enumerate(name))
    return {
        "top1": float(expected.mean() if name == "random" else correct.mean()),
        "sampled_top1": float(correct.mean()),
        "delta_vs_random": float(0.0 if name == "random" else delta.mean()),
        "paired_bootstrap_ci95": [0.0, 0.0]
        if name == "random"
        else _paired_ci(delta, seed + name_seed),
        "movement_cost": float(cost.mean()),
        "body_visibility_gain": float(body.mean()),
        "interaction_visibility_gain": float(interaction.mean()),
        "uncertainty_reduction": float(uncertainty.mean()),
        "complementarity": float(complementarity.mean()),
        "action_entropy": float(action_entropy),
        "action_histogram": distribution.tolist(),
        "oracle_regret": float(
            (np.asarray(arrays["oracle_utility"]) - utility[rows, action]).mean()
        ),
    }


def make_eval_arrays(data: Mapping[str, Any], classes: Sequence[int]) -> dict[str, np.ndarray]:
    cls = _classification_arrays(data, classes)
    _, _, info = candidate_inputs(data["raw"], data["episodes"], data["features"])
    body, interaction = weighted_future_scores(data["features"])
    current_body = body[:, CURRENT_VIEW]
    current_interaction = interaction[:, CURRENT_VIEW]
    cost = np.asarray(info["cost"], dtype=np.float32)
    rel = np.asarray(info["relative"], dtype=np.float32)
    complementarity = np.exp(
        -((np.abs(rel) - np.pi / 2.0) / (np.pi / 3.0)) ** 2
    ).astype(np.float32)
    candidate_correct = cls["correct"]
    valid = info["valid"]
    candidate_correct &= valid
    random_expected = candidate_correct[:, CANDIDATE_VIEWS].mean(axis=1).astype(np.float32)
    utility = cls["utility"].copy()
    utility[~valid] = -np.inf
    oracle_utility = np.max(utility[:, CANDIDATE_VIEWS], axis=1)
    return {
        "candidate_correct": candidate_correct,
        "valid": valid,
        "utility": utility,
        "oracle_utility": oracle_utility,
        "random_expected": random_expected,
        "cost": cost,
        "body_gain": (body - current_body[:, None]).astype(np.float32),
        "interaction_gain": (interaction - current_interaction[:, None]).astype(np.float32),
        "uncertainty_gain": (
            cls["current_entropy"][:, None] - cls["future_entropy"]
        ).astype(np.float32),
        "complementarity": complementarity,
        "utility_full": cls["utility"],
        "labels": cls["labels"],
    }


def _select_max(score: np.ndarray, valid: np.ndarray) -> np.ndarray:
    score = np.asarray(score, dtype=np.float32).copy()
    score[~np.asarray(valid, dtype=bool)] = -np.inf
    return score.argmax(axis=-1).astype(np.int64)


def tune_future_combination(
    data: Mapping[str, Any], classes: Sequence[int]
) -> tuple[list[float], dict[str, Any]]:
    arrays = make_eval_arrays(data, classes)
    body = arrays["body_gain"]
    interaction = arrays["interaction_gain"]
    valid = arrays["valid"]
    best: tuple[tuple[float, float], list[float]] | None = None
    grid = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
    for alpha in grid:
        for beta in grid:
            action = _select_max(alpha * body + beta * interaction, valid)
            correct = arrays["candidate_correct"][np.arange(len(action)), action]
            gain = float((correct.astype(np.float32) - arrays["random_expected"]).mean())
            key = (gain, float(correct.mean()))
            if best is None or key > best[0]:
                best = (key, [alpha, beta])
    assert best is not None
    return best[1], {"tuning_split": "pseudo_unseen", "gain": best[0][0], "grid": grid}


def future_oracle_report(
    prepared: Mapping[str, Mapping[str, Any]],
    classes_by_name: Mapping[str, Sequence[int]],
    tuned_weights: Sequence[float],
) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for split_name, data in prepared.items():
        classes = classes_by_name[split_name]
        arrays = make_eval_arrays(data, classes)
        valid = arrays["valid"]
        body_score = arrays["body_gain"]
        interaction_score = arrays["interaction_gain"]
        combined = body_score + interaction_score
        tuned = tuned_weights[0] * body_score + tuned_weights[1] * interaction_score
        actions = {
            "random": _random_action(valid, 6000 + len(split_name)),
            "gt_visibility_oracle": _select_max(body_score, valid),
            "gt_interaction_oracle": _select_max(interaction_score, valid),
            "gt_visibility_interaction_fixed": _select_max(combined, valid),
            "gt_visibility_interaction_tuned": _select_max(tuned, valid),
        }
        # Full HAR oracle: take a correct candidate whenever one exists; if
        # none exists, use the largest true-label margin as a tie fallback.
        candidate_correct = arrays["candidate_correct"]
        margin = arrays["utility"].copy()
        margin[~valid] = -np.inf
        full_oracle = np.empty(len(margin), dtype=np.int64)
        for index in range(len(margin)):
            good = CANDIDATE_VIEWS[candidate_correct[index, CANDIDATE_VIEWS]]
            full_oracle[index] = int(good[0] if len(good) else margin[index].argmax())
        actions["full_har_oracle"] = full_oracle
        metrics = {
            name: action_metrics(data, classes, arrays, action, name, 7000 + len(split_name))
            for name, action in actions.items()
        }
        metrics["full_har_oracle"]["top1"] = float(candidate_correct[:, CANDIDATE_VIEWS].any(axis=1).mean())
        metrics["full_har_oracle"]["selection_is_upper_bound"] = True
        report[split_name] = {
            "episodes": int(len(data["episodes"])),
            "random_expected_top1": float(arrays["random_expected"].mean()),
            "full_har_oracle_top1": float(candidate_correct[:, CANDIDATE_VIEWS].any(axis=1).mean()),
            "methods": metrics,
            "oracle_actions": {
                name: action.tolist() for name, action in actions.items()
            },
            "arrays": arrays,
        }
    return report


def recoverable_report(
    data: Mapping[str, Any],
    classes: Sequence[int],
    experiment_root: Path,
    device: torch.device,
    handcrafted_weights: Sequence[float],
) -> dict[str, Any]:
    arrays = make_eval_arrays(data, classes)
    candidate_correct = arrays["candidate_correct"][:, CANDIDATE_VIEWS]
    utility = arrays["utility"][:, CANDIDATE_VIEWS]
    utility_range = np.ptp(utility, axis=1)
    all_correct = candidate_correct.all(axis=1)
    all_wrong = ~candidate_correct.any(axis=1)
    view_insensitive = all_correct | (utility_range <= 0.05)
    recoverable = (~view_insensitive) & (~all_correct) & (~all_wrong)
    unrecoverable = (~view_insensitive) & all_wrong
    residual = ~(view_insensitive | recoverable | unrecoverable)
    if residual.any():
        raise AssertionError("recoverable categories do not partition the data")
    full_oracle = np.empty(len(candidate_correct), dtype=np.int64)
    for index in range(len(full_oracle)):
        good = CANDIDATE_VIEWS[candidate_correct[index]]
        full_oracle[index] = int(good[0] if len(good) else utility[index].argmax())

    valid = arrays["valid"]
    action_map: dict[str, np.ndarray] = {
        "random": _random_action(valid, 8123),
        "handcrafted_fixed": handcrafted_actions(
            handcrafted_components(dict(data)), [1.0, 1.0, 1.0, 0.20], valid
        ),
        "handcrafted_tuned": handcrafted_actions(
            handcrafted_components(dict(data)),
            handcrafted_weights,
            valid,
        ),
        "oracle": full_oracle,
    }
    checkpoint_dir = experiment_root
    old_checkpoint = (
        checkpoint_dir.parent
        / "active_view_candidate_rank_v6_new50_5"
        / "seed_20260911"
        / "best.pt"
    )
    if old_checkpoint.exists():
        action_map["old_rl_legacy"] = load_old_actions(data, classes, old_checkpoint, device)
    for variant in ["structured_rl", "structured_rl_har_aux", "structured_rl_large"]:
        for seed in [20260909, 20260910, 20260911]:
            checkpoint = experiment_root / variant / f"seed_{seed}" / "best.pt"
            if not checkpoint.exists():
                continue
            saved = torch.load(checkpoint, map_location=device, weights_only=False)
            model = StructuredScorer(hidden=int(saved.get("hidden", 128))).to(device)
            model.load_state_dict(saved["model"])
            context, candidate, info = candidate_inputs(
                data["raw"], data["episodes"], data["features"]
            )
            action_map[f"{variant}_seed_{seed}"] = action_from_model(
                model, context, candidate, info["valid"], device
            )
            del model
    if not recoverable.any():
        return {
            "episodes": int(len(candidate_correct)),
            "categories": {
                "view_insensitive": int(view_insensitive.sum()),
                "unrecoverable": int(unrecoverable.sum()),
                "recoverable": 0,
            },
            "recoverable_methods": {},
        }
    subset_arrays = dict(arrays)
    subset_arrays["candidate_correct"] = arrays["candidate_correct"][recoverable]
    subset_arrays["valid"] = arrays["valid"][recoverable]
    subset_arrays["utility"] = arrays["utility"][recoverable]
    subset_arrays["oracle_utility"] = arrays["oracle_utility"][recoverable]
    for key in ["random_expected", "cost", "body_gain", "interaction_gain", "uncertainty_gain", "complementarity"]:
        subset_arrays[key] = arrays[key][recoverable]
    method_report = {
        name: action_metrics(
            data,
            classes,
            subset_arrays,
            action[recoverable],
            name,
            8200,
        )
        for name, action in action_map.items()
    }
    return {
        "episodes": int(len(candidate_correct)),
        "categories": {
            "view_insensitive": int(view_insensitive.sum()),
            "view_insensitive_fraction": float(view_insensitive.mean()),
            "unrecoverable": int(unrecoverable.sum()),
            "unrecoverable_fraction": float(unrecoverable.mean()),
            "recoverable": int(recoverable.sum()),
            "recoverable_fraction": float(recoverable.mean()),
        },
        "diagnostic_breakdown": {
            "all_candidates_correct": int(all_correct.sum()),
            "all_candidates_wrong": int(all_wrong.sum()),
            "utility_tie_threshold": 0.05,
        },
        "recoverable_methods": method_report,
        "recoverable_indices": np.flatnonzero(recoverable).astype(int).tolist(),
    }


def pseudo_zsl_audit(raw_root: Path, classes_by_name: Mapping[str, Sequence[int]]) -> dict[str, Any]:
    cache_meta = {
        split: json.loads((raw_root / split / "metadata.json").read_text(encoding="utf-8"))
        for split in ["dqn_train", "val", "seen_test", "test"]
    }
    current_pseudo = set(int(x) for x in classes_by_name["pseudo_unseen"])
    # ``test/coverage.classes`` contains only the test classes, not the
    # recognizer's training pool.  The complete seen_test cache is the
    # authoritative 50-class coverage for this old VPOCLIP checkpoint.
    backbone_seen = set(
        int(x) for x in cache_meta["seen_test"].get("coverage", {}).get("classes", [])
    )
    true_unseen = set(int(x) for x in classes_by_name["true_unseen"])
    # The cache coverage classes list the recognizer's 50 seen classes.  The
    # current pseudo pool is a selector holdout only, not a recognizer holdout.
    missing_from_backbone = sorted(current_pseudo - backbone_seen)
    candidate_true_zsl = sorted(backbone_seen - current_pseudo)
    # Do not propose any class that is already in the recognizer's seen pool.
    valid_unseen_pool = sorted(set(range(55)) - backbone_seen)
    return {
        "recognizer": cache_meta["test"].get("recognizer"),
        "cache_protocol": cache_meta["test"].get("coverage"),
        "current_pseudo_classes": sorted(current_pseudo),
        "current_pseudo_in_backbone_seen": sorted(current_pseudo & backbone_seen),
        "current_pseudo_missing_from_backbone": missing_from_backbone,
        "true_unseen_classes": sorted(true_unseen),
        "classes_absent_from_backbone_in_0_to_54": valid_unseen_pool,
        "strict_pseudo_zsl_available_without_vpoclip_retraining": False,
        "reason": (
            "The current 10-class pseudo split is present in the VPOCLIP seen "
            "coverage; only the five true-unseen classes are absent. Creating "
            "a new 10-class recognizer holdout requires retraining VPOCLIP and "
            "is intentionally not done in this diagnostic run."
        ),
        "backbone_seen_classes_source": "seen_test/coverage.classes",
        "split_metadata_consistency": {
            split: {
                "coverage_classes": cache_meta[split].get("coverage", {}).get("classes", []),
                "recognizer_checkpoint": cache_meta[split].get("recognizer", {}).get("checkpoint"),
            }
            for split in cache_meta
        },
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=Path("data/rl_candidate_rank_v6_new50_5/cache"))
    parser.add_argument("--structured-experiment", type=Path, default=Path("work_dir/structured_vs_handcrafted_v1"))
    parser.add_argument("--output", type=Path, default=Path("work_dir/structured_viewpoint_diagnostics_v1"))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    raw_root = args.raw_root if args.raw_root.is_absolute() else root / args.raw_root
    experiment_root = args.structured_experiment if args.structured_experiment.is_absolute() else root / args.structured_experiment
    output = args.output if args.output.is_absolute() else root / args.output
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    seed_all(20260910)
    classes_by_name = {
        "seen": SEEN_CLASSES,
        "pseudo_unseen": PSEUDO_CLASSES,
        "true_unseen": UNSEEN_CLASSES,
    }
    split_by_name = {
        "seen": ("seen_test", SEEN_CLASSES),
        "pseudo_unseen": ("val", PSEUDO_CLASSES),
        "true_unseen": ("test", UNSEEN_CLASSES),
    }
    print(json.dumps({
        "DIAGNOSTIC_START": True,
        "raw_root": str(raw_root),
        "output": str(output),
        "device": str(device),
        "no_new_policy_training": True,
        "protocol": "fixed view0, physical candidates 1/2/3, one move, terminal",
    }), flush=True)
    prepared: dict[str, dict[str, Any]] = {}
    for name, (split, classes) in split_by_name.items():
        prepared[name] = prepare_data(raw_root, split, classes)
    split_stats = {
        name: {
            "split": split_by_name[name][0],
            "episodes": int(len(data["episodes"])),
            "source_episode_ids": int(len(data["source_episode_ids"])),
            "classes": [int(x) for x in split_by_name[name][1]],
        }
        for name, data in prepared.items()
    }
    (output / "split_stats.json").write_text(json.dumps(split_stats, indent=2), encoding="utf-8")

    proxy_reports: dict[str, Any] = {}
    proxy_arrays: dict[str, Any] = {}
    for name, data in prepared.items():
        rows, proxy_summary = build_proxy_rows(data, classes_by_name[name])
        write_rows(output / f"proxy_utility_{name}.csv", rows)
        proxy_reports[name] = proxy_correlation_report(proxy_summary)
        proxy_arrays[name] = proxy_summary
        np.savez_compressed(
            output / f"proxy_utility_{name}.npz",
            **{
                key: value
                for key, value in proxy_summary["arrays"].items()
                if isinstance(value, np.ndarray)
            },
        )

    tuned_weights, tuning_info = tune_future_combination(
        prepared["pseudo_unseen"], PSEUDO_CLASSES
    )
    handcrafted_tuning_path = experiment_root / "handcrafted_tuning.json"
    if handcrafted_tuning_path.exists():
        handcrafted_weights = json.loads(
            handcrafted_tuning_path.read_text(encoding="utf-8")
        )["weights"]
    else:
        handcrafted_weights = [1.0, 1.0, 1.0, 0.20]
    future = future_oracle_report(prepared, classes_by_name, tuned_weights)
    future_json = _json_safe(future)
    (output / "future_proxy_ceiling.json").write_text(
        json.dumps(future_json, indent=2), encoding="utf-8"
    )

    recoverable = recoverable_report(
        prepared["true_unseen"],
        UNSEEN_CLASSES,
        experiment_root,
        device,
        handcrafted_weights,
    )
    (output / "recoverable_true_unseen.json").write_text(
        json.dumps(_json_safe(recoverable), indent=2), encoding="utf-8"
    )
    pseudo_audit = pseudo_zsl_audit(raw_root, classes_by_name)
    (output / "pseudo_zsl_audit.json").write_text(
        json.dumps(_json_safe(pseudo_audit), indent=2), encoding="utf-8"
    )
    summary = {
        "diagnostic_version": "v1",
        "protocol": "fixed view0, physical candidates 1/2/3, one move, terminal",
        "recognizer_cache": str(raw_root),
        "structured_experiment": str(experiment_root),
        "no_new_policy_training": True,
        "visibility_definition": (
            "2-D pose presence plus image-edge proxy; object aggregate map; "
            "no per-joint confidence in cache"
        ),
        "split_stats": split_stats,
        "diagnostic_a_proxy_utility_correlation": proxy_reports,
        "diagnostic_b_future_proxy_ceiling": {
            "tuned_weights_from_pseudo_only": tuned_weights,
            "tuning_info": tuning_info,
            "results": future_json,
        },
        "handcrafted_weights_from_structured_experiment": handcrafted_weights,
        "diagnostic_c_recoverable_subset": _json_safe(recoverable),
        "diagnostic_d_pseudo_zsl_audit": _json_safe(pseudo_audit),
        "decision_rule": {
            "proxy_near_zero_and_future_oracle_near_random": "stop visibility/interaction selector",
            "proxy_near_zero_but_future_oracle_above_random": "study future visibility/interaction forecasting, not PPO first",
            "proxy_positive_but_policy_near_random": "then inspect policy optimization",
            "recoverable_subset_small": "consider need-to-move detector before candidate ranking",
        },
    }
    (output / "summary.json").write_text(
        json.dumps(_json_safe(summary), indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "STRUCTURED_DIAGNOSTICS_COMPLETE": True,
        "output": str(output),
        "split_stats": split_stats,
        "tuned_future_weights": tuned_weights,
        "recoverable_categories": recoverable.get("categories", {}),
        "pseudo_zsl_strict_available": pseudo_audit["strict_pseudo_zsl_available_without_vpoclip_retraining"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
