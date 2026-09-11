"""Train one seen-class Need-to-Move binary classifier and test on unseen.

The label is created by comparing the same recording's fixed current view 0
with its three legal destination views.  This is an offline supervision label
only; the deployed classifier receives current-view causal features and never
receives future candidate logits/features or the class label.

For a fair end-to-end test, ``move=0`` keeps view 0 and ``move=1`` invokes a
destination policy.  We report random, fixed, and the existing structured
destination selector so the binary gate is not confused with the where-to-
move policy.
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
from torch import nn

from .diagnose_structured_viewpoint import _paired_ci
from .need_to_move_diagnostics import (
    _classification_metrics,
    _current_state_features,
    _stratified_group_folds,
)
from .structured_sa_experiment import (
    CURRENT_VIEW,
    NUM_VIEWS,
    PSEUDO_CLASSES,
    SEEN_CLASSES,
    StructuredScorer,
    UNSEEN_CLASSES,
    action_from_model,
    candidate_inputs,
    load_old_actions,
    margin_scores,
    prepare_data,
)


CANDIDATE_VIEWS = np.asarray([1, 2, 3], dtype=np.int64)
MOVE_MARGIN_EPS = 0.05


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def _labels_from_same_action_views(
    data: Mapping[str, Any], classes: Sequence[int]
) -> dict[str, np.ndarray]:
    """Create move labels from same-recording view comparisons.

    ``move=1`` means at least one legal candidate gives a meaningful positive
    GT-margin gain over view 0, or changes a wrong current prediction into a
    correct fused prediction.  A candidate that is merely different but
    worse is not considered worth moving to.
    """

    raw = data["raw"]
    episodes = np.asarray(data["episodes"], dtype=np.int64)
    labels = np.asarray(raw["labels"])[episodes]
    logits = np.asarray(raw["logits"][episodes], dtype=np.float32)
    current = logits[:, CURRENT_VIEW]
    fused = 0.5 * (current[:, None, :] + logits)
    bank = np.asarray(sorted({int(x) for x in classes}), dtype=np.int64)
    current_scores = current[..., bank]
    fused_scores = fused[..., bank]
    local = np.searchsorted(bank, labels)
    if np.any(local >= len(bank)) or np.any(bank[np.minimum(local, len(bank) - 1)] != labels):
        raise ValueError("labels outside requested class bank")
    current_correct = current_scores.argmax(-1) == local
    fused_correct = fused_scores.argmax(-1) == local[:, None]
    current_margin = margin_scores(current, labels, classes)
    fused_margin = margin_scores(fused, labels[:, None], classes)
    gain = fused_margin[:, CANDIDATE_VIEWS] - current_margin[:, None]
    candidate_correct = fused_correct[:, CANDIDATE_VIEWS]
    recoverable = (~current_correct) & candidate_correct.any(axis=1)
    move = (gain.max(axis=1) > MOVE_MARGIN_EPS) | recoverable
    return {
        "labels": labels.astype(np.int64),
        "current_correct": current_correct.astype(bool),
        "candidate_correct": candidate_correct.astype(bool),
        "current_margin": current_margin.astype(np.float32),
        "candidate_margin_gain": gain.astype(np.float32),
        "best_margin_gain": gain.max(axis=1).astype(np.float32),
        "move": move.astype(np.int64),
        "recoverable": recoverable.astype(bool),
    }


def _fit_one_model(
    x: np.ndarray,
    y: np.ndarray,
    train_index: np.ndarray,
    val_index: np.ndarray,
    device: torch.device,
    seed: int,
    epochs: int = 400,
) -> tuple[MoveMLP, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    seed_all(seed)
    mean = x[train_index].mean(axis=0).astype(np.float32)
    std = np.maximum(x[train_index].std(axis=0), 1e-5).astype(np.float32)
    x_train = ((x[train_index] - mean) / std).astype(np.float32)
    x_val = ((x[val_index] - mean) / std).astype(np.float32)
    train_x = torch.from_numpy(x_train).to(device)
    train_y = torch.from_numpy(y[train_index].astype(np.float32)).to(device)
    val_x = torch.from_numpy(x_val).to(device)
    positive = max(float(train_y.sum().item()), 1.0)
    negative = max(float(len(train_y) - train_y.sum().item()), 1.0)
    model = MoveMLP(x.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(negative / positive, device=device)
    )
    for _ in range(int(epochs)):
        model.train()
        loss = loss_fn(model(train_x), train_y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.inference_mode():
        train_probability = torch.sigmoid(model(train_x)).cpu().numpy()
        val_probability = torch.sigmoid(model(val_x)).cpu().numpy()
    return model, train_probability, val_probability, {"mean": mean, "std": std}


def _predict_model(
    model: MoveMLP,
    data: Mapping[str, Any],
    normalization: Mapping[str, np.ndarray],
    device: torch.device,
) -> np.ndarray:
    x, _ = _current_state_features(data)
    x = ((x - normalization["mean"]) / normalization["std"]).astype(np.float32)
    model.eval()
    with torch.inference_mode():
        return torch.sigmoid(model(torch.from_numpy(x).to(device))).cpu().numpy()


def _random_action(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(1, 4, size=n, dtype=np.int64)


def _load_structured_destination(
    data: Mapping[str, Any], checkpoint: Path, device: torch.device
) -> np.ndarray:
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    model = StructuredScorer(hidden=int(saved.get("hidden", 128))).to(device)
    model.load_state_dict(saved["model"])
    context, candidate, info = candidate_inputs(
        data["raw"], data["episodes"], data["features"]
    )
    action = action_from_model(model, context, candidate, info["valid"], device)
    del model
    return action


def _evaluate_protocols(
    data: Mapping[str, Any],
    classes: Sequence[int],
    move_probability: np.ndarray,
    threshold: float,
    destination_checkpoint: Path | None,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    labels = _labels_from_same_action_views(data, classes)
    raw = data["raw"]
    episodes = np.asarray(data["episodes"], dtype=np.int64)
    logits = np.asarray(raw["logits"][episodes], dtype=np.float32)
    current = logits[:, CURRENT_VIEW]
    fused = 0.5 * (current[:, None, :] + logits)
    bank = np.asarray(sorted({int(x) for x in classes}), dtype=np.int64)
    local = np.searchsorted(bank, labels["labels"])
    current_correct = current[..., bank].argmax(-1) == local
    candidate_correct = fused[..., bank].argmax(-1) == local[:, None]
    candidate_correct = candidate_correct[:, CANDIDATE_VIEWS]
    move_pred = np.asarray(move_probability >= threshold, dtype=bool)
    random_destination = _random_action(len(episodes), seed)
    fixed_destination = np.ones(len(episodes), dtype=np.int64)
    structured_destination = None
    if destination_checkpoint is not None and destination_checkpoint.exists():
        structured_destination = _load_structured_destination(
            data, destination_checkpoint, device
        )

    actions: dict[str, tuple[np.ndarray, np.ndarray]] = {
        # Action value 0 means no movement and therefore single-view view 0.
        "single": (np.zeros(len(episodes), dtype=np.int64), np.zeros(len(episodes), dtype=bool)),
        "always_random": (random_destination, np.ones(len(episodes), dtype=bool)),
        "always_fixed": (fixed_destination, np.ones(len(episodes), dtype=bool)),
        "gate_random": (random_destination, move_pred),
        "gate_fixed": (fixed_destination, move_pred),
    }
    if structured_destination is not None:
        actions["gate_structured"] = (structured_destination, move_pred)

    # Diagnostic oracle: gate with the offline label and select the best
    # candidate by true-label margin.  This never affects model training.
    candidate_gain = labels["candidate_margin_gain"]
    oracle_destination = CANDIDATE_VIEWS[candidate_gain.argmax(axis=1)]
    actions["oracle_gate"] = (oracle_destination, labels["move"].astype(bool))

    result: dict[str, Any] = {
        "episodes": int(len(episodes)),
        "move_label_positive_rate": float(labels["move"].mean()),
        "predicted_move_rate": float(move_pred.mean()),
        "methods": {},
    }
    for name, (destination, should_move) in actions.items():
        rows = np.arange(len(episodes))
        chosen = np.where(should_move, destination, CURRENT_VIEW)
        correct = np.where(
            should_move,
            candidate_correct[rows, destination - 1],
            current_correct,
        )
        # destination - 1 maps physical view 1/2/3 to local columns 0/1/2.
        cost = np.asarray(raw["cost"][episodes, CURRENT_VIEW], dtype=np.float32)
        movement_cost = np.where(
            should_move, cost[rows, destination], 0.0
        )
        gain = correct.astype(np.float32) - current_correct.astype(np.float32)
        result["methods"][name] = {
            "top1": float(correct.mean()),
            "delta_vs_single": float(gain.mean()),
            "paired_bootstrap_ci95_vs_single": _paired_ci(gain, seed + len(name)),
            "move_rate": float(should_move.mean()),
            "movement_cost": float(movement_cost.mean()),
            "selected_physical_views": np.bincount(chosen, minlength=NUM_VIEWS).tolist(),
        }
    result["single_top1"] = float(current_correct.mean())
    result["label_diagnostics"] = {
        "move_positive": int(labels["move"].sum()),
        "move_negative": int((1 - labels["move"]).sum()),
        "recoverable_positive": int(labels["recoverable"].sum()),
        "best_margin_gain_mean": float(labels["best_margin_gain"].mean()),
        "margin_gain_threshold": MOVE_MARGIN_EPS,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=Path("data/rl_candidate_rank_v6_new50_5/cache"))
    parser.add_argument("--output", type=Path, default=Path("work_dir/need_to_move_seen_binary_v1"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument(
        "--destination-checkpoint",
        type=Path,
        default=Path("work_dir/structured_vs_handcrafted_v1/structured_rl/seed_20260909/best.pt"),
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    raw_root = args.raw_root if args.raw_root.is_absolute() else root / args.raw_root
    output = args.output if args.output.is_absolute() else root / args.output
    destination_checkpoint = (
        args.destination_checkpoint
        if args.destination_checkpoint.is_absolute()
        else root / args.destination_checkpoint
    )
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    seed_all(args.seed)
    print(json.dumps({
        "NEED_TO_MOVE_SEEN_BINARY_START": True,
        "raw_root": str(raw_root), "output": str(output), "device": str(device),
        "label": "seen same-action view0 vs candidate fused GT-margin comparison",
        "future_candidate_inputs_to_binary_model": False,
        "protocol": "train dqn_train seen, held-out subject validation, test seen_test and true unseen",
    }), flush=True)

    train_data = prepare_data(raw_root, "dqn_train", SEEN_CLASSES)
    seen_eval = prepare_data(raw_root, "seen_test", SEEN_CLASSES)
    unseen_eval = prepare_data(raw_root, "test", UNSEEN_CLASSES)
    train_labels = _labels_from_same_action_views(train_data, SEEN_CLASSES)
    x, feature_info = _current_state_features(train_data)
    subjects = np.asarray([
        str(train_data["raw"]["metadata"]["episodes"][int(source)].get("subject", "unknown"))
        for source in train_data["source_episode_ids"]
    ])
    folds = _stratified_group_folds(
        train_labels["move"], subjects, n_splits=5, seed=args.seed
    )
    train_index, val_index = folds[0]
    model, train_prob, val_prob, normalization = _fit_one_model(
        x, train_labels["move"], train_index, val_index, device, args.seed
    )
    threshold = 0.5
    train_metrics = _classification_metrics(
        train_labels["move"][train_index], train_prob, threshold
    )
    val_metrics = _classification_metrics(
        train_labels["move"][val_index], val_prob, threshold
    )
    checkpoint = output / "need_to_move_binary.pt"
    torch.save({
        "model": model.state_dict(), "width": int(x.shape[-1]),
        "normalization": normalization, "threshold": threshold,
        "seed": args.seed, "label_definition": "best GT margin gain > 0.05 or recoverable correction",
    }, checkpoint)
    (output / "train_metrics.json").write_text(
        json.dumps(_json_safe({
            "episodes": int(len(x)), "train_subjects": sorted(set(subjects[train_index].tolist())),
            "validation_subjects": sorted(set(subjects[val_index].tolist())),
            "train_label_counts": np.bincount(train_labels["move"][train_index], minlength=2),
            "validation_label_counts": np.bincount(train_labels["move"][val_index], minlength=2),
            "train_metrics": train_metrics, "validation_metrics": val_metrics,
            "feature_info": feature_info, "threshold": threshold,
        }), indent=2), encoding="utf-8"
    )
    seen_probability = _predict_model(model, seen_eval, normalization, device)
    unseen_probability = _predict_model(model, unseen_eval, normalization, device)
    seen_labels = _labels_from_same_action_views(seen_eval, SEEN_CLASSES)
    unseen_labels = _labels_from_same_action_views(unseen_eval, UNSEEN_CLASSES)
    seen_binary_metrics = _classification_metrics(seen_labels["move"], seen_probability, threshold)
    unseen_binary_metrics = _classification_metrics(unseen_labels["move"], unseen_probability, threshold)
    results = {
        "seen_test": _evaluate_protocols(
            seen_eval, SEEN_CLASSES, seen_probability, threshold,
            destination_checkpoint, device, 33000,
        ),
        "true_unseen": _evaluate_protocols(
            unseen_eval, UNSEEN_CLASSES, unseen_probability, threshold,
            destination_checkpoint, device, 34000,
        ),
    }
    summary = {
        "version": "need_to_move_seen_binary_v1",
        "checkpoint": str(checkpoint),
        "trained_on": "dqn_train seen classes",
        "tested_on": ["seen_test", "true_unseen"],
        "classes": {"seen": SEEN_CLASSES, "true_unseen": UNSEEN_CLASSES},
        "label_definition": (
            "move=1 if max(candidate fused GT-margin gain over view0) > 0.05 "
            "or view0 is wrong and some candidate correct; else move=0"
        ),
        "causal_input_only": True,
        "binary_model_metrics": {
            "train_subject_holdout": {"train": train_metrics, "validation": val_metrics},
            "seen_test": seen_binary_metrics,
            "true_unseen": unseen_binary_metrics,
        },
        "results": results,
        "interpretation": {
            "single": "do not move; classify view0 only",
            "always_random": "always move to random candidate and fuse view0+candidate",
            "gate_*": "move only when binary classifier predicts move=1",
            "oracle_gate": "diagnostic upper bound using offline move label and GT destination ranking",
        },
    }
    (output / "summary.json").write_text(
        json.dumps(_json_safe(summary), indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "NEED_TO_MOVE_SEEN_BINARY_COMPLETE": True,
        "output": str(output),
        "seen_validation": val_metrics,
        "true_unseen_binary": unseen_binary_metrics,
        "true_unseen_protocols": results["true_unseen"]["methods"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
