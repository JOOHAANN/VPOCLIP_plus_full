"""VoI and view-sensitivity experiment.

This is the implementation of ``Next_Stage_VoI_and_ViewSensitivity.md``.
It deliberately keeps the old RL experiments untouched and uses only causal
current-view inputs for prediction.  Future candidate logits and the GT label
are used only to create offline targets and to report oracle diagnostics.

The current cache contains clip-level VPOCLIP logits, one static pose/object
summary per view, image quality, and geometry.  It does *not* contain
per-segment logits, branch-specific RGB/skeleton/object logits, or keypoint
confidence.  The missing temporal/cross-modal fields are therefore reported
as unavailable instead of being fabricated.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from torch import nn

from .diagnose_structured_viewpoint import _paired_ci
from .need_to_move_diagnostics import _current_state_features, _stratified_group_folds
from .structured_sa_experiment import (
    CANDIDATE_DIM,
    CURRENT_VIEW,
    NUM_VIEWS,
    SEEN_CLASSES,
    UNSEEN_CLASSES,
    StructuredScorer,
    candidate_inputs,
    margin_scores,
    prepare_data,
)


CANDIDATE_VIEWS = np.asarray([1, 2, 3], dtype=np.int64)
SENSITIVITY_QUANTILE = 0.70
MLP_EPOCHS = 300
RANKER_EPOCHS = 140


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def causal_features(data: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Return current-view-only features.

    ``_current_state_features`` contains current recognition confidence,
    structured pose/object proxies, image quality, and current camera
    geometry.  No future candidate information is concatenated here.
    """

    x, info = _current_state_features(data)
    info = dict(info)
    info["feature_dim"] = int(x.shape[-1])
    info["source"] = "current_view_only_cached_features"
    return np.asarray(x, dtype=np.float32), info


def compute_targets(
    data: Mapping[str, Any], classes: Sequence[int]
) -> dict[str, np.ndarray]:
    """Compute offline utility, VoI, and view-sensitivity targets."""

    raw = data["raw"]
    n = len(data["episodes"])
    logits = np.asarray(raw["logits"][:n], dtype=np.float32)
    labels = np.asarray(raw["labels"][:n], dtype=np.int64)
    current = logits[:, CURRENT_VIEW]
    fused = 0.5 * (current[:, None, :] + logits)
    current_margin = margin_scores(current, labels, classes)
    fused_margin = margin_scores(fused, labels[:, None], classes)
    utility = fused_margin - current_margin[:, None]
    _, _, action_info = candidate_inputs(raw, np.arange(n), data["features"])
    valid = np.asarray(action_info["valid"], dtype=bool)
    valid[:, CURRENT_VIEW] = False
    candidate_utility = utility[:, CANDIDATE_VIEWS].astype(np.float32)
    candidate_valid = valid[:, CANDIDATE_VIEWS]
    masked = np.where(candidate_valid, candidate_utility, np.nan)
    if np.any(np.sum(candidate_valid, axis=1) == 0):
        raise ValueError("an episode has no valid candidate viewpoint")
    best = np.nanmax(masked, axis=1)
    worst = np.nanmin(masked, axis=1)
    random_mean = np.nanmean(masked, axis=1)
    # Current view is the baseline, so its incremental utility is zero.
    voi_current = best
    voi_random = best - random_mean
    sensitivity = best - worst
    best_action = CANDIDATE_VIEWS[np.nanargmax(masked, axis=1)]
    return {
        "utility": utility.astype(np.float32),
        "candidate_utility": candidate_utility,
        "candidate_valid": candidate_valid,
        "voi_current": voi_current.astype(np.float32),
        "voi_random": voi_random.astype(np.float32),
        "view_sensitivity": sensitivity.astype(np.float32),
        "best_action": best_action.astype(np.int64),
        "current_correct": (
            current[..., np.asarray(sorted(set(int(x) for x in classes)))].argmax(-1)
            == np.searchsorted(np.asarray(sorted(set(int(x) for x in classes))), labels)
        ),
    }


def standardize_fit(x: np.ndarray, index: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x[index].mean(axis=0).astype(np.float32)
    std = np.maximum(x[index].std(axis=0), 1e-5).astype(np.float32)
    return ((x - mean) / std).astype(np.float32), mean, std


class RidgeRegressor:
    def __init__(self, alpha: float = 10.0) -> None:
        self.alpha = float(alpha)
        self.x_mean: np.ndarray | None = None
        self.x_std: np.ndarray | None = None
        self.y_mean = 0.0
        self.y_std = 1.0
        self.coef: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RidgeRegressor":
        self.x_mean = x.mean(axis=0).astype(np.float32)
        self.x_std = np.maximum(x.std(axis=0), 1e-5).astype(np.float32)
        xs = ((x - self.x_mean) / self.x_std).astype(np.float64)
        self.y_mean = float(y.mean())
        self.y_std = max(float(y.std()), 1e-5)
        ys = (y - self.y_mean) / self.y_std
        design = np.concatenate([xs, np.ones((len(xs), 1), dtype=np.float64)], axis=1)
        reg = np.eye(design.shape[1], dtype=np.float64) * self.alpha
        reg[-1, -1] = 0.0
        self.coef = np.linalg.solve(design.T @ design + reg, design.T @ ys)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.coef is None or self.x_mean is None or self.x_std is None:
            raise RuntimeError("RidgeRegressor is not fitted")
        xs = ((x - self.x_mean) / self.x_std).astype(np.float64)
        design = np.concatenate([xs, np.ones((len(xs), 1), dtype=np.float64)], axis=1)
        return (design @ self.coef * self.y_std + self.y_mean).astype(np.float32)


class TinyRandomForest:
    """Dependency-free randomized regression forest for the cached features."""

    def __init__(self, n_estimators: int = 32, max_depth: int = 7,
                 min_leaf: int = 20, max_features: int = 7, seed: int = 0) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_leaf = min_leaf
        self.max_features = max_features
        self.rng = np.random.default_rng(seed)
        self.trees: list[Any] = []

    def _leaf(self, y: np.ndarray) -> tuple[str, float]:
        return ("leaf", float(y.mean()) if len(y) else 0.0)

    def _build(self, x: np.ndarray, y: np.ndarray, index: np.ndarray, depth: int) -> Any:
        yi = y[index]
        if (depth >= self.max_depth or len(index) < 2 * self.min_leaf
                or np.var(yi) < 1e-10):
            return self._leaf(yi)
        feature_count = min(self.max_features, x.shape[1])
        features = self.rng.choice(x.shape[1], size=feature_count, replace=False)
        best: tuple[float, int, float, np.ndarray, np.ndarray] | None = None
        parent_sum = float(yi.sum())
        parent_sq = float((yi * yi).sum())
        for feature in features:
            values = x[index, int(feature)]
            finite = np.isfinite(values)
            if not np.any(finite) or float(values[finite].min()) == float(values[finite].max()):
                continue
            low = float(values[finite].min())
            high = float(values[finite].max())
            thresholds = self.rng.uniform(low, high, size=12)
            for threshold in thresholds:
                left = index[values <= threshold]
                right = index[values > threshold]
                if len(left) < self.min_leaf or len(right) < self.min_leaf:
                    continue
                left_y, right_y = y[left], y[right]
                left_sse = float((left_y * left_y).sum()) - float(left_y.sum()) ** 2 / len(left_y)
                right_sse = float((right_y * right_y).sum()) - float(right_y.sum()) ** 2 / len(right_y)
                score = left_sse + right_sse
                if best is None or score < best[0]:
                    best = (score, int(feature), float(threshold), left, right)
        if best is None:
            return self._leaf(yi)
        _, feature, threshold, left, right = best
        return (
            "node", feature, threshold,
            self._build(x, y, left, depth + 1),
            self._build(x, y, right, depth + 1),
        )

    def fit(self, x: np.ndarray, y: np.ndarray) -> "TinyRandomForest":
        self.trees = []
        n = len(x)
        for _ in range(self.n_estimators):
            bootstrap = self.rng.integers(0, n, size=n, dtype=np.int64)
            self.trees.append(self._build(x, y, bootstrap, 0))
        return self

    def _predict_one(self, tree: Any, row: np.ndarray) -> float:
        while tree[0] == "node":
            tree = tree[3] if row[tree[1]] <= tree[2] else tree[4]
        return float(tree[1])

    def predict(self, x: np.ndarray) -> np.ndarray:
        values = np.asarray(
            [[self._predict_one(tree, row) for tree in self.trees] for row in x],
            dtype=np.float32,
        )
        return values.mean(axis=1)


class VoIMLP(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(width, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def fit_mlp_predict(
    x_train: np.ndarray, y_train: np.ndarray, x_query: np.ndarray,
    device: torch.device, seed: int, epochs: int = MLP_EPOCHS,
) -> tuple[VoIMLP, np.ndarray, dict[str, np.ndarray]]:
    seed_all(seed)
    mean = x_train.mean(axis=0).astype(np.float32)
    std = np.maximum(x_train.std(axis=0), 1e-5).astype(np.float32)
    y_mean = float(y_train.mean())
    y_std = max(float(y_train.std()), 1e-5)
    tx = torch.from_numpy(((x_train - mean) / std).astype(np.float32)).to(device)
    ty = torch.from_numpy(((y_train - y_mean) / y_std).astype(np.float32)).to(device)
    model = VoIMLP(x_train.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)
    loss_fn = nn.SmoothL1Loss()
    for _ in range(int(epochs)):
        loss = loss_fn(model(tx), ty)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    qx = torch.from_numpy(((x_query - mean) / std).astype(np.float32)).to(device)
    model.eval()
    with torch.inference_mode():
        prediction = (model(qx).cpu().numpy() * y_std + y_mean).astype(np.float32)
    return model, prediction, {"mean": mean, "std": std, "y_mean": np.asarray(y_mean), "y_std": np.asarray(y_std)}


def scalar_metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    finite = np.isfinite(y) & np.isfinite(prediction)
    y, prediction = y[finite], prediction[finite]
    ss_res = float(np.sum((y - prediction) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    result: dict[str, Any] = {
        "n": int(len(y)),
        "r2": float(1.0 - ss_res / max(ss_tot, 1e-12)),
        "target_mean": float(y.mean()),
        "prediction_mean": float(prediction.mean()),
        "target_std": float(y.std()),
        "prediction_std": float(prediction.std()),
    }
    if len(y) > 1 and y.std() > 1e-12 and prediction.std() > 1e-12:
        result["pearson"] = float(pearsonr(y, prediction).statistic)
        result["spearman"] = float(spearmanr(y, prediction).statistic)
    else:
        result["pearson"] = None
        result["spearman"] = None
    order = np.argsort(-prediction)
    all_mean = float(y.mean())
    result["top_fraction"] = {}
    for fraction in (0.10, 0.20, 0.30):
        count = max(1, int(math.ceil(len(y) * fraction)))
        top_mean = float(y[order[:count]].mean())
        result["top_fraction"][str(int(fraction * 100))] = {
            "count": count,
            "mean_true_voi": top_mean,
            "population_mean_true_voi": all_mean,
            "lift": top_mean - all_mean,
        }
    return result


def fit_scalar_models(
    x: np.ndarray, y: np.ndarray, sensitivity: np.ndarray,
    groups: np.ndarray, device: torch.device, seed: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray], list[dict[str, Any]]]:
    """Five subject-held-out folds plus final full-seen fits."""

    high = (sensitivity >= np.quantile(sensitivity, SENSITIVITY_QUANTILE)).astype(np.int64)
    folds = _stratified_group_folds(high, groups, n_splits=5, seed=seed)
    oof = {name: np.full(len(y), np.nan, dtype=np.float32)
           for name in ("ridge", "random_forest", "mlp")}
    fold_info: list[dict[str, Any]] = []
    for fold, (train_idx, val_idx) in enumerate(folds):
        ridge = RidgeRegressor(alpha=10.0).fit(x[train_idx], y[train_idx])
        oof["ridge"][val_idx] = ridge.predict(x[val_idx])
        forest = TinyRandomForest(seed=seed + 100 + fold).fit(x[train_idx], y[train_idx])
        oof["random_forest"][val_idx] = forest.predict(x[val_idx])
        _, mlp_pred, _ = fit_mlp_predict(
            x[train_idx], y[train_idx], x[val_idx], device, seed + 200 + fold
        )
        oof["mlp"][val_idx] = mlp_pred
        fold_info.append({
            "fold": int(fold), "train_episodes": int(len(train_idx)),
            "validation_episodes": int(len(val_idx)),
            "validation_subjects": sorted(set(groups[val_idx].tolist())),
            "train_high_sensitivity": int(high[train_idx].sum()),
            "validation_high_sensitivity": int(high[val_idx].sum()),
        })

    final_models: dict[str, Any] = {}
    final_models["ridge"] = RidgeRegressor(alpha=10.0).fit(x, y)
    final_models["random_forest"] = TinyRandomForest(seed=seed + 1000).fit(x, y)
    final_models["mlp"], _, final_mlp_norm = fit_mlp_predict(
        x, y, x[:1], device, seed + 2000
    )
    test_predictions: dict[str, np.ndarray] = {"model_normalization": final_mlp_norm}
    return final_models, oof, fold_info


def predict_final_model(
    model_name: str, model: Any, x: np.ndarray,
    mlp_normalization: Mapping[str, np.ndarray] | None,
    device: torch.device,
) -> np.ndarray:
    if model_name == "mlp":
        assert mlp_normalization is not None
        mean = np.asarray(mlp_normalization["mean"], dtype=np.float32)
        std = np.asarray(mlp_normalization["std"], dtype=np.float32)
        y_mean = float(np.asarray(mlp_normalization["y_mean"]))
        y_std = float(np.asarray(mlp_normalization["y_std"]))
        with torch.inference_mode():
            value = model(torch.from_numpy(((x - mean) / std).astype(np.float32)).to(device))
        return (value.cpu().numpy() * y_std + y_mean).astype(np.float32)
    return model.predict(x)


def train_sensitivity_ranker(
    train_data: Mapping[str, Any], train_targets: Mapping[str, np.ndarray],
    device: torch.device, seed: int, output: Path,
) -> dict[str, Any]:
    """Train a candidate scorer on the top 30% high-sensitivity train episodes."""

    seed_all(seed)
    context, candidate, info = candidate_inputs(
        train_data["raw"], train_data["episodes"], train_data["features"]
    )
    utility = np.asarray(train_targets["utility"], dtype=np.float32).copy()
    valid = np.asarray(info["valid"], dtype=bool)
    utility[~valid] = 0.0
    sensitivity = np.asarray(train_targets["view_sensitivity"])
    cutoff = float(np.quantile(sensitivity, SENSITIVITY_QUANTILE))
    selected = sensitivity >= cutoff
    tx_context = torch.from_numpy(context[selected]).to(device)
    tx_candidate = torch.from_numpy(candidate[selected]).to(device)
    tx_target = torch.from_numpy(utility[selected]).to(device)
    tx_valid = torch.from_numpy(valid[selected]).to(device)
    model = StructuredScorer(hidden=128).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=2e-2)
    for _ in range(RANKER_EPOCHS):
        prediction = model(tx_context, tx_candidate)
        mask = tx_valid
        mse = ((prediction - tx_target) ** 2)[mask].mean()
        pair_losses: list[torch.Tensor] = []
        for left in CANDIDATE_VIEWS:
            for right in CANDIDATE_VIEWS:
                if int(left) >= int(right):
                    continue
                ok = tx_valid[:, int(left)] & tx_valid[:, int(right)]
                delta = tx_target[:, int(left)] - tx_target[:, int(right)]
                informative = ok & (delta.abs() > 0.02)
                if informative.any():
                    pair_losses.append(
                        torch.relu(0.10 - delta[informative].sign()
                                   * (prediction[informative, int(left)]
                                      - prediction[informative, int(right)]))
                    )
        pair = torch.cat(pair_losses).mean() if pair_losses else mse.new_zeros(())
        loss = mse + 0.5 * pair
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    checkpoint = output / "high_sensitivity_ranker.pt"
    torch.save({
        "model": model.state_dict(), "hidden": 128, "seed": seed,
        "sensitivity_quantile": SENSITIVITY_QUANTILE,
        "sensitivity_cutoff": cutoff,
        "selected_train_episodes": int(selected.sum()),
        "context_dim": int(context.shape[1]), "candidate_dim": int(candidate.shape[2]),
        "future_inputs_to_model": False,
    }, checkpoint)
    return {
        "model": model, "checkpoint": checkpoint, "context": context,
        "candidate": candidate, "valid": valid, "cutoff": cutoff,
        "selected": selected,
        "selected_train_episodes": int(selected.sum()),
    }


def ranker_actions(
    ranker: Mapping[str, Any], data: Mapping[str, Any], device: torch.device,
) -> np.ndarray:
    context, candidate, info = candidate_inputs(
        data["raw"], data["episodes"], data["features"]
    )
    model = ranker["model"]
    model.eval()
    with torch.inference_mode():
        scores = model(
            torch.from_numpy(context).to(device),
            torch.from_numpy(candidate).to(device),
        )
        valid = torch.from_numpy(info["valid"]).to(device)
        return scores.masked_fill(~valid, -torch.inf).argmax(-1).cpu().numpy().astype(np.int64)


def classification_protocol(
    data: Mapping[str, Any], classes: Sequence[int], action: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    raw = data["raw"]
    n = len(data["episodes"])
    logits = np.asarray(raw["logits"][:n], dtype=np.float32)
    labels = np.asarray(raw["labels"][:n], dtype=np.int64)
    bank = np.asarray(sorted(set(int(x) for x in classes)), dtype=np.int64)
    local = np.searchsorted(bank, labels)
    current = logits[:, CURRENT_VIEW]
    fused = 0.5 * (current[:, None, :] + logits)
    current_correct = current[..., bank].argmax(-1) == local
    candidate_correct = fused[..., bank].argmax(-1) == local[:, None]
    _, _, info = candidate_inputs(raw, np.arange(n), data["features"])
    valid = info["valid"]
    rows = np.arange(n)
    policy_correct = candidate_correct[rows, action]
    random_expected = candidate_correct[:, CANDIDATE_VIEWS].mean(axis=1)
    fixed_correct = candidate_correct[:, 1]
    delta_policy = policy_correct.astype(np.float32) - current_correct.astype(np.float32)
    delta_random = random_expected.astype(np.float32) - current_correct.astype(np.float32)
    cost = np.asarray(raw["cost"][:n, CURRENT_VIEW], dtype=np.float32)
    chosen_cost = cost[rows, action]
    return {
        "episodes": int(n),
        "single_view0_top1": float(current_correct.mean()),
        "random_expected_top1": float(random_expected.mean()),
        "fixed_view1_top1": float(fixed_correct.mean()),
        "policy_top1": float(policy_correct.mean()),
        "policy_delta_vs_single": float(delta_policy.mean()),
        "random_delta_vs_single": float(delta_random.mean()),
        "policy_minus_random": float((policy_correct - random_expected).mean()),
        "policy_movement_cost": float(chosen_cost.mean()),
        "policy_action_histogram": np.bincount(action, minlength=NUM_VIEWS).tolist(),
        "oracle_top1": float(candidate_correct[rows, np.asarray(
            np.argmax(np.where(valid[:, CANDIDATE_VIEWS],
                               candidate_correct[:, CANDIDATE_VIEWS].astype(np.float32), -1.0), axis=1)
        ) + 1].mean()),
        "paired_ci_policy_vs_single": _paired_ci(delta_policy, seed),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=Path("data/rl_candidate_rank_v6_new50_5/cache"))
    parser.add_argument("--output", type=Path, default=Path("work_dir/voi_view_sensitivity_v1"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260910)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    raw_root = args.raw_root if args.raw_root.is_absolute() else root / args.raw_root
    output = args.output if args.output.is_absolute() else root / args.output
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    seed_all(args.seed)
    print(json.dumps({
        "VOI_VIEW_SENSITIVITY_START": True,
        "raw_root": str(raw_root), "output": str(output), "device": str(device),
        "protocol": "train seen dqn_train; subject-held-out OOF; test seen_test and strict true unseen",
        "future_candidate_inputs_to_predictors": False,
        "target": "U=fused GT margin-current GT margin; VoI=max(U)-mean_random(U); VS=max(U)-min(U)",
    }), flush=True)

    train_data = prepare_data(raw_root, "dqn_train", SEEN_CLASSES)
    seen_data = prepare_data(raw_root, "seen_test", SEEN_CLASSES)
    unseen_data = prepare_data(raw_root, "test", UNSEEN_CLASSES)
    datasets = [("train", train_data, SEEN_CLASSES), ("seen_test", seen_data, SEEN_CLASSES),
                ("true_unseen", unseen_data, UNSEEN_CLASSES)]
    targets = {name: compute_targets(data, classes) for name, data, classes in datasets}
    x_train, feature_info = causal_features(train_data)
    groups = np.asarray([
        str(train_data["raw"]["metadata"]["episodes"][int(source)].get("subject", "unknown"))
        for source in train_data["source_episode_ids"]
    ])
    y = targets["train"]["voi_random"]
    sensitivity = targets["train"]["view_sensitivity"]
    final_models, oof, fold_info = fit_scalar_models(
        x_train, y, sensitivity, groups, device, args.seed
    )
    # Refit MLP once on all seen data; fit_scalar_models used a one-query call
    # to obtain its normalization, so repeat with the actual test inputs here.
    final_mlp, _, mlp_norm = fit_mlp_predict(x_train, y, x_train[:1], device, args.seed + 2000)
    final_models["mlp"] = final_mlp
    scalar_report: dict[str, Any] = {
        "target": "voi_random",
        "folds": fold_info,
        "oof_subject_holdout": {
            name: scalar_metrics(y, prediction) for name, prediction in oof.items()
        },
        "test": {},
        "feature_info": feature_info,
    }
    for split_name, data, _ in datasets[1:]:
        x_query, _ = causal_features(data)
        scalar_report["test"][split_name] = {}
        for model_name, model in final_models.items():
            prediction = predict_final_model(
                model_name, model, x_query,
                mlp_norm if model_name == "mlp" else None, device
            )
            scalar_report["test"][split_name][model_name] = scalar_metrics(
                targets[split_name]["voi_random"], prediction
            )
    (output / "voi_predictor_metrics.json").write_text(
        json.dumps(json_safe(scalar_report), indent=2), encoding="utf-8"
    )
    with (output / "voi_predictor_models.pkl").open("wb") as handle:
        pickle.dump({"ridge": final_models["ridge"], "random_forest": final_models["random_forest"]}, handle)
    torch.save({
        "model": final_models["mlp"].state_dict(), "width": int(x_train.shape[1]),
        "normalization": mlp_norm, "target": "voi_random", "seed": args.seed,
    }, output / "voi_mlp.pt")

    ranker = train_sensitivity_ranker(train_data, targets["train"], device, args.seed, output)
    rank_report: dict[str, Any] = {
        "training": {
            "episodes": int(len(train_data["episodes"])),
            "high_sensitivity_cutoff": ranker["cutoff"],
            "high_sensitivity_episodes": ranker["selected_train_episodes"],
            "high_sensitivity_fraction": float(ranker["selected"].mean()),
            "future_candidate_inputs_to_ranker": False,
        },
        "results": {},
    }
    for split_name, data, classes in datasets[1:]:
        action = ranker_actions(ranker, data, device)
        rank_report["results"][split_name] = classification_protocol(
            data, classes, action, args.seed + len(split_name)
        )
    (output / "high_sensitivity_ranker_metrics.json").write_text(
        json.dumps(json_safe(rank_report), indent=2), encoding="utf-8"
    )

    summary = {
        "version": "voi_view_sensitivity_v1",
        "cache_limitation": {
            "available": [
                "clip_level_current_logits", "current_pose_object_proxies",
                "image_quality", "camera_geometry",
            ],
            "unavailable_not_fabricated": [
                "per_frame_or_temporal_segment_logits",
                "branch_specific_rgb_skeleton_object_logits",
                "per_joint_skeleton_confidence", "object_detection_stability_over_time",
            ],
        },
        "splits": {
            name: {"episodes": int(len(data["episodes"])), "classes": list(classes)}
            for name, data, classes in datasets
        },
        "target_definition": {
            "utility": "fused(view0,candidate) GT margin minus view0 GT margin",
            "voi_current": "max_candidate_utility; current incremental utility is zero",
            "voi_random": "max_candidate_utility minus mean valid-candidate utility",
            "view_sensitivity": "max_candidate_utility minus min_candidate_utility",
            "high_sensitivity_selection": "top 30 percent of dqn_train by VS, cutoff fixed before evaluation",
        },
        "voi_predictor": scalar_report,
        "high_sensitivity_ranker": rank_report,
        "notes": [
            "GT labels and future candidate logits are offline supervision/evaluation only.",
            "The predictor input is current-view causal state and contains no future candidate data.",
            "The ranker is a supervised one-step utility scorer, not an RL/random gate.",
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(json_safe(summary), indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "VOI_VIEW_SENSITIVITY_COMPLETE": True,
        "output": str(output),
        "splits": summary["splits"],
        "oof": scalar_report["oof_subject_holdout"],
        "unseen_scalar": scalar_report["test"]["true_unseen"],
        "unseen_ranker": rank_report["results"]["true_unseen"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
