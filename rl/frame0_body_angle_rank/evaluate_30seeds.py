"""Thirty-seed evaluation for the current frame-0 50/5 policy sweep.

The trained policy checkpoints are kept fixed.  The thirty evaluation seeds
change the random starting view and the sampled random comparison path.  The
evaluator is invoked once per variant so that the historical adapter
monkey-patches cannot leak between variants.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import train_ranked_variant as adapter
from .. import multistep_angle_object_trajectory_policy_v1 as base


TRAIN_SEEDS = [20260909, 20260910, 20260911]
DEFAULT_EVAL_SEEDS = list(range(20260920, 20260950))
TRUE_UNSEEN = [14, 15, 25, 39, 46]
SEEN = [x for x in range(55) if x not in TRUE_UNSEEN]
VARIANTS = (
    "v1",
    "v2_legacy",
    "v3",
    "v4",
    "v5",
    "v6",
    "object_only",
    "geometry_only",
)


def _ci95(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return 1.96 * float(np.std(values, ddof=1)) / math.sqrt(len(values))


def _metrics_row(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        "top1": float(metrics.get("top1", 0.0)),
        "mean_entropy": float(metrics.get("mean_entropy", 0.0)),
        "mean_movement_cost": float(metrics.get("mean_movement_cost", 0.0)),
        "episodes": float(metrics.get("episodes", 0.0)),
    }


def _empty_metrics() -> dict[str, float]:
    return {
        "top1": 0.0,
        "mean_entropy": 0.0,
        "mean_movement_cost": 0.0,
        "episodes": 0.0,
    }


def _configure(args: argparse.Namespace) -> tuple[Any, Path | None, dict[str, Any]]:
    # The adapter applies the exact state/model implementation used during
    # the frame-0 training sweep.  It also replaces GPUCache with the cache
    # class that loads the frame-0 yaw array required by v4.
    return adapter.configure_variant(args, args.output_root)


def _load_raw(
    module: Any,
    cache_root: Path,
    track_root: Path | None,
    device: torch.device,
) -> Any:
    raw = module.GPUCache(cache_root / "test", device)
    module.attach_view_valid(raw, cache_root, "test")
    if track_root is not None:
        module.attach_object_tracks(raw, track_root, "test")
    return raw


def _policy_rollout(
    variant: str,
    module: Any,
    model: torch.nn.Module,
    raw: Any,
    episodes: torch.Tensor,
    starts: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if variant == "v2_legacy":
        return module.policy_rollout_angle(model, raw, episodes, starts)
    return module.policy_rollout_trajectory(model, raw, episodes, starts)


def _policy_metrics(
    variant: str,
    module: Any,
    model: torch.nn.Module,
    raw: Any,
    episodes: torch.Tensor,
    starts: torch.Tensor,
    bank: torch.Tensor,
) -> tuple[dict[str, float], dict[str, Any]]:
    rollout = _policy_rollout(variant, module, model, raw, episodes, starts)
    policy_episodes = rollout["episodes"]
    policy_paths = torch.stack(
        (rollout["starts"], rollout["action1"], rollout["action2"]), 1
    )
    if len(policy_episodes):
        metrics = base.metrics_for_paths(raw, policy_episodes, policy_paths, bank)
        return _metrics_row(metrics), {
            "episodes_before_filter": int(len(episodes)),
            "episodes_after_filter": int(len(policy_episodes)),
            "completion_rate": float(len(policy_episodes) / max(len(episodes), 1)),
        }
    return _empty_metrics(), {
        "episodes_before_filter": int(len(episodes)),
        "episodes_after_filter": 0,
        "completion_rate": 0.0,
    }


def _evaluate_condition(
    variant: str,
    module: Any,
    model: torch.nn.Module,
    raw: Any,
    episodes: torch.Tensor,
    starts: torch.Tensor,
    bank: torch.Tensor,
    random_seed: int,
) -> dict[str, Any]:
    single = (
        base._single_metrics(raw, episodes, starts, bank)
        if len(episodes)
        else _empty_metrics()
    )
    policy, policy_info = _policy_metrics(
        variant, module, model, raw, episodes, starts, bank
    )

    random_episodes, random_paths, _ = base.sample_random_paths(
        raw, episodes, starts, random_seed + 17
    )
    random = (
        base.metrics_for_paths(raw, random_episodes, random_paths, bank)
        if len(random_episodes)
        else _empty_metrics()
    )
    oracle_episodes, oracle_paths = base.oracle_paths(raw, episodes, starts, bank)
    oracle = (
        base.metrics_for_paths(raw, oracle_episodes, oracle_paths, bank)
        if len(oracle_episodes)
        else _empty_metrics()
    )
    return {
        "single": _metrics_row(single),
        "policy": policy,
        "random": _metrics_row(random),
        "oracle": _metrics_row(oracle),
        "policy_info": policy_info,
        "random_episodes": int(len(random_episodes)),
        "oracle_episodes": int(len(oracle_episodes)),
    }


def _series_summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)) if values else 0.0,
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "ci95": _ci95(values),
    }


def _aggregate_random_start(
    per_train: dict[str, Any], eval_seeds: list[int]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for metric in ("top1", "mean_entropy", "mean_movement_cost"):
        policy_values = [
            float(
                np.mean(
                    [
                        per_train[str(train_seed)]["random_start"][str(eval_seed)][
                            "policy"
                        ][metric]
                        for train_seed in TRAIN_SEEDS
                    ]
                )
            )
            for eval_seed in eval_seeds
        ]
        random_values = [
            float(
                np.mean(
                    [
                        per_train[str(train_seed)]["random_start"][str(eval_seed)][
                            "random"
                        ][metric]
                        for train_seed in TRAIN_SEEDS
                    ]
                )
            )
            for eval_seed in eval_seeds
        ]
        oracle_values = [
            float(
                np.mean(
                    [
                        per_train[str(train_seed)]["random_start"][str(eval_seed)][
                            "oracle"
                        ][metric]
                        for train_seed in TRAIN_SEEDS
                    ]
                )
            )
            for eval_seed in eval_seeds
        ]
        delta_values = [p - r for p, r in zip(policy_values, random_values)]
        result[metric] = {
            "policy": _series_summary(policy_values),
            "random": _series_summary(random_values),
            "oracle": _series_summary(oracle_values),
            "policy_minus_random": _series_summary(delta_values),
        }

    single_values = [
        float(
            np.mean(
                [
                    per_train[str(TRAIN_SEEDS[0])]["random_start"][str(eval_seed)][
                        "single"
                    ][metric]
                    for eval_seed in eval_seeds
                ]
            )
        )
        for metric in ("top1", "mean_entropy")
    ]
    result["single"] = {
        "top1_mean_over_eval_seeds": single_values[0],
        "mean_entropy_mean_over_eval_seeds": single_values[1],
    }
    result["episodes"] = int(
        per_train[str(TRAIN_SEEDS[0])]["random_start"][str(eval_seeds[0])][
            "policy"
        ]["episodes"]
    )
    return result


def _aggregate_fixed0(
    per_train: dict[str, Any], eval_seeds: list[int]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for metric in ("top1", "mean_entropy", "mean_movement_cost"):
        policy_train_values = [
            float(per_train[str(train_seed)]["fixed0_policy"][metric])
            for train_seed in TRAIN_SEEDS
        ]
        random_by_seed = [
            float(
                np.mean(
                    [
                        per_train[str(train_seed)]["fixed0"][str(eval_seed)][
                            "random"
                        ][metric]
                        for train_seed in TRAIN_SEEDS
                    ]
                )
            )
            for eval_seed in eval_seeds
        ]
        oracle_train_values = [
            float(per_train[str(train_seed)]["fixed0_oracle"][metric])
            for train_seed in TRAIN_SEEDS
        ]
        policy_mean = float(np.mean(policy_train_values))
        random_mean = float(np.mean(random_by_seed))
        result[metric] = {
            "policy": {
                "mean_over_train_seeds": policy_mean,
                "std_over_train_seeds": float(
                    np.std(policy_train_values, ddof=1)
                ),
                "eval_seed_std": 0.0,
                "ci95_over_eval_seeds": 0.0,
            },
            "random": {
                **_series_summary(random_by_seed),
                "mean_over_eval_seeds": random_mean,
            },
            "oracle": {
                "mean_over_train_seeds": float(np.mean(oracle_train_values)),
                "std_over_train_seeds": float(
                    np.std(oracle_train_values, ddof=1)
                ),
            },
            "policy_minus_random": policy_mean - random_mean,
        }

    result["single"] = {
        "top1": float(
            np.mean(
                [
                    per_train[str(train_seed)]["fixed0_single"]["top1"]
                    for train_seed in TRAIN_SEEDS
                ]
            )
        ),
        "mean_entropy": float(
            np.mean(
                [
                    per_train[str(train_seed)]["fixed0_single"]["mean_entropy"]
                    for train_seed in TRAIN_SEEDS
                ]
            )
        ),
    }
    result["episodes"] = int(
        per_train[str(TRAIN_SEEDS[0])]["fixed0_single"]["episodes"]
    )
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    if len(args.eval_seeds) != 30:
        raise ValueError(
            f"exactly 30 evaluation seeds are required, got {len(args.eval_seeds)}"
        )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for the current cache/evaluation setup")

    module, effective_track_root, provenance = _configure(args)
    raw = _load_raw(module, args.cache_root, effective_track_root, device)

    result: dict[str, Any] = {
        "experiment": "frame0_body_angle_rank_new50_5_eval_30_seeds",
        "variant": args.variant,
        "train_seeds": TRAIN_SEEDS,
        "eval_seeds": args.eval_seeds,
        "seed_role": {
            "training_checkpoints_fixed": True,
            "eval_seed_controls": ["random_start_view", "sampled_random_path"],
            "fixed0_policy_is_deterministic": True,
        },
        "class_split": {
            "seen_50way": SEEN,
            "true_unseen_5way": TRUE_UNSEEN,
        },
        "protocol": "test split; start fixed at view 0 or sampled uniformly from valid views; two policy moves; three-view mean-logit fusion",
        "cache_root": str(args.cache_root),
        "policy_root": str(args.policy_root),
        "provenance": provenance,
        "splits": {},
    }

    for split_name, class_ids in (
        ("seen_50way", SEEN),
        ("true_unseen_5way", TRUE_UNSEEN),
    ):
        bank = base.class_bank_tensor(class_ids, device)
        eligible = torch.isin(raw.labels, bank) & (raw.valid.sum(-1) >= 3)
        episodes = eligible.nonzero().flatten()
        fixed_episodes = (eligible & raw.valid[:, 0]).nonzero().flatten()
        fixed_starts = torch.zeros_like(fixed_episodes)
        per_train: dict[str, Any] = {}

        for train_seed in TRAIN_SEEDS:
            checkpoint = args.policy_root / f"seed_{train_seed}" / "best.pt"
            if not checkpoint.exists():
                raise FileNotFoundError(checkpoint)
            if args.variant == "v2_legacy":
                model = module.AngleObjectPolicy().to(device)
            else:
                model = module.AngleObjectTrajectoryPolicy().to(device)
            payload = torch.load(checkpoint, map_location=device, weights_only=False)
            model.load_state_dict(payload["online"], strict=True)
            model.eval()

            fixed_policy, fixed_info = _policy_metrics(
                args.variant,
                module,
                model,
                raw,
                fixed_episodes,
                fixed_starts,
                bank,
            )
            fixed_single = base._single_metrics(
                raw, fixed_episodes, fixed_starts, bank
            )
            fixed_single_row = _metrics_row(fixed_single)
            fixed_single_row["episodes"] = float(len(fixed_episodes))
            fixed_oracle_episodes, fixed_oracle_paths = base.oracle_paths(
                raw, fixed_episodes, fixed_starts, bank
            )
            fixed_oracle = (
                base.metrics_for_paths(
                    raw, fixed_oracle_episodes, fixed_oracle_paths, bank
                )
                if len(fixed_oracle_episodes)
                else _empty_metrics()
            )

            fixed_rows: dict[str, Any] = {}
            for eval_seed in args.eval_seeds:
                fixed_rows[str(eval_seed)] = _evaluate_condition(
                    args.variant,
                    module,
                    model,
                    raw,
                    fixed_episodes,
                    fixed_starts,
                    bank,
                    eval_seed,
                )

            random_rows: dict[str, Any] = {}
            for eval_seed in args.eval_seeds:
                starts = base.choose_random_starts(raw, episodes, eval_seed)
                random_rows[str(eval_seed)] = _evaluate_condition(
                    args.variant,
                    module,
                    model,
                    raw,
                    episodes,
                    starts,
                    bank,
                    eval_seed,
                )

            per_train[str(train_seed)] = {
                "checkpoint": str(checkpoint),
                "checkpoint_epoch": int(payload.get("epoch", -1)),
                "fixed0_single": fixed_single_row,
                "fixed0_policy": fixed_policy,
                "fixed0_policy_info": fixed_info,
                "fixed0_oracle": _metrics_row(fixed_oracle),
                "fixed0": fixed_rows,
                "random_start": random_rows,
            }
            del model
            torch.cuda.empty_cache()

        result["splits"][split_name] = {
            "class_ids": class_ids,
            "episodes": int(len(episodes)),
            "fixed0_episodes": int(len(fixed_episodes)),
            "per_train_seed": per_train,
            "aggregates": {
                "fixed0": _aggregate_fixed0(per_train, args.eval_seeds),
                "random_start": _aggregate_random_start(
                    per_train, args.eval_seeds
                ),
            },
        }

    args.output_root.mkdir(parents=True, exist_ok=True)
    output = args.output_root / "summary.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--track-root", type=Path, required=True)
    parser.add_argument("--track-root-v3", type=Path, required=True)
    parser.add_argument("--policy-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--eval-seeds", type=int, nargs="+", default=DEFAULT_EVAL_SEEDS
    )
    args = parser.parse_args()
    summary = run(args)
    compact = {
        split: data["aggregates"] for split, data in summary["splits"].items()
    }
    print("FRAME0_30_SEED_EVALUATION_COMPLETE", args.variant, flush=True)
    print(json.dumps(compact, indent=2), flush=True)


if __name__ == "__main__":
    main()
