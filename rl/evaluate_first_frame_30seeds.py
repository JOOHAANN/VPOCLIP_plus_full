"""Thirty-seed evaluation for the first-sampled-frame trajectory variants.

This evaluates the already trained first-frame v1/v3/v4/v5/v6 and ablation
checkpoints.  It does not train new models.  Each variant is run in a fresh
process so the first-frame monkeypatches cannot leak between variants.

The random seeds control both random starting views and random comparison
paths.  Fixed-view evaluation uses view0 as the starting view.  Both seen-50
and true-unseen-5 class banks are reported, for one and two moves.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import multistep_angle_object_trajectory_policy_v1 as base
from .first_frame_analysis.train_first_frame import (
    DEFAULT_TRUE_UNSEEN,
    _configure_variant,
    _first_frame_pose_cache,
)


TRAIN_SEEDS = [20260909, 20260910, 20260911]
EVAL_SEEDS = list(range(20260920, 20260950))
VARIANTS = ("v1", "v3", "v4", "v5", "v6", "object_only", "geometry_only")


def _ci95(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return 1.96 * float(np.std(values, ddof=1)) / math.sqrt(len(values))


def _metric_row(metrics: dict[str, float]) -> dict[str, float]:
    return {
        "top1": float(metrics["top1"]),
        "mean_entropy": float(metrics.get("mean_entropy", float("nan"))),
        "mean_movement_cost": float(metrics.get("mean_movement_cost", float("nan"))),
    }


def _configure_first_frame(
    variant: str,
    cache_root: Path,
    track_root: Path,
    v3_track_root: Path,
) -> Path:
    original_cache = base.GPUCache
    base.GPUCache = _first_frame_pose_cache(original_cache)
    base.NUM_FRAMES = 1
    base.CACHE_ROOT = cache_root
    base.TRACK_ROOT = track_root

    effective_track_root = v3_track_root if variant == "v3" else track_root
    attach_tracks, state_builder, policy_class = _configure_variant(variant, base)
    base.attach_object_tracks = attach_tracks
    base.build_trajectory_state = state_builder
    base.AngleObjectTrajectoryPolicy = policy_class
    return effective_track_root


def _evaluate_bank(
    model: torch.nn.Module,
    raw: Any,
    episodes: torch.Tensor,
    bank: torch.Tensor,
    eval_seeds: list[int],
    fixed_start: int = 0,
) -> dict[str, Any]:
    # Import this only after the first-frame variant has been configured.  The
    # historical evaluator imports all variant modules at import time; doing
    # that before setting v1.NUM_FRAMES would leave v6's cached NUM_FRAMES at
    # 13 and make its first-frame checkpoint incompatible.
    from .evaluate_trajectory_variants_old_split import (
        oracle_path,
        path_metrics,
        policy_path,
        random_path,
        single_metrics,
    )

    device = raw.device
    fixed_episodes = (raw.valid[episodes, fixed_start]).nonzero().flatten()
    fixed_episodes = episodes.index_select(0, fixed_episodes)
    fixed_starts = torch.full_like(fixed_episodes, fixed_start)

    fixed: dict[str, Any] = {}
    for moves in (1, 2):
        policy = policy_path(model, raw, fixed_episodes, fixed_starts, moves)
        oracle = oracle_path(raw, fixed_episodes, fixed_starts, bank, moves)
        random_rows = []
        for seed in eval_seeds:
            random = random_path(
                fixed_starts, moves, seed + 100 * fixed_start + 1000 * moves, device
            )
            random_rows.append(_metric_row(path_metrics(raw, fixed_episodes, random, bank)))
        fixed[f"moves_{moves}"] = {
            "episodes": int(len(fixed_episodes)),
            "single": _metric_row(single_metrics(raw, fixed_episodes, fixed_starts, bank)),
            "policy": _metric_row(path_metrics(raw, fixed_episodes, policy, bank)),
            "oracle": _metric_row(path_metrics(raw, fixed_episodes, oracle, bank)),
            "random_mean": {
                key: float(np.mean([row[key] for row in random_rows]))
                for key in random_rows[0]
            },
            "random_std": {
                key: float(np.std([row[key] for row in random_rows], ddof=1))
                for key in random_rows[0]
            },
        }

    random_start: dict[str, Any] = {}
    for seed in eval_seeds:
        starts = base.choose_random_starts(raw, episodes, seed)
        random_start[str(seed)] = {}
        for moves in (1, 2):
            policy = policy_path(model, raw, episodes, starts, moves)
            random = random_path(starts, moves, seed + 1000 * moves, device)
            oracle = oracle_path(raw, episodes, starts, bank, moves)
            random_start[str(seed)][f"moves_{moves}"] = {
                "episodes": int(len(episodes)),
                "single": _metric_row(single_metrics(raw, episodes, starts, bank)),
                "policy": _metric_row(path_metrics(raw, episodes, policy, bank)),
                "random": _metric_row(path_metrics(raw, episodes, random, bank)),
                "oracle": _metric_row(path_metrics(raw, episodes, oracle, bank)),
            }
    return {"fixed0": fixed, "random_start": random_start}


def _aggregate(
    per_train: dict[str, Any],
    eval_seeds: list[int],
) -> dict[str, Any]:
    fixed: dict[str, Any] = {}
    random_start: dict[str, Any] = {}
    for moves in (1, 2):
        rows = [per_train[str(seed)]["fixed0"][f"moves_{moves}"] for seed in TRAIN_SEEDS]
        policy = [row["policy"]["top1"] for row in rows]
        random = [row["random_mean"]["top1"] for row in rows]
        fixed[f"moves_{moves}"] = {
            "episodes": rows[0]["episodes"],
            "single_top1": rows[0]["single"]["top1"],
            "policy_top1_mean_over_train_seeds": float(np.mean(policy)),
            "policy_top1_std_over_train_seeds": float(np.std(policy, ddof=1)),
            "random_top1_mean_over_eval_seeds": float(np.mean(random)),
            "random_top1_std_over_eval_seeds": float(
                np.mean([row["random_std"]["top1"] for row in rows])
            ),
            "random_top1_ci95_over_eval_seeds": float(
                1.96
                * np.mean([row["random_std"]["top1"] for row in rows])
                / math.sqrt(len(eval_seeds))
            ),
            "oracle_top1_mean_over_train_seeds": float(
                np.mean([row["oracle"]["top1"] for row in rows])
            ),
            "policy_minus_random": float(np.mean(policy) - np.mean(random)),
        }

        policy_by_eval = [
            float(
                np.mean([
                    per_train[str(seed)]["random_start"][str(eval_seed)]
                    [f"moves_{moves}"]["policy"]["top1"]
                    for seed in TRAIN_SEEDS
                ])
            )
            for eval_seed in eval_seeds
        ]
        random_by_eval = [
            per_train[str(TRAIN_SEEDS[0])]["random_start"][str(eval_seed)]
            [f"moves_{moves}"]["random"]["top1"]
            for eval_seed in eval_seeds
        ]
        oracle_by_eval = [
            float(
                np.mean([
                    per_train[str(seed)]["random_start"][str(eval_seed)]
                    [f"moves_{moves}"]["oracle"]["top1"]
                    for seed in TRAIN_SEEDS
                ])
            )
            for eval_seed in eval_seeds
        ]
        random_start[f"moves_{moves}"] = {
            "episodes": per_train[str(TRAIN_SEEDS[0])]["random_start"]
            [str(eval_seeds[0])][f"moves_{moves}"]["episodes"],
            "single_top1_mean_over_eval_seeds": float(
                np.mean([
                    per_train[str(TRAIN_SEEDS[0])]["random_start"][str(seed)]
                    [f"moves_{moves}"]["single"]["top1"]
                    for seed in eval_seeds
                ])
            ),
            "policy_top1_mean_over_eval_seeds": float(np.mean(policy_by_eval)),
            "policy_top1_std_over_eval_seeds": float(np.std(policy_by_eval, ddof=1)),
            "policy_top1_ci95": _ci95(policy_by_eval),
            "random_top1_mean_over_eval_seeds": float(np.mean(random_by_eval)),
            "random_top1_std_over_eval_seeds": float(np.std(random_by_eval, ddof=1)),
            "random_top1_ci95": _ci95(random_by_eval),
            "oracle_top1_mean_over_eval_seeds": float(np.mean(oracle_by_eval)),
            "policy_minus_random": float(np.mean(policy_by_eval) - np.mean(random_by_eval)),
        }
    return {"fixed0": fixed, "random_start": random_start}


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    effective_track_root = _configure_first_frame(
        args.variant, args.cache_root, args.track_root, args.v3_track_root
    )
    raw = base.GPUCache(args.cache_root / "test", device)
    base.attach_view_valid(raw, args.cache_root, "test")
    base.attach_object_tracks(raw, effective_track_root, "test")

    banks = {
        "seen_50way": [x for x in range(55) if x not in DEFAULT_TRUE_UNSEEN],
        "true_unseen_5way": list(DEFAULT_TRUE_UNSEEN),
    }
    result: dict[str, Any] = {
        "experiment": "trajectory_first_sampled_frame_30_eval_seeds",
        "variant": args.variant,
        "train_seeds": TRAIN_SEEDS,
        "eval_seeds": args.eval_seeds,
        "protocol": "fixed view0 and random start; one/two moves; 2/3-view fusion",
        "temporal_input_frames": 1,
        "original_temporal_input_frames": 13,
        "splits": {},
    }

    for split_name, class_ids in banks.items():
        bank = base.class_bank_tensor(class_ids, device)
        eligible = torch.isin(raw.labels, bank) & (raw.valid.sum(-1) >= 3)
        episodes = eligible.nonzero().flatten()
        per_train: dict[str, Any] = {}
        for train_seed in TRAIN_SEEDS:
            checkpoint = args.policy_root / f"seed_{train_seed}" / "best.pt"
            if not checkpoint.exists():
                raise FileNotFoundError(checkpoint)
            model_cls = base.AngleObjectTrajectoryPolicy
            model = model_cls().to(device)
            payload = torch.load(checkpoint, map_location=device, weights_only=False)
            model.load_state_dict(payload["online"], strict=True)
            model.eval()
            per_train[str(train_seed)] = _evaluate_bank(
                model, raw, episodes, bank, args.eval_seeds
            )
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        result["splits"][split_name] = {
            "class_ids": class_ids,
            "episodes": int(len(episodes)),
            "per_train_seed": per_train,
            "aggregates": _aggregate(per_train, args.eval_seeds),
        }

    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--track-root", type=Path, required=True)
    parser.add_argument("--v3-track-root", type=Path, required=True)
    parser.add_argument("--policy-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=EVAL_SEEDS)
    args = parser.parse_args()
    result = run(args)
    compact = {
        split: data["aggregates"] for split, data in result["splits"].items()
    }
    print("FIRST_FRAME_30_SEED_EVALUATION_COMPLETE", args.variant, flush=True)
    print(json.dumps(compact, indent=2), flush=True)


if __name__ == "__main__":
    main()
