"""Random-start 30-seed evaluation for v4 + first-frame depth on new 50/5.

This is an evaluation-only script.  It does not retrain or overwrite the
v4+depth checkpoints.  For every evaluation seed, each episode independently
gets a valid random starting view.  Policy and Random then use the same start
and the same random path seed, making the comparison paired.
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
from . import multistep_angle_object_trajectory_policy_v4_depth as depth_variant
from .evaluate_trajectory_variants_old_split import (
    path_metrics,
    policy_path,
    random_path,
    single_metrics,
)


TRAIN_SEEDS = [20260909, 20260910, 20260911]
EVAL_SEEDS = list(range(20260920, 20260950))
NEW_UNSEEN = [14, 15, 25, 39, 46]
T_CRIT_95_N30 = 2.045229642


def ci95(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(T_CRIT_95_N30 * np.std(values, ddof=1) / math.sqrt(len(values)))


def mean_std_ci(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "ci95_half_width": ci95(values),
    }


def _configure_depth_variant() -> None:
    # The helper rollout functions use the mutable v1 extension points.  Keep
    # this configuration local to this fresh evaluator process.
    base.build_trajectory_state = depth_variant.build_trajectory_state_v4_depth
    base.AngleObjectTrajectoryPolicy = depth_variant.BodyOrientationDepthTrajectoryPolicy


@torch.inference_mode()
def run(
    cache_root: Path,
    track_root: Path,
    depth_root: Path,
    policy_root: Path,
    output_root: Path,
    eval_seeds: list[int],
    unseen_classes: list[int],
    expected_episodes: int,
) -> dict[str, Any]:
    _configure_depth_variant()
    device = torch.device("cuda:0")
    raw = base.GPUCache(cache_root / "test", device)
    base.attach_view_valid(raw, cache_root, "test")
    base.attach_object_tracks(raw, track_root, "test")
    depth_audit = depth_variant.attach_depth_tracks(raw, cache_root, depth_root, "test")

    bank = base.class_bank_tensor(unseen_classes, device)
    eligible = torch.isin(raw.labels, bank) & (raw.valid.sum(-1) >= 3)
    episodes = eligible.nonzero().flatten()
    if len(episodes) != expected_episodes:
        raise RuntimeError(
            f"new 50/5 unseen expected {expected_episodes} episodes, got {len(episodes)}"
        )

    per_train_seed: dict[str, Any] = {}
    for train_seed in TRAIN_SEEDS:
        checkpoint = policy_root / f"seed_{train_seed}" / "best.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        model = depth_variant.BodyOrientationDepthTrajectoryPolicy().to(device)
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(payload["online"], strict=True)
        model.eval()
        train_rows: dict[str, Any] = {}
        for eval_seed in eval_seeds:
            starts = base.choose_random_starts(raw, episodes, int(eval_seed))
            seed_rows: dict[str, Any] = {
                "start_view_histogram": {
                    str(v): int((starts == v).sum().item()) for v in range(base.NUM_VIEWS)
                }
            }
            for moves in (1, 2):
                policy = policy_path(model, raw, episodes, starts, moves)
                random = random_path(
                    starts, moves, int(eval_seed) + 1000 * moves, device
                )
                seed_rows[f"moves_{moves}"] = {
                    "single": single_metrics(raw, episodes, starts, bank),
                    "policy": path_metrics(raw, episodes, policy, bank),
                    "random": path_metrics(raw, episodes, random, bank),
                    "episodes": int(len(episodes)),
                    "moves": moves,
                    "fusion_views": moves + 1,
                }
            train_rows[str(eval_seed)] = seed_rows
        per_train_seed[str(train_seed)] = train_rows
        del model
        torch.cuda.empty_cache()

    aggregates: dict[str, Any] = {}
    for moves in (1, 2):
        policy_by_eval: list[float] = []
        random_by_eval: list[float] = []
        single_by_eval: list[float] = []
        for eval_seed in eval_seeds:
            key = str(eval_seed)
            policy_by_eval.append(float(np.mean([
                per_train_seed[str(train_seed)][key][f"moves_{moves}"]["policy"]["top1"]
                for train_seed in TRAIN_SEEDS
            ])))
            # Random is independent of the trained checkpoint; use one copy.
            random_by_eval.append(
                per_train_seed[str(TRAIN_SEEDS[0])][key][f"moves_{moves}"]["random"]["top1"]
            )
            single_by_eval.append(
                per_train_seed[str(TRAIN_SEEDS[0])][key][f"moves_{moves}"]["single"]["top1"]
            )
        delta_by_eval = [p - r for p, r in zip(policy_by_eval, random_by_eval)]
        aggregates[f"random_start_moves_{moves}"] = {
            "episodes": int(len(episodes)),
            "fusion_views": moves + 1,
            "single": mean_std_ci(single_by_eval),
            "policy": mean_std_ci(policy_by_eval),
            "random": mean_std_ci(random_by_eval),
            "policy_minus_random": mean_std_ci(delta_by_eval),
        }

    output_root.mkdir(parents=True, exist_ok=True)
    result = {
        "experiment": "v4_depth_new50_5_random_start_eval_30seeds",
        "variant": "multistep_angle_object_trajectory_policy_v4_depth",
        "class_bank": unseen_classes,
        "episodes": int(len(episodes)),
        "cache_root": str(cache_root),
        "track_root": str(track_root),
        "depth_root": str(depth_root),
        "policy_root": str(policy_root),
        "train_seeds": TRAIN_SEEDS,
        "eval_seeds": eval_seeds,
        "random_start_protocol": (
            "each episode independently samples a valid start view; "
            "policy and random share the same starts and path seeds"
        ),
        "depth_join_audit": depth_audit,
        "per_train_seed": per_train_seed,
        "aggregates": aggregates,
    }
    (output_root / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=root / "data/trajectory_latest_vpoclip_new50_5/cache_compat",
    )
    parser.add_argument(
        "--track-root",
        type=Path,
        default=root / "data/trajectory_latest_vpoclip_new50_5/object_tracks_full",
    )
    parser.add_argument(
        "--depth-root",
        type=Path,
        default=root / "data/new50_5_group1_zsl_depth_first_frame_fp32",
    )
    parser.add_argument(
        "--policy-root",
        type=Path,
        default=root / "work_dir/multistep_angle_object_trajectory_policy_v4_depth_new50_5",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=root / "work_dir/multistep_angle_object_trajectory_policy_v4_depth_new50_5/random_start_30seeds",
    )
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=EVAL_SEEDS)
    parser.add_argument("--unseen-classes", type=int, nargs="+", default=NEW_UNSEEN)
    parser.add_argument("--expected-episodes", type=int, default=271)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    result = run(
        args.cache_root,
        args.track_root,
        args.depth_root,
        args.policy_root,
        args.output_root,
        args.eval_seeds,
        args.unseen_classes,
        args.expected_episodes,
    )
    print("V4_DEPTH_RANDOM_START_EVAL_COMPLETE", flush=True)
    print(json.dumps(result["aggregates"], indent=2), flush=True)


if __name__ == "__main__":
    main()
