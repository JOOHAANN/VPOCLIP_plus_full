"""Unified 30-seed unseen evaluation for depth trajectory variants."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import multistep_angle_object_trajectory_policy_v1 as base
from . import multistep_angle_object_trajectory_depth_variants as depth_variants
from .evaluate_trajectory_variants_old_split import (
    oracle_path,
    path_metrics,
    policy_path,
    random_path,
    single_metrics,
)


TRAIN_SEEDS = [20260909, 20260910, 20260911]
DEFAULT_EVAL_SEEDS = list(range(20260920, 20260950))
NEW_UNSEEN = [14, 15, 25, 39, 46]
T_CRIT_95_N30 = 2.045229642


def ci95(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(T_CRIT_95_N30 * np.std(values, ddof=1) / math.sqrt(len(values)))


def aggregate(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "ci95_half_width": ci95(values),
    }


@torch.inference_mode()
def evaluate(
    variant: str,
    cache_root: Path,
    track_root: Path,
    depth_root: Path,
    policy_root: Path,
    output_root: Path,
    eval_seeds: list[int],
    class_bank: list[int],
    expected_episodes: int,
) -> dict[str, Any]:
    depth_variants.CACHE_ROOT = cache_root
    depth_variants.TRACK_ROOT = track_root
    depth_variants.DEPTH_ROOT = depth_root
    spec = depth_variants.configure_variant(variant)
    device = torch.device("cuda:0")

    raw = base.GPUCache(cache_root / "test", device)
    base.attach_view_valid(raw, cache_root, "test")
    base.attach_object_tracks(raw, track_root, "test")
    depth_audit = depth_variants.v4_depth.attach_depth_tracks(
        raw, cache_root, depth_root, "test"
    )
    bank = base.class_bank_tensor(class_bank, device)
    eligible = torch.isin(raw.labels, bank) & (raw.valid.sum(-1) >= 3)
    episodes = eligible.nonzero().flatten()
    if len(episodes) != expected_episodes:
        raise RuntimeError(f"expected {expected_episodes} unseen episodes, got {len(episodes)}")

    fixed_by_seed: dict[str, Any] = {}
    random_by_seed: dict[str, Any] = {}
    for train_seed in TRAIN_SEEDS:
        checkpoint = policy_root / f"seed_{train_seed}" / "best.pt"
        model = spec["model"]().to(device)
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(payload["online"], strict=True)
        model.eval()
        fixed_by_seed[str(train_seed)] = {}

        for start in range(base.NUM_VIEWS):
            fixed_episodes = (eligible & raw.valid[:, start]).nonzero().flatten()
            starts = torch.full_like(fixed_episodes, start)
            fixed_by_seed[str(train_seed)][f"fixed{start}"] = {}
            for moves in (1, 2):
                policy = policy_path(model, raw, fixed_episodes, starts, moves)
                oracle = oracle_path(raw, fixed_episodes, starts, bank, moves)
                random_rows = []
                for eval_seed in eval_seeds:
                    random = random_path(
                        starts, moves, int(eval_seed) + 100 * start + 1000 * moves, device
                    )
                    random_rows.append(path_metrics(raw, fixed_episodes, random, bank))
                fixed_by_seed[str(train_seed)][f"fixed{start}"][f"moves_{moves}"] = {
                    "episodes": int(len(fixed_episodes)),
                    "single": single_metrics(raw, fixed_episodes, starts, bank),
                    "policy": path_metrics(raw, fixed_episodes, policy, bank),
                    "random_mean": {
                        key: float(np.mean([row[key] for row in random_rows]))
                        for key in random_rows[0]
                    },
                    "random_std": {
                        key: float(np.std([row[key] for row in random_rows], ddof=1))
                        for key in random_rows[0]
                    },
                    "oracle": path_metrics(raw, fixed_episodes, oracle, bank),
                    "moves": moves,
                    "fusion_views": moves + 1,
                }

        random_by_seed[str(train_seed)] = {}
        for eval_seed in eval_seeds:
            starts = base.choose_random_starts(raw, episodes, int(eval_seed))
            random_by_seed[str(train_seed)][str(eval_seed)] = {
                "start_view_histogram": {
                    str(v): int((starts == v).sum().item()) for v in range(base.NUM_VIEWS)
                }
            }
            for moves in (1, 2):
                policy = policy_path(model, raw, episodes, starts, moves)
                random = random_path(starts, moves, int(eval_seed) + 1000 * moves, device)
                oracle = oracle_path(raw, episodes, starts, bank, moves)
                random_by_seed[str(train_seed)][str(eval_seed)][f"moves_{moves}"] = {
                    "episodes": int(len(episodes)),
                    "single": single_metrics(raw, episodes, starts, bank),
                    "policy": path_metrics(raw, episodes, policy, bank),
                    "random": path_metrics(raw, episodes, random, bank),
                    "oracle": path_metrics(raw, episodes, oracle, bank),
                    "moves": moves,
                    "fusion_views": moves + 1,
                }
        del model
        torch.cuda.empty_cache()

    fixed_aggregates: dict[str, Any] = {}
    for start in range(base.NUM_VIEWS):
        for moves in (1, 2):
            rows = [
                fixed_by_seed[str(seed)][f"fixed{start}"][f"moves_{moves}"]
                for seed in TRAIN_SEEDS
            ]
            policy = [row["policy"]["top1"] for row in rows]
            random = [row["random_mean"]["top1"] for row in rows]
            fixed_aggregates[f"fixed{start}_moves_{moves}"] = {
                "episodes": rows[0]["episodes"],
                "single_top1": rows[0]["single"]["top1"],
                "policy": aggregate(policy),
                "random": aggregate(random),
                "oracle_top1_mean": float(np.mean([row["oracle"]["top1"] for row in rows])),
                "policy_minus_random": aggregate([p - r for p, r in zip(policy, random)]),
            }

    random_aggregates: dict[str, Any] = {}
    for moves in (1, 2):
        policy_rows = []
        random_rows = []
        single_rows = []
        oracle_rows = []
        for eval_seed in eval_seeds:
            key = str(eval_seed)
            policy_rows.append(float(np.mean([
                random_by_seed[str(seed)][key][f"moves_{moves}"]["policy"]["top1"]
                for seed in TRAIN_SEEDS
            ])))
            random_rows.append(
                random_by_seed[str(TRAIN_SEEDS[0])][key][f"moves_{moves}"]["random"]["top1"]
            )
            single_rows.append(
                random_by_seed[str(TRAIN_SEEDS[0])][key][f"moves_{moves}"]["single"]["top1"]
            )
            oracle_rows.append(float(np.mean([
                random_by_seed[str(seed)][key][f"moves_{moves}"]["oracle"]["top1"]
                for seed in TRAIN_SEEDS
            ])))
        random_aggregates[f"random_start_moves_{moves}"] = {
            "episodes": int(len(episodes)),
            "fusion_views": moves + 1,
            "single": aggregate(single_rows),
            "policy": aggregate(policy_rows),
            "random": aggregate(random_rows),
            "oracle": aggregate(oracle_rows),
            "policy_minus_random": aggregate([p - r for p, r in zip(policy_rows, random_rows)]),
        }

    output_root.mkdir(parents=True, exist_ok=True)
    result = {
        "experiment": "trajectory_depth_variants_new50_5_unseen_30seeds",
        "variant": variant,
        "base_2d_variant": spec["base_2d"],
        "class_bank": class_bank,
        "episodes": int(len(episodes)),
        "train_seeds": TRAIN_SEEDS,
        "eval_seeds": eval_seeds,
        "policy_root": str(policy_root),
        "depth_root": str(depth_root),
        "depth_join_audit": depth_audit,
        "random_start_protocol": "per-episode random valid start; paired policy/random path seeds",
        "fixed_aggregates": fixed_aggregates,
        "random_start_aggregates": random_aggregates,
        "fixed_results_by_train_seed": fixed_by_seed,
        "random_start_results_by_train_seed": random_by_seed,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=tuple(depth_variants.VARIANTS), required=True)
    parser.add_argument("--cache-root", type=Path, default=depth_variants.CACHE_ROOT)
    parser.add_argument("--track-root", type=Path, default=depth_variants.TRACK_ROOT)
    parser.add_argument("--depth-root", type=Path, default=depth_variants.DEPTH_ROOT)
    parser.add_argument("--policy-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=DEFAULT_EVAL_SEEDS)
    parser.add_argument("--class-bank", type=int, nargs="+", default=NEW_UNSEEN)
    parser.add_argument("--expected-episodes", type=int, default=271)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    result = evaluate(
        args.variant,
        args.cache_root,
        args.track_root,
        args.depth_root,
        args.policy_root,
        args.output_root,
        args.eval_seeds,
        args.class_bank,
        args.expected_episodes,
    )
    print("TRAJECTORY_DEPTH_VARIANT_EVALUATION_COMPLETE", args.variant, flush=True)
    print(json.dumps({
        "fixed": result["fixed_aggregates"],
        "random_start": result["random_start_aggregates"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
