"""30-seed fixed-start and random-start evaluation on the new 50/5 split.

This evaluator is deliberately separate from the historical old-split
evaluator.  It evaluates the true new unseen bank ``[14, 15, 25, 39, 46]``
using the same 1-move/2-move trajectory protocol as the v1 ablation report.
Each invocation evaluates one variant in a fresh process, so the state/model
adapter monkeypatches cannot leak between variants.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from . import multistep_angle_object_trajectory_policy_v1 as base
from .evaluate_trajectory_variants_old_split import (
    configure_variant,
    oracle_path,
    path_metrics,
    policy_path,
    random_path,
    single_metrics,
)


TRAIN_SEEDS = [20260909, 20260910, 20260911]
DEFAULT_EVAL_SEEDS = list(range(20260920, 20260950))
NEW_UNSEEN = [14, 15, 25, 39, 46]


def evaluate_variant(
    name: str,
    cache_root: Path,
    track_root: Path,
    policy_root: Path,
    output_root: Path,
    eval_seeds: list[int],
    class_bank: list[int],
    expected_episodes: int,
) -> dict[str, Any]:
    model_cls, attach_tracks = configure_variant(name)
    device = torch.device("cuda:0")
    raw = base.GPUCache(cache_root / "test", device)
    base.attach_view_valid(raw, cache_root, "test")
    attach_tracks(raw, track_root, "test")
    bank = base.class_bank_tensor(class_bank, device)
    eligible = torch.isin(raw.labels, bank) & (raw.valid.sum(-1) >= 3)
    episodes = eligible.nonzero().flatten()
    if len(episodes) != expected_episodes:
        raise RuntimeError(
            f"new protocol expected {expected_episodes} test episodes, got {len(episodes)}"
        )

    fixed_results: dict[str, Any] = {}
    random_results: dict[str, Any] = {}
    for train_seed in TRAIN_SEEDS:
        checkpoint = policy_root / f"seed_{train_seed}" / "best.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        model = model_cls().to(device)
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(payload["online"], strict=True)
        model.eval()

        fixed_results[str(train_seed)] = {}
        for start in range(base.NUM_VIEWS):
            starts = torch.full_like(episodes, start)
            fixed_results[str(train_seed)][f"fixed{start}"] = {}
            fixed_episodes = (eligible & raw.valid[:, start]).nonzero().flatten()
            fixed_starts = torch.full_like(fixed_episodes, start)
            for moves in (1, 2):
                policy = policy_path(model, raw, fixed_episodes, fixed_starts, moves)
                oracle = oracle_path(raw, fixed_episodes, fixed_starts, bank, moves)
                random_values = []
                for eval_seed in eval_seeds:
                    random = random_path(
                        fixed_starts,
                        moves,
                        eval_seed + 100 * start + 1000 * moves,
                        device,
                    )
                    random_values.append(path_metrics(raw, fixed_episodes, random, bank))
                fixed_results[str(train_seed)][f"fixed{start}"][f"moves_{moves}"] = {
                    "single": single_metrics(raw, fixed_episodes, fixed_starts, bank),
                    "policy": path_metrics(raw, fixed_episodes, policy, bank),
                    "oracle": path_metrics(raw, fixed_episodes, oracle, bank),
                    "random_mean": {
                        metric: float(np.mean([row[metric] for row in random_values]))
                        for metric in random_values[0]
                    },
                    "random_std": {
                        metric: float(np.std([row[metric] for row in random_values]))
                        for metric in random_values[0]
                    },
                    "episodes": int(len(fixed_episodes)),
                    "moves": moves,
                    "fusion_views": moves + 1,
                }

        random_results[str(train_seed)] = {}
        for eval_seed in eval_seeds:
            starts = base.choose_random_starts(raw, episodes, eval_seed)
            random_results[str(train_seed)][str(eval_seed)] = {}
            for moves in (1, 2):
                policy = policy_path(model, raw, episodes, starts, moves)
                random = random_path(starts, moves, eval_seed + 1000 * moves, device)
                oracle = oracle_path(raw, episodes, starts, bank, moves)
                random_results[str(train_seed)][str(eval_seed)][f"moves_{moves}"] = {
                    "single": single_metrics(raw, episodes, starts, bank),
                    "policy": path_metrics(raw, episodes, policy, bank),
                    "random": path_metrics(raw, episodes, random, bank),
                    "oracle": path_metrics(raw, episodes, oracle, bank),
                    "episodes": int(len(episodes)),
                    "moves": moves,
                    "fusion_views": moves + 1,
                }
        del model
        torch.cuda.empty_cache()

    fixed_aggregates: dict[str, Any] = {}
    for start in range(base.NUM_VIEWS):
        for moves in (1, 2):
            rows = [
                fixed_results[str(seed)][f"fixed{start}"][f"moves_{moves}"]
                for seed in TRAIN_SEEDS
            ]
            fixed_aggregates[f"fixed{start}_moves_{moves}"] = {
                "episodes": rows[0]["episodes"],
                "single_top1": rows[0]["single"]["top1"],
                "policy_top1_mean": float(np.mean([r["policy"]["top1"] for r in rows])),
                "policy_top1_std_over_train_seeds": float(
                    np.std([r["policy"]["top1"] for r in rows])
                ),
                "random_top1_mean": float(
                    np.mean([r["random_mean"]["top1"] for r in rows])
                ),
                "random_top1_std_mean": float(
                    np.mean([r["random_std"]["top1"] for r in rows])
                ),
                "oracle_top1_mean": float(np.mean([r["oracle"]["top1"] for r in rows])),
                "policy_minus_random": float(
                    np.mean([r["policy"]["top1"] for r in rows])
                    - np.mean([r["random_mean"]["top1"] for r in rows])
                ),
            }

    random_aggregates: dict[str, Any] = {}
    for moves in (1, 2):
        policy_rows = []
        random_rows = []
        oracle_rows = []
        for eval_seed in eval_seeds:
            policy_rows.append(
                np.mean([
                    random_results[str(seed)][str(eval_seed)][f"moves_{moves}"]["policy"]["top1"]
                    for seed in TRAIN_SEEDS
                ])
            )
            random_rows.append(
                random_results[str(TRAIN_SEEDS[0])][str(eval_seed)][f"moves_{moves}"]["random"]["top1"]
            )
            oracle_rows.append(
                np.mean([
                    random_results[str(seed)][str(eval_seed)][f"moves_{moves}"]["oracle"]["top1"]
                    for seed in TRAIN_SEEDS
                ])
            )
        random_aggregates[f"random_start_moves_{moves}"] = {
            "episodes": int(len(episodes)),
            "single_top1_mean": float(np.mean([
                random_results[str(TRAIN_SEEDS[0])][str(s)][f"moves_{moves}"]["single"]["top1"]
                for s in eval_seeds
            ])),
            "policy_top1_mean_over_train_seeds": float(np.mean(policy_rows)),
            "policy_top1_std_over_eval_seeds": float(np.std(policy_rows)),
            "random_top1_mean_over_eval_seeds": float(np.mean(random_rows)),
            "random_top1_std_over_eval_seeds": float(np.std(random_rows)),
            "oracle_top1_mean_over_train_seeds": float(np.mean(oracle_rows)),
            "policy_minus_random": float(np.mean(policy_rows) - np.mean(random_rows)),
        }

    output_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "experiment": "trajectory_policy_new_50_5_unseen_30_seed_ablation",
        "variant": name,
        "class_bank": class_bank,
        "episodes": int(len(episodes)),
        "cache_root": str(cache_root),
        "track_root": str(track_root),
        "policy_root": str(policy_root),
        "train_seeds": TRAIN_SEEDS,
        "random_start_eval_seeds": eval_seeds,
        "fixed_protocol": "fixed view0/view1/view2/view3; each random baseline repeated over eval seeds",
        "random_start_protocol": "each sample independently selects one valid view from 0..3 per evaluation seed",
        "fixed_results_by_train_seed": fixed_results,
        "random_start_results_by_train_seed": random_results,
        "fixed_aggregates": fixed_aggregates,
        "random_start_aggregates": random_aggregates,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        choices=("v1", "v3", "v4", "v5", "v6", "object_only", "geometry_only"),
        required=True,
    )
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--track-root", type=Path, required=True)
    parser.add_argument("--policy-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--class-bank", type=int, nargs="+", default=NEW_UNSEEN)
    parser.add_argument("--expected-episodes", type=int, default=271)
    parser.add_argument(
        "--eval-seeds", type=int, nargs="+", default=DEFAULT_EVAL_SEEDS
    )
    args = parser.parse_args()
    summary = evaluate_variant(
        args.variant,
        args.cache_root,
        args.track_root,
        args.policy_root,
        args.output_root,
        args.eval_seeds,
        args.class_bank,
        args.expected_episodes,
    )
    print("NEW_SPLIT_TRAJECTORY_EVALUATION_COMPLETE", args.variant, flush=True)
    print(json.dumps({
        "fixed": summary["fixed_aggregates"],
        "random_start": summary["random_start_aggregates"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
