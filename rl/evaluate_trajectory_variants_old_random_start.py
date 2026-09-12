"""Evaluate old-split v4/v5/v6 with a random starting viewpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from . import multistep_angle_object_trajectory_policy_v1 as base
from .evaluate_trajectory_variants_old_split import (
    EVAL_SEEDS,
    OLD_UNSEEN,
    TRAIN_SEEDS,
    configure_variant,
    path_metrics,
    policy_path,
    random_path,
    single_metrics,
)


def evaluate_variant(
    name: str,
    cache_root: Path,
    track_root: Path,
    policy_root: Path,
    output_root: Path,
    eval_seeds: list[int],
) -> dict:
    model_cls, attach_tracks = configure_variant(name)
    device = torch.device("cuda:0")
    raw = base.GPUCache(cache_root / "test", device)
    base.attach_view_valid(raw, cache_root, "test")
    attach_tracks(raw, track_root, "test")
    bank = base.class_bank_tensor(OLD_UNSEEN, device)
    eligible = torch.isin(raw.labels, bank) & (raw.valid.sum(-1) >= 3)
    episodes = eligible.nonzero().flatten()
    if len(episodes) != 414:
        raise RuntimeError(f"old protocol expected 414 test episodes, got {len(episodes)}")

    by_train_seed = {}
    for train_seed in TRAIN_SEEDS:
        checkpoint = policy_root / f"seed_{train_seed}" / "best.pt"
        model = model_cls().to(device)
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(payload["online"], strict=True)
        model.eval()
        by_eval_seed = {}
        for eval_seed in eval_seeds:
            starts = base.choose_random_starts(raw, episodes, eval_seed)
            row = {}
            for moves in (1, 2):
                policy = policy_path(model, raw, episodes, starts, moves)
                random = random_path(starts, moves, eval_seed + 1000 * moves, device)
                row[f"moves_{moves}"] = {
                    "single": single_metrics(raw, episodes, starts, bank),
                    "policy": path_metrics(raw, episodes, policy, bank),
                    "random": path_metrics(raw, episodes, random, bank),
                }
            by_eval_seed[str(eval_seed)] = row
        by_train_seed[str(train_seed)] = by_eval_seed
        del model
        torch.cuda.empty_cache()

    aggregates = {}
    for moves in (1, 2):
        train_rows = []
        for train_seed in TRAIN_SEEDS:
            rows = [by_train_seed[str(train_seed)][str(s)][f"moves_{moves}"] for s in eval_seeds]
            train_rows.append({
                "single": float(np.mean([r["single"]["top1"] for r in rows])),
                "policy": float(np.mean([r["policy"]["top1"] for r in rows])),
                "policy_std_over_start_seeds": float(np.std([r["policy"]["top1"] for r in rows])),
            })
        random_rows = [by_train_seed[str(TRAIN_SEEDS[0])][str(s)][f"moves_{moves}"]["random"] for s in eval_seeds]
        policy_mean = float(np.mean([r["policy"] for r in train_rows]))
        random_mean = float(np.mean([r["top1"] for r in random_rows]))
        aggregates[f"random_start_moves_{moves}"] = {
            "single_top1_mean": float(np.mean([r["single"] for r in train_rows])),
            "policy_top1_mean_over_train_seeds": policy_mean,
            "policy_top1_std_over_train_seeds": float(np.std([r["policy"] for r in train_rows])),
            "random_top1_mean_over_eval_seeds": random_mean,
            "random_top1_std_over_eval_seeds": float(np.std([r["top1"] for r in random_rows])),
            "policy_minus_random": policy_mean - random_mean,
            "episodes": int(len(episodes)),
        }

    output_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "experiment": "trajectory_policy_random_start_old_split",
        "variant": name,
        "class_bank": OLD_UNSEEN,
        "episodes": int(len(episodes)),
        "train_seeds": TRAIN_SEEDS,
        "random_start_eval_seeds": eval_seeds,
        "start_protocol": "each sample independently selects one valid view from 0..3 per evaluation seed",
        "results_by_train_seed": by_train_seed,
        "aggregates": aggregates,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        choices=("v1", "v4", "v5", "v6", "object_only", "geometry_only"),
        required=True,
    )
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--track-root", type=Path, required=True)
    parser.add_argument("--policy-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--eval-seeds", type=int, nargs="+", default=list(range(20260920, 20260930))
    )
    args = parser.parse_args()
    summary = evaluate_variant(
        args.variant,
        args.cache_root,
        args.track_root,
        args.policy_root,
        args.output_root,
        args.eval_seeds,
    )
    print("OLD_SPLIT_RANDOM_START_COMPLETE", args.variant, flush=True)
    print(json.dumps(summary["aggregates"], indent=2), flush=True)


if __name__ == "__main__":
    main()
