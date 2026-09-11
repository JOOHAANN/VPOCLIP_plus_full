"""Repeat one/two-move unseen evaluation over multiple random-path seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .evaluate_trajectory_policy_one_two_moves import (
    OUTPUT_ROOT as ONE_TWO_ROOT,
    path_metrics,
    policy_path,
    random_path,
    single_metrics,
)
from .multistep_angle_object_trajectory_policy_v1 import (
    CACHE_ROOT,
    TRACK_ROOT,
    TRUE_UNSEEN_CLASSES,
    AngleObjectTrajectoryPolicy,
    GPUCache,
    attach_object_tracks,
    attach_view_valid,
    class_bank_tensor,
)


ROOT = CACHE_ROOT.parents[2]
POLICY_ROOT = ROOT / "work_dir" / "multistep_angle_object_trajectory_policy_v1"
OUTPUT_ROOT = ONE_TWO_ROOT / "random_seed_sweep"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--track-root", type=Path, default=TRACK_ROOT)
    parser.add_argument("--policy-root", type=Path, default=POLICY_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--train-seeds", type=int, nargs="+", default=[20260909, 20260910, 20260911])
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=list(range(20260920, 20260930)))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda:0")
    raw = GPUCache(args.cache_root / "test", device)
    attach_view_valid(raw, args.cache_root, "test")
    attach_object_tracks(raw, args.track_root, "test")
    bank = class_bank_tensor(TRUE_UNSEEN_CLASSES, device)
    episodes = (torch.isin(raw.labels, bank) & (raw.valid.sum(-1) >= 3)).nonzero().flatten()
    args.output_root.mkdir(parents=True, exist_ok=True)

    policy_values: dict[str, dict[str, list[float]]] = {}
    random_values: dict[str, dict[str, list[float]]] = {}
    single_values: dict[str, float] = {}
    for start in range(4):
        for moves in (1, 2):
            key = f"fixed{start}_moves_{moves}"
            policy_values[key] = {}
            random_values[key] = {}
            starts = torch.full_like(episodes, start)
            for seed in args.train_seeds:
                checkpoint = args.policy_root / f"seed_{seed}" / "best.pt"
                model = AngleObjectTrajectoryPolicy().to(device)
                payload = torch.load(checkpoint, map_location=device, weights_only=False)
                model.load_state_dict(payload["online"], strict=True)
                model.eval()
                policy = policy_path(model, raw, episodes, starts, moves)
                value = path_metrics(raw, episodes, policy, bank)
                policy_values[key].setdefault("top1", []).append(value["top1"])
                policy_values[key].setdefault("cost", []).append(value["mean_movement_cost"])
                del model
                torch.cuda.empty_cache()
            for eval_seed in args.eval_seeds:
                random = random_path(starts, moves, int(eval_seed) + 100 * start + 1000 * moves, device)
                value = path_metrics(raw, episodes, random, bank)
                random_values[key].setdefault("top1", []).append(value["top1"])
                random_values[key].setdefault("cost", []).append(value["mean_movement_cost"])
            single_values[key] = single_metrics(raw, episodes, starts, bank)["top1"]
            print("RANDOM_SEED_SWEEP", key, json.dumps({
                "single": single_values[key],
                "policy_top1_mean": float(np.mean(policy_values[key]["top1"])),
                "policy_top1_std": float(np.std(policy_values[key]["top1"])),
                "random_top1_mean": float(np.mean(random_values[key]["top1"])),
                "random_top1_std": float(np.std(random_values[key]["top1"])),
                "policy_minus_random": float(np.mean(policy_values[key]["top1"]) - np.mean(random_values[key]["top1"])),
            }), flush=True)

    aggregates: dict[str, Any] = {}
    for key in policy_values:
        aggregates[key] = {
            "single_top1": single_values[key],
            "policy_top1_mean_over_train_seeds": float(np.mean(policy_values[key]["top1"])),
            "policy_top1_std_over_train_seeds": float(np.std(policy_values[key]["top1"])),
            "policy_cost_mean_over_train_seeds": float(np.mean(policy_values[key]["cost"])),
            "random_top1_mean_over_eval_seeds": float(np.mean(random_values[key]["top1"])),
            "random_top1_std_over_eval_seeds": float(np.std(random_values[key]["top1"])),
            "random_cost_mean_over_eval_seeds": float(np.mean(random_values[key]["cost"])),
            "policy_minus_random_mean": float(np.mean(policy_values[key]["top1"]) - np.mean(random_values[key]["top1"])),
            "train_seed_count": len(args.train_seeds),
            "random_eval_seed_count": len(args.eval_seeds),
        }
    summary = {
        "experiment": "one_two_move_random_evaluation_seed_sweep_true_unseen",
        "class_bank": [int(x) for x in TRUE_UNSEEN_CLASSES],
        "episodes": int(len(episodes)),
        "train_seeds": [int(x) for x in args.train_seeds],
        "random_eval_seeds": [int(x) for x in args.eval_seeds],
        "raw_values": {"policy": policy_values, "random": random_values},
        "aggregates": aggregates,
    }
    (args.output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("RANDOM_SEED_SWEEP_COMPLETE", json.dumps(aggregates), flush=True)


if __name__ == "__main__":
    main()
