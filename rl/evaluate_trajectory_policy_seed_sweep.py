"""Evaluate every trained trajectory-policy seed on the same unseen protocols.

This is evaluation only: it does not retrain VPOCLIP or the viewpoint policy,
and it uses identical random-baseline seeds for every checkpoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .multistep_angle_object_trajectory_policy_v1 import (
    CACHE_ROOT,
    TRACK_ROOT,
    TRUE_UNSEEN_CLASSES,
    GPUCache,
    AngleObjectTrajectoryPolicy,
    attach_view_valid,
    attach_object_tracks,
    evaluate_protocol,
)


ROOT = CACHE_ROOT.parents[2]
DEFAULT_POLICY_ROOT = ROOT / "work_dir" / "multistep_angle_object_trajectory_policy_v1"
DEFAULT_OUTPUT = DEFAULT_POLICY_ROOT / "seed_sweep_true_unseen"


def compact(report: dict[str, Any]) -> dict[str, Any]:
    metrics = report["metrics"]
    return {
        "single": float(metrics["single_start"]["top1"]),
        "random_exact": float(metrics["random_3view_exact"]["top1"]),
        "random_sampled": float(metrics["random_3view_sampled"]["top1"]),
        "policy": float(metrics["policy_3view"]["top1"]),
        "oracle": float(metrics["oracle_3view"]["top1"]),
        "all_valid": float(metrics["all_valid_views"]["top1"]),
        "policy_cost": float(metrics["policy_3view"]["mean_movement_cost"]),
        "episodes": int(report["episodes_policy_completed"]),
    }


def mean_std(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    keys = [key for key in rows[0] if key not in {"episodes"}]
    return {
        key: {
            "mean": float(np.mean([row[key] for row in rows])),
            "std": float(np.std([row[key] for row in rows], ddof=0)),
        }
        for key in keys
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--track-root", type=Path, default=TRACK_ROOT)
    parser.add_argument("--policy-root", type=Path, default=DEFAULT_POLICY_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260909, 20260910, 20260911])
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda:0")
    raw = GPUCache(args.cache_root / "test", device)
    attach_view_valid(raw, args.cache_root, "test")
    attach_object_tracks(raw, args.track_root, "test")
    args.output_root.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict[str, Any]] = {}
    fixed_protocols = ("fixed0", "fixed1", "fixed2", "fixed3", "random_start")
    for seed in args.seeds:
        checkpoint = args.policy_root / f"seed_{seed}" / "best.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        model = AngleObjectTrajectoryPolicy().to(device)
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(state["online"], strict=True)
        model.eval()
        results[str(seed)] = {}
        for protocol in fixed_protocols:
            # Keep stochastic baselines identical across policy checkpoints.
            report = evaluate_protocol(
                model, raw, TRUE_UNSEEN_CLASSES, protocol, 20260910 + len(protocol)
            )
            results[str(seed)][protocol] = compact(report)
            report["evaluated_checkpoint"] = str(checkpoint)
            report["evaluation_seed"] = 20260910 + len(protocol)
            (args.output_root / f"seed_{seed}").mkdir(parents=True, exist_ok=True)
            (args.output_root / f"seed_{seed}" / f"evaluation_{protocol}.json").write_text(
                json.dumps(report, indent=2), encoding="utf-8"
            )
            print("TRAJECTORY_SEED_SWEEP", seed, protocol, json.dumps(results[str(seed)][protocol]), flush=True)
        del model
        torch.cuda.empty_cache()

    aggregates = {}
    for protocol in fixed_protocols:
        rows = [results[str(seed)][protocol] for seed in args.seeds]
        aggregates[protocol] = {
            "seeds": [int(seed) for seed in args.seeds],
            "mean_std": mean_std(rows),
        }
    summary = {
        "experiment": "trajectory_policy_seed_sweep_true_unseen",
        "class_bank": [int(x) for x in TRUE_UNSEEN_CLASSES],
        "protocols": list(fixed_protocols),
        "baseline_seed_policy": "identical random evaluation seed for every checkpoint",
        "results_by_seed": results,
        "aggregates": aggregates,
    }
    (args.output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("TRAJECTORY_SEED_SWEEP_COMPLETE", json.dumps(aggregates), flush=True)


if __name__ == "__main__":
    main()
