"""Evaluate the current-cache non-RL v5/v6 policies over 30 random starts.

The v5/v6 policies are the historical utility-ranking controllers, not DQN
policies.  This evaluator keeps their original one-robot, three-observation
protocol and adds the same 30-seed / 95% CI reporting used elsewhere in the
project.  It is intentionally separate from the two-robot sequential replay
so that the protocol difference is explicit rather than hidden.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from . import multistep_angle_object_trajectory_policy_v1 as v1
from . import multistep_angle_object_trajectory_policy_v5_no_category_id as v5
from . import multistep_angle_object_trajectory_policy_v6_no_object as v6


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_ROOT = ROOT / "data" / "frame0_body_angle_rank_g1_pseudo_selected_stage_b" / "cache"
DEFAULT_TRACK_ROOT = ROOT / "data" / "frame0_body_angle_rank_g1_pseudo_selected_stage_b" / "object_tracks_full"
DEFAULT_SEEDS = list(range(20260920, 20260950))


def ci(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean()) if len(array) else 0.0
    std = float(array.std(ddof=1)) if len(array) > 1 else 0.0
    half = 1.96 * std / math.sqrt(len(array)) if len(array) > 1 else 0.0
    return {
        "n": int(len(array)),
        "mean": mean,
        "std": std,
        "ci95_half_width": half,
        "ci95_low": mean - half,
        "ci95_high": mean + half,
    }


def load_raw(cache_root: Path, track_root: Path, split: str, device: torch.device, with_objects: bool):
    raw = v1.GPUCache(cache_root / split, device)
    v1.attach_view_valid(raw, cache_root, split)
    if with_objects:
        v1.attach_object_tracks(raw, track_root, split)
    return raw


def selected_checkpoint(output_root: Path) -> Path:
    summary = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
    return Path(summary["selection"]["chosen"]["checkpoint"])


def evaluate_one(
    name: str,
    output_root: Path,
    cache_root: Path,
    track_root: Path,
    device: torch.device,
    seeds: list[int],
) -> dict[str, Any]:
    if name == "v5":
        model_cls = v5.CategoryFreeObjectTrajectoryPolicy
        builder = v5.build_trajectory_state_v5
        with_objects = True
    elif name == "v6":
        model_cls = v6.PersonGeometryTrajectoryPolicy
        builder = v6.build_trajectory_state_v5
        with_objects = False
    else:
        raise ValueError(name)

    v1.build_trajectory_state = builder
    raw = load_raw(cache_root, track_root, "test", device, with_objects)
    model = model_cls().to(device)
    checkpoint = selected_checkpoint(output_root)
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(saved["online"], strict=True)
    model.eval()

    rows: list[dict[str, Any]] = []
    for seed in seeds:
        report = v1.evaluate_protocol(
            model,
            raw,
            v1.TRUE_UNSEEN_CLASSES,
            "random_start",
            int(seed),
        )
        metrics = report["metrics"]
        rows.append(
            {
                "seed": int(seed),
                "policy_accuracy": float(metrics["policy_3view"]["top1"]),
                "policy_move_cost": float(metrics["policy_3view"]["mean_movement_cost"]),
                "random_exact_accuracy": float(metrics["random_3view_exact"]["top1"]),
                "random_exact_move_cost": float(metrics["random_3view_exact"]["mean_movement_cost"]),
                "single_accuracy": float(metrics["single_start"]["top1"]),
            }
        )

    summary = {
        "variant": name,
        "protocol": "historical non-RL one-robot random-start three-view evaluation",
        "checkpoint": str(checkpoint),
        "class_bank": [int(x) for x in v1.TRUE_UNSEEN_CLASSES],
        "seeds": [int(x) for x in seeds],
        "metrics": {
            key: ci([row[key] for row in rows])
            for key in (
                "policy_accuracy",
                "policy_move_cost",
                "random_exact_accuracy",
                "random_exact_move_cost",
                "single_accuracy",
            )
        },
        "per_seed": rows,
    }
    output = output_root / "random_start_30seeds.json"
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--track-root", type=Path, default=DEFAULT_TRACK_ROOT)
    parser.add_argument("--v5-output", type=Path, required=True)
    parser.add_argument("--v6-output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument(
        "--unseen-classes",
        type=int,
        nargs="+",
        default=None,
        help="Explicit test-only unseen class bank; overrides the historical default.",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.unseen_classes is not None:
        unseen = sorted(set(int(x) for x in args.unseen_classes))
        if not unseen:
            raise ValueError("--unseen-classes must not be empty")
        v1.TRUE_UNSEEN_CLASSES = unseen
    device = torch.device("cuda:0")
    torch.set_float32_matmul_precision("high")

    reports = {
        "v5": evaluate_one("v5", args.v5_output, args.cache_root, args.track_root, device, args.seeds),
        "v6": evaluate_one("v6", args.v6_output, args.cache_root, args.track_root, device, args.seeds),
    }
    print(json.dumps(reports, indent=2), flush=True)


if __name__ == "__main__":
    main()
