#!/usr/bin/env python3
"""Evaluate the frozen strict-45/5/5 VPOCLIP without active movement.

The no-movement baseline uses the same final unseen test episodes and the
same 30 evaluation seeds as the DDQN report.  Robot A's independently sampled
initial candidate is the only observation; no target selection or fusion gate
is applied.  Fixed rank views and the per-episode four-view average are also
reported as diagnostics.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


CACHE = Path("/home/youhan/ws/VPOCLIP_plus_full/work_dir/frame0_body_angle_rank_strict45_10/cache/test")
OUT = Path("/home/youhan/ws/VPOCLIP_plus_full/work_dir/dual_robot_depth_ddqn_strict45_5/vpoclip_nomove.json")
UNSEEN = np.asarray([25, 39, 46, 52, 54], dtype=np.int64)
SEEDS = list(range(20260920, 20260950))


def ci(values: list[float]) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float64)
    mean = float(x.mean())
    std = float(x.std(ddof=1))
    half = 1.96 * std / math.sqrt(len(x))
    return {
        "n": int(len(x)),
        "mean": mean,
        "std": std,
        "ci95_half": half,
        "ci95_low": mean - half,
        "ci95_high": mean + half,
    }


def main() -> None:
    labels = np.load(CACHE / "labels.npy", allow_pickle=False).astype(np.int64)
    logits = np.load(CACHE / "logits.npy", mmap_mode="r", allow_pickle=False)
    valid = np.load(CACHE / "view_valid.npy", allow_pickle=False).astype(bool)
    episodes = np.flatnonzero(np.isin(labels, UNSEEN))
    bank = UNSEEN

    fixed = {}
    for view in range(logits.shape[1]):
        scores = logits[episodes, view][:, bank]
        fixed[f"rank_{view}"] = float((scores.argmax(-1) == (labels[episodes, None] == bank).argmax(-1)).mean())

    per_seed_random_start = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        starts = np.asarray([int(rng.choice(np.flatnonzero(row))) for row in valid[episodes]], dtype=np.int64)
        scores = logits[episodes, starts][:, bank]
        correct = scores.argmax(-1) == (labels[episodes, None] == bank).argmax(-1)
        per_seed_random_start.append(float(correct.mean()))

    all_view_correct = logits[episodes][:, :, bank].argmax(-1) == (labels[episodes, None, None] == bank).argmax(-1)
    per_episode_view_average = all_view_correct.mean(axis=1)

    result = {
        "protocol": "strict_45_5_5",
        "cache": str(CACHE),
        "final_unseen_classes": UNSEEN.tolist(),
        "episodes": int(len(episodes)),
        "evaluation_seeds": SEEDS,
        "recognizer": "frozen VPOCLIP strict-45/10 checkpoint evaluated on the final five-class unseen bank",
        "primary_no_movement_definition": "one VPOCLIP view from Robot A's sampled initial candidate; no DDQN, no target movement, no multi-view fusion",
        "random_initial_single_view_accuracy": ci(per_seed_random_start),
        "fixed_rank_accuracy": fixed,
        "four_view_mean_diagnostic": {
            "mean_over_episodes": float(per_episode_view_average.mean()),
            "std_over_episodes": float(per_episode_view_average.std(ddof=1)),
        },
        "per_seed_random_initial_single_view": per_seed_random_start,
    }
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
