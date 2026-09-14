#!/usr/bin/env python3
"""Select VPOCLIP stopping epochs for the merged group-A/group-B 45/10 run.

The ten real unseen classes are never read from the test split by this tool.
Two disjoint five-class pseudo-unseen sets are carved from the 45 training
classes, and their held-out ZSL curves are combined with an earliest
one-standard-error rule.  The resulting epoch counts are then used for the
full 45-class VPOCLIP run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


TRUE_UNSEEN = [1, 7, 14, 15, 18, 25, 39, 46, 52, 54]
PSEUDO_GROUPS = {
    "p3": [8, 27, 28, 35, 50],
    "p4": [3, 13, 20, 30, 41],
}


def read_scores(path: Path) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload.get("pseudo_zsl")
    if values is None:
        raise ValueError(f"{path} has no pseudo_zsl history")
    scores = np.asarray(
        [np.nan if value is None else float(value) for value in values], dtype=float
    )
    if scores.size == 0 or not np.isfinite(scores).any():
        raise ValueError(f"{path} contains no finite pseudo_zsl scores")
    return scores


def count_clips(data_dir: Path, classes: list[int]) -> int:
    total = 0
    for prefix in ("trimodal_train", "trimodal_val"):
        labels = np.load(data_dir / f"{prefix}_labels.npy", mmap_mode="r")
        total += int(np.isin(labels, classes).sum())
    return total


def select(scores: np.ndarray, num_clips: int, window: int) -> dict:
    window = max(1, min(int(window), len(scores)))
    valid = np.isfinite(scores)
    if not valid.all():
        indices = np.arange(len(scores))
        if valid.sum() == 1:
            scores = np.full_like(scores, scores[valid][0])
        else:
            scores = np.interp(indices, indices[valid], scores[valid])

    if window == 1:
        smoothed = scores.copy()
        offset = 0
    else:
        smoothed = np.convolve(scores, np.ones(window) / window, mode="valid")
        offset = window // 2

    peak = float(np.nanmax(smoothed))
    standard_error = float(
        np.sqrt(max(peak * (1.0 - peak), 0.0) / max(int(num_clips), 1))
    )
    eligible = np.flatnonzero(smoothed >= peak - standard_error)
    if eligible.size == 0:
        raise RuntimeError("No epoch is within one standard error of the pseudo peak")
    return {
        "epochs_run": int(len(scores)),
        "num_clips": int(num_clips),
        "peak": peak,
        "peak_percent": peak * 100.0,
        "standard_error": standard_error,
        "standard_error_percent": standard_error * 100.0,
        "peak_epoch": int(np.nanargmax(smoothed)) + offset + 1,
        "one_se_epoch": int(eligible[0]) + offset + 1,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-a", nargs=2, required=True, metavar=("P3_HISTORY", "P4_HISTORY"))
    parser.add_argument("--stage-b", nargs=2, required=True, metavar=("P3_HISTORY", "P4_HISTORY"))
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smooth-a", type=int, default=9)
    parser.add_argument("--smooth-b", type=int, default=3)
    args = parser.parse_args()

    true_unseen = set(TRUE_UNSEEN)
    for name, classes in PSEUDO_GROUPS.items():
        if true_unseen.intersection(classes):
            raise ValueError(f"pseudo group {name} overlaps true unseen classes")

    rows: dict[str, dict[str, dict]] = {"stage_a": {}, "stage_b": {}}
    for stage, paths, window in (
        ("stage_a", args.stage_a, args.smooth_a),
        ("stage_b", args.stage_b, args.smooth_b),
    ):
        for name, path in zip(("p3", "p4"), paths):
            classes = PSEUDO_GROUPS[name]
            rows[stage][name] = select(
                read_scores(Path(path)), count_clips(args.data_dir, classes), window
            )

    selected_a = int(np.median([rows["stage_a"][name]["one_se_epoch"] for name in PSEUDO_GROUPS]))
    selected_b = int(np.median([rows["stage_b"][name]["one_se_epoch"] for name in PSEUDO_GROUPS]))
    result = {
        "protocol": "merged_groupAB_strict_zsl_45_10",
        "true_unseen_excluded_from_selection": TRUE_UNSEEN,
        "true_unseen_actions": [f"A{x + 1:03d}" for x in TRUE_UNSEEN],
        "pseudo_groups": PSEUDO_GROUPS,
        "stage_a": rows["stage_a"],
        "stage_b": rows["stage_b"],
        "recommended_stage_a_epochs": selected_a,
        "recommended_stage_b_epochs": selected_b,
        "rule": "median of earliest epochs within one binomial SE of smoothed pseudo-ZSL peaks",
    }

    print("Pseudo-unseen checkpoint selection (true 10-class test never read)")
    for stage in ("stage_a", "stage_b"):
        print(f"\n{stage}:")
        for name in PSEUDO_GROUPS:
            row = rows[stage][name]
            print(
                f"  {name}: n={row['num_clips']} peak={row['peak_percent']:.2f}% "
                f"@epoch {row['peak_epoch']} 1SE={row['standard_error_percent']:.2f}% "
                f"-> epoch {row['one_se_epoch']}"
            )
    print(f"\nrecommended stage A epochs: {selected_a}")
    print(f"recommended stage B epochs: {selected_b}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
