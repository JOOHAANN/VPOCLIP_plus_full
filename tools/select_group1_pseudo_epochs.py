"""Select stopping epochs from group1 pseudo-unseen probes only.

The real group1 unseen classes are A015/A016/A026/A040/A047 (labels
14,15,25,39,46).  This utility intentionally reads only the two disjoint
pseudo groups [8,27,28,35,50] and [3,13,30,41,52].  It reports both the
smoothed peak and the earliest point within one binomial standard error of that
peak; the final recommendation is the median one-SE epoch across the probes.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def _scores(path):
    values = json.loads(Path(path).read_text())["pseudo_zsl"]
    scores = np.asarray([np.nan if value is None else float(value) for value in values], dtype=float)
    if not np.isfinite(scores).any():
        raise ValueError(f"No finite pseudo_zsl scores in {path}")
    return scores


def _num_clips(data_dir, classes):
    total = 0
    for prefix in ("trimodal_train", "trimodal_val"):
        labels = np.load(Path(data_dir) / f"{prefix}_labels.npy", mmap_mode="r")
        total += int(np.isin(labels, classes).sum())
    return total


def _select(scores, num_clips, window):
    # A centered moving average avoids selecting a one-epoch sampling spike.
    window = max(1, min(int(window), len(scores)))
    if window == 1:
        smoothed = scores.copy()
        offset = 0
    else:
        valid = np.isfinite(scores)
        # Validation is performed every epoch in these configs.  Interpolate
        # only if an interrupted/legacy history has a missing entry.
        if not valid.all():
            indices = np.arange(len(scores))
            scores = np.interp(indices, indices[valid], scores[valid])
        smoothed = np.convolve(scores, np.ones(window) / window, mode="valid")
        offset = window // 2
    peak = float(np.nanmax(smoothed))
    se = float(np.sqrt(max(peak * (1.0 - peak), 0.0) / max(num_clips, 1)))
    eligible = np.flatnonzero(smoothed >= peak - se)
    one_se = int(eligible[0]) + offset + 1
    peak_epoch = int(np.nanargmax(smoothed)) + offset + 1
    return {
        "epochs_run": int(len(scores)),
        "num_clips": int(num_clips),
        "peak": peak,
        "peak_percent": peak * 100.0,
        "standard_error": se,
        "standard_error_percent": se * 100.0,
        "peak_epoch": peak_epoch,
        "one_se_epoch": one_se,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-a", nargs=2, required=True, metavar=("P3_HISTORY", "P4_HISTORY"))
    parser.add_argument("--stage-b", nargs=2, required=True, metavar=("P3_HISTORY", "P4_HISTORY"))
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--smooth-a", type=int, default=9)
    parser.add_argument("--smooth-b", type=int, default=3)
    args = parser.parse_args()

    groups = {
        "p3": [8, 27, 28, 35, 50],
        "p4": [3, 13, 30, 41, 52],
    }
    rows = {"stage_a": {}, "stage_b": {}}
    for stage, paths, window in (
        ("stage_a", args.stage_a, args.smooth_a),
        ("stage_b", args.stage_b, args.smooth_b),
    ):
        for name, path in zip(("p3", "p4"), paths):
            clips = _num_clips(args.data_dir, groups[name])
            rows[stage][name] = _select(_scores(path), clips, window)

    selected_a = int(np.median([rows["stage_a"][name]["one_se_epoch"] for name in ("p3", "p4")]))
    selected_b = int(np.median([rows["stage_b"][name]["one_se_epoch"] for name in ("p3", "p4")]))
    result = {
        "true_unseen_excluded_from_selection": [14, 15, 25, 39, 46],
        "pseudo_groups": groups,
        "stage_a": rows["stage_a"],
        "stage_b": rows["stage_b"],
        "recommended_stage_a_epochs": selected_a,
        "recommended_stage_b_epochs": selected_b,
    }

    print("Pseudo-unseen checkpoint selection (no real unseen test access)")
    for stage in ("stage_a", "stage_b"):
        print(f"\n{stage}:")
        for name in ("p3", "p4"):
            row = rows[stage][name]
            print(
                f"  {name}: n={row['num_clips']} peak={row['peak_percent']:.2f}% "
                f"@epoch {row['peak_epoch']} 1SE={row['standard_error_percent']:.2f}% "
                f"-> epoch {row['one_se_epoch']}"
            )
    print(f"\nrecommended stage A epochs: {selected_a}")
    print(f"recommended stage B epochs: {selected_b}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
