"""Turn several pseudo-unseen runs into one stopping epoch.

Each run holds out a different group of 5 classes from the 50 seen ones and
tracks ZSL accuracy on them, so its best epoch is one noisy sample from the
distribution of "when does zero-shot transfer peak". Two rules are reported:

  median            - the middle of the per-group best epochs
  one-standard-error - the earliest epoch already within one standard error of
                       the peak, which avoids stopping late just because a
                       later epoch caught a favourable draw

The real unseen classes are never involved, so the chosen epoch can be applied
to a full 50-class run without touching the test set.

Usage:
  cd /workspace/VPOCLIP_plus
  python tools/aggregate_pseudo_unseen_epochs.py --groups 1 2 3 4
"""

import argparse
import glob
import json

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate pseudo-unseen selection epochs.")
    parser.add_argument("--groups", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--variant", default="objconcat")
    parser.add_argument("--smooth", type=int, default=9, help="Moving-average window over epochs.")
    return parser.parse_args()


def one_standard_error_epoch(scores, num_clips, smooth):
    """Earliest epoch whose smoothed score is within 1 SE of the smoothed peak."""

    smoothed = np.convolve(scores, np.ones(smooth) / smooth, mode="valid")
    peak = np.nanmax(smoothed)
    # Binomial standard error of an accuracy estimated on num_clips clips.
    standard_error = np.sqrt(peak * (1 - peak) / num_clips)
    reached = np.flatnonzero(smoothed >= peak - standard_error)
    offset = smooth // 2
    return int(reached[0]) + offset + 1, int(np.nanargmax(smoothed)) + offset + 1, peak, standard_error


def main():
    args = parse_args()
    rows = []
    for group in args.groups:
        paths = sorted(glob.glob(f"work_dir/pug{group}_{args.variant}/run_*/history.json"))
        if not paths:
            print(f"[group {group}] no history yet")
            continue
        history = json.load(open(paths[-1]))
        scores = np.array([np.nan if v is None else v for v in history["pseudo_zsl"]], dtype=float)
        if np.isnan(scores).all():
            print(f"[group {group}] no pseudo_zsl recorded")
            continue
        num_clips = 635  # ~585 removed training clips plus ~50 validation clips
        early, peak_epoch, peak, se = one_standard_error_epoch(scores, num_clips, args.smooth)
        rows.append({"group": group, "epochs_run": len(scores), "peak_epoch": peak_epoch,
                     "one_se_epoch": early, "peak": peak * 100, "se": se * 100})

    if not rows:
        return
    print(f"{'group':>6s} {'ran':>5s} {'peak%':>7s} {'SE':>5s} {'peak@ep':>8s} {'1SE@ep':>7s}")
    print("-" * 46)
    for r in rows:
        print(f"{r['group']:6d} {r['epochs_run']:5d} {r['peak']:6.2f}% {r['se']:5.2f} "
              f"{r['peak_epoch']:8d} {r['one_se_epoch']:7d}")

    peaks = [r["peak_epoch"] for r in rows]
    earlies = [r["one_se_epoch"] for r in rows]
    print(f"\nbest epochs (peak)  : {sorted(peaks)}  -> median {int(np.median(peaks))}")
    print(f"best epochs (1 SE)  : {sorted(earlies)}  -> median {int(np.median(earlies))}")
    print(f"\nrecommended stopping epoch: {int(np.median(earlies))}")


if __name__ == "__main__":
    main()
