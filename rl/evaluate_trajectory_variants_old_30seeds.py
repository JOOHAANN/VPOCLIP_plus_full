"""Thirty-seed trajectory evaluation on the historical 50/5 split.

This wrapper reuses the tested fixed/random-start evaluator implementation but
supplies the historical unseen bank and episode count.  It is kept separate
from both the new-split evaluator and the legacy ten-seed evaluator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .evaluate_trajectory_variants_new_split import evaluate_variant


OLD_UNSEEN = [0, 2, 26, 34, 50]
DEFAULT_EVAL_SEEDS = list(range(20260920, 20260950))


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
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=DEFAULT_EVAL_SEEDS)
    args = parser.parse_args()

    summary = evaluate_variant(
        args.variant,
        args.cache_root,
        args.track_root,
        args.policy_root,
        args.output_root,
        list(args.eval_seeds),
        OLD_UNSEEN,
        expected_episodes=414,
    )
    summary["experiment"] = "trajectory_policy_old_50_5_unseen_30_seed_ablation"
    summary["random_start_eval_seeds"] = list(args.eval_seeds)
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("OLD_SPLIT_30SEED_TRAJECTORY_EVALUATION_COMPLETE", args.variant, flush=True)


if __name__ == "__main__":
    main()
