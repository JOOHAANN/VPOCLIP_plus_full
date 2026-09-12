"""Train one depth-augmented trajectory variant in an isolated output root."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import multistep_angle_object_trajectory_policy_v1 as base
from . import multistep_angle_object_trajectory_depth_variants as depth_variants


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=tuple(depth_variants.VARIANTS), required=True)
    parser.add_argument("--cache-root", type=Path, default=depth_variants.CACHE_ROOT)
    parser.add_argument("--track-root", type=Path, default=depth_variants.TRACK_ROOT)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--updates-per-epoch", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260909, 20260910, 20260911])
    parser.add_argument("--unseen-classes", type=int, nargs="+", default=[14, 15, 25, 39, 46])
    parser.add_argument("--pseudo-unseen-classes", type=int, nargs="+", default=[5, 6, 7, 22, 38, 42, 44, 45, 51, 52])
    args = parser.parse_args()

    spec = depth_variants.configure_variant(args.variant)
    base.CACHE_ROOT = args.cache_root
    base.TRACK_ROOT = args.track_root
    base.OUTPUT_ROOT = args.output_root
    depth_variants.CACHE_ROOT = args.cache_root
    depth_variants.TRACK_ROOT = args.track_root
    base.OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # v1 owns the complete, already validated training/evaluation loop.  A
    # fresh process is used for every variant so monkeypatches cannot leak.
    forwarded = [
        sys.argv[0],
        "--cache-root", str(args.cache_root),
        "--track-root", str(args.track_root),
        "--output-root", str(args.output_root),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--updates-per-epoch", str(args.updates_per_epoch),
        "--learning-rate", str(args.learning_rate),
        "--seeds", *(str(x) for x in args.seeds),
        "--unseen-classes", *(str(x) for x in args.unseen_classes),
        "--pseudo-unseen-classes", *(str(x) for x in args.pseudo_unseen_classes),
    ]
    original_argv = sys.argv
    sys.argv = forwarded
    try:
        base.main()
    finally:
        sys.argv = original_argv
    depth_variants.rewrite_metadata(args.output_root, args.variant, spec)
    print("TRAJECTORY_DEPTH_TRAIN_EVAL_COMPLETE", args.variant, flush=True)


if __name__ == "__main__":
    main()
