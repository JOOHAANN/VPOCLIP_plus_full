"""Train the preserved 2-D trajectory variants on a new VPOCLIP cache.

The policy inputs are intentionally the original 2-D trajectory inputs.  The
new recognizer checkpoint affects only the offline per-view logits/z used by
the training target and evaluation; no depth feature is added here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import multistep_angle_object_trajectory_policy_v1 as base
from . import multistep_angle_object_trajectory_policy_v4 as v4
from . import multistep_angle_object_trajectory_policy_v5_no_category_id as v5
from . import multistep_angle_object_trajectory_policy_v6_no_object as v6
from . import multistep_angle_object_trajectory_ablation as ablation


ROOT = Path(__file__).resolve().parents[1]
RECOGNIZER_CHECKPOINT = (
    ROOT / "work_dir" / "g1_pseudo_selected_stage_b" / "last_model.pth"
)
DEFAULT_TRACK_ROOT = (
    ROOT / "data" / "trajectory_latest_vpoclip_new50_5" / "object_tracks_full"
)


def _configure(args: argparse.Namespace) -> None:
    base.CACHE_ROOT = args.cache_root
    base.TRACK_ROOT = args.track_root
    base.OUTPUT_ROOT = args.output_root

    if args.variant == "v4":
        base.build_trajectory_state = v4.build_trajectory_state_v4
        base.AngleObjectTrajectoryPolicy = v4.BodyOrientationTrajectoryPolicy
        base.attach_object_tracks = base.attach_object_tracks
    elif args.variant == "v5":
        base.build_trajectory_state = v5.build_trajectory_state_v5
        base.AngleObjectTrajectoryPolicy = v5.CategoryFreeObjectTrajectoryPolicy
        base.attach_object_tracks = base.attach_object_tracks
    elif args.variant == "v6":
        base.build_trajectory_state = v6.build_trajectory_state_v5
        base.AngleObjectTrajectoryPolicy = v6.PersonGeometryTrajectoryPolicy
        base.attach_object_tracks = v6.attach_no_object_tracks
    elif args.variant == "object_only":
        base.build_trajectory_state = ablation.build_object_only_state
        base.AngleObjectTrajectoryPolicy = ablation.ObjectOnlyTrajectoryPolicy
        base.attach_object_tracks = base.attach_object_tracks
    elif args.variant == "geometry_only":
        base.build_trajectory_state = ablation.build_geometry_only_state
        base.AngleObjectTrajectoryPolicy = ablation.GeometryOnlyPolicy
        base.attach_object_tracks = ablation.attach_no_object_tracks
    else:
        raise ValueError(args.variant)


def _forward_argv(args: argparse.Namespace) -> list[str]:
    values = [
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
    return values


def _write_experiment_metadata(args: argparse.Namespace) -> None:
    payload = {
        "variant": args.variant,
        "policy_family": "original_2d_trajectory_fusion",
        "depth_features": False,
        "input_contract": {
            "frame_size": [640, 480],
            "frames_per_view": 13,
            "views": ["C001", "C003", "C005", "C007"],
            "new_vpoclip_checkpoint": str(RECOGNIZER_CHECKPOINT),
            "new_cache_root": str(args.cache_root),
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "updates_per_epoch": args.updates_per_epoch,
            "learning_rate": args.learning_rate,
            "seeds": args.seeds,
            "seen_classes": sorted(set(range(55)) - set(args.unseen_classes)),
            "pseudo_unseen_classes": args.pseudo_unseen_classes,
            "true_unseen_classes": args.unseen_classes,
        },
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "experiment_metadata.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        choices=("v4", "v5", "v6", "object_only", "geometry_only"),
        required=True,
    )
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--track-root", type=Path, default=DEFAULT_TRACK_ROOT)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--updates-per-epoch", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[20260909, 20260910, 20260911]
    )
    parser.add_argument(
        "--unseen-classes", type=int, nargs="+", default=[14, 15, 25, 39, 46]
    )
    parser.add_argument(
        "--pseudo-unseen-classes",
        type=int,
        nargs="+",
        default=[0, 2, 9, 10, 11, 17, 26, 34, 49, 50],
    )
    args = parser.parse_args()
    _configure(args)
    _write_experiment_metadata(args)

    original_argv = sys.argv
    sys.argv = [original_argv[0], *_forward_argv(args)]
    try:
        base.main()
    finally:
        sys.argv = original_argv

    # Make the provenance unambiguous even though base.main writes the shared
    # historical summary schema.
    summary_path = args.output_root / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary.update(
            {
                "variant": args.variant,
                "policy_family": "original_2d_trajectory_fusion",
                "depth_features": False,
                "new_vpoclip_checkpoint": str(RECOGNIZER_CHECKPOINT),
                "cache_root": str(args.cache_root),
            }
        )
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
