"""Retrain the original trajectory policies with only the first cached frame.

This adapter intentionally leaves the historical v1--v6 and ablation modules
untouched.  It reuses their original two-view/multi-step target, loss,
optimizer, split, and evaluation code, but changes only the sensor temporal
length:

    original pose:          [13, 17, 2] -> [1, 17, 2]
    original object track:  [13, 50, C] -> [1, 50, C]

The first frame is the first entry in ``window.sample_frames`` (index 0 of the
already cached 13 samples).  If multiple viewpoints have already been
observed, their one-frame states are averaged exactly as in the original
trajectory implementation.  VPOCLIP logits and the delta-margin reward are
unchanged and are used only by the offline target/evaluation path.

Each variant is launched in its own process by the surrounding tmux pipeline,
so monkeypatches cannot leak between variants.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_ROOT = (
    ROOT / "data" / "trajectory_latest_vpoclip_new50_5" / "cache_compat"
)
DEFAULT_TRACK_ROOT = (
    ROOT / "data" / "trajectory_latest_vpoclip_new50_5" / "object_tracks_full"
)
DEFAULT_V3_TRACK_ROOT = ROOT / "data" / "angle_object_trajectory_v3_bbox_size"
DEFAULT_OUTPUT_ROOT = ROOT / "work_dir" / "trajectory_first_frame_new50_5"

VARIANTS = (
    "v1",
    "v3",
    "v4",
    "v5",
    "v6",
    "object_only",
    "geometry_only",
)

# These are the exact class lists used by the existing new 50/5 trajectory
# report.  They are passed explicitly so the result cannot silently fall back
# to the historical old split.
DEFAULT_TRUE_UNSEEN = [14, 15, 25, 39, 46]
DEFAULT_PSEUDO_UNSEEN = [0, 2, 9, 10, 11, 17, 26, 34, 49, 50]


def _first_frame_pose_cache(original_gpu_cache: type) -> type:
    """Return a GPUCache that retains only the first 34 pose values."""

    class FirstFrameGPUCache(original_gpu_cache):
        def __init__(self, root: Path, device: torch.device) -> None:
            super().__init__(root, device)
            expected = 13 * 17 * 2
            if self.pose.shape[-1] != expected:
                raise ValueError(
                    f"unexpected flattened pose width {self.pose.shape[-1]} != {expected}"
                )
            # The cache stores 13 consecutive [17,2] samples.  The first
            # sample is the first 34 values, not a middle-frame shortcut.
            self.pose = self.pose[..., : 17 * 2].contiguous()

    return FirstFrameGPUCache


def _attach_first_frame_tracks(
    raw: Any,
    track_root: Path,
    split: str,
    channels: int,
) -> None:
    """Attach the first sampled frame from a 4- or 6-channel track cache."""

    path = Path(track_root) / split / "object_position_track.npy"
    if not path.exists():
        raise FileNotFoundError(path)
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.ndim != 5:
        raise ValueError(f"track {path} has shape {array.shape}, expected 5-D")
    expected_prefix = (raw.num_episodes, 4, 13, 50, channels)
    if tuple(array.shape) != expected_prefix:
        raise ValueError(
            f"track {path} shape {array.shape} != {expected_prefix}; "
            "the first-frame experiment must use the same 13-frame cache"
        )
    first = np.array(array[:, :, :1, :, :], copy=True)
    raw.object_position_track = torch.from_numpy(first).to(
        device=raw.device, dtype=torch.float32, non_blocking=False
    )


def _attach_first_frame_tracks_4(raw: Any, track_root: Path, split: str) -> None:
    _attach_first_frame_tracks(raw, track_root, split, channels=4)


def _attach_first_frame_tracks_6(raw: Any, track_root: Path, split: str) -> None:
    _attach_first_frame_tracks(raw, track_root, split, channels=6)


def _attach_no_object_tracks(raw: Any, track_root: Path, split: str) -> None:
    del raw, track_root, split


def _configure_variant(variant: str, base: Any) -> tuple[Callable[..., None], Callable[..., Any], type]:
    """Return (track attachment, state builder, policy class)."""

    if variant == "v1":
        return _attach_first_frame_tracks_4, base.build_trajectory_state, base.AngleObjectTrajectoryPolicy

    if variant == "v3":
        from .. import multistep_angle_object_trajectory_policy_v3 as module

        return _attach_first_frame_tracks_6, module.build_trajectory_state_v3, module.ObjectSizeTrajectoryPolicy

    if variant == "v4":
        from .. import multistep_angle_object_trajectory_policy_v4 as module

        return _attach_first_frame_tracks_4, module.build_trajectory_state_v4, module.BodyOrientationTrajectoryPolicy

    if variant == "v5":
        from .. import multistep_angle_object_trajectory_policy_v5_no_category_id as module

        return _attach_first_frame_tracks_4, module.build_trajectory_state_v5, module.CategoryFreeObjectTrajectoryPolicy

    if variant == "v6":
        from .. import multistep_angle_object_trajectory_policy_v5_no_object as module

        return _attach_no_object_tracks, module.build_trajectory_state_v5, module.PersonGeometryTrajectoryPolicy

    if variant in {"object_only", "geometry_only"}:
        from .. import multistep_angle_object_trajectory_ablation as module

        if variant == "object_only":
            return _attach_first_frame_tracks_4, module.build_object_only_state, module.ObjectOnlyTrajectoryPolicy
        return _attach_no_object_tracks, module.build_geometry_only_state, module.GeometryOnlyPolicy

    raise ValueError(f"unsupported variant {variant}")


def _replace_frame_descriptions(value: Any) -> Any:
    if isinstance(value, str):
        return value.replace("13-frame", "1-frame").replace("13 frame", "1 frame")
    if isinstance(value, list):
        return [_replace_frame_descriptions(item) for item in value]
    if isinstance(value, dict):
        return {key: _replace_frame_descriptions(item) for key, item in value.items()}
    return value


def _rewrite_metadata(output_root: Path, variant: str) -> None:
    """Make the temporal ablation explicit in every generated artifact."""

    config_path = output_root / "config_resolved.json"
    config: dict[str, Any] = {}
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
    config.update(
        {
            "experiment": "trajectory_first_sampled_frame",
            "variant": variant,
            "temporal_input_frames": 1,
            "selected_sample_frame_index": 0,
            "original_temporal_input_frames": 13,
            "reward_and_fusion": "unchanged from original v-series two-view fusion",
            "true_unseen_classes": DEFAULT_TRUE_UNSEEN,
            "pseudo_unseen_classes": DEFAULT_PSEUDO_UNSEEN,
            "isolated_output_root": str(output_root),
        }
    )
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    manifest = {
        "experiment": "trajectory_first_sampled_frame",
        "variant": variant,
        "what_changed": "Each 13-frame per-view sensor sequence is truncated to its first sampled frame before state construction.",
        "what_did_not_change": [
            "VPOCLIP logits and frozen recognizer",
            "seen/pseudo-unseen/true-unseen split",
            "delta-margin target reward",
            "listwise + pairwise ranking loss",
            "candidate geometry and reachability",
            "two-view/multi-step evaluation protocol",
        ],
        "temporal_input": {
            "original_frames": 13,
            "used_frames": 1,
            "sample_frame_index": 0,
            "pose_shape_per_view": [1, 17, 2],
            "object_track_shape_per_view": [1, 50],
        },
        "class_split": {
            "true_unseen": DEFAULT_TRUE_UNSEEN,
            "pseudo_unseen": DEFAULT_PSEUDO_UNSEEN,
        },
    }
    (output_root / "first_frame_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    for path in output_root.rglob("*.json"):
        if path.name in {"config_resolved.json", "first_frame_manifest.json"}:
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if path.name in {"audit.json", "summary.json", "selection.json"}:
            value = _replace_frame_descriptions(value)
            if isinstance(value, dict):
                value.update(
                    {
                        "experiment": "trajectory_first_sampled_frame",
                        "variant": variant,
                        "temporal_input_frames": 1,
                        "selected_sample_frame_index": 0,
                        "original_temporal_input_frames": 13,
                    }
                )
        path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def run_variant(args: argparse.Namespace) -> None:
    # Importing the base module here, inside the per-variant process, keeps
    # every original module's monkeypatch surface local to this invocation.
    from .. import multistep_angle_object_trajectory_policy_v1 as base

    original_gpu_cache = base.GPUCache
    base.GPUCache = _first_frame_pose_cache(original_gpu_cache)
    base.NUM_FRAMES = 1
    effective_track_root = (
        args.v3_track_root if args.variant == "v3" else args.track_root
    )
    base.CACHE_ROOT = args.cache_root
    base.TRACK_ROOT = effective_track_root
    base.OUTPUT_ROOT = args.output_root

    attach_tracks, state_builder, policy_class = _configure_variant(args.variant, base)
    base.attach_object_tracks = attach_tracks
    base.build_trajectory_state = state_builder
    base.AngleObjectTrajectoryPolicy = policy_class

    # v1.main owns the tested training/evaluation loop.  Forward only its
    # normal CLI arguments so the protocol remains byte-for-byte comparable.
    forwarded = [
        "trajectory_first_frame",
        "--cache-root", str(args.cache_root),
        "--track-root", str(effective_track_root),
        "--output-root", str(args.output_root),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--updates-per-epoch", str(args.updates_per_epoch),
        "--learning-rate", str(args.learning_rate),
        "--seeds", *(str(seed) for seed in args.seeds),
        "--unseen-classes", *(str(x) for x in args.true_unseen),
        "--pseudo-unseen-classes", *(str(x) for x in args.pseudo_unseen),
    ]
    original_argv = sys.argv
    sys.argv = forwarded
    try:
        base.main()
    finally:
        sys.argv = original_argv
    _rewrite_metadata(args.output_root, args.variant)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--track-root", type=Path, default=DEFAULT_TRACK_ROOT)
    parser.add_argument("--v3-track-root", type=Path, default=DEFAULT_V3_TRACK_ROOT)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--updates-per-epoch", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260909, 20260910, 20260911])
    parser.add_argument("--true-unseen", type=int, nargs="+", default=DEFAULT_TRUE_UNSEEN)
    parser.add_argument("--pseudo-unseen", type=int, nargs="+", default=DEFAULT_PSEUDO_UNSEEN)
    return parser.parse_args()


if __name__ == "__main__":
    run_variant(parse_args())
