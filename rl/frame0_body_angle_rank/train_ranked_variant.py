"""Train one preserved trajectory variant on the frame-0 angle-rank cache.

The historical implementations are imported and configured in this process,
but their source files and outputs are never rewritten.  ``v2_legacy`` is
included because the repository has no independent trajectory-v2 module: it
uses the preserved old single-state angle+object implementation and is marked
as such in its provenance files rather than being mislabeled as a canonical
v2.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .. import multistep_angle_object_policy_v1 as v2_legacy
from .. import multistep_angle_object_trajectory_ablation as ablation
from .. import multistep_angle_object_trajectory_policy_v1 as v1
from .. import multistep_angle_object_trajectory_policy_v3 as v3
from .. import multistep_angle_object_trajectory_policy_v4 as v4
from .. import multistep_angle_object_trajectory_policy_v5_no_category_id as v5
from .. import multistep_angle_object_trajectory_policy_v6_no_object as v6
from . import true_yaw_v4
from ..train_rl_fast import GPUCache as BaseGPUCache


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_ROOT = ROOT / "data" / "frame0_body_angle_rank_g1_pseudo_selected_stage_b" / "cache"
DEFAULT_TRACK_ROOT = ROOT / "data" / "frame0_body_angle_rank_g1_pseudo_selected_stage_b" / "object_tracks_full"
DEFAULT_TRACK_ROOT_V3 = ROOT / "data" / "frame0_body_angle_rank_g1_pseudo_selected_stage_b" / "object_tracks_v3_bbox_size"
DEFAULT_OUTPUT_ROOT = ROOT / "work_dir" / "frame0_body_angle_rank_v1_v6"
DEFAULT_SEEDS = [20260909, 20260910, 20260911]


class Frame0YawGPUCache(BaseGPUCache):
    """Historical GPU cache plus the isolated frame-0 skeleton yaw array."""

    def __init__(self, root: Path, device: torch.device) -> None:
        super().__init__(root, device)
        path = root / "body_yaw_deg.npy"
        if not path.exists():
            raise FileNotFoundError(
                f"true-yaw cache is missing {path}; rebuild the isolated cache"
            )
        array = np.asarray(np.load(path, mmap_mode="r", allow_pickle=False), dtype=np.float32)
        expected = (self.num_episodes, 4)
        if tuple(array.shape) != expected:
            raise ValueError(f"body yaw shape {array.shape} != {expected}")
        self.body_yaw_deg = torch.from_numpy(np.array(array, copy=True)).to(
            device=device, dtype=torch.float32, non_blocking=False
        )


def configure_common(module: Any, args: argparse.Namespace, output_root: Path) -> None:
    module.CACHE_ROOT = args.cache_root
    module.OUTPUT_ROOT = output_root
    module.GPUCache = Frame0YawGPUCache


def run_main(module: Any, argv: list[str]) -> None:
    original_argv = sys.argv
    sys.argv = [original_argv[0], *argv]
    try:
        module.main()
    finally:
        sys.argv = original_argv


def base_args(args: argparse.Namespace, track_root: Path | None, output_root: Path) -> list[str]:
    values = [
        "--cache-root", str(args.cache_root),
        "--output-root", str(output_root),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--updates-per-epoch", str(args.updates_per_epoch),
        "--learning-rate", str(args.learning_rate),
        "--seeds", *(str(seed) for seed in args.seeds),
    ]
    if track_root is not None:
        values.extend(["--track-root", str(track_root)])
    if args.unseen_classes is not None:
        values.extend(["--unseen-classes", *(str(x) for x in args.unseen_classes)])
        values.extend(["--pseudo-unseen-classes", *(str(x) for x in args.pseudo_unseen_classes)])
    return values


def configure_variant(args: argparse.Namespace, output_root: Path) -> tuple[Any, Path | None, dict[str, Any]]:
    variant = args.variant
    if variant == "v1":
        configure_common(v1, args, output_root)
        v1.TRACK_ROOT = args.track_root
        v1.build_trajectory_state = v1.build_trajectory_state
        v1.AngleObjectTrajectoryPolicy = v1.AngleObjectTrajectoryPolicy
        v1.attach_object_tracks = v1.attach_object_tracks
        return v1, args.track_root, {"requested_variant": "v1", "implementation": "multistep_angle_object_trajectory_policy_v1.py"}
    if variant == "v2_legacy":
        configure_common(v2_legacy, args, output_root)
        return v2_legacy, None, {
            "requested_variant": "v2",
            "reported_variant": "v2_legacy_single_state",
            "implementation": "multistep_angle_object_policy_v1.py",
            "warning": "No independent trajectory-v2 implementation/checkpoint exists in the repository; this is the preserved legacy angle+object model.",
        }
    if variant == "v3":
        configure_common(v1, args, output_root)
        v1.TRACK_ROOT = args.track_root_v3
        v1.build_trajectory_state = v3.build_trajectory_state_v3
        v1.AngleObjectTrajectoryPolicy = v3.ObjectSizeTrajectoryPolicy
        v1.attach_object_tracks = v3.attach_object_tracks_v3
        return v1, args.track_root_v3, {"requested_variant": "v3", "implementation": "multistep_angle_object_trajectory_policy_v3.py", "track_channels": 6}
    if variant == "v4":
        configure_common(v1, args, output_root)
        v1.TRACK_ROOT = args.track_root
        v1.build_trajectory_state = true_yaw_v4.build_trajectory_state_v4_true_yaw
        v1.AngleObjectTrajectoryPolicy = v4.BodyOrientationTrajectoryPolicy
        v1.attach_object_tracks = v1.attach_object_tracks
        return v1, args.track_root, {
            "requested_variant": "v4",
            "implementation": "multistep_angle_object_trajectory_policy_v4.py",
            "human_yaw_source": "frame0_body_angle_rank/body_yaw_deg.npy",
            "rtmpose_used_for_yaw": False,
        }
    if variant == "v5":
        configure_common(v1, args, output_root)
        v1.TRACK_ROOT = args.track_root
        v1.build_trajectory_state = v5.build_trajectory_state_v5
        v1.AngleObjectTrajectoryPolicy = v5.CategoryFreeObjectTrajectoryPolicy
        v1.attach_object_tracks = v1.attach_object_tracks
        return v1, args.track_root, {"requested_variant": "v5", "implementation": "multistep_angle_object_trajectory_policy_v5_no_category_id.py"}
    if variant == "v6":
        configure_common(v1, args, output_root)
        v1.TRACK_ROOT = args.track_root
        v1.build_trajectory_state = v6.build_trajectory_state_v5
        v1.AngleObjectTrajectoryPolicy = v6.PersonGeometryTrajectoryPolicy
        v1.attach_object_tracks = v6.attach_no_object_tracks
        return v1, args.track_root, {"requested_variant": "v6", "implementation": "multistep_angle_object_trajectory_policy_v6_no_object.py"}
    if variant == "object_only":
        configure_common(v1, args, output_root)
        v1.TRACK_ROOT = args.track_root
        v1.build_trajectory_state = ablation.build_object_only_state
        v1.AngleObjectTrajectoryPolicy = ablation.ObjectOnlyTrajectoryPolicy
        v1.attach_object_tracks = v1.attach_object_tracks
        return v1, args.track_root, {"requested_variant": "object_only", "implementation": "multistep_angle_object_trajectory_ablation.py"}
    if variant == "geometry_only":
        configure_common(v1, args, output_root)
        v1.TRACK_ROOT = args.track_root
        v1.build_trajectory_state = ablation.build_geometry_only_state
        v1.AngleObjectTrajectoryPolicy = ablation.GeometryOnlyPolicy
        v1.attach_object_tracks = ablation.attach_no_object_tracks
        return v1, args.track_root, {"requested_variant": "geometry_only", "implementation": "multistep_angle_object_trajectory_ablation.py"}
    raise ValueError(variant)


def write_provenance(output_root: Path, args: argparse.Namespace, provenance: dict[str, Any]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    payload = {
        **provenance,
        "experiment": "frame0_body_angle_rank_v1_v6",
        "cache_root": str(args.cache_root),
        "track_root": str(args.track_root),
        "track_root_v3": str(args.track_root_v3),
        "angle_contract": {
            "source": "released 25-joint 3-D skeleton at video frame 0",
            "value": "signed angle from anatomical body-forward to person-to-camera",
            "sign": "negative body-left, positive body-right",
            "action": "rank 0..3 after ascending signed-angle sort; rank_to_original maps to source video",
            "camera_id_is_not_order": True,
        },
        "v_poclip": "frozen logits/z from source cache; selector never receives class logits/z",
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "updates_per_epoch": args.updates_per_epoch,
            "learning_rate": args.learning_rate,
            "seeds": args.seeds,
            "device": "cuda:0",
            "per_process_memory_fraction": args.memory_fraction,
        },
        "split": {
            "unseen_classes": args.unseen_classes,
            "pseudo_unseen_classes": args.pseudo_unseen_classes,
            "split_override_forwarded_to_variant": args.unseen_classes is not None,
        },
        "old_outputs_modified": False,
    }
    (output_root / "experiment_metadata.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("v1", "v2_legacy", "v3", "v4", "v5", "v6", "object_only", "geometry_only"), required=True)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--track-root", type=Path, default=DEFAULT_TRACK_ROOT)
    parser.add_argument("--track-root-v3", type=Path, default=DEFAULT_TRACK_ROOT_V3)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--updates-per-epoch", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--memory-fraction", type=float, default=0.16)
    parser.add_argument(
        "--unseen-classes", type=int, nargs="+", default=None,
        help="Strict test classes; forwarded to the trajectory implementation.",
    )
    parser.add_argument(
        "--pseudo-unseen-classes", type=int, nargs="+", default=None,
        help="Held-out seen classes used for validation/model selection.",
    )
    args = parser.parse_args()

    if args.unseen_classes is not None:
        if args.pseudo_unseen_classes is None:
            raise ValueError("--pseudo-unseen-classes is required with --unseen-classes")
        unseen = sorted(set(int(x) for x in args.unseen_classes))
        pseudo = sorted(set(int(x) for x in args.pseudo_unseen_classes))
        if not unseen or any(x < 0 or x >= 55 for x in unseen):
            raise ValueError(f"invalid unseen classes: {unseen}")
        if not pseudo or not set(pseudo).isdisjoint(unseen):
            raise ValueError(f"pseudo-unseen classes must be non-empty and disjoint: {pseudo}")
        args.unseen_classes = unseen
        args.pseudo_unseen_classes = pseudo

    if not args.cache_root.exists():
        raise FileNotFoundError(f"ranked cache is not ready: {args.cache_root}")
    if args.variant != "v2_legacy" and not args.track_root.exists():
        raise FileNotFoundError(f"ranked tracks are not ready: {args.track_root}")
    if args.variant == "v3" and not args.track_root_v3.exists():
        raise FileNotFoundError(f"ranked v3 tracks are not ready: {args.track_root_v3}")
    if args.output_root.exists() and (args.output_root / "complete.json").exists():
        print(f"FRAME0_ANGLE_VARIANT_ALREADY_COMPLETE {args.output_root}", flush=True)
        return

    # One RL process at a time is launched by the outer shell.  This cap leaves
    # a large headroom for the concurrently running X3D process.
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing an unplanned CPU training run")
    torch.cuda.set_per_process_memory_fraction(float(args.memory_fraction), 0)
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

    module, track_root, provenance = configure_variant(args, args.output_root)
    write_provenance(args.output_root, args, provenance)
    argv = base_args(args, track_root, args.output_root)
    print("FRAME0_ANGLE_VARIANT_START", json.dumps(provenance, ensure_ascii=False), flush=True)
    run_main(module, argv)

    summary_path = args.output_root / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary.update(provenance)
        summary["angle_contract"] = json.loads((args.output_root / "experiment_metadata.json").read_text(encoding="utf-8"))["angle_contract"]
        summary["ranked_cache"] = str(args.cache_root)
        summary["old_outputs_modified"] = False
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("FRAME0_ANGLE_VARIANT_COMPLETE", args.variant, flush=True)


if __name__ == "__main__":
    main()
