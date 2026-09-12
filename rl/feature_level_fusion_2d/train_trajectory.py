"""Train the preserved 2-D trajectory policies with 512-D fusion rewards.

This is intentionally a new entry point.  It imports the historical policy
definitions for the five requested variants but never edits or overwrites
their source files or their old output directories.
"""

from __future__ import annotations

import argparse
import json
import sys
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .. import multistep_angle_object_trajectory_policy_v1 as base
from .. import multistep_angle_object_trajectory_policy_v4 as v4
from .. import multistep_angle_object_trajectory_policy_v5_no_category_id as v5
from .. import multistep_angle_object_trajectory_policy_v6_no_object as v6
from .. import multistep_angle_object_trajectory_ablation as ablation
from . import feature_fusion as ff


TRUE_UNSEEN = [14, 15, 25, 39, 46]
PSEUDO_UNSEEN = [0, 2, 9, 10, 11, 17, 26, 34, 49, 50]
TRAIN_SEEDS = [20260909, 20260910, 20260911]


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    tmp.replace(path)


def configure_variant(name: str) -> None:
    """Apply the same isolated variant selection as the 2-D anchor runner."""

    if name == "v4":
        base.build_trajectory_state = v4.build_trajectory_state_v4
        base.AngleObjectTrajectoryPolicy = v4.BodyOrientationTrajectoryPolicy
        base.attach_object_tracks = base.attach_object_tracks
    elif name == "v5":
        base.build_trajectory_state = v5.build_trajectory_state_v5
        base.AngleObjectTrajectoryPolicy = v5.CategoryFreeObjectTrajectoryPolicy
        base.attach_object_tracks = base.attach_object_tracks
    elif name == "v6":
        base.build_trajectory_state = v6.build_trajectory_state_v5
        base.AngleObjectTrajectoryPolicy = v6.PersonGeometryTrajectoryPolicy
        base.attach_object_tracks = v6.attach_no_object_tracks
    elif name == "object_only":
        base.build_trajectory_state = ablation.build_object_only_state
        base.AngleObjectTrajectoryPolicy = ablation.ObjectOnlyTrajectoryPolicy
        base.attach_object_tracks = base.attach_object_tracks
    elif name == "geometry_only":
        base.build_trajectory_state = ablation.build_geometry_only_state
        base.AngleObjectTrajectoryPolicy = ablation.GeometryOnlyPolicy
        base.attach_object_tracks = ablation.attach_no_object_tracks
    else:
        raise ValueError(name)


def load_fusion(path: Path, device: torch.device) -> tuple[ff.FeatureFusion512, float]:
    fusion = ff.FeatureFusion512().to(device)
    payload = torch.load(path, map_location=device, weights_only=False)
    fusion.load_state_dict(payload["model"], strict=True)
    fusion.eval()
    return fusion, float(payload.get("logit_scale", 14.56))


@torch.no_grad()
def evaluate_protocol_feature(
    model: torch.nn.Module,
    raw: Any,
    classes: list[int],
    protocol: str,
    seed: int,
    fusion: ff.FeatureFusion512,
    scale: float,
) -> dict[str, Any]:
    """Validation protocol with feature-level rewards and final scores."""

    bank = base.class_bank_tensor(classes, raw.device)
    eligible = torch.isin(raw.labels, bank) & (raw.valid.sum(-1) >= 3)
    episodes = eligible.nonzero().flatten()
    if protocol.startswith("fixed"):
        start = int(protocol.removeprefix("fixed"))
        episodes = (eligible & raw.valid[:, start]).nonzero().flatten()
        starts = torch.full_like(episodes, start)
    elif protocol == "random_start":
        starts = base.choose_random_starts(raw, episodes, seed)
    else:
        raise ValueError(protocol)
    if not len(episodes):
        empty = {"top1": 0.0, "mean_entropy": 0.0, "mean_movement_cost": 0.0, "episodes": 0.0}
        return {"metrics": {"policy_3view": empty, "random_3view_exact": empty}}

    rollout = base.policy_rollout_trajectory(model, raw, episodes, starts)
    policy_episodes = rollout["episodes"]
    policy_paths = torch.stack((rollout["starts"], rollout["action1"], rollout["action2"]), 1)
    policy = ff.path_metrics(raw, fusion, policy_episodes, policy_paths, bank, scale)
    random_episodes, random_paths, _ = base.sample_random_paths(raw, episodes, starts, seed + 17)
    random = ff.path_metrics(raw, fusion, random_episodes, random_paths, bank, scale)
    return {
        "protocol": protocol,
        "episodes_before_two_move_filter": int(len(episodes)),
        "episodes_policy_completed": int(len(policy_episodes)),
        "metrics": {"policy_3view": policy, "random_3view_exact": random},
    }


def attach_split(raw: Any, cache_root: Path, track_root: Path, split: str) -> None:
    base.attach_view_valid(raw, cache_root, split)
    base.attach_object_tracks(raw, track_root, split)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("v4", "v5", "v6", "object_only", "geometry_only"), required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--track-root", type=Path, required=True)
    parser.add_argument("--fusion-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--updates-per-epoch", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seeds", type=int, nargs="+", default=TRAIN_SEEDS)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda:0")
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    configure_variant(args.variant)
    base.CACHE_ROOT = args.cache_root
    base.TRACK_ROOT = args.track_root
    base.OUTPUT_ROOT = args.output_root

    args.output_root.mkdir(parents=True, exist_ok=True)
    dump(args.output_root / "config_resolved.json", {
        "variant": args.variant,
        "cache_root": str(args.cache_root),
        "track_root": str(args.track_root),
        "fusion_root": str(args.fusion_root),
        "output_root": str(args.output_root),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "updates_per_epoch": args.updates_per_epoch,
        "learning_rate": args.learning_rate,
        "seeds": args.seeds,
        "seen_classes": sorted(set(range(55)) - set(TRUE_UNSEEN)),
        "pseudo_unseen_classes": PSEUDO_UNSEEN,
        "true_unseen_classes": TRUE_UNSEEN,
        "recognizer_fusion": "FeatureFusion512 then cosine similarity to normalized text prototypes",
        "anchor": "/home/youhan/ws/VPOCLIP_plus_full/work_dir/trajectory_g1_pseudo_selected_stage_b_2d",
        "policy_inputs_unchanged": True,
    })

    raw: dict[str, Any] = {}
    for split in ("dqn_train", "val"):
        raw[split] = base.GPUCache(args.cache_root / split, device)
        attach_split(raw[split], args.cache_root, args.track_root, split)
    summaries: list[dict[str, Any]] = []
    for seed in args.seeds:
        fusion_path = args.fusion_root / f"seed_{seed}" / "best.pt"
        if not fusion_path.exists():
            raise FileNotFoundError(fusion_path)
        fusion, scale = load_fusion(fusion_path, device)
        base.utility_targets = partial(
            ff.feature_utility_targets,
            fusion=fusion,
            scale=scale,
        )
        base.evaluate_protocol = partial(
            evaluate_protocol_feature,
            fusion=fusion,
            scale=scale,
        )
        args.seed = seed
        summary = base.train_seed(
            seed,
            raw["dqn_train"],
            raw["val"],
            device,
            args,
            args.output_root,
        )
        summaries.append(summary)
        # Make the provenance explicit in each copied/adapted run.
        audit_path = args.output_root / f"seed_{seed}" / "audit.json"
        if audit_path.exists():
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            audit.update({
                "variant": args.variant,
                "training_target": "feature-level fused 512-D delta margin",
                "final_fusion": "FeatureFusion512 recursively over selected views",
                "fusion_checkpoint": str(fusion_path),
                "fusion_logit_scale": scale,
                "old_anchor_preserved": True,
            })
            dump(audit_path, audit)
        del fusion
        torch.cuda.empty_cache()
    dump(args.output_root / "training_summary.json", summaries)
    print("FEATURE_FUSION_TRAJECTORY_TRAIN_COMPLETE", args.variant, flush=True)


if __name__ == "__main__":
    main()

