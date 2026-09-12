"""Controlled data ablations for the historical v1 trajectory policy.

The preserved v1 policy has three input groups:

* candidate/view geometry and the current human-camera relative direction;
* the causal 13-frame person center/velocity trajectory;
* the causal 13-frame 50-slot object trajectory.

This module trains two missing group ablations with the exact v1 training loop:

``object_only``
    geometry + object trajectory; the explicit person trajectory branch is
    removed.  Human-relative object coordinates remain because they are part
    of the object sensor definition in v1.

``geometry_only``
    geometry/current relative direction only; both temporal sensor branches
    are removed.

The existing v1, v4, v5 and v6 checkpoints are evaluated alongside these two
models by the separate old-split evaluator.  This file intentionally does not
modify any preserved v1-v9 implementation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn

from . import multistep_angle_object_trajectory_policy_v1 as v1


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_ROOT = ROOT / "data" / "rl_candidate_rank_v6_new50_5" / "cache"
DEFAULT_TRACK_ROOT = ROOT / "data" / "angle_object_trajectory_v1"


def attach_no_object_tracks(raw: Any, track_root: Path, split: str) -> None:
    """Geometry-only ablation deliberately does not load object tracks."""

    del raw, track_root, split


def _causal_masks(
    raw: Any,
    episodes: torch.Tensor,
    history: torch.Tensor,
    history_count: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    episodes = episodes.long()
    history = history.long()
    count = history_count.long().clamp(1, history.shape[1])
    batch = episodes.shape[0]
    positions = torch.arange(history.shape[1], device=episodes.device)[None, :]
    observed = positions < count[:, None]
    current = history.gather(1, (count - 1)[:, None]).squeeze(1)
    observed_views = torch.zeros(
        (batch, v1.NUM_VIEWS), dtype=torch.bool, device=episodes.device
    )
    observed_views.scatter_(1, history, observed)
    current_relative_angle = raw.geometry[episodes, current, :2]
    geometry = raw.geometry[episodes, :, :2]
    sin_delta = (
        geometry[..., 0] * current_relative_angle[:, None, 1]
        - geometry[..., 1] * current_relative_angle[:, None, 0]
    )
    cos_delta = (geometry * current_relative_angle[:, None]).sum(-1)
    valid = (
        raw.valid[episodes]
        & raw.reachable[episodes, current]
        & (~observed_views)
    )
    candidate = torch.cat(
        (
            geometry,
            torch.stack((sin_delta, cos_delta), -1),
            current_relative_angle[:, None].expand(-1, v1.NUM_VIEWS, -1),
            valid[..., None].float(),
        ),
        -1,
    )
    return observed.float(), current_relative_angle, candidate, valid


def build_object_only_state(
    raw: Any,
    episodes: torch.Tensor,
    history: torch.Tensor,
    history_count: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Build v1 state without the explicit person trajectory branch."""

    observed_f, current_relative_angle, candidate, valid = _causal_masks(
        raw, episodes, history, history_count
    )
    count = history_count.long().clamp(1, history.shape[1])
    denominator = count.float().clamp_min(1.0)[:, None]
    observed_pose = raw.pose[episodes[:, None], history]
    person_per_view = v1.person_trajectory_from_pose(observed_pose)
    object_track = raw.object_position_track[episodes[:, None], history]
    object_xy = object_track[..., 1:3]
    relative_xy = object_xy - person_per_view[..., :2].unsqueeze(-2)
    object_sensor = torch.cat((object_track, relative_xy), -1)
    object_sensor = (
        object_sensor * observed_f[:, :, None, None, None]
    ).sum(1) / denominator[:, :, None, None]
    return {
        "object_trajectory": object_sensor,
        "current_relative_angle": current_relative_angle,
        "candidate": candidate,
        "mask": valid,
    }


def build_geometry_only_state(
    raw: Any,
    episodes: torch.Tensor,
    history: torch.Tensor,
    history_count: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Build state with only sample-specific candidate geometry."""

    _, current_relative_angle, candidate, valid = _causal_masks(
        raw, episodes, history, history_count
    )
    return {
        "current_relative_angle": current_relative_angle,
        "candidate": candidate,
        "mask": valid,
    }


class _CandidateHead(nn.Module):
    """Shared v1 candidate scorer used by both ablation models."""

    def __init__(self, context_width: int) -> None:
        super().__init__()
        self.angle = nn.Sequential(nn.Linear(2, 16), nn.LayerNorm(16), nn.GELU())
        self.context = nn.Sequential(
            nn.Linear(context_width + 16, 64), nn.LayerNorm(64), nn.GELU()
        )
        self.candidate = nn.Sequential(
            nn.Linear(10, 64), nn.LayerNorm(64), nn.GELU()
        )
        self.context_to_action = nn.Linear(64, 64, bias=False)
        self.residual = nn.Sequential(
            nn.Linear(128, 32), nn.GELU(), nn.Linear(32, 1)
        )

    def score(self, context_input: torch.Tensor, state: dict[str, torch.Tensor]) -> torch.Tensor:
        angle = self.angle(state["current_relative_angle"])
        context = self.context(torch.cat((context_input, angle), -1))
        candidate = self.candidate(v1.angle_candidate_features(state["candidate"]))
        expanded = context[:, None].expand(-1, v1.NUM_VIEWS, -1)
        q = (self.context_to_action(context)[:, None] * candidate).sum(-1)
        q = q / (64.0 ** 0.5)
        return q + 0.20 * self.residual(torch.cat((expanded, candidate), -1)).squeeze(-1)


class ObjectOnlyTrajectoryPolicy(nn.Module):
    """Geometry + raw 50-slot object trajectory, without person branch."""

    def __init__(self) -> None:
        super().__init__()
        self.object_temporal = nn.Sequential(
            nn.Conv1d(v1.NUM_OBJECT_CLASSES * v1.OBJECT_CHANNELS, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
        )
        self.object_summary = nn.Sequential(
            nn.Linear(64 * v1.NUM_FRAMES, 64), nn.LayerNorm(64), nn.GELU()
        )
        self.head = _CandidateHead(64)

    def forward(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        obj = state["object_trajectory"]
        obj = obj.reshape(obj.shape[0], v1.NUM_FRAMES, v1.NUM_OBJECT_CLASSES * v1.OBJECT_CHANNELS)
        obj = self.object_temporal(obj.transpose(1, 2)).flatten(1)
        obj = self.object_summary(obj)
        return self.head.score(obj, state)


class GeometryOnlyPolicy(nn.Module):
    """Geometry/current relative direction only."""

    def __init__(self) -> None:
        super().__init__()
        self.head = _CandidateHead(0)

    def forward(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.head.score(
            torch.zeros(
                (state["current_relative_angle"].shape[0], 0),
                device=state["current_relative_angle"].device,
                dtype=state["current_relative_angle"].dtype,
            ),
            state,
        )


def _rewrite_metadata(variant: str, output_root: Path) -> None:
    policy_inputs = {
        "object_only": [
            "sample-specific human/camera relative angle",
            "candidate-view geometry and reachability",
            "causal 13-frame per-class object presence/position/confidence trajectory",
            "object position relative to person center",
        ],
        "geometry_only": [
            "sample-specific human/camera relative angle",
            "candidate-view geometry and reachability",
        ],
    }[variant]
    excluded = [
        "RGB/video embedding", "VPOCLIP z", "skeleton embedding",
        "semantic belief", "current logits", "class ID", "future-view features",
    ]
    if variant == "object_only":
        excluded.append("explicit person center/velocity trajectory branch")
    else:
        excluded.extend([
            "causal person center/velocity trajectory",
            "all YOLO object features",
            "all 50 object categories",
        ])
    summary_path = output_root / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary.update({
            "variant": f"multistep_angle_object_trajectory_ablation_{variant}",
            "base_comparison": "multistep_angle_object_trajectory_policy_v1",
            "policy_inputs": policy_inputs,
            "policy_inputs_excluded": excluded,
            "ablation_contract": (
                "person branch removed; human-relative object coordinates retained"
                if variant == "object_only"
                else "both temporal person and object branches removed"
            ),
        })
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for audit_path in output_root.glob("seed_*/audit.json"):
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        audit.update({
            "variant": f"ablation_{variant}",
            "policy_inputs_only": policy_inputs,
            "policy_inputs_excluded": excluded,
            "ablation_contract": (
                "person branch removed; human-relative object coordinates retained"
                if variant == "object_only"
                else "both temporal person and object branches removed"
            ),
        })
        audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")


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
    ]
    if args.unseen_classes is not None:
        values.extend(["--unseen-classes", *(str(x) for x in args.unseen_classes)])
        values.extend(["--pseudo-unseen-classes", *(str(x) for x in args.pseudo_unseen_classes)])
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("object_only", "geometry_only"), required=True)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--track-root", type=Path, default=DEFAULT_TRACK_ROOT)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--updates-per-epoch", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260909, 20260910, 20260911])
    parser.add_argument("--unseen-classes", type=int, nargs="+", default=None)
    parser.add_argument("--pseudo-unseen-classes", type=int, nargs="+", default=None)
    args = parser.parse_args()
    if (args.unseen_classes is None) != (args.pseudo_unseen_classes is None):
        raise ValueError("--unseen-classes and --pseudo-unseen-classes must be supplied together")

    v1.CACHE_ROOT = args.cache_root
    v1.TRACK_ROOT = args.track_root
    v1.OUTPUT_ROOT = args.output_root
    if args.variant == "object_only":
        v1.build_trajectory_state = build_object_only_state
        v1.AngleObjectTrajectoryPolicy = ObjectOnlyTrajectoryPolicy
        v1.attach_object_tracks = v1.attach_object_tracks
    else:
        v1.build_trajectory_state = build_geometry_only_state
        v1.AngleObjectTrajectoryPolicy = GeometryOnlyPolicy
        v1.attach_object_tracks = attach_no_object_tracks

    original_argv = sys.argv
    sys.argv = [original_argv[0], *_forward_argv(args)]
    try:
        v1.main()
    finally:
        sys.argv = original_argv
    _rewrite_metadata(args.variant, args.output_root)


if __name__ == "__main__":
    main()
