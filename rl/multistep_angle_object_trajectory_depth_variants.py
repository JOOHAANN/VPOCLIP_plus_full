"""Depth-augmented trajectory variants for the new 50/5 experiment.

The preserved 2-D v4/v5/v6/object-only implementations are not modified.
This module adds one causal first-frame relative-depth summary to each state:
the depth vectors of already observed views are mean pooled before entering a
small 32-D branch.  Depth is joined by exact ``sample_name`` in
``multistep_angle_object_trajectory_policy_v4_depth``; camera IDs are never
used as an ordering or direction assumption.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn

from . import multistep_angle_object_trajectory_ablation as ablation
from . import multistep_angle_object_trajectory_policy_v1 as v1
from . import multistep_angle_object_trajectory_policy_v4 as v4
from . import multistep_angle_object_trajectory_policy_v4_depth as v4_depth
from . import multistep_angle_object_trajectory_policy_v5_no_category_id as v5
from . import multistep_angle_object_trajectory_policy_v6_no_object as v6


ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = ROOT / "data" / "trajectory_latest_vpoclip_new50_5" / "cache_compat"
TRACK_ROOT = ROOT / "data" / "trajectory_latest_vpoclip_new50_5" / "object_tracks_full"
DEPTH_ROOT = ROOT / "data" / "new50_5_group1_zsl_depth_first_frame_fp32"
DEPTH_DIM = 50
DEPTH_HIDDEN = 32

BASE_CHECKPOINTS = {
    "v5_depth": ROOT / "work_dir" / "multistep_angle_object_trajectory_policy_v5_no_category_id_no_slots_group1",
    "v6_depth": ROOT / "work_dir" / "multistep_angle_object_trajectory_policy_v6_no_object_group1",
    "object_only_depth": ROOT / "work_dir" / "trajectory_ablation_new50_5" / "object_only",
}


def add_observed_depth(
    raw: Any,
    episodes: torch.Tensor,
    history: torch.Tensor,
    history_count: torch.Tensor,
) -> torch.Tensor:
    """Mean-pool depth from the views already present in ``history``."""

    episodes = episodes.long()
    history = history.long()
    count = history_count.long().clamp(1, history.shape[1])
    positions = torch.arange(history.shape[1], device=episodes.device)[None, :]
    observed = (positions < count[:, None]).float()
    denominator = count.float().clamp_min(1.0)[:, None]
    depth = raw.depth[episodes[:, None], history]
    return (depth * observed[..., None]).sum(1) / denominator


def _with_depth(builder: Callable[..., dict[str, torch.Tensor]]) -> Callable[..., dict[str, torch.Tensor]]:
    def build(
        raw: Any,
        episodes: torch.Tensor,
        history: torch.Tensor,
        history_count: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        state = builder(raw, episodes, history, history_count)
        state["depth_prior"] = add_observed_depth(raw, episodes, history, history_count)
        return state

    return build


def build_v4_depth(raw: Any, episodes: torch.Tensor, history: torch.Tensor, history_count: torch.Tensor):
    return v4_depth.build_trajectory_state_v4_depth(raw, episodes, history, history_count)


build_v5_depth = _with_depth(v5.build_trajectory_state_v5)
build_v6_depth = _with_depth(v6.build_trajectory_state_v5)
build_object_only_depth = _with_depth(ablation.build_object_only_state)


def _copy_checkpoint_with_wider_context(
    model: nn.Module,
    checkpoint_root: Path,
    context_key: str,
    old_context_width: int,
) -> None:
    """Initialize the depth model from the corresponding preserved 2-D model."""

    checkpoint = checkpoint_root / "seed_20260909" / "best.pt"
    if not checkpoint.exists():
        return
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    old = saved.get("online", saved)
    current = model.state_dict()
    with torch.no_grad():
        for name, value in old.items():
            if name not in current:
                continue
            if name == context_key + ".0.weight":
                if value.shape[1] != old_context_width:
                    raise ValueError(
                        f"unexpected {name} shape {tuple(value.shape)} in {checkpoint}"
                    )
                current[name][:, :old_context_width].copy_(value)
                current[name][:, old_context_width:].zero_()
            elif current[name].shape == value.shape:
                current[name].copy_(value)
        # Start as the original 2-D policy and let the new branch learn the
        # incremental value of depth.
        current["depth.0.weight"].normal_(mean=0.0, std=0.02)
        current["depth.0.bias"].zero_()
    model.load_state_dict(current, strict=True)


class DepthCategoryFreeObjectTrajectoryPolicy(v5.CategoryFreeObjectTrajectoryPolicy):
    """v5 plus a compact depth branch."""

    def __init__(self) -> None:
        super().__init__()
        self.depth = nn.Sequential(
            nn.Linear(DEPTH_DIM, DEPTH_HIDDEN),
            nn.LayerNorm(DEPTH_HIDDEN),
            nn.GELU(),
        )
        self.context = nn.Sequential(
            nn.Linear(32 + 32 + 16 + DEPTH_HIDDEN, 64),
            nn.LayerNorm(64),
            nn.GELU(),
        )
        _copy_checkpoint_with_wider_context(
            self,
            BASE_CHECKPOINTS["v5_depth"],
            "context",
            old_context_width=80,
        )

    def forward(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        person = self.person_temporal(
            state["person_trajectory"].transpose(1, 2)
        ).flatten(1)
        person = self.person_summary(person)
        obj = state["object_trajectory"]
        if obj.shape[-1] != v5.POOLED_OBJECT_CHANNELS:
            raise ValueError(f"v5 expects pooled object features, got {tuple(obj.shape)}")
        obj = self.object_temporal(obj.transpose(1, 2)).flatten(1)
        obj = self.object_summary(obj)
        angle = self.angle(state["current_relative_angle"])
        depth = self.depth(state["depth_prior"])
        context = self.context(torch.cat((person, obj, angle, depth), -1))
        candidate = self.candidate(v1.angle_candidate_features(state["candidate"]))
        expanded = context[:, None].expand(-1, v1.NUM_VIEWS, -1)
        q = (self.context_to_action(context)[:, None] * candidate).sum(-1)
        q = q / (64.0 ** 0.5)
        return q + 0.20 * self.residual(torch.cat((expanded, candidate), -1)).squeeze(-1)


class DepthPersonGeometryTrajectoryPolicy(v6.PersonGeometryTrajectoryPolicy):
    """v6 plus depth; all YOLO object inputs remain removed."""

    def __init__(self) -> None:
        super().__init__()
        self.depth = nn.Sequential(
            nn.Linear(DEPTH_DIM, DEPTH_HIDDEN),
            nn.LayerNorm(DEPTH_HIDDEN),
            nn.GELU(),
        )
        self.context = nn.Sequential(
            nn.Linear(32 + 16 + DEPTH_HIDDEN, 64),
            nn.LayerNorm(64),
            nn.GELU(),
        )
        _copy_checkpoint_with_wider_context(
            self,
            BASE_CHECKPOINTS["v6_depth"],
            "context",
            old_context_width=48,
        )

    def forward(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        person = self.person_summary(
            self.person_temporal(state["person_trajectory"].transpose(1, 2)).flatten(1)
        )
        angle = self.angle(state["current_relative_angle"])
        depth = self.depth(state["depth_prior"])
        context = self.context(torch.cat((person, angle, depth), -1))
        candidate = self.candidate(v1.angle_candidate_features(state["candidate"]))
        expanded = context[:, None].expand(-1, v1.NUM_VIEWS, -1)
        q = (self.context_to_action(context)[:, None] * candidate).sum(-1) / (64.0 ** 0.5)
        return q + 0.20 * self.residual(torch.cat((expanded, candidate), -1)).squeeze(-1)


class DepthObjectOnlyTrajectoryPolicy(ablation.ObjectOnlyTrajectoryPolicy):
    """object-only ablation plus depth; the explicit person branch stays absent."""

    def __init__(self) -> None:
        super().__init__()
        self.depth = nn.Sequential(
            nn.Linear(DEPTH_DIM, DEPTH_HIDDEN),
            nn.LayerNorm(DEPTH_HIDDEN),
            nn.GELU(),
        )
        self.head.context = nn.Sequential(
            nn.Linear(64 + DEPTH_HIDDEN + 16, 64),
            nn.LayerNorm(64),
            nn.GELU(),
        )
        _copy_checkpoint_with_wider_context(
            self,
            BASE_CHECKPOINTS["object_only_depth"],
            "head.context",
            old_context_width=80,
        )

    def forward(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        obj = state["object_trajectory"]
        obj = obj.reshape(
            obj.shape[0], v1.NUM_FRAMES, v1.NUM_OBJECT_CLASSES * v1.OBJECT_CHANNELS
        )
        obj = self.object_temporal(obj.transpose(1, 2)).flatten(1)
        obj = self.object_summary(obj)
        depth = self.depth(state["depth_prior"])
        return self.head.score(torch.cat((obj, depth), -1), state)


VARIANTS: dict[str, dict[str, Any]] = {
    "v4_depth": {
        "builder": build_v4_depth,
        "model": v4_depth.BodyOrientationDepthTrajectoryPolicy,
        "needs_objects": True,
        "base_2d": "v4_body_orientation",
    },
    "v5_depth": {
        "builder": build_v5_depth,
        "model": DepthCategoryFreeObjectTrajectoryPolicy,
        "needs_objects": True,
        "base_2d": "v5_no_category_id_no_slots",
    },
    "v6_depth": {
        "builder": build_v6_depth,
        "model": DepthPersonGeometryTrajectoryPolicy,
        "needs_objects": False,
        "base_2d": "v6_no_object",
    },
    "object_only_depth": {
        "builder": build_object_only_depth,
        "model": DepthObjectOnlyTrajectoryPolicy,
        "needs_objects": True,
        "base_2d": "object_only",
    },
}


def configure_variant(name: str) -> dict[str, Any]:
    if name not in VARIANTS:
        raise ValueError(f"unknown depth trajectory variant: {name}")
    spec = VARIANTS[name]
    v1.build_trajectory_state = spec["builder"]
    v1.AngleObjectTrajectoryPolicy = spec["model"]
    original_attach_object_tracks = v1.attach_object_tracks

    if spec["needs_objects"]:
        def attach(raw: Any, track_root: Path, split: str) -> None:
            original_attach_object_tracks(raw, track_root, split)
            v4_depth.attach_depth_tracks(raw, CACHE_ROOT, DEPTH_ROOT, split)
    else:
        def attach(raw: Any, track_root: Path, split: str) -> None:
            del track_root
            v4_depth.attach_depth_tracks(raw, CACHE_ROOT, DEPTH_ROOT, split)

    v1.attach_object_tracks = attach
    return spec


def rewrite_metadata(output_root: Path, name: str, spec: dict[str, Any]) -> None:
    summary_path = output_root / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary.update(
            {
                "variant": name,
                "base_2d_variant": spec["base_2d"],
                "depth_definition": {
                    "shape_per_view": [DEPTH_DIM],
                    "dtype": "float32",
                    "normalization": "per-frame 5th-95th percentile to [0,1]",
                    "causal_pooling": "mean over already observed views only",
                    "join": "exact sample_name; no camera-ID ordering assumption",
                    "source": str(DEPTH_ROOT),
                },
            }
        )
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    for audit_path in output_root.glob("seed_*/audit.json"):
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        audit.update(
            {
                "variant": name,
                "base_2d_variant": spec["base_2d"],
                "depth_input": {
                    "shape_per_observed_state": [DEPTH_DIM],
                    "dtype": "float32",
                    "source": str(DEPTH_ROOT),
                    "frame": "first decoded frame at 640x480",
                    "causal_pooling": "mean over observed views",
                },
            }
        )
        audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
