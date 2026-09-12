"""v5: category-free, slot-free object trajectory state.

The raw detector cache is still stored as ``[view, frame, 50, 4]`` because
that is the efficient on-disk representation produced by the detector.  The
policy does *not* receive those 50 ordered channels.  Before the state enters
the network, all present objects are treated as an unordered set and pooled
per frame into count, mean, minimum and maximum statistics for:

``center_x, center_y, confidence, relative_x_to_person, relative_y_to_person``.

Thus the policy sees 16 values per frame (1 + 5 + 5 + 5), not 50 slots and
not an object category-ID.  This preserves coarse object geometry while
removing the category/slot shortcut.  The v1 training and evaluation loop is
kept intact and is only supplied with this alternate state builder and model.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn

from . import multistep_angle_object_trajectory_policy_v1 as v1


ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = ROOT / "data" / "trajectory_latest_vpoclip_new50_5" / "cache_compat"
TRACK_ROOT = ROOT / "data" / "trajectory_latest_vpoclip_new50_5" / "object_tracks_full"
OUTPUT_ROOT = ROOT / "work_dir" / "multistep_angle_object_trajectory_policy_v5_no_category_id_no_slots_group1"
POOLED_OBJECT_CHANNELS = 16


def _pool_unordered_objects(sensor: torch.Tensor) -> torch.Tensor:
    """Pool ``[B,H,T,O,6]`` object sensors without using slot identity."""

    presence = sensor[..., 0].clamp(0.0, 1.0)
    numeric = sensor[..., 1:]  # x, y, confidence, relative x/y: five values
    count = presence.sum(dim=-1, keepdim=True)
    denominator = count.clamp_min(1.0)
    weights = presence.unsqueeze(-1)

    mean = (numeric * weights).sum(dim=-2) / denominator
    present_mask = weights > 0.0
    minimum = numeric.masked_fill(~present_mask, float("inf")).amin(dim=-2)
    maximum = numeric.masked_fill(~present_mask, float("-inf")).amax(dim=-2)
    has_object = count > 0.0
    minimum = torch.where(has_object, minimum, torch.zeros_like(minimum))
    maximum = torch.where(has_object, maximum, torch.zeros_like(maximum))

    count_ratio = count / float(v1.NUM_OBJECT_CLASSES)
    pooled = torch.cat((count_ratio, mean, minimum, maximum), dim=-1)
    if pooled.shape[-1] != POOLED_OBJECT_CHANNELS:
        raise AssertionError(f"unexpected pooled object width: {pooled.shape}")
    return pooled


def build_trajectory_state_v5(
    raw: v1.GPUCache,
    episodes: torch.Tensor,
    history: torch.Tensor,
    history_count: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Build the causal geometry/person/pooled-object state."""

    episodes = episodes.long()
    history = history.long()
    count = history_count.long().clamp(1, history.shape[1])
    batch = episodes.shape[0]
    positions = torch.arange(history.shape[1], device=episodes.device)[None, :]
    observed = positions < count[:, None]
    observed_f = observed.float()
    denominator = count.float().clamp_min(1.0)[:, None]
    current = history.gather(1, (count - 1)[:, None]).squeeze(1)

    observed_pose = raw.pose[episodes[:, None], history]
    person_per_view = v1.person_trajectory_from_pose(observed_pose)
    person = (person_per_view * observed_f[:, :, None, None]).sum(1)
    person = person / denominator[:, :, None]

    # [B,H,T,O,4] -> [B,H,T,O,6].  The six raw per-object values are used
    # only as an intermediate representation; the policy receives no slots.
    object_track = raw.object_position_track[episodes[:, None], history]
    relative_xy = object_track[..., 1:3] - person_per_view[..., :2].unsqueeze(-2)
    object_sensor = torch.cat((object_track, relative_xy), dim=-1)
    pooled_per_view = _pool_unordered_objects(object_sensor)
    object_summary = (pooled_per_view * observed_f[:, :, None, None]).sum(1)
    object_summary = object_summary / denominator[:, :, None]

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
    return {
        "person_trajectory": person,
        "object_trajectory": object_summary,
        "current_relative_angle": current_relative_angle,
        "candidate": candidate,
        "mask": valid,
    }


class CategoryFreeObjectTrajectoryPolicy(nn.Module):
    """Low-capacity scorer whose object branch is permutation-invariant."""

    def __init__(self) -> None:
        super().__init__()
        self.person_temporal = nn.Sequential(
            nn.Conv1d(4, 16, kernel_size=3, padding=1),
            nn.GroupNorm(4, 16),
            nn.GELU(),
        )
        self.person_summary = nn.Sequential(
            nn.Linear(16 * v1.NUM_FRAMES, 32), nn.LayerNorm(32), nn.GELU()
        )
        self.object_temporal = nn.Sequential(
            nn.Conv1d(POOLED_OBJECT_CHANNELS, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.GELU(),
        )
        self.object_summary = nn.Sequential(
            nn.Linear(32 * v1.NUM_FRAMES, 32), nn.LayerNorm(32), nn.GELU()
        )
        self.angle = nn.Sequential(nn.Linear(2, 16), nn.LayerNorm(16), nn.GELU())
        self.context = nn.Sequential(
            nn.Linear(32 + 32 + 16, 64), nn.LayerNorm(64), nn.GELU()
        )
        self.candidate = nn.Sequential(
            nn.Linear(10, 64), nn.LayerNorm(64), nn.GELU()
        )
        self.context_to_action = nn.Linear(64, 64, bias=False)
        self.residual = nn.Sequential(
            nn.Linear(128, 32), nn.GELU(), nn.Linear(32, 1)
        )

    def forward(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        person = self.person_temporal(
            state["person_trajectory"].transpose(1, 2)
        ).flatten(1)
        person = self.person_summary(person)

        obj = state["object_trajectory"]
        if obj.shape[-1] != POOLED_OBJECT_CHANNELS:
            raise ValueError(
                "v5 expects pooled object features with no slot axis; "
                f"got {tuple(obj.shape)}"
            )
        obj = self.object_temporal(obj.transpose(1, 2)).flatten(1)
        obj = self.object_summary(obj)

        angle = self.angle(state["current_relative_angle"])
        context = self.context(torch.cat((person, obj, angle), -1))
        candidate = self.candidate(v1.angle_candidate_features(state["candidate"]))
        expanded = context[:, None].expand(-1, v1.NUM_VIEWS, -1)
        q = (self.context_to_action(context)[:, None] * candidate).sum(-1)
        q = q / math.sqrt(64.0)
        q = q + 0.20 * self.residual(torch.cat((expanded, candidate), -1)).squeeze(-1)
        return q


def _rewrite_metadata() -> None:
    summary_path = OUTPUT_ROOT / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary.update(
            {
                "variant": "multistep_angle_object_trajectory_policy_v5_no_category_id_no_slots",
                "base_comparison": "multistep_angle_object_trajectory_policy_v1",
                "policy_inputs": [
                    "sample-specific human/camera relative angle",
                    "causal 13-frame person center/velocity trajectory",
                    "causal 13-frame unordered object summary trajectory",
                    "per-object human-relative center coordinates before pooling",
                ],
                "raw_object_track_shape_per_episode": [4, 13, 50, 4],
                "policy_object_shape_per_frame": [POOLED_OBJECT_CHANNELS],
                "object_summary_fields": [
                    "present_count_ratio",
                    "mean_center_x",
                    "mean_center_y",
                    "mean_confidence",
                    "mean_relative_x_to_person",
                    "mean_relative_y_to_person",
                    "min_center_x",
                    "min_center_y",
                    "min_confidence",
                    "min_relative_x_to_person",
                    "min_relative_y_to_person",
                    "max_center_x",
                    "max_center_y",
                    "max_confidence",
                    "max_relative_x_to_person",
                    "max_relative_y_to_person",
                ],
                "object_slots_seen_by_policy": 0,
                "object_pooling": "per-frame permutation-invariant count + mean/min/max over present objects",
                "explicit_object_category_id_dimension": False,
                "object_category_encoding": "none; detector slot identity is discarded before the policy",
                "formal_unseen_split": [14, 15, 25, 39, 46],
            }
        )
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    for audit_path in OUTPUT_ROOT.glob("seed_*/audit.json"):
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        audit.update(
            {
                "variant": "v5_no_category_id_no_slots",
                "raw_object_track_shape": [4, 13, 50, 4],
                "policy_object_shape_per_frame": [POOLED_OBJECT_CHANNELS],
                "object_slots_seen_by_policy": 0,
                "object_pooling": "per-frame permutation-invariant count + mean/min/max over present objects",
                "explicit_object_category_id_dimension": False,
                "object_category_encoding": "none; detector slot identity is discarded before the policy",
            }
        )
        audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")


def main() -> None:
    v1.CACHE_ROOT = CACHE_ROOT
    v1.TRACK_ROOT = TRACK_ROOT
    v1.OUTPUT_ROOT = OUTPUT_ROOT
    v1.build_trajectory_state = build_trajectory_state_v5
    v1.AngleObjectTrajectoryPolicy = CategoryFreeObjectTrajectoryPolicy
    v1.main()
    _rewrite_metadata()


if __name__ == "__main__":
    main()
