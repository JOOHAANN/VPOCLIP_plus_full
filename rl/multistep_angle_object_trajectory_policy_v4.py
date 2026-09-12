"""v4 trajectory policy: v1 plus a causal human-body orientation signal.

v4 keeps the successful v1 object/geometry inputs and adds two per-frame
orientation channels derived from the cached 2-D skeleton:

    theta_body = atan2(shoulder_center_y - hip_center_y,
                       shoulder_center_x - hip_center_x)
    [sin(theta_body), cos(theta_body)]

The angle is a torso-axis orientation proxy, not a claim that 2-D keypoints
perfectly recover the person's semantic facing direction.  v1/v2/v3 remain
untouched.  The v1 training/evaluation protocol is reused exactly.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch
from torch import nn

from . import multistep_angle_object_trajectory_policy_v1 as v1


ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = (
    ROOT / "data" / "trajectory_latest_vpoclip_new50_5" / "cache_compat"
)
TRACK_ROOT = (
    ROOT / "data" / "trajectory_latest_vpoclip_new50_5" / "object_tracks_full"
)
OUTPUT_ROOT = ROOT / "work_dir" / "multistep_angle_object_trajectory_policy_v4_body_orientation"
_INITIALIZATION_CHECKPOINT = os.environ.get("VPOCLIP_V1_INIT_CHECKPOINT")
V1_CHECKPOINT = (
    Path(_INITIALIZATION_CHECKPOINT)
    if _INITIALIZATION_CHECKPOINT
    else ROOT
    / "work_dir"
    / "multistep_angle_object_trajectory_latest_vpoclip_new50_5"
    / "seed_20260909"
    / "best.pt"
)

# The actual v1 checkpoint lives here.  Keep the fallback explicit so v4 can
# still be initialized when the historical output path is used.
if not V1_CHECKPOINT.exists():
    V1_CHECKPOINT = (
        ROOT
        / "work_dir"
        / "multistep_angle_object_trajectory_policy_latest_vpoclip_new50_5"
        / "seed_20260909"
        / "best.pt"
    )

PERSON_CHANNELS_V4 = 6  # x, y, vx, vy, sin(body_angle), cos(body_angle)


def person_trajectory_with_orientation(pose: torch.Tensor) -> torch.Tensor:
    """Return v1 center/velocity plus a causal torso-axis orientation."""

    points = pose.reshape(*pose.shape[:-1], v1.NUM_FRAMES, 17, 2)
    base = v1.person_trajectory_from_pose(pose)
    shoulder_center = (points[..., 5, :] + points[..., 6, :]) * 0.5
    hip_center = (points[..., 11, :] + points[..., 12, :]) * 0.5
    torso = shoulder_center - hip_center
    present = (
        points[..., [5, 6, 11, 12], :].norm(dim=-1).mean(-1) > 0.02
    ).float()
    theta = torch.atan2(torso[..., 1], torso[..., 0])
    orientation = torch.stack((theta.sin(), theta.cos()), -1) * present[..., None]
    return torch.cat((base, orientation), -1)


def build_trajectory_state_v4(
    raw: Any,
    episodes: torch.Tensor,
    history: torch.Tensor,
    history_count: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Build the v1 state with the additional body-orientation history."""

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
    person_per_view = person_trajectory_with_orientation(observed_pose)
    person = (person_per_view * observed_f[:, :, None, None]).sum(1) / denominator[:, :, None]

    object_track = raw.object_position_track[episodes[:, None], history]
    object_xy = object_track[..., 1:3]
    relative_xy = object_xy - person_per_view[..., :2][..., None, :]
    object_sensor = torch.cat((object_track, relative_xy), -1)
    object_sensor = (
        object_sensor * observed_f[:, :, None, None, None]
    ).sum(1) / denominator[:, :, None, None]

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
    valid = raw.valid[episodes] & raw.reachable[episodes, current] & (~observed_views)
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
        "object_trajectory": object_sensor,
        "current_relative_angle": current_relative_angle,
        "candidate": candidate,
        "mask": valid,
    }


class BodyOrientationTrajectoryPolicy(nn.Module):
    """v1 factorized scorer with a 6-channel human trajectory."""

    def __init__(self) -> None:
        super().__init__()
        self.person_temporal = nn.Sequential(
            nn.Conv1d(PERSON_CHANNELS_V4, 16, kernel_size=3, padding=1),
            nn.GroupNorm(4, 16),
            nn.GELU(),
        )
        self.person_summary = nn.Sequential(
            nn.Linear(16 * v1.NUM_FRAMES, 32), nn.LayerNorm(32), nn.GELU()
        )
        self.object_temporal = nn.Sequential(
            nn.Conv1d(v1.NUM_OBJECT_CLASSES * v1.OBJECT_CHANNELS, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
        )
        self.object_summary = nn.Sequential(
            nn.Linear(64 * v1.NUM_FRAMES, 64), nn.LayerNorm(64), nn.GELU()
        )
        self.angle = nn.Sequential(nn.Linear(2, 16), nn.LayerNorm(16), nn.GELU())
        self.context = nn.Sequential(
            nn.Linear(32 + 64 + 16, 64), nn.LayerNorm(64), nn.GELU()
        )
        self.candidate = nn.Sequential(
            nn.Linear(10, 64), nn.LayerNorm(64), nn.GELU()
        )
        self.context_to_action = nn.Linear(64, 64, bias=False)
        self.residual = nn.Sequential(
            nn.Linear(128, 32), nn.GELU(), nn.Linear(32, 1)
        )
        self._initialize_matching_v1_weights()

    def _initialize_matching_v1_weights(self) -> None:
        """Copy v1 weights; initialize only the two new orientation channels."""

        if not V1_CHECKPOINT.exists():
            return
        saved = torch.load(V1_CHECKPOINT, map_location="cpu", weights_only=False)
        old = saved.get("online", saved)
        current = self.state_dict()
        with torch.no_grad():
            for name, value in old.items():
                if name not in current:
                    continue
                if name == "person_temporal.0.weight":
                    current[name][:, : value.shape[1], :].copy_(value)
                    current[name][:, value.shape[1] :, :].zero_()
                elif current[name].shape == value.shape:
                    current[name].copy_(value)
        self.load_state_dict(current, strict=True)

    def forward(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        person = self.person_summary(
            self.person_temporal(state["person_trajectory"].transpose(1, 2)).flatten(1)
        )
        obj = state["object_trajectory"].reshape(
            state["object_trajectory"].shape[0],
            v1.NUM_FRAMES,
            v1.NUM_OBJECT_CLASSES * v1.OBJECT_CHANNELS,
        )
        obj = self.object_summary(self.object_temporal(obj.transpose(1, 2)).flatten(1))
        angle = self.angle(state["current_relative_angle"])
        context = self.context(torch.cat((person, obj, angle), -1))
        candidate = self.candidate(v1.angle_candidate_features(state["candidate"]))
        expanded = context[:, None].expand(-1, v1.NUM_VIEWS, -1)
        q = (self.context_to_action(context)[:, None] * candidate).sum(-1) / (64.0 ** 0.5)
        return q + 0.20 * self.residual(torch.cat((expanded, candidate), -1)).squeeze(-1)


def _rewrite_metadata() -> None:
    summary_path = OUTPUT_ROOT / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["variant"] = "multistep_angle_object_trajectory_policy_v4_body_orientation"
        summary["base_comparison"] = "multistep_angle_object_trajectory_policy_v1"
        summary["policy_inputs"] = [
            "sample-specific human/camera relative angle",
            "causal 13-frame person center/velocity trajectory",
            "causal 13-frame torso-axis orientation sin/cos trajectory",
            "causal 13-frame per-class object presence/position/confidence trajectory",
            "object position relative to person center",
        ]
        summary["body_orientation_definition"] = (
            "sin/cos of atan2(shoulder_center_y-hip_center_y, "
            "shoulder_center_x-hip_center_x)"
        )
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    for audit_path in OUTPUT_ROOT.glob("seed_*/audit.json"):
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        audit["variant"] = "v4_body_orientation"
        audit["person_track_channels"] = [
            "center_x", "center_y", "velocity_x", "velocity_y", "body_sin", "body_cos",
        ]
        audit["body_orientation_definition"] = (
            "torso axis from shoulder midpoint to hip midpoint, encoded as sin/cos"
        )
        audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")


def main() -> None:
    # Reuse the exact v1 data split, target, optimizer and evaluation protocols.
    v1.CACHE_ROOT = CACHE_ROOT
    v1.TRACK_ROOT = TRACK_ROOT
    v1.OUTPUT_ROOT = OUTPUT_ROOT
    v1.build_trajectory_state = build_trajectory_state_v4
    v1.AngleObjectTrajectoryPolicy = BodyOrientationTrajectoryPolicy
    v1.main()
    _rewrite_metadata()


if __name__ == "__main__":
    main()
