"""v3 trajectory policy with YOLO bounding-box size features.

This file is an isolated extension of ``multistep_angle_object_trajectory_policy_v1``.
The historical v1/v2 code and caches remain unchanged.  v3 consumes a six-channel
YOLO track per object and appends the relative object centre to it:

    raw YOLO track: [presence, centre_x, centre_y, box_width, box_height, confidence]
    policy object sensor: raw track + [relative_x_to_person, relative_y_to_person]

Width and height are normalized by the 640x480 frame dimensions.  The policy
still sees only causal observed views; VPOCLIP logits are used for offline
utility targets and final fusion, not as policy inputs.

The training/evaluation loop is reused from v1 through a narrow adapter so the
protocol stays identical.  v1 and v2 are not modified by this module.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from . import multistep_angle_object_trajectory_policy_v1 as v1


ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = (
    ROOT / "data" / "trajectory_latest_vpoclip_new50_5" / "cache_compat"
)
TRACK_ROOT = ROOT / "data" / "angle_object_trajectory_v3_bbox_size"
OUTPUT_ROOT = ROOT / "work_dir" / "multistep_angle_object_trajectory_policy_v3_bbox_size"
V1_CHECKPOINT = (
    ROOT
    / "work_dir"
    / "multistep_angle_object_trajectory_policy_latest_vpoclip_new50_5"
    / "seed_20260909"
    / "best.pt"
)

NUM_OBJECT_CHANNELS = 6
OBJECT_SENSOR_CHANNELS = 8


def attach_object_tracks_v3(raw: Any, track_root: Path, split: str) -> None:
    """Attach the six-channel bbox-size track without touching v1 caches."""

    path = Path(track_root) / split / "object_position_track.npy"
    if not path.exists():
        raise FileNotFoundError(
            f"missing v3 bbox-size track {path}; run build_object_position_tracks.py "
            "with --include-box-size first"
        )
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    expected = (
        raw.num_episodes,
        v1.NUM_VIEWS,
        v1.NUM_FRAMES,
        v1.NUM_OBJECT_CLASSES,
        NUM_OBJECT_CHANNELS,
    )
    if tuple(array.shape) != expected:
        raise ValueError(f"{path} shape {array.shape} != {expected}")
    raw.object_position_track = torch.from_numpy(np.array(array, copy=True)).to(
        device=raw.device, dtype=torch.float32
    )


def build_trajectory_state_v3(
    raw: Any,
    episodes: torch.Tensor,
    history: torch.Tensor,
    history_count: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Build the v1 causal state plus normalized object bbox size."""

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


class ObjectSizeTrajectoryPolicy(nn.Module):
    """v1 scorer with 2 additional object-size channels per object."""

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
            nn.Conv1d(
                v1.NUM_OBJECT_CLASSES * OBJECT_SENSOR_CHANNELS,
                64,
                kernel_size=3,
                padding=1,
            ),
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
        """Warm-start unchanged branches from the successful v1 checkpoint."""

        if not V1_CHECKPOINT.exists():
            return
        saved = torch.load(V1_CHECKPOINT, map_location="cpu", weights_only=False)
        old = saved.get("online", saved)
        current = self.state_dict()
        with torch.no_grad():
            for name, value in old.items():
                if name not in current:
                    continue
                if name == "object_temporal.0.weight":
                    current[name][:, : value.shape[1], :].copy_(value)
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
            v1.NUM_OBJECT_CLASSES * OBJECT_SENSOR_CHANNELS,
        )
        obj = self.object_summary(self.object_temporal(obj.transpose(1, 2)).flatten(1))
        angle = self.angle(state["current_relative_angle"])
        context = self.context(torch.cat((person, obj, angle), -1))
        candidate = self.candidate(v1.angle_candidate_features(state["candidate"]))
        expanded = context[:, None].expand(-1, v1.NUM_VIEWS, -1)
        q = (self.context_to_action(context)[:, None] * candidate).sum(-1) / (64.0 ** 0.5)
        return q + 0.20 * self.residual(torch.cat((expanded, candidate), -1)).squeeze(-1)


def _rewrite_metadata() -> None:
    """Label reused v1 reports as v3 and record the changed feature shape."""

    summary_path = OUTPUT_ROOT / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["variant"] = "multistep_angle_object_trajectory_policy_v3_bbox_size"
        summary["base_comparison"] = "multistep_angle_object_trajectory_policy_v1"
        summary["policy_inputs"] = [
            "sample-specific human/camera relative angle",
            "causal 13-frame person center/velocity trajectory",
            "causal 13-frame per-class object presence/position/width/height/confidence trajectory",
            "object position relative to person center",
        ]
        summary["object_track_shape_per_episode"] = [
            v1.NUM_VIEWS,
            v1.NUM_FRAMES,
            v1.NUM_OBJECT_CLASSES,
            NUM_OBJECT_CHANNELS,
        ]
        summary["object_sensor_channels_per_class"] = OBJECT_SENSOR_CHANNELS
        summary["bbox_size_normalization"] = "width/640 and height/480"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    for audit_path in OUTPUT_ROOT.glob("seed_*/audit.json"):
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        audit["variant"] = "v3_bbox_size"
        audit["object_track_fields"] = [
            "presence", "center_x", "center_y", "box_width", "box_height", "confidence",
        ]
        audit["object_track_shape"] = [v1.NUM_VIEWS, v1.NUM_FRAMES, v1.NUM_OBJECT_CLASSES, NUM_OBJECT_CHANNELS]
        audit["object_sensor"] = "six raw YOLO channels + relative_x + relative_y"
        audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")


def main() -> None:
    # Reuse the tested v1 loop while replacing only the track reader, state
    # builder, and scorer.  This keeps the protocol and comparison fair.
    v1.CACHE_ROOT = CACHE_ROOT
    v1.TRACK_ROOT = TRACK_ROOT
    v1.OUTPUT_ROOT = OUTPUT_ROOT
    v1.attach_object_tracks = attach_object_tracks_v3
    v1.build_trajectory_state = build_trajectory_state_v3
    v1.AngleObjectTrajectoryPolicy = ObjectSizeTrajectoryPolicy
    v1.main()
    _rewrite_metadata()


if __name__ == "__main__":
    main()
