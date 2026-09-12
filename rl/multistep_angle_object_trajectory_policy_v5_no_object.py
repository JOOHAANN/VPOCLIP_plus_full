"""v5: v1 geometry/person trajectory policy with the whole object branch removed.

This is an isolated ablation of v1.  The state contains only the causal human
center/velocity trajectory, the sample-specific human/camera relative angle,
and candidate-view geometry.  No YOLO presence, object position, confidence,
box size, or 50-object-class channels are loaded or passed to the network.
The v1 training/evaluation protocol is reused; the caller supplies the
group1-correct 50/5 class split through v1's existing CLI arguments.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from . import multistep_angle_object_trajectory_policy_v1 as v1


ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = ROOT / "data" / "trajectory_latest_vpoclip_new50_5" / "cache_compat"
OUTPUT_ROOT = ROOT / "work_dir" / "multistep_angle_object_trajectory_policy_v5_no_object_group1"
NUM_FRAMES = v1.NUM_FRAMES


def attach_no_object_tracks(raw: Any, track_root: Path, split: str) -> None:
    """Do not load object tracks: v5 deliberately has no object state."""

    del raw, track_root, split


def build_trajectory_state_v5(
    raw: Any,
    episodes: torch.Tensor,
    history: torch.Tensor,
    history_count: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Build the causal person-and-geometry state without object features."""

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
        "current_relative_angle": current_relative_angle,
        "candidate": candidate,
        "mask": valid,
    }


class PersonGeometryTrajectoryPolicy(nn.Module):
    """v1 scorer after removing all 50-object-class features."""

    def __init__(self) -> None:
        super().__init__()
        self.person_temporal = nn.Sequential(
            nn.Conv1d(4, 16, kernel_size=3, padding=1),
            nn.GroupNorm(4, 16),
            nn.GELU(),
        )
        self.person_summary = nn.Sequential(
            nn.Linear(16 * NUM_FRAMES, 32), nn.LayerNorm(32), nn.GELU()
        )
        self.angle = nn.Sequential(nn.Linear(2, 16), nn.LayerNorm(16), nn.GELU())
        self.context = nn.Sequential(
            nn.Linear(32 + 16, 64), nn.LayerNorm(64), nn.GELU()
        )
        self.candidate = nn.Sequential(
            nn.Linear(10, 64), nn.LayerNorm(64), nn.GELU()
        )
        self.context_to_action = nn.Linear(64, 64, bias=False)
        self.residual = nn.Sequential(
            nn.Linear(128, 32), nn.GELU(), nn.Linear(32, 1)
        )

    def forward(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        person = self.person_summary(
            self.person_temporal(state["person_trajectory"].transpose(1, 2)).flatten(1)
        )
        angle = self.angle(state["current_relative_angle"])
        context = self.context(torch.cat((person, angle), -1))
        candidate = self.candidate(v1.angle_candidate_features(state["candidate"]))
        expanded = context[:, None].expand(-1, v1.NUM_VIEWS, -1)
        q = (self.context_to_action(context)[:, None] * candidate).sum(-1) / (64.0 ** 0.5)
        return q + 0.20 * self.residual(torch.cat((expanded, candidate), -1)).squeeze(-1)


def _rewrite_metadata() -> None:
    summary_path = OUTPUT_ROOT / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary.update(
            {
                "variant": "multistep_angle_object_trajectory_policy_v5_no_object_group1",
                "base_comparison": "multistep_angle_object_trajectory_policy_v1",
                "policy_inputs": [
                    "sample-specific human/camera relative angle",
                    "causal 13-frame person center/velocity trajectory",
                    "candidate-view geometry and reachability",
                ],
                "object_features": "removed_entirely",
                "object_classes_in_state": 0,
                "object_channels_in_state": 0,
                "formal_unseen_split": [14, 15, 25, 39, 46],
            }
        )
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    for audit_path in OUTPUT_ROOT.glob("seed_*/audit.json"):
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        audit.update(
            {
                "variant": "v5_no_object",
                "policy_inputs_only": [
                    "sample-specific human/camera relative angle",
                    "causal 13-frame person center/velocity trajectory",
                    "candidate-view geometry and reachability",
                ],
                "policy_inputs_excluded": [
                    "RGB/video embedding",
                    "VPOCLIP z",
                    "skeleton embedding",
                    "semantic belief",
                    "current logits",
                    "class ID",
                    "future-view features",
                    "all YOLO object features",
                    "all 50 object categories",
                ],
                "object_features": "removed_entirely",
                "object_classes_in_state": 0,
                "object_channels_in_state": 0,
            }
        )
        audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")


def main() -> None:
    v1.CACHE_ROOT = CACHE_ROOT
    v1.TRACK_ROOT = ROOT / "data" / "no_object_tracks_not_used"
    v1.OUTPUT_ROOT = OUTPUT_ROOT
    v1.attach_object_tracks = attach_no_object_tracks
    v1.build_trajectory_state = build_trajectory_state_v5
    v1.AngleObjectTrajectoryPolicy = PersonGeometryTrajectoryPolicy

    original_dump = v1.dump

    def dump_v5(path: Path, value: Any) -> None:
        if path.name == "audit.json":
            value = dict(value)
            value["policy_inputs_only"] = [
                "sample-specific human/camera relative angle",
                "causal 13-frame person center/velocity trajectory",
                "candidate-view geometry and reachability",
            ]
            value["object_features"] = "removed_entirely"
            value["object_classes_in_state"] = 0
            value["object_channels_in_state"] = 0
        original_dump(path, value)

    v1.dump = dump_v5
    v1.main()
    _rewrite_metadata()


if __name__ == "__main__":
    main()
