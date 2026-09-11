"""Neural-network skeleton for the fixed-three-slot Dueling Q architecture."""

from __future__ import annotations

from typing import Mapping

import torch
from torch import nn

from .contracts import PolicyState


def mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    """Construct the small two-layer blocks used throughout the policy."""

    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, output_dim),
    )


class ViewEncoder(nn.Module):
    """Shared encoder for one A/B/C observation slot."""

    def __init__(self, config: Mapping[str, int | bool]) -> None:
        super().__init__()
        self.use_pose = bool(config.get("use_pose", False))
        self.use_object = bool(config.get("use_object", False))
        self.semantic = mlp(512, 256, 128)
        self.pose_input_dim = int(config.get("pose_input_dim", 442))
        self.object_input_dim = int(config.get("object_input_dim", 1800))
        self.quality_dim = int(config.get("quality_dim", 4))
        self.pose = mlp(self.pose_input_dim, 128, 64)
        self.object = mlp(self.object_input_dim, 128, 64)
        quality_view_dim = int(config.get("quality_dim", 4)) + int(config.get("view_encoding_dim", 8))
        self.quality_view = mlp(quality_view_dim, 64, 32)
        self.fuse = mlp(128 + 64 + 64 + 32, 256, 256)

    def forward(self, slot: Mapping[str, torch.Tensor], slot_mask: torch.Tensor) -> torch.Tensor:
        """Encode one slot as h_v[256], returning exact zeros for padded slots."""

        z = slot["z"]
        if z.ndim == 1:
            z = z.unsqueeze(0)
        parts = [self.semantic(z)]
        if self.use_pose:
            parts.append(self.pose(slot["pose"]))
        else:
            parts.append(torch.zeros((*z.shape[:-1], 64), device=z.device, dtype=z.dtype))
        if self.use_object:
            parts.append(self.object(slot["object_map"]))
        else:
            parts.append(torch.zeros((*z.shape[:-1], 64), device=z.device, dtype=z.dtype))
        qv = torch.cat((slot["quality"], slot["view_encoding"]), dim=-1)
        parts.append(self.quality_view(qv))
        encoded = self.fuse(torch.cat(parts, dim=-1))
        return encoded * slot_mask.unsqueeze(-1)


class EvidenceEncoder(nn.Module):
    """Compress e_top5[512], p_top5[5], entropy, and margin to 128 dimensions."""

    def __init__(self, input_dim: int = 519) -> None:
        super().__init__()
        self.net = mlp(input_dim, 256, 128)

    def forward(self, evidence: torch.Tensor) -> torch.Tensor:
        return self.net(evidence)


class RobotEncoder(nn.Module):
    """Compress masks/costs plus current body-relative angle/confidence."""

    def __init__(self, num_views: int) -> None:
        super().__init__()
        self.net = mlp(4 * num_views + 4, 128, 64)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return self.net(context)


class DuelingQNetwork(nn.Module):
    """Return destination Q-values; this network has no action-classification head."""

    def __init__(self, num_views: int, view_config: Mapping[str, int | bool]) -> None:
        super().__init__()
        self.num_views = num_views
        self.view_encoder = ViewEncoder(view_config)
        self.evidence_encoder = EvidenceEncoder()
        self.robot_encoder = RobotEncoder(num_views)
        self.candidate_dim = int(view_config.get("candidate_dim", 8))
        self.candidate = mlp(self.candidate_dim, 64, 32)
        self.trunk = nn.Sequential(
            nn.Linear(3 * 256 + 128 + 64, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
        )
        self.value_head = mlp(256, 128, 1)
        self.advantage_head = mlp(256 + 32, 128, 1)

    def forward(self, state: PolicyState) -> torch.Tensor:
        """Return Q[B,V] using V(s)+A(s,a)-mean_a A(s,a)."""

        view = state["view_features"]
        pose = state["pose_features"]
        obj = state["object_features"]
        quality = state["quality_features"]
        slots = state["slot_mask"]
        evidence = state["evidence"]
        robot = state["robot_context"]
        candidate_geometry = state["candidate_geometry"]
        if view.ndim == 2:
            view, pose, obj, quality, slots, evidence, robot, candidate_geometry = (
                x.unsqueeze(0) for x in (view, pose, obj, quality, slots, evidence, robot, candidate_geometry)
            )
        encoded_slots = []
        view_encoding = torch.eye(3, device=view.device, dtype=view.dtype)
        for index in range(3):
            encoded_slots.append(
                self.view_encoder(
                    {
                        "z": view[:, index],
                        "pose": pose[:, index],
                        "object_map": obj[:, index],
                        "quality": quality[:, index],
                        "view_encoding": view_encoding[index].expand(view.shape[0], -1),
                    },
                    slots[:, index],
                )
            )
        h = self.trunk(torch.cat((
            torch.cat(encoded_slots, dim=-1),
            self.evidence_encoder(evidence),
            self.robot_encoder(robot),
        ), dim=-1))
        value = self.value_head(h)
        candidate_h = self.candidate(candidate_geometry)
        advantage = self.advantage_head(torch.cat((h.unsqueeze(1).expand(-1, self.num_views, -1), candidate_h), dim=-1)).squeeze(-1)
        return value + advantage - advantage.mean(dim=-1, keepdim=True)


# Implementation guide
# 1. Keep one shared ViewEncoder for A, B, and C; never create slot-specific weights.
# 2. M0 should zero/disable pose and object branches while preserving the same
#    public state contract. M1 adds Top-5 evidence; M2 enables pose/object inputs.
# 3. Flatten pose and object inputs only inside their branch or replace them with
#    structured encoders later without changing h_v[256].
# 4. Multiply every h_v by slot_mask after the final linear layer to cancel bias.
# 5. Apply action masks outside forward so online and target networks share the
#    exact same unmasked Q definition.
