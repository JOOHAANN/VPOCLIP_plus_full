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
        self.pose = mlp(int(config.get("pose_input_dim", 663)), 128, 64)
        self.object = mlp(int(config.get("object_input_dim", 1800)), 128, 64)
        quality_view_dim = int(config.get("quality_dim", 4)) + int(config.get("view_encoding_dim", 8))
        self.quality_view = mlp(quality_view_dim, 64, 32)
        self.fuse = mlp(128 + 64 + 64 + 32, 256, 256)

    def forward(self, slot: Mapping[str, torch.Tensor], slot_mask: torch.Tensor) -> torch.Tensor:
        """Encode one slot as h_v[256], returning exact zeros for padded slots."""

        raise NotImplementedError("Implement optional branch masking and post-bias slot masking here.")


class EvidenceEncoder(nn.Module):
    """Compress e_top5[512], p_top5[5], entropy, and margin to 128 dimensions."""

    def __init__(self, input_dim: int = 519) -> None:
        super().__init__()
        self.net = mlp(input_dim, 256, 128)

    def forward(self, evidence: torch.Tensor) -> torch.Tensor:
        return self.net(evidence)


class RobotEncoder(nn.Module):
    """Compress the 4V+1 robot context to 64 dimensions."""

    def __init__(self, num_views: int) -> None:
        super().__init__()
        self.net = mlp(4 * num_views + 1, 128, 64)

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
        self.trunk = nn.Sequential(
            nn.Linear(3 * 256 + 128 + 64 + 3, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
        )
        self.value_head = mlp(256, 128, 1)
        self.advantage_head = mlp(256, 128, num_views)

    def forward(self, state: PolicyState) -> torch.Tensor:
        """Return Q[B,V] using V(s)+A(s,a)-mean_a A(s,a)."""

        raise NotImplementedError("Implement fixed-slot encoding and dueling aggregation here.")


# Implementation guide
# 1. Keep one shared ViewEncoder for A, B, and C; never create slot-specific weights.
# 2. M0 should zero/disable pose and object branches while preserving the same
#    public state contract. M1 adds Top-5 evidence; M2 enables pose/object inputs.
# 3. Flatten pose and object inputs only inside their branch or replace them with
#    structured encoders later without changing h_v[256].
# 4. Multiply every h_v by slot_mask after the final linear layer to cancel bias.
# 5. Apply action masks outside forward so online and target networks share the
#    exact same unmasked Q definition.
