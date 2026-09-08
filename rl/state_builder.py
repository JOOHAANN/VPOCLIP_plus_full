"""Single source of truth for constructing fixed-slot policy states."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from .contracts import PolicyState
from .logit_fusion import LogitFusion
from .multiview_cache import MultiViewCache


class StateBuilder:
    """Build s_A, s_AB, and terminal s_ABC from cache indices."""

    def __init__(
        self,
        cache: MultiViewCache,
        fusion: LogitFusion,
        text_prototypes: torch.Tensor,
        num_views: int,
        max_selected_views: int = 3,
    ) -> None:
        self.cache = cache
        self.fusion = fusion
        self.text_prototypes = text_prototypes
        self.num_views = num_views
        self.max_selected_views = max_selected_views

    def build(self, episode_id: int, views: Sequence[int]) -> PolicyState:
        """Return a label-free, fixed-shape state and valid destination mask."""
        views = tuple(int(v) for v in views)
        if not views or len(views) > self.max_selected_views:
            raise ValueError(f"selected views must contain 1..{self.max_selected_views} items")
        if len(set(views)) != len(views):
            raise ValueError("selected views must be unique")
        observations = [self.cache.get_view(episode_id, view) for view in views]
        device = self.text_prototypes.device
        logits = torch.stack([obs["logits"].to(device=device, dtype=torch.float32) for obs in observations])
        fusion = self.fusion.fuse(logits, self.text_prototypes)

        z = torch.zeros((3, 512), dtype=torch.float32, device=device)
        pose_dim = int(self.cache.arrays.get("pose", torch.empty(0)).shape[-1]) if "pose" in self.cache.arrays else 0
        object_dim = int(self.cache.arrays.get("object_map", torch.empty(0)).reshape(len(self.cache), self.num_views, -1).shape[-1]) if "object_map" in self.cache.arrays else 0
        pose = torch.zeros((3, max(pose_dim, 1)), dtype=torch.float32, device=device)
        obj = torch.zeros((3, max(object_dim, 1)), dtype=torch.float32, device=device)
        quality = torch.zeros((3, 4), dtype=torch.float32, device=device)
        slot_mask = torch.zeros(3, dtype=torch.float32, device=device)
        for slot, obs in enumerate(observations):
            z[slot] = obs["z"].to(device=device, dtype=torch.float32)
            if "pose" in obs:
                flat = obs["pose"].to(device=device, dtype=torch.float32).flatten()
                pose[slot, : flat.numel()] = flat[: pose.shape[-1]]
            if "object_map" in obs:
                flat = obs["object_map"].to(device=device, dtype=torch.float32).flatten()
                obj[slot, : flat.numel()] = flat[: obj.shape[-1]]
            if "image_quality" in obs:
                q = obs["image_quality"].to(device=device, dtype=torch.float32).flatten()
                quality[slot, : min(4, q.numel())] = q[:4]
            slot_mask[slot] = 1.0

        evidence = torch.cat(
            (
                fusion.semantic_summary.reshape(-1),
                fusion.top_probabilities.reshape(-1)[:5],
                fusion.entropy.reshape(-1)[:1],
                fusion.margin.reshape(-1)[:1],
            )
        )
        if evidence.numel() != 519:
            raise RuntimeError(f"evidence contract must be 519-D, got {evidence.numel()}")

        current = views[-1]
        reachable = torch.tensor(np.array(self.cache.arrays["reachable"][episode_id, current], copy=True), device=device, dtype=torch.float32)
        costs = torch.tensor(np.array(self.cache.arrays["move_cost"][episode_id, current], copy=True), device=device, dtype=torch.float32)
        costs = torch.nan_to_num(costs, nan=1.0, posinf=1.0, neginf=1.0).clamp(0.0, 1.0)
        selected_mask = torch.zeros(self.num_views, device=device)
        selected_mask[list(views)] = 1.0
        current_one_hot = torch.zeros(self.num_views, device=device)
        current_one_hot[current] = 1.0
        remaining = torch.tensor([float(max(0, 2 - len(views)))], device=device)
        if "view_geometry" in self.cache.arrays:
            current_angle = torch.tensor(
                np.array(self.cache.arrays["view_geometry"][episode_id, current, :2], copy=True),
                device=device,
                dtype=torch.float32,
            )
        else:
            default_angle = torch.tensor(current * 0.5 * torch.pi, device=device)
            current_angle = torch.stack((default_angle.sin(), default_angle.cos()))
        yaw_confidence = quality[len(views) - 1, 3:4]
        robot_context = torch.cat(
            (
                current_one_hot,
                selected_mask,
                reachable,
                costs,
                remaining,
                current_angle,
                yaw_confidence,
            )
        )

        if "view_geometry" in self.cache.arrays:
            geometry = torch.tensor(np.array(self.cache.arrays["view_geometry"][episode_id], copy=True), device=device, dtype=torch.float32)
            if geometry.shape[-1] == 2:
                geometry = torch.cat((geometry, costs[:, None], reachable[:, None]), dim=-1)
            elif geometry.shape[-1] >= 4:
                geometry = geometry[:, :4].clone()
                geometry[:, 2] = costs
                geometry[:, 3] = reachable
        else:
            geometry = torch.zeros((self.num_views, 4), device=device)
            geometry[:, 2] = costs
            geometry[:, 3] = reachable
            # Fall back only when an old cache has no per-sample geometry.
            default_angles = torch.linspace(0.0, 1.5 * torch.pi, self.num_views, device=device)
            geometry[:, 0] = default_angles.sin()
            geometry[:, 1] = default_angles.cos()
        return {
            "view_features": z,
            "evidence": evidence,
            "robot_context": robot_context,
            "slot_mask": slot_mask,
            "action_mask": self.valid_action_mask(episode_id, views).to(device),
            "candidate_geometry": geometry,
            "pose_features": pose,
            "object_features": obj,
            "quality_features": quality,
        }

    def valid_action_mask(self, episode_id: int, views: Sequence[int]) -> torch.Tensor:
        """Mask selected and unreachable destinations from the current view."""

        if not views:
            raise ValueError("valid_action_mask needs a current view")
        current = int(views[-1])
        reachable = torch.as_tensor(
            np.array(self.cache.arrays["reachable"][episode_id, current], copy=True),
            dtype=torch.bool,
        )
        mask = reachable.clone()
        mask[list(int(v) for v in views)] = False
        return mask


# Implementation guide
# 1. Read each selected view independently, place it in A/B/C order, and zero-pad
#    unused slots. Keep the same shapes for one, two, and three observations.
# 2. Build u_X only through LogitFusion and build robot_context as current one-hot,
#    selected/reachable masks, costs, budget, sin/cos(delta_A), and yaw confidence.
# 3. Return no label or target column. Reward code must request those separately.
# 4. Apply slot_mask after ViewEncoder as well as before it so layer bias cannot
#    turn a padded slot into non-zero evidence.
# 5. Mark terminal s_ABC with no valid action; it exists only as next_state.
