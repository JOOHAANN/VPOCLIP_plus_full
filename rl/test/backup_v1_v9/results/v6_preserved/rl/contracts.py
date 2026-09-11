"""Shared typed contracts for cache, state, environment, and replay modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, TypedDict

import torch


class ViewObservation(TypedDict, total=False):
    """One independently encoded camera observation from the cache."""

    z: torch.Tensor
    logits: torch.Tensor
    pose: torch.Tensor
    object_map: torch.Tensor
    image_quality: torch.Tensor
    view_geometry: torch.Tensor


class PolicyState(TypedDict):
    """Label-free fixed-shape state at the navigation-selected view A.

    The policy observes A only.  The four candidate rows contain geometry and
    navigation information, never visual features from an unvisited camera.
    """

    view_features: torch.Tensor
    evidence: torch.Tensor
    robot_context: torch.Tensor
    slot_mask: torch.Tensor
    action_mask: torch.Tensor
    candidate_geometry: torch.Tensor
    pose_features: torch.Tensor
    object_features: torch.Tensor
    quality_features: torch.Tensor


class StepInfo(TypedDict, total=False):
    """Diagnostics returned by ``ActiveViewEnv.reset`` and ``step``."""

    episode_id: int
    selected_views: List[int]
    valid_action_mask: torch.Tensor
    probabilities: torch.Tensor
    fused_logits: torch.Tensor
    target_col: int
    prediction: int


@dataclass(frozen=True)
class IndexTransition:
    """Compact one-step replay item reconstructed through ``StateBuilder``."""

    episode_id: int
    views: Tuple[int, ...]
    action: int
    reward: float
    next_views: Tuple[int, ...]
    done: bool


@dataclass(frozen=True)
class RewardTerms:
    """Named reward components used for logging and reward unit tests."""

    ce_improvement: float
    terminal_correctness: float
    movement_penalty: float
    total: float


@dataclass(frozen=True)
class CacheShape:
    """Expected dimensions used when validating a multi-view cache."""

    episodes: int
    views: int = 4
    embedding_dim: int = 512
    classes: int = 55
    frames: int = 13
    joints: int = 17


TensorDict = Dict[str, torch.Tensor]
OptionalTensor = Optional[torch.Tensor]


# Implementation guide
# 1. Treat these types as cross-module API contracts, not storage classes.
# 2. Add fields only when every producer and consumer has a migration plan.
# 3. Keep replay transitions index-based; never duplicate full tensor states.
# 4. Keep labels out of PolicyState so unseen evaluation cannot leak targets.
