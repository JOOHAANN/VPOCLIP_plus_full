"""Two-decision active-view environment: initial A, choose B, then choose C."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from .contracts import PolicyState, RewardTerms, StepInfo
from .multiview_cache import MultiViewCache
from .state_builder import StateBuilder


@dataclass(frozen=True)
class RewardConfig:
    """Weights for CE improvement, terminal correctness, and movement cost."""

    ce_weight: float = 0.3
    terminal_weight: float = 1.0
    movement_weight: float = 0.05
    ce_clip: float = 1.0


class ActiveViewEnv:
    """Expose exactly two destination decisions for each synchronized episode."""

    def __init__(
        self,
        cache: MultiViewCache,
        state_builder: StateBuilder,
        reward_config: RewardConfig,
        training: bool = True,
    ) -> None:
        self.cache = cache
        self.state_builder = state_builder
        self.reward_config = reward_config
        self.training = training
        self.episode_id: Optional[int] = None
        self.selected_views: Tuple[int, ...] = ()
        self.terminated = False

    def reset(self, episode_id: int, initial_view: Optional[int] = None) -> Tuple[PolicyState, StepInfo]:
        """Initialize A and return s_A with the legal destination mask."""

        raise NotImplementedError("Implement legal initial-view selection and state reset here.")

    def step(self, action_view_id: int) -> Tuple[PolicyState, float, bool, bool, StepInfo]:
        """Observe B or C; terminate immediately after the second decision."""

        raise NotImplementedError("Implement the A-to-B-to-C transition logic here.")

    def compute_reward(self, previous_views: Tuple[int, ...], next_views: Tuple[int, ...]) -> RewardTerms:
        """Compute CE shaping, final correctness, and movement penalty in training."""

        raise NotImplementedError("Implement reward without exposing targets to policy state here.")


# Implementation guide
# 1. reset must select one legal A, set budget=2, and return s_A plus diagnostics.
# 2. step must reject selected/unreachable actions before reading the next view.
# 3. The first step returns s_AB and done=False; the second returns s_ABC and
#    done=True. Never allow a fourth observation or bootstrap from s_ABC.
# 4. Use CE_A-CE_AB and CE_AB-CE_ABC shaping, clipped before weighting; add the
#    +/-1 correctness term only at C and subtract the directed movement cost.
# 5. In evaluation mode, do not request labels, compute reward, or update models.
