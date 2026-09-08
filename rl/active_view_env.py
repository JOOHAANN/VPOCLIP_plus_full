"""Single-step active second-view environment from the Rosbot design."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

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
    """Navigation supplies A; the policy selects one legal complementary B."""

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
        """Initialize A and return the label-free policy state."""
        if not (0 <= int(episode_id) < len(self.cache)):
            raise IndexError(episode_id)
        self.episode_id = int(episode_id)
        if initial_view is None:
            # In deployment navigation chooses A.  Training randomizes the
            # legal starting view so the policy cannot memorize C001.
            reachable = self.cache.arrays["reachable"][self.episode_id]
            candidates = [i for i in range(self.cache.expected.views) if bool(reachable[i, i])]
            if not candidates:
                raise RuntimeError(f"episode {episode_id} has no legal initial view")
            import random
            initial_view = random.choice(candidates) if self.training else candidates[0]
        initial_view = int(initial_view)
        self.selected_views = (initial_view,)
        self.terminated = False
        state = self.state_builder.build(self.episode_id, self.selected_views)
        return state, {
            "episode_id": self.episode_id,
            "selected_views": [initial_view],
            "valid_action_mask": state["action_mask"],
        }

    def step(self, action_view_id: int) -> Tuple[PolicyState, float, bool, bool, StepInfo]:
        """Observe B and terminate immediately."""
        if self.episode_id is None or self.terminated:
            raise RuntimeError("reset must be called before step and only one step is allowed")
        action_view_id = int(action_view_id)
        valid = self.state_builder.valid_action_mask(self.episode_id, self.selected_views)
        if action_view_id < 0 or action_view_id >= self.cache.expected.views or not bool(valid[action_view_id]):
            raise ValueError(f"illegal second view {action_view_id}; valid={valid.tolist()}")
        previous = self.selected_views
        self.selected_views = previous + (action_view_id,)
        self.terminated = True
        next_state = self.state_builder.build(self.episode_id, self.selected_views)
        terms = self.compute_reward(previous, self.selected_views) if self.training else RewardTerms(0.0, 0.0, 0.0, 0.0)
        fused = self.state_builder.fusion.fuse(
            torch.stack([self.cache.get_view(self.episode_id, v)["logits"] for v in self.selected_views]).to(self.state_builder.text_prototypes.device),
            self.state_builder.text_prototypes,
        )
        meta = self.cache.get_episode_meta(self.episode_id)
        target = int(meta.get("target_col", meta.get("label", -1)))
        return next_state, terms.total, True, False, {
            "episode_id": self.episode_id,
            "selected_views": list(self.selected_views),
            "valid_action_mask": next_state["action_mask"],
            "fused_logits": fused.logits,
            "target_col": target,
            "prediction": int(torch.argmax(fused.logits).item()),
        }

    def compute_reward(self, previous_views: Tuple[int, ...], next_views: Tuple[int, ...]) -> RewardTerms:
        """Compute ``CE_A - CE_AB``, terminal correctness, and movement cost."""
        if self.episode_id is None:
            raise RuntimeError("environment has not been reset")
        device = self.state_builder.text_prototypes.device
        target = int(self.cache.arrays["target_columns"][self.episode_id])
        def ce(views: Tuple[int, ...]) -> float:
            scores = torch.stack([self.cache.get_view(self.episode_id, v)["logits"] for v in views]).to(device)
            fused = self.state_builder.fusion.fuse(scores, self.state_builder.text_prototypes)
            return float(-torch.log(fused.probabilities[target].clamp_min(1e-8)).item())
        ce_improvement = max(-self.reward_config.ce_clip, min(self.reward_config.ce_clip, ce(previous_views) - ce(next_views)))
        scores = torch.stack([self.cache.get_view(self.episode_id, v)["logits"] for v in next_views]).to(device)
        prediction = int(torch.argmax(self.state_builder.fusion.fuse(scores, self.state_builder.text_prototypes).logits).item())
        terminal = 1.0 if prediction == target else -1.0
        movement = self.cache.get_cost(self.episode_id, previous_views[-1], next_views[-1])
        if not torch.isfinite(torch.tensor(movement)):
            movement = 1.0
        movement = float(max(0.0, min(1.0, movement)))
        total = (
            self.reward_config.ce_weight * ce_improvement
            + self.reward_config.terminal_weight * terminal
            - self.reward_config.movement_weight * movement
        )
        return RewardTerms(
            ce_improvement=float(self.reward_config.ce_weight * ce_improvement),
            terminal_correctness=float(self.reward_config.terminal_weight * terminal),
            movement_penalty=float(-self.reward_config.movement_weight * movement),
            total=float(total),
        )


# Implementation guide
# 1. reset must select one legal A, set budget=2, and return s_A plus diagnostics.
# 2. step must reject selected/unreachable actions before reading the next view.
# 3. The first step returns s_AB and done=False; the second returns s_ABC and
#    done=True. Never allow a fourth observation or bootstrap from s_ABC.
# 4. Use CE_A-CE_AB and CE_AB-CE_ABC shaping, clipped before weighting; add the
#    +/-1 correctness term only at C and subtract the directed movement cost.
# 5. In evaluation mode, do not request labels, compute reward, or update models.
