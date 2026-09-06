"""Masked Double DQN action selection, Bellman update, and target synchronization."""

from __future__ import annotations

from typing import Dict, Sequence

import torch
from torch import nn


from .contracts import IndexTransition, PolicyState
from .networks import DuelingQNetwork
from .state_builder import StateBuilder


class DoubleDQNAgent:
    """Train only the view policy while the VPOCLIP recognizer stays frozen."""

    def __init__(
        self,
        online: DuelingQNetwork,
        target: DuelingQNetwork,
        state_builder: StateBuilder,
        optimizer: torch.optim.Optimizer,
        gamma: float = 0.95,
        grad_clip: float = 10.0,
    ) -> None:
        self.online = online
        self.target = target
        self.state_builder = state_builder
        self.optimizer = optimizer
        self.gamma = gamma
        self.grad_clip = grad_clip

    def select_action(self, state: PolicyState, epsilon: float) -> int:
        """Choose a random valid destination or masked online-network argmax."""

        raise NotImplementedError("Implement epsilon-greedy selection over valid destinations here.")

    def update(self, transitions: Sequence[IndexTransition]) -> Dict[str, float]:
        """Perform one masked Double DQN SmoothL1 update."""

        raise NotImplementedError("Implement indexed state reconstruction and Bellman loss here.")

    def sync_target(self) -> None:
        """Hard-copy online weights and leave the target network in evaluation mode."""

        raise NotImplementedError("Implement target hard synchronization here.")

    @staticmethod
    def masked_q(q_values: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """Replace invalid destination scores with a stable large negative value."""

        raise NotImplementedError("Implement shape-checked action masking here.")


# Implementation guide
# 1. Apply masks during epsilon sampling, online argmax, and next-state argmax.
# 2. Select a_star with Q_online(next_state), then evaluate that action with
#    Q_target(next_state); do not take the maximum directly from the target net.
# 3. Compute target = r + gamma*(1-done)*Q_target(next,a_star) under no_grad.
# 4. Terminal ABC transitions must have target exactly equal to reward.
# 5. Use SmoothL1Loss, clip DQN gradients, and never include VPOCLIP parameters
#    in the optimizer.
