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
        mask = state["action_mask"].bool()
        valid = torch.nonzero(mask, as_tuple=False).flatten()
        if valid.numel() == 0:
            raise RuntimeError("no legal action remains")
        import random
        if random.random() < float(epsilon):
            return int(valid[torch.randint(valid.numel(), (1,))].item())
        with torch.no_grad():
            q = self.masked_q(self.online(self._batch_state(state)), mask.unsqueeze(0))
        return int(torch.argmax(q[0]).item())

    def update(self, transitions: Sequence[IndexTransition]) -> Dict[str, float]:
        """Perform one masked Double DQN SmoothL1 update."""

        if not transitions:
            return {"loss": 0.0, "q": 0.0, "target": 0.0}
        device = next(self.online.parameters()).device
        states = [self.state_builder.build(t.episode_id, t.views) for t in transitions]
        next_states = [self.state_builder.build(t.episode_id, t.next_views) for t in transitions]
        state = self._stack_states(states, device)
        next_state = self._stack_states(next_states, device)
        actions = torch.as_tensor([t.action for t in transitions], dtype=torch.long, device=device)
        rewards = torch.as_tensor([t.reward for t in transitions], dtype=torch.float32, device=device)
        done = torch.as_tensor([t.done for t in transitions], dtype=torch.float32, device=device)
        q = self.online(state).gather(1, actions[:, None]).squeeze(1)
        with torch.no_grad():
            online_next = self.masked_q(self.online(next_state), next_state["action_mask"].bool())
            next_action = torch.argmax(online_next, dim=1)
            target_next = self.masked_q(self.target(next_state), next_state["action_mask"].bool())
            bootstrap = target_next.gather(1, next_action[:, None]).squeeze(1)
            target = rewards + self.gamma * (1.0 - done) * bootstrap
        loss = torch.nn.functional.smooth_l1_loss(q, target)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), self.grad_clip)
        self.optimizer.step()
        return {"loss": float(loss.item()), "q": float(q.mean().item()), "target": float(target.mean().item())}

    def sync_target(self) -> None:
        """Hard-copy online weights and leave the target network in evaluation mode."""

        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

    @staticmethod
    def masked_q(q_values: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """Replace invalid destination scores with a stable large negative value."""

        if q_values.ndim != 2 or valid_mask.shape != q_values.shape:
            raise ValueError(f"q/mask shape mismatch: {tuple(q_values.shape)} vs {tuple(valid_mask.shape)}")
        if not valid_mask.any(dim=1).all():
            raise ValueError("each batch row must contain a valid action")
        return q_values.masked_fill(~valid_mask.bool(), torch.finfo(q_values.dtype).min)

    @staticmethod
    def _batch_state(state: PolicyState) -> PolicyState:
        return {key: value.unsqueeze(0) if value.ndim in (1, 2) else value for key, value in state.items()}

    @staticmethod
    def _stack_states(states: Sequence[PolicyState], device: torch.device) -> PolicyState:
        keys = states[0].keys()
        return {key: torch.stack([state[key].to(device) for state in states], dim=0) for key in keys}


# Implementation guide
# 1. Apply masks during epsilon sampling, online argmax, and next-state argmax.
# 2. Select a_star with Q_online(next_state), then evaluate that action with
#    Q_target(next_state); do not take the maximum directly from the target net.
# 3. Compute target = r + gamma*(1-done)*Q_target(next,a_star) under no_grad.
# 4. Terminal ABC transitions must have target exactly equal to reward.
# 5. Use SmoothL1Loss, clip DQN gradients, and never include VPOCLIP parameters
#    in the optimizer.
