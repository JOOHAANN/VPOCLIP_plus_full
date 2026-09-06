"""Index-based replay buffer for the two transitions in each episode."""

from __future__ import annotations

from collections import deque
from random import Random
from typing import Deque, List, Optional

from .contracts import IndexTransition


class ReplayBuffer:
    """Store compact episode/view indices instead of repeated tensor states."""

    def __init__(self, capacity: int, seed: Optional[int] = None) -> None:
        self.capacity = int(capacity)
        self._items: Deque[IndexTransition] = deque(maxlen=self.capacity)
        self._rng = Random(seed)

    def add(self, transition: IndexTransition) -> None:
        """Append one validated A->AB or AB->ABC transition."""

        raise NotImplementedError("Implement transition-contract validation before append here.")

    def sample(self, batch_size: int) -> List[IndexTransition]:
        """Sample indices; the agent uses StateBuilder to reconstruct tensors."""

        raise NotImplementedError("Implement reproducible uniform sampling here.")

    def __len__(self) -> int:
        return len(self._items)


# Implementation guide
# 1. Accept only views lengths 1->2 for non-terminal and 2->3 for terminal items.
# 2. Verify action equals the newly appended destination and all views are unique.
# 3. Keep episode_id and tuples only; reconstruct both states during agent.update.
# 4. Add prioritized replay only after the uniform baseline is correct and tested.
