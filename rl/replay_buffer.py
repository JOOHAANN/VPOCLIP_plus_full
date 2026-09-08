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
        if len(transition.views) != 1 or len(transition.next_views) != 2:
            raise ValueError("Rosbot active-view replay expects exactly A -> AB")
        if transition.next_views[0] != transition.views[0] or transition.next_views[1] != transition.action:
            raise ValueError("next_views must append the selected action to the previous view")
        if len(set(transition.next_views)) != 2 or not transition.done:
            raise ValueError("one-step active-view transitions must terminate at AB")
        if not all(int(view) >= 0 for view in transition.next_views):
            raise ValueError("view ids must be non-negative")
        self._items.append(transition)

    def sample(self, batch_size: int) -> List[IndexTransition]:
        """Sample indices; the agent uses StateBuilder to reconstruct tensors."""

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if batch_size > len(self._items):
            raise ValueError(f"cannot sample {batch_size} items from {len(self._items)}")
        return self._rng.sample(list(self._items), int(batch_size))

    def __len__(self) -> int:
        return len(self._items)


# Implementation guide
# 1. Accept only views lengths 1->2 for non-terminal and 2->3 for terminal items.
# 2. Verify action equals the newly appended destination and all views are unique.
# 3. Keep episode_id and tuples only; reconstruct both states during agent.update.
# 4. Add prioritized replay only after the uniform baseline is correct and tested.
