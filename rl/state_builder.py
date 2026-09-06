"""Single source of truth for constructing fixed-slot policy states."""

from __future__ import annotations

from typing import Sequence

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

        raise NotImplementedError("Implement the only train/eval state-building path here.")

    def valid_action_mask(self, episode_id: int, views: Sequence[int]) -> torch.Tensor:
        """Mask selected and unreachable destinations from the current view."""

        raise NotImplementedError("Implement selected/reachable action masking here.")


# Implementation guide
# 1. Read each selected view independently, place it in A/B/C order, and zero-pad
#    unused slots. Keep the same shapes for one, two, and three observations.
# 2. Build u_X only through LogitFusion and build robot_context as current one-hot,
#    selected mask, reachable mask, remaining budget, and all destination costs.
# 3. Return no label or target column. Reward code must request those separately.
# 4. Apply slot_mask after ViewEncoder as well as before it so layer bias cannot
#    turn a padded slot into non-zero evidence.
# 5. Mark terminal s_ABC with no valid action; it exists only as next_state.
