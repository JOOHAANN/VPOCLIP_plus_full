"""Memory-mapped reader and validator for aligned multi-view cache arrays."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Union

import numpy as np

from .contracts import CacheShape, ViewObservation


Index = Union[int, np.integer]


class MultiViewCache:
    """Read cache arrays by episode and view without loading them into RAM."""

    REQUIRED_ARRAYS = ("z", "logits", "labels", "target_columns", "reachable", "move_cost")

    def __init__(self, root: Union[str, Path], expected: CacheShape, mmap: bool = True) -> None:
        self.root = Path(root)
        self.expected = expected
        self.mmap = mmap
        self.arrays: Dict[str, np.ndarray] = {}

    def open(self) -> "MultiViewCache":
        """Load required and optional ``.npy`` arrays and validate the contract."""

        raise NotImplementedError("Implement mmap loading and startup validation here.")

    def validate(self) -> None:
        """Check dimensions, dtypes, labels, metadata alignment, and graph masks."""

        raise NotImplementedError("Implement strict cache validation here.")

    def get_view(self, episode_id: Index, view_id: Index) -> ViewObservation:
        """Return one cached observation without exposing a writable array view."""

        raise NotImplementedError("Implement per-view lookup here.")

    def get_cost(self, episode_id: Index, source: Index, destination: Index) -> float:
        """Return normalized movement cost for one directed edge."""

        raise NotImplementedError("Implement movement-cost lookup here.")

    def get_episode_meta(self, episode_id: Index) -> Mapping[str, Any]:
        """Return labels and non-policy metadata for logging and reward only."""

        raise NotImplementedError("Implement metadata lookup here.")

    def __len__(self) -> int:
        return self.expected.episodes


# Implementation guide
# 1. Use np.load(..., mmap_mode="r", allow_pickle=False) for tensor arrays.
# 2. Require z[N,V,512], logits[N,V,55], reachable[N,V,V], and cost[N,V,V].
# 3. Validate optional pose/object/quality/geometry arrays when their state stage
#    is enabled; fail early instead of silently substituting unrelated features.
# 4. Confirm every view in one episode has identical grouping metadata and label.
# 5. Keep labels available to the environment reward path but never return them
#    from get_view or include them in a policy state.
