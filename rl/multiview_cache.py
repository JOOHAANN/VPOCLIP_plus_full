"""Memory-mapped reader and validator for aligned multi-view cache arrays."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Union

import numpy as np
import torch

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
        if not self.root.is_dir():
            raise FileNotFoundError(f"cache directory does not exist: {self.root}")
        for name in self.REQUIRED_ARRAYS:
            path = self.root / f"{name}.npy"
            if not path.exists():
                raise FileNotFoundError(f"missing required cache array: {path}")
            self.arrays[name] = np.load(path, mmap_mode="r" if self.mmap else None, allow_pickle=False)
        for name in ("pose", "object_map", "image_quality", "view_geometry"):
            path = self.root / f"{name}.npy"
            if path.exists():
                self.arrays[name] = np.load(path, mmap_mode="r" if self.mmap else None, allow_pickle=False)
        self.validate()
        return self

    def validate(self) -> None:
        """Check dimensions, dtypes, labels, metadata alignment, and graph masks."""

        z = self.arrays["z"]
        logits = self.arrays["logits"]
        labels = self.arrays["labels"]
        columns = self.arrays["target_columns"]
        reachable = self.arrays["reachable"]
        cost = self.arrays["move_cost"]
        n, v, d = z.shape
        if (n, v) != (self.expected.episodes, self.expected.views):
            raise ValueError(f"z shape {z.shape} does not match expected episodes/views")
        if d != self.expected.embedding_dim:
            raise ValueError(f"z embedding dim must be {self.expected.embedding_dim}, got {d}")
        if logits.shape != (n, v, self.expected.classes):
            raise ValueError(f"logits must be {(n, v, self.expected.classes)}, got {logits.shape}")
        if labels.shape != (n,) or columns.shape != (n,):
            raise ValueError("labels and target_columns must both have shape [episodes]")
        if reachable.shape != (n, v, v) or cost.shape != (n, v, v):
            raise ValueError("reachable and move_cost must have shape [episodes,views,views]")
        if reachable.dtype != np.bool_:
            raise ValueError(f"reachable must be bool, got {reachable.dtype}")
        if not np.isfinite(z).all() or not np.isfinite(logits).all() or not np.isfinite(cost).all():
            raise ValueError("cache contains non-finite z/logits/move_cost values")
        if np.any(labels < 0) or np.any(labels >= self.expected.classes):
            raise ValueError("labels are outside the 55-class bank")
        for name, shape_prefix in (("pose", (n, v)), ("object_map", (n, v)), ("image_quality", (n, v)), ("view_geometry", (n, v))):
            if name in self.arrays and self.arrays[name].shape[:2] != shape_prefix:
                raise ValueError(f"{name} first dimensions must be {shape_prefix}")
        if "view_geometry" in self.arrays and self.arrays["view_geometry"].shape[-1] < 2:
            raise ValueError("view_geometry needs at least [sin(delta),cos(delta)]")
        if np.any(np.diagonal(reachable, axis1=1, axis2=2) == False):
            raise ValueError("each view must be reachable from itself for reset/navigation bookkeeping")

    def get_view(self, episode_id: Index, view_id: Index) -> ViewObservation:
        """Return one cached observation without exposing a writable array view."""

        e, v = int(episode_id), int(view_id)
        if not (0 <= e < len(self) and 0 <= v < self.expected.views):
            raise IndexError((e, v))
        result: ViewObservation = {
            "z": torch.from_numpy(np.array(self.arrays["z"][e, v], copy=True)),
            "logits": torch.from_numpy(np.array(self.arrays["logits"][e, v], copy=True)),
        }
        for key in ("pose", "object_map", "image_quality", "view_geometry"):
            if key in self.arrays:
                result[key] = torch.from_numpy(np.array(self.arrays[key][e, v], copy=True))
        return result

    def get_cost(self, episode_id: Index, source: Index, destination: Index) -> float:
        """Return normalized movement cost for one directed edge."""

        e, s, d = int(episode_id), int(source), int(destination)
        if not bool(self.arrays["reachable"][e, s, d]):
            return float("inf")
        return float(self.arrays["move_cost"][e, s, d])

    def get_episode_meta(self, episode_id: Index) -> Mapping[str, Any]:
        """Return labels and non-policy metadata for logging and reward only."""

        e = int(episode_id)
        metadata_path = self.root / "metadata.json"
        metadata: Dict[str, Any] = {}
        if metadata_path.exists():
            import json
            all_meta = json.loads(metadata_path.read_text(encoding="utf-8"))
            records = all_meta.get("episodes", [])
            if e < len(records):
                metadata.update(records[e])
        metadata.setdefault("label", int(self.arrays["labels"][e]))
        metadata.setdefault("target_col", int(self.arrays["target_columns"][e]))
        return metadata

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
