"""Frozen VPOCLIP adapter used only to export per-view embeddings and logits."""

from __future__ import annotations

from typing import Dict, Mapping, Optional

import torch
from torch import nn

from .contracts import TensorDict


class VPOCLIPAdapter:
    """Own the frozen recognizer and expose a cache-oriented batch interface."""

    def __init__(self, model: nn.Module, device: torch.device) -> None:
        self.model = model.to(device)
        self.device = device
        self.freeze()

    def freeze(self) -> None:
        """Put the recognizer in evaluation mode and disable all gradients."""

        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def encode_all_views(self, batch: Mapping[str, torch.Tensor]) -> TensorDict:
        """Return ``z[B,V,512]`` and ``logits[B,V,55]`` without gradients."""

        raise NotImplementedError("Implement flattened B*V VPOCLIP inference here.")

    def assert_frozen(self, reference: Optional[Dict[str, torch.Tensor]] = None) -> None:
        """Check gradient flags and optionally compare against a weight snapshot."""

        raise NotImplementedError("Implement freeze and bitwise-weight validation here.")

    @classmethod
    def from_config(cls, config_path: str, checkpoint_path: str, device: str) -> "VPOCLIPAdapter":
        """Build the existing model, text bank, and checkpoint exactly as test.py does."""

        raise NotImplementedError("Reuse model.py and test.py checkpoint loading semantics here.")


# Implementation guide
# 1. Load the recognizer through model.build_model_from_config and populate the
#    same eight-prompt text bank used by the selected final_aug checkpoint.
# 2. Load checkpoints with the same full-model/fusion-head rules as test.py.
# 3. Flatten every [B,V,...] input to [B*V,...], call encode_visual once under
#    torch.inference_mode(), compute scaled cosine logits, then restore B and V.
# 4. Never add an RL projection to z or modify the 512-D VPOCLIP semantic space.
# 5. Snapshot state_dict before RL training and verify bitwise equality afterward.
