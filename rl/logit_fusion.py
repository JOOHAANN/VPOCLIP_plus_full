"""Independent-view logit fusion and uncertainty summaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch


@dataclass(frozen=True)
class FusionOutput:
    """Fused scores and probability-derived state features."""

    logits: torch.Tensor
    probabilities: torch.Tensor
    top_indices: torch.Tensor
    top_probabilities: torch.Tensor
    semantic_summary: torch.Tensor
    entropy: torch.Tensor
    margin: torch.Tensor


class LogitFusion:
    """Fuse A/AB/ABC only at the recognizer logit level."""

    def __init__(self, num_classes: int = 55, top_k: int = 5) -> None:
        self.num_classes = num_classes
        self.top_k = top_k

    def fuse(
        self,
        logits: torch.Tensor,
        text_prototypes: torch.Tensor,
        weights: Optional[Sequence[float]] = None,
    ) -> FusionOutput:
        """Fuse ``K`` independent logits and build the cumulative evidence summary."""

        raise NotImplementedError("Implement equal or configured weighted logit fusion here.")


# Implementation guide
# 1. Accept logits shaped [K,55] or [B,K,55] and default to equal weights.
# 2. Compute L_X as a weighted sum; never fuse z_A/z_B/z_C before VPOCLIP logits.
# 3. Use softmax only for CE reward, normalized entropy, margin, and Top-5 weights.
# 4. Build e_top5 as the probability-weighted sum of the selected 512-D text
#    prototypes; concatenate it with five probabilities, H, and margin.
# 5. Add a test proving argmax(L_X) equals argmax(softmax(L_X)).
