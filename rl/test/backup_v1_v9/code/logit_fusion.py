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

        if logits.ndim not in (2, 3):
            raise ValueError(f"logits must be [K,C] or [B,K,C], got {tuple(logits.shape)}")
        if logits.shape[-1] != self.num_classes:
            raise ValueError(
                f"expected {self.num_classes} classes, got {logits.shape[-1]}"
            )
        if text_prototypes.ndim != 2 or text_prototypes.shape != (self.num_classes, 512):
            raise ValueError(
                "text_prototypes must have shape [num_classes,512], "
                f"got {tuple(text_prototypes.shape)}"
            )
        if weights is None:
            weight = torch.ones(logits.shape[-2], device=logits.device, dtype=logits.dtype)
        else:
            if len(weights) != logits.shape[-2]:
                raise ValueError("number of fusion weights must equal the number of views")
            weight = torch.as_tensor(weights, device=logits.device, dtype=logits.dtype)
        weight = weight / weight.sum().clamp_min(torch.finfo(weight.dtype).eps)
        fused = (logits * weight.reshape(*([1] * (logits.ndim - 2)), -1, 1)).sum(dim=-2)
        probabilities = torch.softmax(fused, dim=-1)
        k = min(self.top_k, self.num_classes)
        top_probabilities, top_indices = torch.topk(probabilities, k=k, dim=-1)
        prototypes = text_prototypes.to(device=logits.device, dtype=logits.dtype)
        if logits.ndim == 2:
            semantic_summary = (top_probabilities.unsqueeze(-1) * prototypes[top_indices]).sum(dim=-2)
            entropy = -(probabilities.clamp_min(1e-8).log() * probabilities).sum()
            entropy = entropy / torch.log(torch.tensor(float(self.num_classes), device=logits.device))
            margin = top_probabilities[..., 0] - top_probabilities[..., 1].clamp_min(0.0) if k > 1 else top_probabilities[..., 0]
        else:
            semantic_summary = (top_probabilities.unsqueeze(-1) * prototypes[top_indices]).sum(dim=-2)
            entropy = -(probabilities.clamp_min(1e-8).log() * probabilities).sum(dim=-1)
            entropy = entropy / torch.log(torch.tensor(float(self.num_classes), device=logits.device))
            margin = top_probabilities[..., 0] - top_probabilities[..., 1] if k > 1 else top_probabilities[..., 0]
        return FusionOutput(
            logits=fused,
            probabilities=probabilities,
            top_indices=top_indices,
            top_probabilities=top_probabilities,
            semantic_summary=semantic_summary,
            entropy=entropy,
            margin=margin,
        )


# Implementation guide
# 1. Accept logits shaped [K,55] or [B,K,55] and default to equal weights.
# 2. Compute L_X as a weighted sum; never fuse z_A/z_B/z_C before VPOCLIP logits.
# 3. Use softmax only for CE reward, normalized entropy, margin, and Top-5 weights.
# 4. Build e_top5 as the probability-weighted sum of the selected 512-D text
#    prototypes; concatenate it with five probabilities, H, and margin.
# 5. Add a test proving argmax(L_X) equals argmax(softmax(L_X)).
