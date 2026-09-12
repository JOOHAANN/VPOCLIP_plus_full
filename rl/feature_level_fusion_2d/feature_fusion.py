"""512-D feature-level fusion for the trajectory active-view experiment.

The recognizer cache contains one 512-D visual embedding per physical view
and one text embedding per class.  This module fuses visual embeddings first
and computes class similarity afterwards:

    f = Fusion(f_view_0, f_view_1)
    score_c = scale * cosine(f, text_c)

The fusion is permutation-symmetric for a pair.  A three-view path applies
the same module recursively, which makes the one-move and two-move protocols
use exactly the same operation.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class FeatureFusion512(nn.Module):
    """Low-capacity residual fusion that preserves the 512-D CLIP space.

    The zero-initialized last layer makes the initial model equal to normalized
    feature averaging.  Training can learn a correction from the pair's mean
    and absolute difference without changing the output dimension.
    """

    def __init__(self, dim: int = 512, hidden: int = 256) -> None:
        super().__init__()
        self.dim = int(dim)
        self.delta = nn.Sequential(
            nn.Linear(2 * dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )
        last = self.delta[-1]
        assert isinstance(last, nn.Linear)
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        first = F.normalize(first, dim=-1, eps=1e-6)
        second = F.normalize(second, dim=-1, eps=1e-6)
        mean = F.normalize(first + second, dim=-1, eps=1e-6)
        difference = (first - second).abs()
        correction = self.delta(torch.cat((mean, difference), dim=-1))
        return F.normalize(mean + 0.25 * correction, dim=-1, eps=1e-6)


def normalized_text(raw: object) -> torch.Tensor:
    """Return the normalized [55,512] text bank carried by GPUCache."""

    prototypes = getattr(raw, "prototypes", None)
    if prototypes is None:
        prototypes = getattr(raw, "text_prototypes")
    return F.normalize(prototypes.float(), dim=-1, eps=1e-6)


def infer_logit_scale(raw: object) -> float:
    """Recover the recognizer's CLIP temperature from the frozen cache.

    The cache logits are (up to numerical precision) a fixed scale times the
    cosine similarity between normalized z and normalized text features.  The
    estimate keeps entropy comparable to the old logit protocol while class
    argmax remains independent of the scale.
    """

    z = getattr(raw, "z").float()
    logits = getattr(raw, "logits").float()
    text = normalized_text(raw)
    sample_z = F.normalize(z.reshape(-1, z.shape[-1]), dim=-1, eps=1e-6)
    cosine = sample_z @ text.t()
    sample_logits = logits.reshape(-1, logits.shape[-1])
    keep = cosine.abs() > 0.05
    ratio = (sample_logits[keep] / cosine[keep]).flatten()
    ratio = ratio[torch.isfinite(ratio)]
    if not len(ratio):
        return 14.56
    scale = float(ratio.median().detach().cpu())
    return float(np.clip(scale, 1.0, 100.0))


def fuse_sequence(fusion: FeatureFusion512, features: torch.Tensor) -> torch.Tensor:
    """Fuse [B,K,512] or [K,512] features recursively to [B,512]/[512]."""

    if features.shape[-1] != fusion.dim:
        raise ValueError(f"expected last dimension {fusion.dim}, got {features.shape}")
    fused = F.normalize(features[..., 0, :], dim=-1, eps=1e-6)
    for index in range(1, features.shape[-2]):
        fused = fusion(fused, features[..., index, :])
    return F.normalize(fused, dim=-1, eps=1e-6)


def score_features(
    raw: object,
    fusion: FeatureFusion512,
    episodes: torch.Tensor,
    paths: torch.Tensor,
    scale: float | None = None,
) -> torch.Tensor:
    """Compute class scores after feature-level fusion of a path."""

    z = getattr(raw, "z")[episodes[:, None], paths]
    fused = fuse_sequence(fusion, z)
    if scale is None:
        scale = infer_logit_scale(raw)
    return float(scale) * (fused @ normalized_text(raw).t())


def margin_against_bank(
    scores: torch.Tensor, labels: torch.Tensor, bank: torch.Tensor
) -> torch.Tensor:
    restricted = scores.index_select(-1, bank)
    local = torch.searchsorted(bank, labels)
    target = restricted.gather(-1, local[..., None]).squeeze(-1)
    rival = restricted.clone()
    rival.scatter_(-1, local[..., None], -torch.inf)
    return target - rival.amax(-1)


@torch.no_grad()
def feature_utility_targets(
    raw: object,
    episodes: torch.Tensor,
    history: torch.Tensor,
    history_count: torch.Tensor,
    valid: torch.Tensor,
    bank: torch.Tensor,
    fusion: FeatureFusion512,
    scale: float | None = None,
) -> torch.Tensor:
    """Return offline candidate utilities using fused features, not logits."""

    fusion.eval()
    count = history_count.long().clamp(1, history.shape[1])
    z_history = getattr(raw, "z")[episodes[:, None], history]
    fused_before = F.normalize(z_history[:, 0], dim=-1, eps=1e-6)
    for index in range(1, history.shape[1]):
        updated = fusion(fused_before, z_history[:, index])
        fused_before = torch.where(
            (count > index)[:, None], updated, fused_before
        )
    if scale is None:
        scale = infer_logit_scale(raw)
    text = normalized_text(raw)
    before_scores = float(scale) * (fused_before @ text.t())
    before = margin_against_bank(before_scores, getattr(raw, "labels")[episodes], bank)

    utilities = []
    candidate_z = getattr(raw, "z")[episodes]
    for candidate in range(candidate_z.shape[1]):
        after = fusion(fused_before, candidate_z[:, candidate])
        scores = float(scale) * (after @ text.t())
        candidate_margin = margin_against_bank(
            scores, getattr(raw, "labels")[episodes], bank
        )
        utilities.append(candidate_margin - before)
    return torch.stack(utilities, dim=1).masked_fill(~valid, 0.0)


def _metrics_from_scores(
    scores: torch.Tensor, labels: torch.Tensor, bank: torch.Tensor
) -> dict[str, float]:
    scores = scores.index_select(-1, bank)
    local = torch.searchsorted(bank, labels)
    probability = F.softmax(scores, dim=-1)
    entropy = -(probability * probability.clamp_min(1e-12).log()).sum(-1)
    entropy = entropy / math.log(float(len(bank)))
    return {
        "top1": float(scores.argmax(-1).eq(local).float().mean()) if len(scores) else 0.0,
        "mean_entropy": float(entropy.mean()) if len(scores) else 0.0,
        "episodes": float(len(scores)),
    }


@torch.no_grad()
def single_metrics(
    raw: object,
    fusion: FeatureFusion512,
    episodes: torch.Tensor,
    starts: torch.Tensor,
    bank: torch.Tensor,
    scale: float | None = None,
) -> dict[str, float]:
    paths = starts[:, None]
    return _metrics_from_scores(
        score_features(raw, fusion, episodes, paths, scale),
        getattr(raw, "labels")[episodes],
        bank,
    )


@torch.no_grad()
def path_metrics(
    raw: object,
    fusion: FeatureFusion512,
    episodes: torch.Tensor,
    paths: torch.Tensor,
    bank: torch.Tensor,
    scale: float | None = None,
) -> dict[str, float]:
    metrics = _metrics_from_scores(
        score_features(raw, fusion, episodes, paths, scale),
        getattr(raw, "labels")[episodes],
        bank,
    )
    if len(episodes) and paths.shape[1] > 1:
        costs = sum(
            getattr(raw, "cost")[episodes, paths[:, i], paths[:, i + 1]]
            for i in range(paths.shape[1] - 1)
        )
        metrics["mean_movement_cost"] = float(costs.nan_to_num(1.0).mean())
    else:
        metrics["mean_movement_cost"] = 0.0
    return metrics

