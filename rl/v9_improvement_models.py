"""Four improvements built around v9 relative candidate ranking."""
import torch
import torch.nn.functional as F
from torch import nn

from .causal_rank_v7 import Ranker


def pairwise_loss(q, utility, valid, hard=False):
    delta = utility[:, :, None] - utility[:, None, :]
    pairs = valid[:, :, None] & valid[:, None, :] & (delta > .05)
    diff = q[:, :, None] - q[:, None, :]
    # Utility gaps determine both which pairs matter and their target margin.
    target_margin = .10 + .50 * delta.clamp_min(0)
    violation = F.softplus((target_margin - diff) / .20)
    weights = delta.clamp_min(0) * pairs
    if hard:
        # Retain the two strongest violations per state; this is hard-negative
        # mining over candidate pairs, while preserving the causal state.
        flat = (violation * pairs).flatten(1)
        keep = torch.zeros_like(flat)
        k = min(2, flat.shape[1])
        ids = flat.topk(k, dim=1).indices
        keep.scatter_(1, ids, 1.)
        weights = weights.flatten(1) * keep
        violation = violation.flatten(1)
        return (violation * weights).sum(1) / weights.sum(1).clamp_min(1e-6)
    return (violation * weights).sum((1, 2)) / weights.sum((1, 2)).clamp_min(1e-6)


def listwise_loss(q, utility, valid):
    safe_q = q.masked_fill(~valid, -1e4)
    safe_u = utility.masked_fill(~valid, -1e4)
    # Soft utility targets retain graded information, while equal utilities
    # naturally produce a uniform target over legal candidates.
    target = F.softmax(safe_u / .20, dim=-1)
    logp = F.log_softmax(safe_q / .20, dim=-1)
    return F.kl_div(logp, target, reduction='none').sum(-1)


class InteractionRanker(Ranker):
    """v14: multiplicative context-candidate interaction before scoring."""

    def __init__(self):
        super().__init__(semantic=True)
        self.h_interaction = nn.Linear(64, 32)
        self.head = nn.Sequential(nn.Linear(64 + 32 + 32, 64), nn.GELU(),
                                  nn.Dropout(.20), nn.Linear(64, 1))

    def forward(self, s):
        parts = [s['quality'], s['evidence'],
                 F.dropout(self.z(s['z']), .4, self.training),
                 F.dropout(self.pose(s['pose']), .4, self.training),
                 F.dropout(self.obj(s['object']), .4, self.training)]
        h = self.context(torch.cat(parts, -1))[:, None].expand(-1, 4, -1)
        g = self.geometry(s['candidate'])
        h_i = self.h_interaction(h)
        return self.head(torch.cat((h, g, h_i * g), -1)).squeeze(-1)


def make_model(version):
    return InteractionRanker() if version == 'v14' else Ranker(True)


def decision_loss(version, q, utility, valid):
    if version == 'v12':
        return pairwise_loss(q, utility, valid, hard=True).mean()
    if version == 'v13':
        pair = pairwise_loss(q, utility, valid, hard=False)
        return (listwise_loss(q, utility, valid) + .20 * pair).mean()
    if version == 'v14':
        return pairwise_loss(q, utility, valid, hard=False).mean()
    if version == 'v15':
        return pairwise_loss(q, utility, valid, hard=False).mean()
    raise ValueError(version)
