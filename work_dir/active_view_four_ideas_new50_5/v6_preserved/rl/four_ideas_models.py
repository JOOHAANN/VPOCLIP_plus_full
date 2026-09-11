"""Four causal selector hypotheses; no unvisited visual features at inference."""
import torch
from torch import nn
import torch.nn.functional as F
from .causal_rank_v7 import Ranker, relative_loss


class GatedRanker(nn.Module):
    def __init__(self):
        super().__init__()
        self.semantic = Ranker(True)
        self.geometry = Ranker(False)
        self.gate = nn.Sequential(nn.Linear(23, 16), nn.GELU(), nn.Linear(16, 1), nn.Sigmoid())

    def forward(self, state):
        gate = self.gate(torch.cat((state['quality'], state['evidence']), -1))
        return gate * self.semantic(state) + (1 - gate) * self.geometry(state)


class Ensemble(nn.Module):
    def __init__(self, members):
        super().__init__()
        self.members = nn.ModuleList(members)

    def forward(self, state):
        # Relative-regression members share the same target units.
        return torch.stack([m(state) for m in self.members]).mean(0)


def make_model(version):
    return GatedRanker() if version == 'v11' else Ranker(True)


def decision_loss(version, q, utility, valid):
    if version in ('v10', 'v11'):
        return relative_loss(q, utility, valid)
    count = valid.sum(-1)
    safe = q.masked_fill(~valid, -1e4)
    if version == 'v8':
        # utility = correctness + bounded margin: > .5 exactly identifies correct candidates.
        good = (utility > .5) & valid
        informative = good.any(-1) & ((~good) & valid).any(-1)
        logp = safe.log_softmax(-1)
        # Maximize probability of the entire acceptable set, without forcing a unique winner.
        mass_loss = -torch.logsumexp(logp.masked_fill(~good, -1e4), -1)
        active = informative.float()
        return (mass_loss * active).sum() / active.sum().clamp_min(1)
    if version == 'v9':
        delta = utility[:, :, None] - utility[:, None, :]
        pairs = valid[:, :, None] & valid[:, None, :] & (delta > .05)
        weights = delta.clamp_min(0) * pairs
        diff = q[:, :, None] - q[:, None, :]
        # Reward differences set both pair importance and target ordering margin.
        per = (F.softplus((delta.clamp_min(0) - diff) / .2) * weights).sum((1, 2))
        per = per / weights.sum((1, 2)).clamp_min(1e-6)
        active = pairs.any(-1).any(-1).float()
        # Weak centered regularization prevents unlimited score scale growth.
        center = q - (q * valid).sum(-1, keepdim=True) / count[:, None].clamp_min(1)
        penalty = (center.square() * valid).sum(-1) / count.clamp_min(1)
        return ((per + .001 * penalty) * active).sum() / active.sum().clamp_min(1)
    raise ValueError(version)
