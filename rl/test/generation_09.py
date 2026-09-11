"""Generation 09: permutation-invariant candidate-bank policy.

The action remains a direct four-way policy argmax.  The network receives a
set summary of the current five most probable classes in the active class
bank (probability plus frozen text prototype), current-view evidence, and
relative geometry.  It never inspects an unvisited view at inference and it
does not contain a random fallback or an RL/random gate.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from . import generation_06 as runner


_GEN6_AUGMENTED_STATE = runner.augmented_state


def augmented_state(cache, episodes, starts, bank):
    """Add a fixed-size, label-free posterior/prototype set to the state."""
    state = _GEN6_AUGMENTED_STATE(cache, episodes, starts, bank)
    logits = cache.logits[episodes, starts]
    bank_logits = logits.masked_fill(~bank, -torch.inf)
    probs = bank_logits.softmax(-1)
    top_p, top_i = probs.topk(5, -1)
    prototypes = cache.prototypes[top_i]
    # Keep both probability and log-probability.  The latter preserves useful
    # relative separation after softmax while the prototype part is class-ID
    # free and therefore transferable to unseen classes.
    set_features = torch.cat((prototypes, top_p[..., None],
                              top_p.clamp_min(1e-8).log()[..., None]), -1)
    state["bank_set"] = set_features
    return state


def geometry_features(candidate):
    angle, delta = candidate[..., :2], candidate[..., 2:4]
    angle2 = torch.stack((2 * angle[..., 0] * angle[..., 1],
                          angle[..., 1].square() - angle[..., 0].square()), -1)
    delta2 = torch.stack((2 * delta[..., 0] * delta[..., 1],
                          delta[..., 1].square() - delta[..., 0].square()), -1)
    return torch.cat((candidate, angle2, delta2), -1)


class SetPolicy(nn.Module):
    """Direct action scorer with a permutation-invariant class-bank encoder."""

    def __init__(self, variant: str):
        super().__init__()
        self.variant = variant
        self.semantic = nn.Sequential(nn.Linear(512, 64), nn.LayerNorm(64), nn.GELU())
        self.stats = nn.Sequential(nn.Linear(7, 16), nn.LayerNorm(16), nn.GELU())
        self.bank_item = nn.Sequential(nn.Linear(514, 64), nn.LayerNorm(64), nn.GELU(),
                                       nn.Linear(64, 64), nn.GELU())
        self.bank_summary = nn.Sequential(nn.Linear(128, 64), nn.LayerNorm(64), nn.GELU())
        self.z = nn.Sequential(nn.Linear(512, 32), nn.LayerNorm(32), nn.GELU())
        self.pose = nn.Sequential(nn.Linear(3, 16), nn.LayerNorm(16), nn.GELU())

        context_dim = 64 + 16 + 64
        if "z" in variant:
            context_dim += 32
        if "pose" in variant:
            context_dim += 16
        self.context = nn.Sequential(nn.Linear(context_dim, 96), nn.LayerNorm(96), nn.GELU(),
                                     nn.Dropout(.10), nn.Linear(96, 48), nn.GELU())
        self.candidate = nn.Sequential(nn.Linear(11, 48), nn.LayerNorm(48), nn.GELU())
        self.head = nn.Sequential(nn.Linear(48 + 48 + 48 + 48, 80), nn.GELU(),
                                  nn.Dropout(.10), nn.Linear(80, 1))

    def forward(self, state):
        item = self.bank_item(state["bank_set"])
        pooled = torch.cat((item.mean(1), item.amax(1)), -1)
        parts = [self.semantic(state["semantic"]), self.stats(state["global_stats"]),
                 self.bank_summary(pooled)]
        if "z" in self.variant:
            parts.append(self.z(state["z"]))
        if "pose" in self.variant:
            pose = state["pose"]
            pose = torch.cat((pose.mean(-1, keepdim=True), pose.std(-1, keepdim=True),
                              pose.abs().mean(-1, keepdim=True)), -1)
            parts.append(self.pose(pose))
        h = self.context(torch.cat(parts, -1))[:, None].expand(-1, 4, -1)
        g = self.candidate(geometry_features(state["candidate"]))
        return self.head(torch.cat((h, g, h * g, h - g), -1)).squeeze(-1)


def relative_gain_loss(q, utility, valid):
    """Rank all materially different legal actions without forcing ties apart."""
    count = valid.sum(-1).clamp_min(1)
    centered_q = q - (q * valid).sum(-1, keepdim=True) / count[:, None]
    centered_u = utility - (utility * valid).sum(-1, keepdim=True) / count[:, None]
    regression = (F.smooth_l1_loss(centered_q, centered_u, reduction="none") * valid).sum(-1) / count

    delta = utility[:, :, None] - utility[:, None, :]
    pairs = valid[:, :, None] & valid[:, None, :] & (delta > .02)
    margin = (.10 + .40 * delta.clamp_min(0)).detach()
    diff = q[:, :, None] - q[:, None, :]
    pair_loss = (F.softplus((margin - diff) / .20) * pairs).sum((1, 2))
    pair_count = pairs.sum((1, 2)).clamp_min(1)
    active = pairs.any(-1).any(-1)
    total = pair_loss / pair_count + .15 * regression
    return (total * active).sum() / active.float().sum().clamp_min(1)


# Patch only this process's runner.  Existing generations and the v1-v9
# backup are untouched on disk.
runner.GENERATION = "generation_09"
runner.OUT = runner.ROOT / "work_dir/active_view_iterative_test/generation_09"
runner.POOL_MODE = "all50"
runner.augmented_state = augmented_state
runner.core.causal_state = augmented_state
runner.SemanticPolicy = SetPolicy
runner.loss_for = lambda variant, q, u, valid: relative_gain_loss(q, u, valid)


def relative_advantage(cache, episodes, starts, banks, valid):
    current = cache.logits[episodes, starts]
    future = cache.logits[episodes]
    fused = (current[:, None, :] + future) * .5
    labels = cache.labels[episodes]
    current_true = current.gather(-1, labels[:, None]).squeeze(-1)
    current_rivals = current.masked_fill(~banks, -torch.inf).clone()
    current_rivals.scatter_(1, labels[:, None], -torch.inf)
    current_margin = current_true - current_rivals.amax(-1)
    fused_true = fused.gather(-1, labels[:, None, None].expand(-1, 4, 1)).squeeze(-1)
    rivals = banks[:, None, :].expand(-1, 4, -1).clone()
    rivals.scatter_(2, labels[:, None, None].expand(-1, 4, 1), False)
    fused_rival = fused.masked_fill(~rivals, -torch.inf).amax(-1)
    gain = fused_true - fused_rival - current_margin[:, None]
    utility = torch.tanh(gain / 10.).masked_fill(~valid, 0.)
    selective = valid.any(-1) & ((utility.max(-1).values - utility.min(-1).values) > .02)
    return utility, selective


runner.utility = relative_advantage
runner.VARIANTS = {
    "g09_set": "posterior-prototype set + semantic summary + geometry",
    "g09_set_z": "posterior-prototype set + semantic summary + video embedding + geometry",
    "g09_set_pose": "posterior-prototype set + semantic summary + pose + geometry",
    "g09_set_zpose": "posterior-prototype set + semantic summary + video + pose + geometry",
}


if __name__ == "__main__":
    runner.main()
