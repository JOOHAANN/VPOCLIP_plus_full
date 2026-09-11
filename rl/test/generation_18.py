"""Generation 18: low-capacity factorized skeleton policy.

This is a regularized alternative to the larger MLP scorers.  The temporal
skeleton state and candidate geometry are encoded separately and combined by
a low-rank dot product, which should make the learned rule more transferable
across action classes and camera slots.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from . import generation_17 as skeleton_base
from .generation_09 import geometry_features


base13 = skeleton_base.base13
base12 = base13.base
runner = base13.runner
OUT = runner.ROOT / "work_dir/active_view_iterative_test/generation_18"
GENERATION = "generation_18"
_ORIGINAL_STATE = skeleton_base._ORIGINAL_STATE
skeleton_features = skeleton_base.skeleton_features


def augmented_state(cache, episodes, starts, bank):
    state = _ORIGINAL_STATE(cache, episodes, starts, bank)
    state["skeleton_prior"] = skeleton_features(state["pose"])
    state["current_body_direction"] = state["candidate"][:, 0, 4:6]
    return state


class FactorizedSkeletonPolicy(nn.Module):
    def __init__(self, variant):
        super().__init__()
        self.variant = variant
        self.skeleton = nn.Sequential(nn.Linear(212, 48), nn.LayerNorm(48), nn.GELU())
        self.direction = nn.Sequential(nn.Linear(2, 8), nn.LayerNorm(8), nn.GELU())
        self.semantic = nn.Sequential(nn.Linear(512, 20), nn.LayerNorm(20), nn.GELU())
        self.stats = nn.Sequential(nn.Linear(7, 8), nn.LayerNorm(8), nn.GELU())
        self.bank_item = nn.Sequential(nn.Linear(514, 20), nn.LayerNorm(20), nn.GELU())
        self.bank_summary = nn.Sequential(nn.Linear(40, 20), nn.LayerNorm(20), nn.GELU())
        self.logits = nn.Sequential(nn.Linear(110, 10), nn.LayerNorm(10), nn.GELU())
        self.z = nn.Sequential(nn.Linear(512, 16), nn.LayerNorm(16), nn.GELU())

        context_dim = 48 + 8
        if "semantic" in variant:
            context_dim += 20 + 8 + 20
        if "z" in variant:
            context_dim += 16
        if "logits" in variant:
            context_dim += 10
        self.context = nn.Sequential(nn.Linear(context_dim, 48), nn.LayerNorm(48), nn.GELU())
        self.candidate = nn.Sequential(nn.Linear(15, 48), nn.LayerNorm(48), nn.GELU())
        self.context_to_action = nn.Linear(48, 48, bias=False)
        self.residual = nn.Sequential(nn.Linear(96, 32), nn.GELU(), nn.Linear(32, 1))
        self.transition = nn.Sequential(nn.Linear(96, 64), nn.GELU(), nn.Linear(64, 55))

    def forward(self, state, return_aux=False):
        parts = [self.skeleton(state["skeleton_prior"]),
                 self.direction(state["current_body_direction"])]
        if "semantic" in self.variant:
            item = self.bank_item(state["bank_set"])
            parts.extend((self.semantic(state["semantic"]), self.stats(state["global_stats"]),
                          self.bank_summary(torch.cat((item.mean(1), item.amax(1)), -1))))
        if "z" in self.variant:
            parts.append(self.z(state["z"]))
        if "logits" in self.variant:
            parts.append(self.logits(torch.cat((state["current_logits"],
                                                state["bank_mask"].float()), -1)))
        h = self.context(torch.cat(parts, -1))
        source = state["current_body_direction"][:, None, :]
        candidate = state["candidate"]
        candidate_dir, delta = candidate[..., :2], candidate[..., 2:4]
        relation = torch.stack(((candidate_dir * source).sum(-1),
                                candidate_dir[..., 0] * source[..., 1] - candidate_dir[..., 1] * source[..., 0],
                                (delta * source).sum(-1),
                                delta[..., 0] * source[..., 1] - delta[..., 1] * source[..., 0]), -1)
        g = self.candidate(torch.cat((geometry_features(candidate), relation), -1))
        hc = h[:, None].expand(-1, 4, -1)
        latent = self.context_to_action(h)[:, None, :] * g
        q = latent.sum(-1) / 48.0**.5 + .20 * self.residual(torch.cat((hc, g), -1)).squeeze(-1)
        if not return_aux:
            return q
        joint = torch.cat((hc, g), -1)
        pred = state["current_logits"][:, None, :] + self.transition(joint)
        return q, pred


def fixed0_validation(net, val, present, seen, hold, proxy_eps, proxy_bank):
    regular = runner.core.evaluate(net, val, present, seen, "fixed0")
    proxy = runner.core.evaluate(net, val, present, hold, "fixed0", proxy_eps, proxy_bank)
    score = float(.75 * proxy["macro_choice_gain"] + .25 * regular["macro_choice_gain"])
    return score, {"fixed0": regular["metrics"]}, {"fixed0": proxy["metrics"]}


base12.augmented_state = augmented_state
base12.SkeletonTransitionPolicy = FactorizedSkeletonPolicy
base13.OUT = OUT
base13.GENERATION = GENERATION
base13.POOL_MODE = "all50_five_way_slot_permutation_factorized_skeleton"
base13.PRIMARY_PROTOCOLS = ("fixed0",)
runner.OUT = OUT
runner.GENERATION = GENERATION
runner.POOL_MODE = "all50_five_way_slot_permutation_factorized_skeleton"
runner.augmented_state = augmented_state
runner.core.causal_state = augmented_state
runner.validation = fixed0_validation
runner.SemanticPolicy = FactorizedSkeletonPolicy
runner.train_variant = base13.train_variant
runner.VARIANTS = {
    "g18_factorized_skeleton": "low-rank temporal skeleton + geometry",
    "g18_factorized_skeleton_semantic": "factorized skeleton + semantic bank",
    "g18_factorized_skeleton_z": "factorized skeleton + video embedding",
    "g18_factorized_skeleton_semantic_zlogits": "factorized skeleton + semantic + video + logits",
}


if __name__ == "__main__":
    # TorchInductor produces non-finite gradients for this low-rank product
    # on the current CUDA/PyTorch build.  Eager mode is numerically stable and
    # still executes all tensor work on the GPU; keep this generation honest.
    torch.compile = lambda model, *args, **kwargs: model
    runner.main()
