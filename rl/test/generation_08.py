"""Generation 08: train on candidate gain over the current view."""

from . import generation_06 as runner


def advantage_utility(cache, episodes, starts, banks, valid):
    """Label-free at inference: train targets are relative fused-margin gains."""
    current = cache.logits[episodes, starts]
    future = cache.logits[episodes]
    fused = (current[:, None, :] + future) * .5
    label = cache.labels[episodes]
    current_true = current.gather(-1, label[:, None]).squeeze(-1)
    current_rivals = current.masked_fill(~banks, -torch.inf).clone()
    current_rivals.scatter_(1, label[:, None], -torch.inf)
    current_margin = current_true - current_rivals.amax(-1)
    fused_true = fused.gather(-1, label[:, None, None].expand(-1, 4, 1)).squeeze(-1)
    rivals = banks[:, None, :].expand(-1, 4, -1).clone()
    rivals.scatter_(2, label[:, None, None].expand(-1, 4, 1), False)
    fused_rival = fused.masked_fill(~rivals, -torch.inf).amax(-1)
    gain = fused_true - fused_rival - current_margin[:, None]
    u = torch.tanh(gain / 10.).masked_fill(~valid, 0.)
    selective = valid.any(-1) & ((u.max(-1).values - u.min(-1).values) > .02)
    return u, selective


import torch


runner.GENERATION = "generation_08"
runner.OUT = runner.ROOT / "work_dir/active_view_iterative_test/generation_08"
runner.POOL_MODE = "all50"
runner.utility = advantage_utility
runner.VARIANTS = {
    "g08_semantic": "relative fused-margin gain + semantic summary",
    "g08_semantic_pose": "relative fused-margin gain + semantic + pose",
    "g08_semantic_z": "relative fused-margin gain + semantic + z",
    "g08_semantic_evidence": "relative fused-margin gain + semantic + evidence",
}


if __name__ == "__main__":
    runner.main()
