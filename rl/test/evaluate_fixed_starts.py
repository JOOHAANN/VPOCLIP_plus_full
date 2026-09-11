"""Evaluate a direct active-view policy from fixed initial camera views.

This is separate from the historical fixed0/random evaluator.  It fixes the
initial view to 1, 2, or 3 and compares policy selection with a random choice
among the same remaining candidates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from ..causal_rank_v6 import class_mask, causal_state
from ..train_rl_fast import GPUCache
from .generation_18 import FactorizedSkeletonPolicy, augmented_state


ROOT = Path(__file__).resolve().parents[2]


@torch.inference_mode()
def evaluate_fixed_start(net, cache, present, classes, start_view, seed=20260909):
    all_episodes = torch.arange(cache.num_episodes, device=cache.device)
    # Validation contains some incomplete-view episodes; use only samples for
    # which the requested initial view actually exists.  The official unseen
    # test cache is four-view complete, so this does not drop test samples.
    episodes = all_episodes[present[:, start_view]]
    n = len(episodes)
    rows = torch.arange(n, device=cache.device)
    starts = torch.full((n,), start_view, dtype=torch.long, device=cache.device)
    real = present[episodes]
    if n == 0:
        raise ValueError(f"start view {start_view} has no valid episodes")
    valid = real.clone()
    valid[rows, starts] = False
    bank = class_mask(classes, n, cache.device)
    actions = torch.empty(n, dtype=torch.long, device=cache.device)
    net.eval()
    for b in range(0, n, 512):
        state = augmented_state(cache, episodes[b:b + 512], starts[b:b + 512], bank[b:b + 512])
        assert torch.equal(state["mask"], valid[b:b + 512])
        actions[b:b + 512] = net(state).masked_fill(~state["mask"], -torch.inf).argmax(-1)

    rng = np.random.default_rng(seed + start_view)
    random_actions = torch.tensor(
        [rng.choice(np.flatnonzero(v)) for v in valid.cpu().numpy()],
        dtype=torch.long, device=cache.device)
    fused = (cache.logits[episodes, starts, None, :] + cache.logits[episodes]) * .5
    scores = fused.masked_fill(~bank[:, None], -torch.inf)
    labels = cache.labels[episodes]
    correct = scores.argmax(-1).eq(labels[:, None])
    true = scores.gather(-1, labels[:, None, None].expand(-1, 4, 1)).squeeze(-1)
    oracle = true.masked_fill(~valid, -torch.inf).argmax(-1)
    choices = {"single": starts, "random": random_actions,
               "policy": actions, "oracle": oracle}
    metrics = {}
    for name, selected in choices.items():
        s = scores[rows, selected]
        ok = s.argmax(-1).eq(labels)
        p = s.softmax(-1)
        metrics[name] = {
            "top1": ok.float().mean().item(),
            "top5": s.topk(5, -1).indices.eq(labels[:, None]).any(-1).float().mean().item(),
            "mean_entropy": (-(p * p.clamp_min(1e-12).log()).sum(-1)
                             / bank.sum(-1).float().log()).mean().item(),
            "mean_movement_cost": cache.cost[episodes, starts, selected].mean().item(),
        }
    expected = (correct * valid).sum(-1) / valid.sum(-1)
    metrics["random_exact"] = {"top1": expected.mean().item()}
    return {
        "start_view": start_view,
        "episodes": n,
        "candidate_classes": classes,
        "metrics": metrics,
        "policy_action_histogram": torch.bincount(actions, minlength=4).cpu().tolist(),
        "random_action_histogram": torch.bincount(random_actions, minlength=4).cpu().tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generation", default="generation_18")
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / "logs/rl_candidate_rank_v6_new50_5/config.yaml").read_text())
    device = torch.device(cfg["runtime"]["device"])
    cache_root = Path(cfg["cache"]["root"])
    cache = GPUCache(cache_root / "test", device)
    present = torch.from_numpy(np.load(cache_root / "test/view_valid.npy")).to(device)
    classes = cfg["evaluation"]["unseen_class_ids"]
    variant = "g18_factorized_skeleton"
    variant_root = ROOT / "work_dir/active_view_iterative_test" / args.generation / variant
    result = {"generation": args.generation, "variant": variant,
              "protocol": "fixed initial view; random only chooses the second view",
              "baseline_reference_fixed0": 0.5314009785652161, "seeds": {}}
    for seed_dir in sorted(variant_root.glob("seed_*")):
        checkpoint = seed_dir / "best.pt"
        if not checkpoint.exists():
            continue
        net = FactorizedSkeletonPolicy(variant).to(device)
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        net.load_state_dict(saved["online"])
        result["seeds"][seed_dir.name] = {
            "checkpoint": str(checkpoint),
            "views": {str(v): evaluate_fixed_start(net, cache, present, classes, v)
                      for v in (1, 2, 3)},
        }
        del net
        torch.cuda.empty_cache()
    out = ROOT / "work_dir/active_view_iterative_test" / args.generation / "fixed_views_1_2_3.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"WROTE {out}")


if __name__ == "__main__":
    main()
