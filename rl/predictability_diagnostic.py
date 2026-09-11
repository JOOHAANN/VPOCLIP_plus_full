"""Causal cross-class predictability diagnostic for active-view selection.

This diagnostic deliberately trains only a small selector on the frozen RL
cache.  The target is the oracle next-view action computed from future-view
logits, while the input contains only the causal current-view state and
candidate geometry.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from .causal_rank_v6 import causal_state, class_mask, task_banks
from .train_rl_fast import GPUCache


SEEDS = (20260909, 20260910, 20260911)


class GeometryOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(7, 64), nn.LayerNorm(64), nn.GELU(),
            nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 1),
        )

    def forward(self, state):
        return self.net(state["candidate"]).squeeze(-1)


class CausalStateMLP(nn.Module):
    """Same causal feature family as v6, trained directly on oracle actions."""

    def __init__(self):
        super().__init__()
        self.z = nn.Sequential(nn.Linear(512, 128), nn.LayerNorm(128), nn.GELU())
        self.pose = nn.Sequential(nn.Linear(442, 64), nn.LayerNorm(64), nn.GELU())
        self.obj = nn.Sequential(nn.Linear(1800, 64), nn.LayerNorm(64), nn.GELU())
        self.context = nn.Sequential(
            nn.Linear(128 + 64 + 64 + 4 + 19, 256),
            nn.LayerNorm(256), nn.GELU(), nn.Dropout(.15),
            nn.Linear(256, 128), nn.GELU(),
        )
        self.geometry = nn.Sequential(nn.Linear(7, 64), nn.GELU(), nn.Linear(64, 128), nn.GELU())
        self.score = nn.Sequential(nn.Linear(384, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, state):
        h = self.context(torch.cat((
            self.z(state["z"]), self.pose(state["pose"]), self.obj(state["object"]),
            state["quality"], state["evidence"],
        ), -1))
        g = self.geometry(state["candidate"])
        h = h[:, None].expand(-1, 4, -1)
        return self.score(torch.cat((h, g, h * g), -1)).squeeze(-1)


def rows_for(cache, present, classes):
    allowed = torch.isin(cache.labels, torch.tensor(classes, device=cache.device))[:, None]
    flat = (present & allowed).flatten().nonzero().flatten()
    return flat // 4, flat % 4


def oracle_target(cache, episodes, starts, bank, valid):
    fused = (cache.logits[episodes, starts, None] + cache.logits[episodes]) * .5
    score = fused.masked_fill(~bank[:, None], -torch.inf)
    labels = cache.labels[episodes]
    true_score = score.gather(-1, labels[:, None, None].expand(-1, 4, 1)).squeeze(-1)
    target = true_score.masked_fill(~valid, -torch.inf).argmax(-1)
    return target


def state_batch(cache, episodes, starts, banks):
    state = causal_state(cache, episodes, starts, banks)
    target = oracle_target(cache, episodes, starts, banks, state["mask"])
    return state, target


def train_one(model_cls, cache, present, episodes, starts, banks, seed, device):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = model_cls().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=.02)
    generator = torch.Generator(device=device).manual_seed(seed)
    n = len(episodes)
    batch = 2048
    for _ in range(60):
        order = torch.randperm(n, device=device, generator=generator)
        model.train()
        for begin in range(0, n, batch):
            ids = order[begin:begin + batch]
            state, target = state_batch(cache, episodes[ids], starts[ids], banks[ids])
            valid = state["mask"]
            selectable = valid.sum(-1) > 1
            if not selectable.any():
                continue
            q = model(state).masked_fill(~valid, -1e4)
            loss = F.cross_entropy(q[selectable], target[selectable])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.)
            opt.step()
    return model


@torch.inference_mode()
def evaluate(model, cache, present, classes, protocol, seed, device, episodes=None):
    if episodes is None:
        episode_ids = torch.arange(cache.num_episodes, device=device)
    else:
        episode_ids = episodes
    n = len(episode_ids)
    rows = torch.arange(n, device=device)
    valid_np = present[episode_ids].cpu().numpy()
    rng = np.random.default_rng(seed)
    if protocol == "fixed0":
        starts_np = np.zeros(n, dtype=np.int64)
    else:
        starts_np = np.asarray([rng.choice(np.flatnonzero(v)) for v in valid_np], dtype=np.int64)
    starts = torch.as_tensor(starts_np, device=device)
    bank = class_mask(classes, n, device)
    valid = present[episode_ids].clone()
    valid[rows, starts] = False
    selected = torch.empty(n, dtype=torch.long, device=device)
    for begin in range(0, n, 512):
        sl = slice(begin, begin + 512)
        state = causal_state(cache, episode_ids[sl], starts[sl], bank[sl])
        q = model(state).masked_fill(~state["mask"], -torch.inf)
        selected[sl] = q.argmax(-1)
    target = oracle_target(cache, episode_ids, starts, bank, valid)
    choiceable = valid.sum(-1) > 1
    agree = (selected == target) & choiceable
    random_agree = (1. / valid.sum(-1).float().clamp_min(1))[choiceable]
    fused = (cache.logits[episode_ids, starts, None] + cache.logits[episode_ids]) * .5
    score = fused.masked_fill(~bank[:, None], -torch.inf)
    labels = cache.labels[episode_ids]
    selected_score = score[rows, selected]
    oracle_score = score[rows, target]
    selected_ok = selected_score.argmax(-1).eq(labels)
    oracle_ok = oracle_score.argmax(-1).eq(labels)
    candidate_correct = score.argmax(-1).eq(labels[:, None]) & valid
    random_top1 = (candidate_correct.float().sum(-1) / valid.sum(-1).float().clamp_min(1))[choiceable]
    return {
        "episodes": int(n),
        "choiceable": int(choiceable.sum()),
        "action_agreement": float(agree[choiceable].float().mean()),
        "random_action_agreement": float(random_agree.mean()),
        "selected_top1": float(selected_ok.float().mean()),
        "random_top1": float(random_top1.mean()),
        "oracle_top1": float(oracle_ok.float().mean()),
        "mean_valid_actions": float(valid.sum(-1)[choiceable].float().mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="logs/rl_candidate_rank_v6_new50_5/config.yaml")
    parser.add_argument("--output", default="work_dir/active_view_predictability_diagnostic_v1")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root / args.config).read_text())
    device = torch.device(cfg["runtime"]["device"])
    cache_root = Path(cfg["cache"]["root"])
    out = root / args.output
    out.mkdir(parents=True, exist_ok=True)

    train = GPUCache(cache_root / "dqn_train", device)
    val = GPUCache(cache_root / "val", device)
    test = GPUCache(cache_root / "test", device)
    train_present = torch.from_numpy(np.load(cache_root / "dqn_train" / "view_valid.npy")).to(device)
    val_present = torch.from_numpy(np.load(cache_root / "val" / "view_valid.npy")).to(device)
    test_present = torch.from_numpy(np.load(cache_root / "test" / "view_valid.npy")).to(device)

    pool = cfg["v6"]["policy_train_classes"]
    hold = cfg["v6"]["policy_holdout_classes"]
    unseen = cfg["evaluation"]["unseen_class_ids"]

    train_ep, train_st = rows_for(train, train_present, pool)
    generator = torch.Generator(device=device).manual_seed(90117)
    train_bank = task_banks(train, train_ep, pool, generator, .25)
    # Holdout and unseen banks match evaluation: the policy knows the candidate
    # class set, but never receives the true label.
    val_ep, val_st = rows_for(val, val_present, hold)
    test_ep, test_st = rows_for(test, test_present, unseen)
    train_eval_ep = torch.arange(train.num_episodes, device=device)
    train_eval_ep = train_eval_ep[torch.isin(train.labels, torch.tensor(pool, device=device))]

    all_results = {
        "protocol": "causal current-view state only; future logits used only for oracle target",
        "train_classes": pool,
        "holdout_classes": hold,
        "unseen_classes": unseen,
        "train_contexts": int(len(train_ep)),
        "holdout_contexts": int(len(val_ep)),
        "unseen_contexts": int(len(test_ep)),
        "models": {},
    }
    for name, model_cls in (("geometry_only", GeometryOnly), ("causal_state", CausalStateMLP)):
        seed_results = []
        for seed in SEEDS:
            model = train_one(model_cls, train, train_present, train_ep, train_st, train_bank, seed, device)
            result = {"seed": seed, "train": {}, "holdout": {}, "unseen": {}}
            for protocol in ("fixed0", "random"):
                result["train"][protocol] = evaluate(model, train, train_present, pool, protocol, seed, device, train_eval_ep)
                result["holdout"][protocol] = evaluate(model, val, val_present, hold, protocol, seed, device)
                result["unseen"][protocol] = evaluate(model, test, test_present, unseen, protocol, seed, device)
            seed_results.append(result)
            del model
            torch.cuda.empty_cache()
        all_results["models"][name] = seed_results

    (out / "summary.json").write_text(json.dumps(all_results, indent=2))
    print(json.dumps(all_results, indent=2), flush=True)


if __name__ == "__main__":
    main()
