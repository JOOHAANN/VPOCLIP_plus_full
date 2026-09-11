"""Feature ablation for causal cross-class active-view predictability."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

from .causal_rank_v6 import task_banks
from .predictability_diagnostic import (
    SEEDS,
    evaluate,
    rows_for,
    train_one,
)
from .train_rl_fast import GPUCache


class FeatureRanker(nn.Module):
    """Shared candidate scorer using an explicitly selected causal feature set."""

    def __init__(self, use_pose: bool, use_evidence: bool, use_full: bool = False):
        super().__init__()
        self.use_pose = use_pose
        self.use_evidence = use_evidence
        self.use_full = use_full
        if use_pose or use_full:
            self.pose = nn.Sequential(nn.Linear(442, 64), nn.LayerNorm(64), nn.GELU())
        if use_evidence or use_full:
            self.evidence = nn.Sequential(nn.Linear(19, 32), nn.LayerNorm(32), nn.GELU())
        if use_full:
            self.z = nn.Sequential(nn.Linear(512, 64), nn.LayerNorm(64), nn.GELU())
            self.obj = nn.Sequential(nn.Linear(1800, 64), nn.LayerNorm(64), nn.GELU())
            self.quality = nn.Sequential(nn.Linear(4, 16), nn.LayerNorm(16), nn.GELU())
        context_dim = 0
        if use_pose or use_full:
            context_dim += 64
        if use_evidence or use_full:
            context_dim += 32
        if use_full:
            context_dim += 64 + 64 + 16
        if context_dim == 0:
            context_dim = 1
            self.constant = nn.Parameter(torch.zeros(1))
        self.geometry = nn.Sequential(nn.Linear(7, 64), nn.LayerNorm(64), nn.GELU())
        self.score = nn.Sequential(
            nn.Linear(context_dim + 64 + context_dim * 0, 128),
            nn.GELU(), nn.Linear(128, 1),
        )

    def forward(self, state):
        parts = []
        if self.use_pose or self.use_full:
            parts.append(self.pose(state["pose"]))
        if self.use_evidence or self.use_full:
            parts.append(self.evidence(state["evidence"]))
        if self.use_full:
            parts.extend((self.z(state["z"]), self.obj(state["object"]), self.quality(state["quality"])))
        if parts:
            h = torch.cat(parts, -1)
        else:
            h = self.constant.expand(state["candidate"].shape[0], -1)
        g = self.geometry(state["candidate"])
        h = h[:, None].expand(-1, 4, -1)
        return self.score(torch.cat((h, g), -1)).squeeze(-1)


def make_model(name):
    if name == "geometry":
        return FeatureRanker(False, False)
    if name == "geometry_pose":
        return FeatureRanker(True, False)
    if name == "geometry_evidence":
        return FeatureRanker(False, True)
    if name == "geometry_pose_evidence":
        return FeatureRanker(True, True)
    if name == "full_causal_state":
        return FeatureRanker(False, False, True)
    raise ValueError(name)


def main():
    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root / "logs/rl_candidate_rank_v6_new50_5/config.yaml").read_text())
    device = torch.device(cfg["runtime"]["device"])
    cache_root = Path(cfg["cache"]["root"])
    out = root / "work_dir/active_view_feature_ablation_v1"
    out.mkdir(parents=True, exist_ok=True)

    train = GPUCache(cache_root / "dqn_train", device)
    val = GPUCache(cache_root / "val", device)
    test = GPUCache(cache_root / "test", device)
    train_present = torch.from_numpy(np.load(cache_root / "dqn_train/view_valid.npy")).to(device)
    val_present = torch.from_numpy(np.load(cache_root / "val/view_valid.npy")).to(device)
    test_present = torch.from_numpy(np.load(cache_root / "test/view_valid.npy")).to(device)

    pool = cfg["v6"]["policy_train_classes"]
    hold = cfg["v6"]["policy_holdout_classes"]
    unseen = cfg["evaluation"]["unseen_class_ids"]
    train_ep, train_st = rows_for(train, train_present, pool)
    train_bank = task_banks(train, train_ep, pool, torch.Generator(device=device).manual_seed(90117), .25)
    val_ep, val_st = rows_for(val, val_present, hold)
    test_ep, test_st = rows_for(test, test_present, unseen)
    train_eval_ep = torch.arange(train.num_episodes, device=device)
    train_eval_ep = train_eval_ep[torch.isin(train.labels, torch.tensor(pool, device=device))]

    names = ["geometry", "geometry_pose", "geometry_evidence", "geometry_pose_evidence", "full_causal_state"]
    results = {
        "protocol": "causal current-view state only; future logits used only for oracle target",
        "feature_definitions": {
            "geometry": "candidate geometry and reachability only",
            "geometry_pose": "geometry + current-view pose",
            "geometry_evidence": "geometry + current-view logits summary",
            "geometry_pose_evidence": "geometry + pose + current-view logits summary",
            "full_causal_state": "geometry + pose + evidence + current z/object/quality reference",
        },
        "contexts": {"train": int(len(train_ep)), "holdout": int(len(val_ep)), "unseen": int(len(test_ep))},
        "models": {},
    }
    for name in names:
        print("ABLATION_START", name, flush=True)
        runs = []
        for seed in SEEDS:
            model = train_one(
                lambda name=name: make_model(name),
                train, train_present, train_ep, train_st, train_bank, seed, device,
            )
            row = {"seed": seed, "train": {}, "holdout": {}, "unseen": {}}
            for protocol in ("fixed0", "random"):
                row["train"][protocol] = evaluate(model, train, train_present, pool, protocol, seed, device, train_eval_ep)
                row["holdout"][protocol] = evaluate(model, val, val_present, hold, protocol, seed, device)
                row["unseen"][protocol] = evaluate(model, test, test_present, unseen, protocol, seed, device)
            runs.append(row)
            del model
            torch.cuda.empty_cache()
            print("ABLATION_SEED_COMPLETE", name, seed, flush=True)
        results["models"][name] = runs
    (out / "summary.json").write_text(json.dumps(results, indent=2))
    print("FEATURE_ABLATION_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
