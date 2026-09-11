"""Generation 03: four-view matched, class-balanced causal policy training.

The test set contains four legal low-view candidates.  Earlier generations
trained on mixed 2/3/4-view episodes and used a context-only scorer whose
candidate interaction was weak.  This generation matches the episode shape,
balances classes, and scores each candidate through an explicit context--view
interaction.  It remains a direct policy: argmax over legal candidates, with
no random fallback or gate.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn

from ..causal_rank_v6 import causal_state, evaluate, task_banks
from ..causal_rank_v7 import relative_loss, utility
from ..feature_ablation_diagnostic import FeatureRanker
from ..four_ideas_models import decision_loss as v9_loss
from ..run_four_ideas import SEEDS, validation, verify_v6
from ..train_rl_fast import GPUCache
from ..v9_improvement_models import listwise_loss, pairwise_loss


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "work_dir/active_view_iterative_test/generation_03"
EPOCHS = 40
BATCH_SIZE = 8192
UPDATES_PER_EPOCH = 48
POOL_MODE = "policy40"
VARIANTS = {
    "g03_4view_interaction": "4-view + pose/evidence/quality interaction + listwise",
    "g03_4view_full_interaction": "4-view + compressed full causal state interaction",
    "g03_4view_pose_pairwise": "4-view + pose/quality interaction + v9 pairwise",
    "g03_4view_geometry_transition": "4-view + geometry/pose transition interaction",
}


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


class InteractionPolicy(nn.Module):
    """Candidate scorer with explicit context x candidate interaction."""

    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.pose = nn.Sequential(nn.Linear(442, 48), nn.LayerNorm(48), nn.GELU())
        self.evidence = nn.Sequential(nn.Linear(19, 40), nn.LayerNorm(40), nn.GELU())
        self.quality = nn.Sequential(nn.Linear(4, 16), nn.LayerNorm(16), nn.GELU())
        if mode == "full":
            self.z = nn.Sequential(nn.Linear(512, 40), nn.LayerNorm(40), nn.GELU())
            self.obj = nn.Sequential(nn.Linear(1800, 40), nn.LayerNorm(40), nn.GELU())
            context_dim = 48 + 40 + 16 + 40 + 40
        elif mode == "geometry_transition":
            context_dim = 48 + 16
        elif mode == "pose_pairwise":
            context_dim = 48 + 16
        else:
            context_dim = 48 + 40 + 16
        self.context = nn.Sequential(nn.Linear(context_dim, 128), nn.LayerNorm(128), nn.GELU(),
                                     nn.Dropout(.10), nn.Linear(128, 64), nn.GELU())
        self.candidate = nn.Sequential(nn.Linear(7, 64), nn.LayerNorm(64), nn.GELU(),
                                       nn.Linear(64, 64), nn.GELU())
        self.head = nn.Sequential(nn.Linear(64 + 64 + 64 + 64, 96), nn.GELU(),
                                  nn.Dropout(.10), nn.Linear(96, 1))

    def forward(self, state):
        parts = [self.pose(state["pose"]), self.quality(state["quality"])]
        if self.mode not in ("geometry_transition", "pose_pairwise"):
            parts.append(self.evidence(state["evidence"]))
        if self.mode == "full":
            parts.extend((self.z(state["z"]), self.obj(state["object"])))
        h = self.context(torch.cat(parts, -1))[:, None].expand(-1, 4, -1)
        g = self.candidate(state["candidate"])
        return self.head(torch.cat((h, g, h * g, h - g), -1)).squeeze(-1)


def make_model(variant):
    if variant.endswith("full_interaction"):
        return InteractionPolicy("full")
    if variant.endswith("pose_pairwise"):
        return InteractionPolicy("pose_pairwise")
    if variant.endswith("geometry_transition"):
        return InteractionPolicy("geometry_transition")
    return InteractionPolicy("pose_evidence")


def permute_candidates(state, utility_value, generator):
    b = state["candidate"].shape[0]
    order = torch.rand((b, 4), device=state["candidate"].device, generator=generator).argsort(-1)
    out = dict(state)
    out["candidate"] = state["candidate"].gather(1, order[..., None].expand(-1, -1, state["candidate"].shape[-1]))
    out["mask"] = state["mask"].gather(1, order)
    return out, utility_value.gather(1, order)


def loss_for(variant, q, u, valid):
    if variant.endswith("pose_pairwise"):
        return v9_loss("v9", q, u, valid)
    if variant.endswith("geometry_transition"):
        # Graded target plus a small pairwise term; all candidates remain in the loss.
        return (F.smooth_l1_loss(
            q - (q * valid).sum(-1, keepdim=True) / valid.sum(-1, keepdim=True).clamp_min(1),
            u - (u * valid).sum(-1, keepdim=True) / valid.sum(-1, keepdim=True).clamp_min(1),
            reduction="none") * valid).sum(-1).div(valid.sum(-1).clamp_min(1)).mean() \
            + .20 * pairwise_loss(q, u, valid).mean()
    # The two interaction variants use a soft set target; tied best views share mass.
    return (listwise_loss(q, u, valid) + .20 * pairwise_loss(q, u, valid)).mean()


def train_variant(variant, train, val, present, vp, cfg, pool, hold, seen, proxy_eps, proxy_bank, device):
    out = OUT / variant
    out.mkdir(parents=True, exist_ok=True)
    if (out / "summary.json").exists():
        return json.loads((out / "summary.json").read_text())

    # Match the test topology exactly: only episodes with four legal low views.
    four = present.sum(-1) == 4
    eligible_episodes = four & torch.isin(train.labels, torch.tensor(pool, device=device))
    flat = (eligible_episodes[:, None] & present).flatten().nonzero().flatten()
    episodes, starts = (flat // 4).repeat(4), (flat % 4).repeat(4)
    labels = train.labels[episodes]
    freq = torch.bincount(labels, minlength=55).float()
    uniform = 1 / freq[labels].clamp_min(1)
    uniform /= uniform.sum()
    records = []
    for seed in SEEDS:
        run = out / f"seed_{seed}"
        run.mkdir(exist_ok=True)
        if (run / "complete.json").exists():
            records.append(json.loads((run / "complete.json").read_text()))
            continue
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        gen = torch.Generator(device=device).manual_seed(seed)
        net = make_model(variant).to(device)
        compiled = torch.compile(net, mode="reduce-overhead")
        opt = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=.04)
        best, last_selective = -float("inf"), 0
        for epoch in range(1, EPOCHS + 1):
            started = time.monotonic()
            # Fresh five-way class tasks every epoch; no test classes enter the bank.
            banks = task_banks(train, episodes, pool, gen, 0.0)
            state = causal_state(train, episodes, starts, banks)
            u, selective = utility(train, episodes, starts, banks, state["mask"])
            priority = uniform * selective
            priority = priority / priority.sum() if priority.sum() > 0 else uniform
            probability = .35 * uniform + .65 * priority
            net.train()
            losses = []
            for _ in range(UPDATES_PER_EPOCH):
                ids = torch.multinomial(probability, BATCH_SIZE, True, generator=gen)
                s = {k: v[ids] for k, v in state.items()}
                uu = u[ids]
                s, uu = permute_candidates(s, uu, gen)
                loss = loss_for(variant, compiled(s), uu, s["mask"])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 2.)
                opt.step()
                losses.append(loss.detach())
            score, regular, proxy = validation(net, val, vp, seen, hold, proxy_eps, proxy_bank)
            improved = score > best
            best = max(best, score)
            last_selective = int(selective.sum())
            saved = dict(online=net.state_dict(), optimizer=opt.state_dict(), generator=gen.get_state(),
                         cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(), best=best,
                         epoch=epoch, variant=variant, seed=seed)
            torch.save(saved, run / "last.pt")
            if improved:
                torch.save(saved, run / "best.pt")
            row = dict(version=variant, seed=seed, epoch=epoch,
                       loss=torch.stack(losses).mean().item(), seconds=time.monotonic() - started,
                       contexts=int(len(episodes)), four_view_contexts=int(len(episodes)),
                       selective=last_selective, selection=score, best=best,
                       validation=regular, proxy=proxy)
            with (run / "train.log").open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps({k: row[k] for k in ("version", "seed", "epoch", "loss", "selection", "best")} ), flush=True)
        result = dict(version=variant, seed=seed, score=best, path=str(run))
        dump(run / "complete.json", result)
        records.append(result)
        del compiled, net, opt

    chosen = max(records, key=lambda x: x["score"])
    selection = dict(version=variant, chosen=chosen, runs=records, criterion="validation_only")
    dump(out / "selection.json", selection)
    net = make_model(variant).to(device)
    saved = torch.load(Path(chosen["path"]) / "best.pt", map_location=device, weights_only=False)
    net.load_state_dict(saved["online"])
    net.eval()
    reports = {}
    for split, classes in (("seen_test", cfg["evaluation"]["seen_class_ids"]),
                           ("test", cfg["evaluation"]["unseen_class_ids"])):
        cache = GPUCache(Path(cfg["cache"]["root"]) / split, device)
        real = torch.from_numpy(np.load(Path(cfg["cache"]["root"]) / split / "view_valid.npy")).to(device)
        for protocol in ("fixed0", "random"):
            report = evaluate(net, cache, real, classes, protocol)
            report["selection"] = selection
            report["coverage"] = json.loads((Path(cfg["cache"]["root"]) / split / "coverage.json").read_text())
            dump(out / f"evaluation_{split}_{protocol}_best.json", report)
            reports[f"{split}_{protocol}"] = report["metrics"]
            print("EVALUATION", variant, split, protocol, json.dumps(report["metrics"]), flush=True)
        del cache
    result = dict(selection=selection, metrics=reports,
                  train_protocol="four-view episodes only; class-balanced; pure direct policy; no gate")
    dump(out / "summary.json", result)
    print("ITERATION_VARIANT_COMPLETE", variant, flush=True)
    return result


def main():
    verify_v6()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load((ROOT / "logs/rl_candidate_rank_v6_new50_5/config.yaml").read_text())
    cfg["iterative_test"] = dict(generation="generation_03", variants=VARIANTS,
                                  epochs=EPOCHS, seeds=SEEDS, batch_size=BATCH_SIZE,
                                  updates_per_epoch=UPDATES_PER_EPOCH, input="causal pose/evidence/quality + candidate interaction",
                                  movement_weight=0., four_view_only=True, class_balanced=True,
                                  gating=False, policy_action_only=True,
                                  selection="validation_only",
                                  unseen_success_rule="all 3 seeds positive over random in both protocols")
    (OUT / "config_resolved.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (OUT / "generation_03.py").write_bytes(Path(__file__).read_bytes())
    device = torch.device(cfg["runtime"]["device"])
    cache_root = Path(cfg["cache"]["root"])
    train, val = [GPUCache(cache_root / x, device) for x in ("dqn_train", "val")]
    present, vp = [torch.from_numpy(np.load(cache_root / x / "view_valid.npy")).to(device) for x in ("dqn_train", "val")]
    pool = cfg["v6"]["policy_train_classes"]
    hold = cfg["v6"]["policy_holdout_classes"]
    seen = cfg["evaluation"]["seen_class_ids"]
    proxy_eps = torch.isin(val.labels, torch.tensor(hold, device=device)).nonzero().flatten().repeat(4)
    proxy_bank = task_banks(val, proxy_eps, hold, torch.Generator(device=device).manual_seed(90117), 0)
    summaries = {}
    for variant in VARIANTS:
        summaries[variant] = train_variant(variant, train, val, present, vp, cfg, pool, hold, seen,
                                           proxy_eps, proxy_bank, device)
        torch.cuda.empty_cache()
    dump(OUT / "comparison.json", dict(variants=summaries,
                                        baseline_random={"fixed0": .5314009785652161, "random": .5386473536491394}))
    verify_v6()
    print("ITERATIVE_GENERATION_COMPLETE generation_03", flush=True)


if __name__ == "__main__":
    main()
