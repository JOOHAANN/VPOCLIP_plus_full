"""Generation 06: causal semantic-summary policy for unseen transfer.

The policy sees only the current observation, body geometry, and the frozen
VPOCLIP semantic summary derived from that observation.  Candidate videos are
not read before the action.  It directly selects one legal candidate by
argmax; there is no random fallback and no gate.
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

from .. import causal_rank_v6 as core
from ..causal_rank_v7 import utility
from ..four_ideas_models import decision_loss as v9_loss
from ..run_four_ideas import SEEDS, validation, verify_v6
from ..train_rl_fast import GPUCache
from ..v9_improvement_models import listwise_loss, pairwise_loss


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "work_dir/active_view_iterative_test/generation_06"
GENERATION = "generation_06"
EPOCHS = 40
BATCH_SIZE = 8192
UPDATES_PER_EPOCH = 48
POOL_MODE = "all50"
VARIANTS = {
    "g06_semantic": "semantic prototype summary + geometry interaction",
    "g06_semantic_pose": "semantic summary + pose + geometry interaction",
    "g06_semantic_z": "semantic summary + video embedding + geometry interaction",
    "g06_semantic_evidence": "semantic summary + current evidence + geometry interaction",
}


_ORIGINAL_STATE = core.causal_state


def augmented_state(cache, episodes, starts, bank):
    """Add label-free current-view semantic features to the frozen causal state."""
    state = _ORIGINAL_STATE(cache, episodes, starts, bank)
    logits = cache.logits[episodes, starts]
    probs = torch.softmax(logits, -1)
    top_p, top_i = probs.topk(5, -1)
    semantic = (top_p[..., None] * cache.prototypes[top_i]).sum(-2)
    bank_probs = torch.softmax(logits.masked_fill(~bank, -torch.inf), -1)
    bank_semantic = bank_probs @ cache.prototypes
    entropy = -(probs * probs.clamp_min(1e-12).log()).sum(-1, keepdim=True)
    entropy = entropy / np.log(float(logits.shape[-1]))
    stats = torch.cat((top_p, entropy, (top_p[:, :1] - top_p[:, 1:2])), -1)
    state["semantic"] = semantic
    state["bank_semantic"] = bank_semantic
    state["global_stats"] = stats
    return state


# The existing evaluation function resolves causal_rank_v6.causal_state at call
# time.  Patching only this module's in-memory reference keeps all old source
# files/results unchanged while making generation_06 use the augmented state.
core.causal_state = augmented_state


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


def geometry_features(c):
    a, d = c[..., :2], c[..., 2:4]
    a2 = torch.stack((2 * a[..., 0] * a[..., 1], a[..., 1].square() - a[..., 0].square()), -1)
    d2 = torch.stack((2 * d[..., 0] * d[..., 1], d[..., 1].square() - d[..., 0].square()), -1)
    return torch.cat((c, a2, d2), -1)


def compact_pose(p):
    return torch.cat((p.mean(-1, keepdim=True), p.std(-1, keepdim=True),
                      p.abs().mean(-1, keepdim=True)), -1)


class SemanticPolicy(nn.Module):
    def __init__(self, variant):
        super().__init__()
        self.variant = variant
        self.semantic = nn.Sequential(nn.Linear(512, 64), nn.LayerNorm(64), nn.GELU())
        self.bank_semantic = nn.Sequential(nn.Linear(512, 64), nn.LayerNorm(64), nn.GELU())
        self.stats = nn.Sequential(nn.Linear(7, 16), nn.LayerNorm(16), nn.GELU())
        self.pose = nn.Sequential(nn.Linear(3, 16), nn.LayerNorm(16), nn.GELU())
        self.z = nn.Sequential(nn.Linear(512, 32), nn.LayerNorm(32), nn.GELU())
        self.evidence = nn.Sequential(nn.Linear(19, 24), nn.LayerNorm(24), nn.GELU())
        self.quality = nn.Sequential(nn.Linear(4, 12), nn.LayerNorm(12), nn.GELU())
        context_dim = 64 + 16
        if "banksemantic" in variant:
            context_dim += 64
        if variant.endswith("semantic_pose"):
            context_dim += 16
        if variant.endswith("semantic_z"):
            context_dim += 32 + 16
        if variant.endswith("semantic_evidence"):
            context_dim += 24 + 12
        self.context = nn.Sequential(nn.Linear(context_dim, 96), nn.LayerNorm(96), nn.GELU(),
                                     nn.Dropout(.10), nn.Linear(96, 48), nn.GELU())
        self.candidate = nn.Sequential(nn.Linear(11, 48), nn.LayerNorm(48), nn.GELU())
        self.head = nn.Sequential(nn.Linear(48 + 48 + 48 + 48, 80), nn.GELU(),
                                  nn.Dropout(.10), nn.Linear(80, 1))

    def forward(self, state):
        parts = [self.semantic(state["semantic"]), self.stats(state["global_stats"])]
        if "banksemantic" in self.variant:
            parts.append(self.bank_semantic(state["bank_semantic"]))
        if self.variant.endswith("semantic_pose"):
            parts.append(self.pose(compact_pose(state["pose"])))
        if self.variant.endswith("semantic_z"):
            parts.extend((self.z(state["z"]), self.pose(compact_pose(state["pose"]))))
        if self.variant.endswith("semantic_evidence"):
            parts.extend((self.evidence(state["evidence"]), self.quality(state["quality"])))
        h = self.context(torch.cat(parts, -1))[:, None].expand(-1, 4, -1)
        g = geometry_features(state["candidate"])
        g = self.candidate(g)
        return self.head(torch.cat((h, g, h * g, h - g), -1)).squeeze(-1)


def permute_candidates(state, utility_value, generator):
    b = state["candidate"].shape[0]
    order = torch.rand((b, 4), device=state["candidate"].device, generator=generator).argsort(-1)
    out = dict(state)
    out["candidate"] = state["candidate"].gather(1, order[..., None].expand(-1, -1, state["candidate"].shape[-1]))
    out["mask"] = state["mask"].gather(1, order)
    return out, utility_value.gather(1, order)


def loss_for(variant, q, u, valid):
    if variant.endswith("semantic_z"):
        return v9_loss("v9", q, u, valid)
    if variant.endswith("semantic_evidence"):
        count = valid.sum(-1).clamp_min(1)
        cq = q - (q * valid).sum(-1, keepdim=True) / count[:, None]
        cu = u - (u * valid).sum(-1, keepdim=True) / count[:, None]
        reg = (F.smooth_l1_loss(cq, cu, reduction="none") * valid).sum(-1) / count
        return reg.mean() + .20 * pairwise_loss(q, u, valid).mean()
    return (listwise_loss(q, u, valid) + .20 * pairwise_loss(q, u, valid)).mean()


def train_variant(variant, train, val, present, vp, cfg, pool, hold, seen, proxy_eps, proxy_bank, device):
    out = OUT / variant
    out.mkdir(parents=True, exist_ok=True)
    if (out / "summary.json").exists():
        return json.loads((out / "summary.json").read_text())
    eligible = (present.sum(-1) == 4) & torch.isin(train.labels, torch.tensor(pool, device=device))
    flat = (eligible[:, None] & present).flatten().nonzero().flatten()
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
        net = SemanticPolicy(variant).to(device)
        compiled = torch.compile(net, mode="reduce-overhead")
        opt = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=.04)
        best = -float("inf")
        for epoch in range(1, EPOCHS + 1):
            started = time.monotonic()
            banks = core.task_banks(train, episodes, pool, gen, 0.0)
            state = augmented_state(train, episodes, starts, banks)
            u, selective = utility(train, episodes, starts, banks, state["mask"])
            priority = uniform * selective
            priority = priority / priority.sum() if priority.sum() > 0 else uniform
            probability = .35 * uniform + .65 * priority
            net.train(); losses = []
            for _ in range(UPDATES_PER_EPOCH):
                ids = torch.multinomial(probability, BATCH_SIZE, True, generator=gen)
                s = {k: v[ids] for k, v in state.items()}
                uu = u[ids]
                s, uu = permute_candidates(s, uu, gen)
                loss = loss_for(variant, compiled(s), uu, s["mask"])
                opt.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 2.); opt.step()
                losses.append(loss.detach())
            score, regular, proxy = validation(net, val, vp, seen, hold, proxy_eps, proxy_bank)
            improved = score > best; best = max(best, score)
            saved = dict(online=net.state_dict(), optimizer=opt.state_dict(), generator=gen.get_state(),
                         cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(), best=best,
                         epoch=epoch, variant=variant, seed=seed)
            torch.save(saved, run / "last.pt")
            if improved: torch.save(saved, run / "best.pt")
            row = dict(version=variant, seed=seed, epoch=epoch,
                       loss=torch.stack(losses).mean().item(), seconds=time.monotonic() - started,
                       contexts=int(len(episodes)), four_view_only=True, train_pool_mode=POOL_MODE,
                       selective=int(selective.sum()), selection=score, best=best,
                       validation=regular, proxy=proxy)
            with (run / "train.log").open("a") as f: f.write(json.dumps(row) + "\n")
            print(json.dumps({k: row[k] for k in ("version", "seed", "epoch", "loss", "selection", "best")} ), flush=True)
        result = dict(version=variant, seed=seed, score=best, path=str(run))
        dump(run / "complete.json", result); records.append(result)
        del compiled, net, opt
    chosen = max(records, key=lambda x: x["score"])
    selection = dict(version=variant, chosen=chosen, runs=records, criterion="validation_only")
    dump(out / "selection.json", selection)
    net = SemanticPolicy(variant).to(device)
    net.load_state_dict(torch.load(Path(chosen["path"]) / "best.pt", map_location=device, weights_only=False)["online"])
    net.eval(); reports = {}
    for split, classes in (("seen_test", cfg["evaluation"]["seen_class_ids"]),
                           ("test", cfg["evaluation"]["unseen_class_ids"])):
        cache = GPUCache(Path(cfg["cache"]["root"]) / split, device)
        real = torch.from_numpy(np.load(Path(cfg["cache"]["root"]) / split / "view_valid.npy")).to(device)
        for protocol in ("fixed0", "random"):
            report = core.evaluate(net, cache, real, classes, protocol)
            report["selection"] = selection
            dump(out / f"evaluation_{split}_{protocol}_best.json", report)
            reports[f"{split}_{protocol}"] = report["metrics"]
            print("EVALUATION", variant, split, protocol, json.dumps(report["metrics"]), flush=True)
        del cache
    result = dict(selection=selection, metrics=reports,
                  train_protocol="all50 seen + four-view + causal semantic summary + direct argmax; no gate")
    dump(out / "summary.json", result)
    print("ITERATION_VARIANT_COMPLETE", variant, flush=True)
    return result


def main():
    verify_v6()
    torch.set_num_threads(4); torch.set_float32_matmul_precision("high")
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load((ROOT / "logs/rl_candidate_rank_v6_new50_5/config.yaml").read_text())
    cfg["iterative_test"] = dict(generation=GENERATION, variants=VARIANTS, epochs=EPOCHS,
                                  seeds=SEEDS, batch_size=BATCH_SIZE, updates_per_epoch=UPDATES_PER_EPOCH,
                                  train_pool_mode=POOL_MODE, four_view_only=True,
                                  input="current VPOCLIP semantic summary + causal evidence + geometry",
                                  movement_weight=0., gating=False, policy_action_only=True,
                                  selection="validation_only",
                                  unseen_success_rule="all 3 seeds positive over random in both protocols")
    (OUT / "config_resolved.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (OUT / "generation_06.py").write_bytes(Path(__file__).read_bytes())
    device = torch.device(cfg["runtime"]["device"]); root = Path(cfg["cache"]["root"])
    train, val = [GPUCache(root / x, device) for x in ("dqn_train", "val")]
    present, vp = [torch.from_numpy(np.load(root / x / "view_valid.npy")).to(device) for x in ("dqn_train", "val")]
    pool = cfg["evaluation"]["seen_class_ids"]
    hold, seen = cfg["v6"]["policy_holdout_classes"], cfg["evaluation"]["seen_class_ids"]
    proxy_eps = torch.isin(val.labels, torch.tensor(hold, device=device)).nonzero().flatten().repeat(4)
    proxy_bank = core.task_banks(val, proxy_eps, hold, torch.Generator(device=device).manual_seed(90117), 0)
    summaries = {}
    for variant in VARIANTS:
        summaries[variant] = train_variant(variant, train, val, present, vp, cfg, pool, hold, seen,
                                           proxy_eps, proxy_bank, device)
        torch.cuda.empty_cache()
    dump(OUT / "comparison.json", dict(variants=summaries,
                                        baseline_random={"fixed0": .5314009785652161, "random": .5386473536491394}))
    verify_v6(); print("ITERATIVE_GENERATION_COMPLETE", GENERATION, flush=True)


if __name__ == "__main__":
    main()
