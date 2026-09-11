"""Generation-based search for causal active-view policies.

This file is intentionally isolated under rl/test. Existing v1-v9 sources and
results are never imported for mutation or overwritten; they are only used as
the frozen data/protocol reference.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from ..causal_rank_v6 import causal_state, evaluate, task_banks
from ..causal_rank_v7 import relative_loss, utility
from ..feature_ablation_diagnostic import FeatureRanker
from ..four_ideas_models import decision_loss as v9_loss
from ..run_four_ideas import SEEDS, validation, verify_v6
from ..train_rl_fast import GPUCache
from ..v9_improvement_models import listwise_loss, pairwise_loss


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "work_dir/active_view_iterative_test"
GENERATION = "generation_01"
EPOCHS = 40
BATCH_SIZE = 8192
UPDATES_PER_EPOCH = 48
VARIANTS = {
    "g01_pairwise_aug": "v9 pairwise + candidate-slot permutation",
    "g01_listwise_aug": "listwise + pairwise + candidate-slot permutation",
    "g01_regression_aug": "relative regression + ranking + candidate-slot permutation",
}


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


def permute_candidates(state, utility_value, generator):
    """Randomize slot order while preserving candidate, mask, and target alignment."""
    b = state["candidate"].shape[0]
    order = torch.rand((b, 4), device=state["candidate"].device, generator=generator).argsort(-1)
    out = dict(state)
    out["candidate"] = state["candidate"].gather(1, order[..., None].expand(-1, -1, state["candidate"].shape[-1]))
    out["mask"] = state["mask"].gather(1, order)
    aligned_u = utility_value.gather(1, order)
    return out, aligned_u


def loss_for(variant, q, u, valid):
    if variant == "g01_pairwise_aug":
        return v9_loss("v9", q, u, valid)
    if variant == "g01_listwise_aug":
        return (listwise_loss(q, u, valid) + .20 * pairwise_loss(q, u, valid)).mean()
    if variant == "g01_regression_aug":
        return relative_loss(q, u, valid)
    raise ValueError(variant)


def train_variant(variant, train, val, present, vp, cfg, pool, hold, seen, proxy_eps, proxy_bank, device):
    out = OUT / GENERATION / variant
    out.mkdir(parents=True, exist_ok=True)
    if (out / "summary.json").exists():
        return json.loads((out / "summary.json").read_text())

    flat = (present & torch.isin(train.labels, torch.tensor(pool, device=device))[:, None]).flatten().nonzero().flatten()
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
        net = FeatureRanker(True, False).to(device)
        compiled = torch.compile(net, mode="reduce-overhead")
        opt = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=.03)
        best, begin = -float("inf"), 1
        for epoch in range(begin, EPOCHS + 1):
            started = time.monotonic()
            banks = task_banks(train, episodes, pool, gen, .25)
            state = causal_state(train, episodes, starts, banks)
            u, selective = utility(train, episodes, starts, banks, state["mask"])
            priority = uniform * selective
            priority = priority / priority.sum() if priority.sum() > 0 else uniform
            probability = .5 * uniform + .5 * priority
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
            if improved:
                best = score
            saved = dict(online=net.state_dict(), optimizer=opt.state_dict(), generator=gen.get_state(),
                         cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(), best=best,
                         epoch=epoch, variant=variant, seed=seed)
            torch.save(saved, run / "last.tmp.pt")
            (run / "last.tmp.pt").replace(run / "last.pt")
            if improved:
                torch.save(saved, run / "best.tmp.pt")
                (run / "best.tmp.pt").replace(run / "best.pt")
            row = dict(version=variant, seed=seed, epoch=epoch,
                       loss=torch.stack(losses).mean().item(), seconds=time.monotonic() - started,
                       selective=int(selective.sum()), selection=score, best=best,
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
    net = FeatureRanker(True, False).to(device)
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
    result = dict(selection=selection, metrics=reports)
    dump(out / "summary.json", result)
    print("ITERATION_VARIANT_COMPLETE", variant, flush=True)
    return result


def main():
    verify_v6()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load((ROOT / "logs/rl_candidate_rank_v6_new50_5/config.yaml").read_text())
    cfg["iterative_test"] = dict(generation=GENERATION, variants=VARIANTS, epochs=EPOCHS,
                                 seeds=SEEDS, batch_size=BATCH_SIZE,
                                 updates_per_epoch=UPDATES_PER_EPOCH,
                                 input="geometry + current-view pose", movement_weight=0.,
                                 objective="v9 pairwise/listwise/relative regression",
                                 slot_permutation=True, selection="validation_only",
                                 unseen_success_rule="all 3 seeds positive over random in both protocols")
    out = OUT / GENERATION
    out.mkdir(parents=True, exist_ok=True)
    (out / "config_resolved.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (out / "iterative_runner.py").write_bytes(Path(__file__).read_bytes())
    device = torch.device(cfg["runtime"]["device"])
    cache_root = Path(cfg["cache"]["root"])
    train, val = [GPUCache(cache_root / x, device) for x in ("dqn_train", "val")]
    present, vp = [torch.from_numpy(np.load(cache_root / x / "view_valid.npy")).to(device) for x in ("dqn_train", "val")]
    pool, hold = cfg["v6"]["policy_train_classes"], cfg["v6"]["policy_holdout_classes"]
    seen = cfg["evaluation"]["seen_class_ids"]
    proxy_eps = torch.isin(val.labels, torch.tensor(hold, device=device)).nonzero().flatten().repeat(4)
    proxy_bank = task_banks(val, proxy_eps, hold, torch.Generator(device=device).manual_seed(90117), 0)
    summaries = {}
    for variant in VARIANTS:
        summaries[variant] = train_variant(variant, train, val, present, vp, cfg, pool, hold, seen, proxy_eps, proxy_bank, device)
        torch.cuda.empty_cache()
    dump(out / "comparison.json", dict(variants=summaries, baseline_random={"fixed0": .5314009785652161, "random": .5386473536491394}))
    verify_v6()
    print("ITERATIVE_GENERATION_COMPLETE", GENERATION, flush=True)


if __name__ == "__main__":
    main()
