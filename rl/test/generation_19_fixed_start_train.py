"""Generation 19: train one skeleton policy per fixed initial view.

Unlike a post-hoc protocol change, each model is trained only with its own
initial view (1, 2, or 3).  The policy remains the gen18 direct argmax policy;
there is no random fallback or RL/random gate.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from . import generation_18 as base
from .evaluate_fixed_starts import evaluate_fixed_start


runner = base.runner
base13 = base.base13
base12 = base.base12
ROOT = runner.ROOT
OUT = ROOT / "work_dir/active_view_iterative_test/generation_19_fixed_start_train"
EPOCHS = 40
BATCH_SIZE = 8192
UPDATES_PER_EPOCH = 48
START_VIEWS = (1, 2, 3)
VARIANT = "g19_factorized_skeleton"


def fixed_start_validation(net, val, present, seen, hold, proxy_eps, proxy_bank, start_view):
    regular = evaluate_fixed_start(net, val, present, seen, start_view)
    proxy = evaluate_fixed_start(net, val, present, hold, start_view)
    regular_gain = regular["metrics"]["policy"]["top1"] - regular["metrics"]["random_exact"]["top1"]
    proxy_gain = proxy["metrics"]["policy"]["top1"] - proxy["metrics"]["random_exact"]["top1"]
    score = float(.75 * proxy_gain + .25 * regular_gain)
    return score, {"fixed_start": regular["metrics"]}, {"fixed_start": proxy["metrics"]}


def train_start(start_view, train, val, present, vp, cfg, pool, hold, seen,
                proxy_eps, proxy_bank, device):
    out = OUT / f"start_view_{start_view}"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "summary.json").exists():
        return json.loads((out / "summary.json").read_text())

    eligible = ((present.sum(-1) == 4)
                & torch.isin(train.labels, torch.tensor(pool, device=device)))
    episodes = eligible.nonzero(as_tuple=False).flatten().repeat(4)
    starts = torch.full_like(episodes, start_view)
    labels = train.labels[episodes]
    freq = torch.bincount(labels, minlength=55).float()
    uniform = 1 / freq[labels].clamp_min(1)
    uniform /= uniform.sum()
    records = []

    for seed in runner.SEEDS:
        run = out / f"seed_{seed}"
        run.mkdir(exist_ok=True)
        if (run / "complete.json").exists():
            records.append(json.loads((run / "complete.json").read_text()))
            continue
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        gen = torch.Generator(device=device).manual_seed(seed)
        net = base.FactorizedSkeletonPolicy(VARIANT).to(device)
        # Eager mode is intentional: gen18's low-rank product triggers a
        # TorchInductor non-finite-gradient bug on this CUDA/PyTorch build.
        opt = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=.04)
        best = -float("inf")
        for epoch in range(1, EPOCHS + 1):
            started = time.monotonic()
            banks = runner.core.task_banks(train, episodes, pool, gen, 0.0)
            state = base.augmented_state(train, episodes, starts, banks)
            state["future_target"] = train.logits[episodes].masked_fill(
                ~banks[:, None, :], 0.) / 10.
            utility, selective = runner.utility(train, episodes, starts, banks, state["mask"])
            priority = uniform * selective
            priority = priority / priority.sum() if priority.sum() > 0 else uniform
            probability = .35 * uniform + .65 * priority
            net.train()
            losses = []
            for _ in range(UPDATES_PER_EPOCH):
                ids = torch.multinomial(probability, BATCH_SIZE, True, generator=gen)
                sample = {k: v[ids] for k, v in state.items()}
                uu = utility[ids]
                sample, uu = base13.permute_action_slots(sample, uu, gen)
                q, pred = net(sample, return_aux=True)
                loss = base13.base.base.transition_loss(
                    q, pred, sample["future_target"], uu, sample["mask"], sample["bank_mask"])
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss at start={start_view}, seed={seed}, epoch={epoch}")
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 2.)
                opt.step()
                losses.append(loss.detach())
            score, regular, proxy = fixed_start_validation(
                net, val, vp, seen, hold, proxy_eps, proxy_bank, start_view)
            improved = score > best
            best = max(best, score)
            saved = dict(online=net.state_dict(), optimizer=opt.state_dict(),
                         generator=gen.get_state(), cpu_rng=torch.get_rng_state(),
                         cuda_rng=torch.cuda.get_rng_state(), best=best, epoch=epoch,
                         variant=VARIANT, seed=seed, start_view=start_view)
            torch.save(saved, run / "last.pt")
            if improved:
                torch.save(saved, run / "best.pt")
            row = dict(version=VARIANT, seed=seed, epoch=epoch, start_view=start_view,
                       loss=torch.stack(losses).mean().item(),
                       seconds=time.monotonic() - started, contexts=int(len(episodes)),
                       unique_contexts=int(eligible.sum()), fixed_start_only=True,
                       candidate_slot_permutation=True, selection=score, best=best,
                       validation=regular, proxy=proxy)
            with (run / "train.log").open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps({k: row[k] for k in
                              ("version", "seed", "epoch", "start_view", "loss", "selection", "best")} ),
                  flush=True)
        result = dict(version=VARIANT, seed=seed, start_view=start_view,
                      score=best, path=str(run))
        (run / "complete.json").write_text(json.dumps(result, indent=2))
        records.append(result)
        del net, opt

    chosen = max(records, key=lambda x: x["score"])
    selection = dict(version=VARIANT, start_view=start_view, chosen=chosen,
                     runs=records, criterion="fixed-start validation only")
    (out / "selection.json").write_text(json.dumps(selection, indent=2))
    net = base.FactorizedSkeletonPolicy(VARIANT).to(device)
    saved = torch.load(Path(chosen["path"]) / "best.pt", map_location=device,
                       weights_only=False)
    net.load_state_dict(saved["online"])
    net.eval()
    reports = {}
    for split, classes in (("seen_test", cfg["evaluation"]["seen_class_ids"]),
                           ("test", cfg["evaluation"]["unseen_class_ids"])):
        cache = runner.GPUCache(Path(cfg["cache"]["root"]) / split, device)
        real = torch.from_numpy(np.load(Path(cfg["cache"]["root"]) /
                                         split / "view_valid.npy")).to(device)
        report = evaluate_fixed_start(net, cache, real, classes, start_view)
        report["selection"] = selection
        path = out / f"evaluation_{split}_fixed{start_view}_best.json"
        path.write_text(json.dumps(report, indent=2))
        reports[f"{split}_fixed{start_view}"] = report["metrics"]
        print("EVALUATION", VARIANT, "start", start_view, split,
              json.dumps(report["metrics"]), flush=True)
        del cache
    result = dict(selection=selection, metrics=reports, start_view=start_view,
                  train_protocol="fixed start only + five-way + slot permutation + skeleton primary + direct argmax; no gate")
    (out / "summary.json").write_text(json.dumps(result, indent=2))
    print("ITERATION_START_COMPLETE", start_view, flush=True)
    return result


def main():
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    cfg = yaml.safe_load((ROOT / "logs/rl_candidate_rank_v6_new50_5/config.yaml").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    cfg["iterative_test"] = dict(generation=GENERATION if "GENERATION" in globals() else
                                  "generation_19_fixed_start_train",
                                  variant=VARIANT, start_views=START_VIEWS,
                                  epochs=EPOCHS, seeds=runner.SEEDS,
                                  batch_size=BATCH_SIZE, updates_per_epoch=UPDATES_PER_EPOCH,
                                  fixed_start_training=True, movement_weight=0., gating=False,
                                  selection="fixed-start validation only")
    (OUT / "config_resolved.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    device = torch.device(cfg["runtime"]["device"])
    root = Path(cfg["cache"]["root"])
    train, val = [runner.GPUCache(root / x, device) for x in ("dqn_train", "val")]
    present, vp = [torch.from_numpy(np.load(root / x / "view_valid.npy")).to(device)
                   for x in ("dqn_train", "val")]
    pool = cfg["evaluation"]["seen_class_ids"]
    hold = cfg["v6"]["policy_holdout_classes"]
    seen = cfg["evaluation"]["seen_class_ids"]
    proxy_eps = torch.isin(val.labels, torch.tensor(hold, device=device)).nonzero().flatten()
    proxy_bank = runner.core.task_banks(
        val, proxy_eps, hold, torch.Generator(device=device).manual_seed(90117), 0)
    summaries = {}
    for start_view in START_VIEWS:
        summaries[str(start_view)] = train_start(
            start_view, train, val, present, vp, cfg, pool, hold, seen,
            proxy_eps, proxy_bank, device)
        torch.cuda.empty_cache()
    (OUT / "comparison.json").write_text(json.dumps({
        "starts": summaries, "protocol": "each model trained and tested with its own fixed initial view",
        "baseline_reference_fixed0": 0.5314009785652161}, indent=2))
    print("GENERATION_19_FIXED_START_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
