"""Train three causal RL selectors with controlled feature ablations.

All variants use the same v9 relative-ranking objective and data protocol. Only
the policy input feature set changes.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from .causal_rank_v6 import causal_state, evaluate, task_banks
from .causal_rank_v7 import utility
from .feature_ablation_diagnostic import FeatureRanker
from .four_ideas_models import decision_loss
from .run_four_ideas import SEEDS, validation, verify_v6
from .train_rl_fast import GPUCache


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "work_dir/active_view_feature_variants_rl_v1"
VERSIONS = ["geometry", "geometry_pose", "geometry_pose_evidence"]
EPOCHS = 40
BATCH_SIZE = 8192
UPDATES_PER_EPOCH = 48


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(path)


def make_model(version):
    if version == "geometry":
        return FeatureRanker(False, False)
    if version == "geometry_pose":
        return FeatureRanker(True, False)
    if version == "geometry_pose_evidence":
        return FeatureRanker(True, True)
    raise ValueError(version)


def train_version(version, train, val, present, vp, cfg, pool, hold, seen, proxy_eps, proxy_bank, device):
    version_out = OUT / version
    version_out.mkdir(parents=True, exist_ok=True)
    if (version_out / "summary.json").exists():
        return json.loads((version_out / "summary.json").read_text())

    flat = (present & torch.isin(train.labels, torch.tensor(pool, device=device))[:, None]).flatten().nonzero().flatten()
    episodes, starts = (flat // 4).repeat(4), (flat % 4).repeat(4)
    labels = train.labels[episodes]
    freq = torch.bincount(labels, minlength=55).float()
    uniform = 1 / freq[labels].clamp_min(1)
    uniform /= uniform.sum()
    records = []

    for seed in SEEDS:
        run = version_out / f"seed_{seed}"
        run.mkdir(exist_ok=True)
        if (run / "complete.json").exists():
            records.append(json.loads((run / "complete.json").read_text()))
            continue
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        generator = torch.Generator(device=device).manual_seed(seed)
        net = make_model(version).to(device)
        compiled = torch.compile(net, mode="reduce-overhead")
        opt = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=.03)
        best, begin = -float("inf"), 1
        for epoch in range(begin, EPOCHS + 1):
            started = time.monotonic()
            banks = task_banks(train, episodes, pool, generator, .25)
            state = causal_state(train, episodes, starts, banks)
            u, selective = utility(train, episodes, starts, banks, state["mask"])
            priority = uniform * selective
            priority = priority / priority.sum() if priority.sum() > 0 else uniform
            probability = .5 * uniform + .5 * priority
            net.train()
            losses = []
            for _ in range(UPDATES_PER_EPOCH):
                ids = torch.multinomial(probability, BATCH_SIZE, True, generator=generator)
                s = {k: v[ids] for k, v in state.items()}
                loss = decision_loss("v9", compiled(s), u[ids], s["mask"])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 2.)
                opt.step()
                losses.append(loss.detach())
            score, regular, proxy = validation(net, val, vp, seen, hold, proxy_eps, proxy_bank)
            improved = score > best
            if improved:
                best = score
            saved = dict(
                online=net.state_dict(), optimizer=opt.state_dict(), generator=generator.get_state(),
                cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(), best=best,
                epoch=epoch, version=version, seed=seed,
            )
            torch.save(saved, run / "last.tmp.pt")
            (run / "last.tmp.pt").replace(run / "last.pt")
            if improved:
                torch.save(saved, run / "best.tmp.pt")
                (run / "best.tmp.pt").replace(run / "best.pt")
            row = dict(
                version=version, seed=seed, epoch=epoch,
                loss=torch.stack(losses).mean().item(),
                seconds=time.monotonic() - started, selective=int(selective.sum()),
                selection=score, best=best, validation=regular, proxy=proxy,
            )
            with (run / "train.log").open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps({k: row[k] for k in ("version", "seed", "epoch", "loss", "selection", "best")} ), flush=True)
        result = dict(version=version, seed=seed, score=best, path=str(run))
        dump(run / "complete.json", result)
        records.append(result)
        del compiled, net, opt

    chosen = max(records, key=lambda x: x["score"])
    selection = dict(version=version, chosen=chosen, runs=records, criterion="validation_only")
    dump(version_out / "selection.json", selection)
    net = make_model(version).to(device)
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
            dump(version_out / f"evaluation_{split}_{protocol}_best.json", report)
            reports[f"{split}_{protocol}"] = report["metrics"]
            print("EVALUATION", version, split, protocol, json.dumps(report["metrics"]), flush=True)
        del cache
    result = dict(selection=selection, metrics=reports)
    dump(version_out / "summary.json", result)
    print("FEATURE_ABLATION_VERSION_COMPLETE", version, flush=True)
    return result


def main():
    verify_v6()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load((ROOT / "logs/rl_candidate_rank_v6_new50_5/config.yaml").read_text())
    cfg["feature_ablation_rl"] = dict(
        versions=VERSIONS, epochs=EPOCHS, seeds=SEEDS, batch_size=BATCH_SIZE,
        updates_per_epoch=UPDATES_PER_EPOCH, objective="v9_relative_pairwise",
        movement_weight=0., feature_sets={
            "geometry": "candidate geometry only",
            "geometry_pose": "candidate geometry + current-view pose",
            "geometry_pose_evidence": "candidate geometry + current-view pose + logits summary",
        }, selection="validation_only",
    )
    (OUT / "config_resolved.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (OUT / "run_feature_ablation_rl.py").write_bytes(Path(__file__).read_bytes())
    (OUT / "feature_ablation_diagnostic.py").write_bytes((ROOT / "rl/feature_ablation_diagnostic.py").read_bytes())

    device = torch.device(cfg["runtime"]["device"])
    cache_root = Path(cfg["cache"]["root"])
    train, val = [GPUCache(cache_root / x, device) for x in ("dqn_train", "val")]
    present, vp = [torch.from_numpy(np.load(cache_root / x / "view_valid.npy")).to(device) for x in ("dqn_train", "val")]
    pool, hold = cfg["v6"]["policy_train_classes"], cfg["v6"]["policy_holdout_classes"]
    seen = cfg["evaluation"]["seen_class_ids"]
    proxy_eps = torch.isin(val.labels, torch.tensor(hold, device=device)).nonzero().flatten().repeat(4)
    proxy_bank = task_banks(val, proxy_eps, hold, torch.Generator(device=device).manual_seed(90117), 0)
    summaries = {}
    for version in VERSIONS:
        summaries[version] = train_version(version, train, val, present, vp, cfg, pool, hold, seen, proxy_eps, proxy_bank, device)
        torch.cuda.empty_cache()
    dump(OUT / "comparison.json", dict(base_v9="work_dir/active_view_four_ideas_new50_5/v9", versions=summaries, v6_preservation_verified=True))
    verify_v6()
    print("FEATURE_ABLATION_RL_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
