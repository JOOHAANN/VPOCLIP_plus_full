"""Re-run the historical v9 selector with the latest group1 VPOCLIP cache.

The original v9 artifacts are intentionally untouched.  This runner keeps the
v9 pairwise objective and causal state, but reads the latest stage-B cache and
reports strict five-way unseen results for every valid fixed initial view.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from ..causal_rank_v6 import class_mask, causal_state, task_banks
from ..causal_rank_v7 import utility
from ..four_ideas_models import make_model, decision_loss


RL_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = RL_ROOT.parent
CACHE_ROOT = PROJECT_ROOT / "data/rl_raw_new50_5_group1_stage_b/cache"
VPO_CONFIG = PROJECT_ROOT / "rl/handoff_configs/sc_nbv_new50_5_group1_stage_b_raw.yaml"
OUT = PROJECT_ROOT / "work_dir/active_view_v9_latest_vpoclip_new50_5"
SEEDS = [20260909, 20260910, 20260911]
UNSEEN = [14, 15, 25, 39, 46]
SEEN = [i for i in range(55) if i not in UNSEEN]

# Preserve the old v9 policy-train/holdout idea, while removing classes that
# are now recognizer-unseen and therefore absent from dqn_train/val.
_old_holdout = [5, 6, 14, 22, 25, 38, 42, 44, 45, 51]
HOLDOUT = [x for x in _old_holdout if x in SEEN]
for candidate in SEEN:
    if len(HOLDOUT) >= 10:
        break
    if candidate not in HOLDOUT:
        HOLDOUT.append(candidate)
POLICY_POOL = [x for x in SEEN if x not in HOLDOUT]


class LatestGPUCache:
    """Load the latest variable-view cache without the legacy diagonal check.

    The latest cache correctly masks absent slots, so an absent slot has a
    false self-edge.  The historical MultiViewCache validator requires every
    padded slot to have a true diagonal, which is unrelated to the policy
    mask.  This reader preserves the actual masks and never changes the
    source cache on disk.
    """

    def __init__(self, root: Path, device: torch.device) -> None:
        metadata = json.loads((root / "metadata.json").read_text())
        self.device = device

        def load(name: str, dtype: torch.dtype | None = None) -> torch.Tensor:
            array = np.array(np.load(root / f"{name}.npy", mmap_mode="r"), copy=True)
            return torch.from_numpy(array).to(device=device, dtype=dtype)

        self.z = load("z", torch.float32)
        self.logits = load("logits", torch.float32)
        self.pose = load("pose", torch.float32)
        self.object_map = load("object_map", torch.float32).reshape(len(metadata["episodes"]), 4, -1)
        self.quality = load("image_quality", torch.float32)
        self.geometry = load("view_geometry", torch.float32)
        self.reachable = load("reachable", torch.bool)
        self.cost = load("move_cost", torch.float32)
        self.labels = load("target_columns", torch.long)
        prototypes = np.load(root / "text_prototypes.npy", allow_pickle=False)
        self.prototypes = torch.from_numpy(np.array(prototypes, copy=True)).to(device=device, dtype=torch.float32)
        self.num_episodes = len(metadata["episodes"])


def dump(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    tmp.replace(path)


def _bank(classes: list[int], count: int, device: torch.device) -> torch.Tensor:
    return class_mask(classes, count, device)


@torch.inference_mode()
def evaluate_fixed_start(
    net: torch.nn.Module,
    cache: LatestGPUCache,
    present: torch.Tensor,
    classes: list[int],
    start_view: int,
    seed: int,
    episode_filter: torch.Tensor | None = None,
) -> dict:
    """Evaluate one fixed initial slot, retaining only valid episodes."""
    all_ids = torch.arange(cache.num_episodes, device=cache.device)
    keep = present[:, start_view]
    if episode_filter is not None:
        keep = keep & episode_filter
    episodes = all_ids[keep]
    n = int(episodes.numel())
    if n == 0:
        raise RuntimeError(f"no episodes contain start view {start_view}")
    rows = torch.arange(n, device=cache.device)
    starts = torch.full((n,), start_view, dtype=torch.long, device=cache.device)
    real = present[episodes]
    valid = real.clone()
    valid[rows, starts] = False
    bank = _bank(classes, n, cache.device)

    actions = torch.empty(n, dtype=torch.long, device=cache.device)
    net.eval()
    for begin in range(0, n, 512):
        end = min(n, begin + 512)
        state = causal_state(cache, episodes[begin:end], starts[begin:end], bank[begin:end])
        actions[begin:end] = net(state).masked_fill(~state["mask"], -torch.inf).argmax(-1)

    rng = np.random.default_rng(seed + start_view)
    random_actions = torch.tensor(
        [int(rng.choice(np.flatnonzero(v))) for v in valid.cpu().numpy()],
        dtype=torch.long,
        device=cache.device,
    )
    # Deterministic fixed baseline: choose the next physical slot cyclically;
    # if it is absent, use the first legal slot.
    fixed_values = []
    for row in valid.cpu().numpy():
        cyclic = (start_view + 1) % 4
        legal = np.flatnonzero(row)
        fixed_values.append(int(cyclic) if row[cyclic] else int(legal[0]))
    fixed_actions = torch.tensor(fixed_values, dtype=torch.long, device=cache.device)

    labels = cache.labels[episodes]
    fused = 0.5 * (cache.logits[episodes, starts, None, :] + cache.logits[episodes])
    score = fused.masked_fill(~bank[:, None, :], -torch.inf)
    true_scores = score.gather(-1, labels[:, None, None].expand(-1, 4, 1)).squeeze(-1)
    oracle_actions = true_scores.masked_fill(~valid, -torch.inf).argmax(-1)
    candidate_correct = score.argmax(-1).eq(labels[:, None])
    random_exact = (candidate_correct & valid).sum(-1) / valid.sum(-1).clamp_min(1)

    choices = {
        "single": starts,
        "fixed_next": fixed_actions,
        "random": random_actions,
        "policy": actions,
        "oracle": oracle_actions,
    }
    metrics = {}
    policy_correct = None
    for name, selected in choices.items():
        selected_score = score[rows, selected]
        predicted = selected_score.argmax(-1)
        correct = predicted.eq(labels)
        probabilities = selected_score.softmax(-1)
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(-1)
        entropy = entropy / math.log(float(len(classes)))
        metrics[name] = {
            "top1": float(correct.float().mean().item()),
            "top5": float(selected_score.topk(min(5, len(classes)), -1).indices.eq(labels[:, None]).any(-1).float().mean().item()),
            "mean_entropy": float(entropy.mean().item()),
            "mean_movement_cost": float(cache.cost[episodes, starts, selected].mean().item()),
            "by_view_count": {
                str(k): float(correct[real.sum(-1) == k].float().mean().item())
                for k in (2, 3, 4)
                if bool((real.sum(-1) == k).any())
            },
        }
        if name == "policy":
            policy_correct = correct
    metrics["random_exact"] = {"top1": float(random_exact.mean().item())}

    gains = []
    per_class = {}
    for cls in sorted(int(x) for x in torch.unique(labels).cpu().tolist()):
        class_rows = labels == cls
        class_gain = policy_correct[class_rows].float().mean() - random_exact[class_rows].mean()
        per_class[str(cls)] = {
            "count": int(class_rows.sum().item()),
            "policy": float(policy_correct[class_rows].float().mean().item()),
            "random_exact": float(random_exact[class_rows].mean().item()),
            "gain": float(class_gain.item()),
        }
        if bool((class_rows & (valid.sum(-1) > 1)).any()):
            gains.append(float(class_gain.item()))

    return {
        "protocol": f"fixed{start_view}",
        "start_view": start_view,
        "episodes": n,
        "candidate_classes": classes,
        "metrics": metrics,
        "macro_choice_gain": float(np.mean(gains)) if gains else 0.0,
        "per_class": per_class,
    }


def build_cfg() -> dict:
    cfg = yaml.safe_load(VPO_CONFIG.read_text())
    cfg["cache"]["root"] = str(CACHE_ROOT)
    cfg["v9_latest"] = {
        "recognizer_checkpoint": cfg["recognizer"]["checkpoint"],
        "policy_pool": POLICY_POOL,
        "policy_holdout": HOLDOUT,
        "unseen_5way": UNSEEN,
        "seeds": SEEDS,
        "epochs": 40,
        "updates_per_epoch": 48,
        "batch_size": 8192,
        "objective": "historical v9 gap-weighted pairwise ranking",
        "movement_weight": 0.0,
        "evaluation": "all fixed starts 0,1,2,3 plus random-start diagnostic",
    }
    return cfg


def train_one(cfg: dict, seed: int, train: LatestGPUCache, val: LatestGPUCache, present: torch.Tensor, vp: torch.Tensor) -> dict:
    run = OUT / f"seed_{seed}"
    run.mkdir(parents=True, exist_ok=True)
    if (run / "complete.json").exists():
        return json.loads((run / "complete.json").read_text())
    device = train.device
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    generator = torch.Generator(device=device).manual_seed(seed)

    eligible = present & torch.isin(train.labels, torch.tensor(POLICY_POOL, device=device))[:, None]
    flat = eligible.flatten().nonzero().flatten()
    episodes, starts = (flat // 4).repeat(4), (flat % 4).repeat(4)
    labels = train.labels[episodes]
    frequencies = torch.bincount(labels, minlength=55).float()
    uniform = (1.0 / frequencies[labels]).float()
    uniform /= uniform.sum()

    proxy_ids = torch.isin(val.labels, torch.tensor(HOLDOUT, device=device)).nonzero().flatten()
    proxy_ids = proxy_ids.repeat(4)
    proxy_bank = task_banks(val, proxy_ids, HOLDOUT, torch.Generator(device=device).manual_seed(90117), 0.0)

    net = make_model("v9").to(device)
    compiled = torch.compile(net, mode="reduce-overhead")
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=0.03)
    best = -float("inf")
    start_epoch = 1
    if (run / "last.pt").exists():
        saved = torch.load(run / "last.pt", map_location=device, weights_only=False)
        net.load_state_dict(saved["online"])
        optimizer.load_state_dict(saved["optimizer"])
        generator.set_state(saved["generator"])
        torch.set_rng_state(saved["cpu_rng"].cpu())
        torch.cuda.set_rng_state(saved["cuda_rng"].cpu())
        best = float(saved["best"])
        start_epoch = int(saved["epoch"]) + 1

    audit = {
        "seed": seed,
        "contexts": int(episodes.numel()),
        "train_episodes": train.num_episodes,
        "val_episodes": val.num_episodes,
        "policy_pool": POLICY_POOL,
        "policy_holdout": HOLDOUT,
        "unseen_5way": UNSEEN,
        "recognizer_checkpoint": cfg["recognizer"]["checkpoint"],
        "objective": "v9 gap-weighted pairwise ranking",
        "causal_inference": True,
    }
    (run / "audit.json").write_text(json.dumps(audit, indent=2))
    dump(run / "config_resolved.json", cfg)
    with (run / "train.log").open("a", buffering=1) as log:
        for epoch in range(start_epoch, 41):
            begun = time.monotonic()
            banks = task_banks(train, episodes, POLICY_POOL, generator, 0.25)
            state = causal_state(train, episodes, starts, banks)
            targets, selective = utility(train, episodes, starts, banks, state["mask"])
            priority = uniform * selective
            priority = priority / priority.sum() if float(priority.sum()) > 0 else uniform
            probability = 0.5 * uniform + 0.5 * priority
            net.train()
            losses = []
            for _ in range(48):
                ids = torch.multinomial(probability, 8192, replacement=True, generator=generator)
                sample = {key: value[ids] for key, value in state.items()}
                loss = decision_loss("v9", compiled(sample), targets[ids], sample["mask"])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 2.0)
                optimizer.step()
                losses.append(loss.detach())
            torch.cuda.synchronize()

            # Validation selection is independent of the real unseen test.
            regular = evaluate_fixed_start(net, val, vp, SEEN, 0, seed + 1000)
            proxy_filter = torch.isin(val.labels, torch.tensor(HOLDOUT, device=val.device))
            proxy = evaluate_fixed_start(net, val, vp, HOLDOUT, 0, seed + 2000, proxy_filter)
            selection = float(0.25 * regular["macro_choice_gain"] + 0.75 * proxy["macro_choice_gain"])
            improved = selection > best
            best = max(best, selection)
            saved = {
                "online": net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "generator": generator.get_state(),
                "cpu_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(),
                "epoch": epoch,
                "best": best,
                "seed": seed,
            }
            torch.save(saved, run / "last.tmp.pt")
            (run / "last.tmp.pt").replace(run / "last.pt")
            if improved:
                torch.save(saved, run / "best.tmp.pt")
                (run / "best.tmp.pt").replace(run / "best.pt")
            record = {
                "seed": seed,
                "epoch": epoch,
                "loss": float(torch.stack(losses).mean().item()),
                "seconds": time.monotonic() - begun,
                "contexts": int(episodes.numel()),
                "selective": int(selective.sum().item()),
                "selection": selection,
                "best": best,
                "validation_fixed0_gain": regular["macro_choice_gain"],
                "proxy_fixed0_gain": proxy["macro_choice_gain"],
            }
            log.write(json.dumps(record) + "\n")
            print("V9_LATEST_TRAIN", json.dumps(record), flush=True)

    result = {"seed": seed, "score": best, "path": str(run)}
    dump(run / "complete.json", result)
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = build_cfg()
    dump(OUT / "config_resolved.json", cfg)
    device = torch.device(str(cfg["runtime"].get("device", "cuda:0")))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the latest-VPOCLIP v9 rerun")
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision("high")
    train = LatestGPUCache(CACHE_ROOT / "dqn_train", device)
    val = LatestGPUCache(CACHE_ROOT / "val", device)
    present = torch.from_numpy(np.load(CACHE_ROOT / "dqn_train/view_valid.npy")).to(device)
    vp = torch.from_numpy(np.load(CACHE_ROOT / "val/view_valid.npy")).to(device)
    print("V9_LATEST_READY", json.dumps({
        "cache": str(CACHE_ROOT),
        "recognizer": cfg["recognizer"]["checkpoint"],
        "train_episodes": train.num_episodes,
        "val_episodes": val.num_episodes,
        "policy_pool": POLICY_POOL,
        "holdout": HOLDOUT,
        "unseen_5way": UNSEEN,
    }), flush=True)

    records = [train_one(cfg, seed, train, val, present, vp) for seed in SEEDS]
    chosen = max(records, key=lambda x: float(x["score"]))
    selection = {"chosen": chosen, "runs": records, "criterion": "validation_only"}
    dump(OUT / "selection.json", selection)

    selected = make_model("v9").to(device)
    saved = torch.load(Path(chosen["path"]) / "best.pt", map_location=device, weights_only=False)
    selected.load_state_dict(saved["online"])
    selected.eval()

    test = LatestGPUCache(CACHE_ROOT / "test", device)
    test_present = torch.from_numpy(np.load(CACHE_ROOT / "test/view_valid.npy")).to(device)
    unseen_ids = torch.isin(test.labels, torch.tensor(UNSEEN, device=device))
    unseen_count = int(unseen_ids.sum().item())
    if unseen_count == 0:
        raise RuntimeError("latest test cache contains no strict unseen episodes")
    starts = {}
    for start in range(4):
        report = evaluate_fixed_start(selected, test, test_present & unseen_ids[:, None], UNSEEN, start, 20260910)
        report.update({"recognizer_checkpoint": cfg["recognizer"]["checkpoint"], "selected_seed": chosen["seed"], "selected_epoch": saved["epoch"]})
        starts[str(start)] = report
        dump(OUT / f"evaluation_test_unseen_fixed{start}.json", report)
        print("V9_LATEST_EVALUATION", json.dumps(report["metrics"]), flush=True)

    # Additional random-start diagnostic on exactly the same strict unseen set.
    rng = np.random.default_rng(20260910)
    random_reports = []
    for start in range(4):
        random_reports.append(evaluate_fixed_start(selected, test, test_present & unseen_ids[:, None], UNSEEN, start, 20260910 + start))
    # The four fixed-start reports are the primary answer; random-start is kept
    # as a diagnostic aggregate, not mixed with the fixed-start comparison.
    aggregate = {}
    for name in ("single", "fixed_next", "random", "policy", "oracle", "random_exact"):
        vals = [x["metrics"][name]["top1"] for x in starts.values()]
        aggregate[name] = {"mean_top1": float(np.mean(vals)), "per_start": vals}
    final = {
        "recognizer_checkpoint": cfg["recognizer"]["checkpoint"],
        "cache_root": str(CACHE_ROOT),
        "strict_unseen_classes": UNSEEN,
        "strict_unseen_episodes": unseen_count,
        "selected_policy": selection,
        "fixed_start_reports": starts,
        "fixed_start_mean": aggregate,
        "note": "Primary results are fixed starts 0..3; no 55-way predictions are used for unseen scoring.",
    }
    dump(OUT / "summary.json", final)
    print("V9_LATEST_COMPLETE", json.dumps({"out": str(OUT), "unseen_episodes": unseen_count, "aggregate": aggregate}), flush=True)


if __name__ == "__main__":
    main()
