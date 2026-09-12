"""Train trajectory policies with a standalone second-view target.

This is intentionally a new entry point.  The preserved trajectory policy
modules are imported as architecture definitions, while their old training
targets and loops are not modified.  A training action receives utility one
when the candidate view alone predicts the ground-truth class in the seen
class bank, and zero otherwise.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
import sys

sys.path.insert(0, str(PROJECT_ROOT))

from rl import multistep_angle_object_trajectory_ablation as ablation  # noqa: E402
from rl import multistep_angle_object_trajectory_policy_v1 as base  # noqa: E402
from rl import multistep_angle_object_trajectory_policy_v4 as v4  # noqa: E402
from rl import multistep_angle_object_trajectory_policy_v5_no_category_id as v5  # noqa: E402
from rl import multistep_angle_object_trajectory_policy_v6_no_object as v6  # noqa: E402


NUM_VIEWS = 4
TRAIN_SEEDS = [20260909, 20260910, 20260911]
EVAL_SEEDS = list(range(20260920, 20260950))
OLD_TRUE_UNSEEN = [0, 2, 26, 34, 50]
OLD_PSEUDO_UNSEEN = [5, 6, 14, 22, 25, 38, 42, 44, 45, 51]
OLD_TRAIN = [x for x in range(55) if x not in OLD_TRUE_UNSEEN]
NEW_TRUE_UNSEEN = [14, 15, 25, 39, 46]
# Keep the pseudo-unseen holdout used by the existing trajectory new-split
# report.  The SC-NBV family has a separate historical pseudo holdout in its
# own cache; its runner records that distinction explicitly.
NEW_PSEUDO_UNSEEN = [0, 2, 9, 10, 11, 17, 26, 34, 49, 50]
NEW_TRAIN = [x for x in range(55) if x not in NEW_TRUE_UNSEEN]


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def configure_variant(
    name: str,
) -> tuple[type[torch.nn.Module], Callable[[Any, Path, str], None], Callable[..., dict[str, torch.Tensor]]]:
    """Select one architecture in a fresh process without changing source files."""

    if name == "v1":
        return base.AngleObjectTrajectoryPolicy, base.attach_object_tracks, base.build_trajectory_state
    if name == "v4":
        return v4.BodyOrientationTrajectoryPolicy, base.attach_object_tracks, v4.build_trajectory_state_v4
    if name == "v5":
        return v5.CategoryFreeObjectTrajectoryPolicy, v5.v1.attach_object_tracks, v5.build_trajectory_state_v5
    if name == "v6":
        return v6.PersonGeometryTrajectoryPolicy, v6.attach_no_object_tracks, v6.build_trajectory_state_v5
    if name == "object_only":
        return ablation.ObjectOnlyTrajectoryPolicy, base.attach_object_tracks, ablation.build_object_only_state
    if name == "geometry_only":
        return ablation.GeometryOnlyPolicy, ablation.attach_no_object_tracks, ablation.build_geometry_only_state
    raise ValueError(f"unknown trajectory variant: {name}")


def load_raw(
    cache_root: Path,
    track_root: Path,
    split: str,
    device: torch.device,
    attach_tracks: Callable[[Any, Path, str], None],
) -> base.GPUCache:
    raw = base.GPUCache(cache_root / split, device)
    base.attach_view_valid(raw, cache_root, split)
    attach_tracks(raw, track_root, split)
    return raw


def standalone_targets(
    raw: base.GPUCache,
    episodes: torch.Tensor,
    valid: torch.Tensor,
    bank: torch.Tensor,
) -> torch.Tensor:
    """Binary standalone recognition target; no fusion and no current logits."""

    scores = raw.logits[episodes].index_select(-1, bank)
    local = torch.searchsorted(bank, raw.labels[episodes])
    correct = scores.argmax(-1).eq(local[:, None]).float()
    return correct.masked_fill(~valid, 0.0)


def standalone_top1(
    raw: base.GPUCache,
    episodes: torch.Tensor,
    views: torch.Tensor,
    bank: torch.Tensor,
) -> torch.Tensor:
    scores = raw.logits[episodes, views].index_select(-1, bank)
    local = torch.searchsorted(bank, raw.labels[episodes])
    return scores.argmax(-1).eq(local)


def fused_top1(
    raw: base.GPUCache,
    episodes: torch.Tensor,
    first: torch.Tensor,
    second: torch.Tensor,
    bank: torch.Tensor,
) -> torch.Tensor:
    scores = 0.5 * (raw.logits[episodes, first] + raw.logits[episodes, second])
    scores = scores.index_select(-1, bank)
    local = torch.searchsorted(bank, raw.labels[episodes])
    return scores.argmax(-1).eq(local)


@torch.inference_mode()
def evaluate_one_step(
    model: torch.nn.Module,
    state_builder: Callable[..., dict[str, torch.Tensor]],
    raw: base.GPUCache,
    classes: Sequence[int],
    protocol: str,
    seed: int,
) -> dict[str, Any]:
    bank = base.class_bank_tensor(classes, raw.device)
    eligible = torch.isin(raw.labels, bank) & (raw.valid.sum(-1) >= 2)
    episodes = eligible.nonzero().flatten()
    if protocol.startswith("fixed"):
        start = int(protocol.removeprefix("fixed"))
        episodes = (eligible & raw.valid[:, start]).nonzero().flatten()
        starts = torch.full_like(episodes, start)
    elif protocol == "random_start":
        starts = base.choose_random_starts(raw, episodes, seed)
    else:
        raise ValueError(protocol)
    if len(episodes) == 0:
        raise RuntimeError(f"no episodes for {protocol} and class bank {list(classes)}")

    count = torch.ones(len(episodes), device=raw.device, dtype=torch.long)
    state = state_builder(raw, episodes, starts[:, None], count)
    choiceable = state["mask"].any(-1)
    episodes = episodes[choiceable]
    starts = starts[choiceable]
    state = {key: value[choiceable] for key, value in state.items()}
    if len(episodes) == 0:
        raise RuntimeError(f"no reachable candidate for {protocol}")
    valid = state["mask"]
    q = model(state)
    policy_second = q.masked_fill(~valid, -torch.inf).argmax(-1)

    rng = np.random.default_rng(seed + 100_003)
    random_second = torch.empty_like(policy_second)
    for row, options in enumerate(valid.detach().cpu().numpy()):
        random_second[row] = int(rng.choice(np.flatnonzero(options)))

    single_policy = standalone_top1(raw, episodes, policy_second, bank)
    fused_policy = fused_top1(raw, episodes, starts, policy_second, bank)
    single_random = standalone_top1(raw, episodes, random_second, bank)
    fused_random = fused_top1(raw, episodes, starts, random_second, bank)

    # Oracle is diagnostic only: it is not used for training or selection.
    candidate_scores = raw.logits[episodes].index_select(-1, bank)
    local = torch.searchsorted(bank, raw.labels[episodes])
    oracle_correct = candidate_scores.argmax(-1).eq(local[:, None]) & valid
    oracle_exists = oracle_correct.any(-1)
    oracle_second = candidate_scores.new_zeros(len(episodes), dtype=torch.long)
    for row, options in enumerate(oracle_correct.detach().cpu().numpy()):
        legal = np.flatnonzero(valid[row].detach().cpu().numpy())
        hits = np.flatnonzero(options)
        oracle_second[row] = int(hits[0] if len(hits) else legal[0])
    oracle_single = standalone_top1(raw, episodes, oracle_second, bank)
    oracle_fused = fused_top1(raw, episodes, starts, oracle_second, bank)

    metrics = {
        "single_start": {"top1": float(standalone_top1(raw, episodes, starts, bank).float().mean())},
        "policy_second_single": {"top1": float(single_policy.float().mean())},
        "policy_fused": {"top1": float(fused_policy.float().mean())},
        "random_second_single": {"top1": float(single_random.float().mean())},
        "random_fused": {"top1": float(fused_random.float().mean())},
        "oracle_second_single": {"top1": float(oracle_single.float().mean())},
        "oracle_fused": {"top1": float(oracle_fused.float().mean())},
    }
    metrics["policy_second_minus_random_second"] = {
        "top1": metrics["policy_second_single"]["top1"] - metrics["random_second_single"]["top1"]
    }
    metrics["policy_fused_minus_random_fused"] = {
        "top1": metrics["policy_fused"]["top1"] - metrics["random_fused"]["top1"]
    }
    return {
        "protocol": protocol,
        "seed": int(seed),
        "class_bank": [int(value) for value in classes],
        "episodes": int(len(episodes)),
        "oracle_has_correct_candidate_rate": float(oracle_exists.float().mean()),
        "metrics": metrics,
    }


def train_seed(
    seed: int,
    model_cls: type[torch.nn.Module],
    state_builder: Callable[..., dict[str, torch.Tensor]],
    train_raw: base.GPUCache,
    val_raw: base.GPUCache,
    train_classes: Sequence[int],
    pseudo_classes: Sequence[int],
    device: torch.device,
    args: argparse.Namespace,
    output_root: Path,
) -> dict[str, Any]:
    run = output_root / f"seed_{seed}"
    run.mkdir(parents=True, exist_ok=True)
    complete_path = run / "complete.json"
    if complete_path.exists() and not args.force:
        return json.loads(complete_path.read_text(encoding="utf-8"))

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    generator = torch.Generator(device=device).manual_seed(seed)
    model = model_cls().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs, eta_min=args.learning_rate * 0.05
    )
    bank = base.class_bank_tensor(train_classes, device)
    eligible = (train_raw.valid.sum(-1) == NUM_VIEWS) & torch.isin(train_raw.labels, bank)
    pool = eligible.nonzero().flatten()
    if not len(pool):
        raise RuntimeError("no complete four-view seen training episodes")
    dump(
        run / "audit.json",
        {
            "experiment": "single_state_analysis",
            "variant": args.variant,
            "seed": seed,
            "training_episodes": int(len(pool)),
            "training_classes": [int(x) for x in train_classes],
            "selection_classes": [int(x) for x in pseudo_classes],
            "target": "standalone candidate-view Top-1 correctness in seen class bank",
            "current_view_logits_in_target": False,
            "fusion_in_target": False,
            "movement_cost_in_target": False,
            "history_length": 1,
            "candidate_action_count": NUM_VIEWS,
            "source_cache": str(train_raw.num_episodes),
        },
    )
    best_score = -float("inf")
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        started = time.monotonic()
        loss_values: list[float] = []
        skipped = 0
        for _ in range(args.updates_per_epoch):
            ids = torch.randint(len(pool), (args.batch_size,), device=device, generator=generator)
            episodes = pool[ids]
            starts = torch.randint(NUM_VIEWS, (args.batch_size,), device=device, generator=generator)
            history = starts[:, None]
            count = torch.ones(args.batch_size, device=device, dtype=torch.long)
            state = state_builder(train_raw, episodes, history, count)
            keep = state["mask"].any(-1)
            if not keep.all():
                skipped += int((~keep).sum())
                state = {key: value[keep] for key, value in state.items()}
                episodes = episodes[keep]
            if not len(episodes):
                continue
            target = standalone_targets(train_raw, episodes, state["mask"], bank)
            state, target = base.permute_candidates(state, target, generator)
            prediction = model(state)
            stage = torch.ones(len(episodes), device=device, dtype=torch.long)
            loss, _parts = base.multistep_loss(prediction, target, state["mask"], stage)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at seed={seed}, epoch={epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_values.append(float(loss.detach()))
        scheduler.step()
        torch.cuda.synchronize(device)
        validation = evaluate_one_step(
            model, state_builder, val_raw, pseudo_classes, "fixed0", seed + 90117
        )
        score = float(validation["metrics"]["policy_second_minus_random_second"]["top1"])
        is_best = score > best_score
        if is_best:
            best_score = score
            best_epoch = epoch
        payload = {
            "online": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_score": best_score,
            "best_epoch": best_epoch,
            "seed": seed,
            "target": "standalone_candidate_top1",
        }
        torch.save(payload, run / "last.pt")
        if is_best:
            torch.save(payload, run / "best.pt")
        elapsed = time.monotonic() - started
        row = {
            "epoch": epoch,
            "seed": seed,
            "loss": float(np.mean(loss_values)) if loss_values else None,
            "seconds": elapsed,
            "updates_per_second": args.updates_per_epoch / max(elapsed, 1e-6),
            "skipped_no_candidate": skipped,
            "validation_selection_score": score,
            "best_score": best_score,
            "best_epoch": best_epoch,
            "validation": validation["metrics"],
        }
        with (run / "train.log").open("a", encoding="utf-8", buffering=1) as handle:
            handle.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
    result = {
        "variant": args.variant,
        "seed": seed,
        "best_score": best_score,
        "best_epoch": best_epoch,
        "checkpoint": str(run / "best.pt"),
        "training_episodes": int(len(pool)),
    }
    dump(complete_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("v1", "v4", "v5", "v6", "object_only", "geometry_only"), required=True)
    parser.add_argument("--split", choices=("old", "new"), required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--track-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--updates-per-epoch", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.04)
    parser.add_argument("--seeds", type=int, nargs="+", default=TRAIN_SEEDS)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    torch.set_float32_matmul_precision("high")
    if not torch.cuda.is_available():
        raise RuntimeError("single-state training requires CUDA")
    device = torch.device("cuda:0")
    model_cls, attach_tracks, state_builder = configure_variant(args.variant)
    if args.split == "old":
        train_classes, pseudo = OLD_TRAIN, OLD_PSEUDO_UNSEEN
    else:
        train_classes, pseudo = NEW_TRAIN, NEW_PSEUDO_UNSEEN
    args.output_root.mkdir(parents=True, exist_ok=True)
    dump(
        args.output_root / "config.json",
        {
            "variant": args.variant,
            "split": args.split,
            "cache_root": str(args.cache_root),
            "track_root": str(args.track_root),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "updates_per_epoch": args.updates_per_epoch,
            "learning_rate": args.learning_rate,
            "seeds": args.seeds,
            "target": "standalone second-view recognition correctness",
        },
    )
    train_raw = load_raw(args.cache_root, args.track_root, "dqn_train", device, attach_tracks)
    val_raw = load_raw(args.cache_root, args.track_root, "val", device, attach_tracks)
    summaries = []
    for seed in args.seeds:
        print("SINGLE_STATE_TRAJECTORY_TRAIN_START", args.split, args.variant, seed, flush=True)
        summaries.append(
            train_seed(
                int(seed), model_cls, state_builder, train_raw, val_raw,
                train_classes, pseudo, device, args, args.output_root,
            )
        )
    dump(
        args.output_root / "complete.json",
        {
            "experiment": "single_state_trajectory",
            "split": args.split,
            "variant": args.variant,
            "target": "standalone candidate-view Top-1",
            "runs": summaries,
        },
    )
    print("SINGLE_STATE_TRAJECTORY_TRAIN_COMPLETE", args.split, args.variant, flush=True)


if __name__ == "__main__":
    main()
