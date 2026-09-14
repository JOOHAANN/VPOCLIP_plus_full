"""Train one frame-0 angle-rank variant with masked offline Double DQN.

The existing v1--v6/object-only/geometry-only comparison uses an offline
utility-ranking loss.  This module keeps each variant's state builder and
network unchanged, but trains two copies of that network with a replay pool,
masked Double-DQN targets, and a periodically synchronized target network.

The cache is an offline deterministic MDP: labels and VPOCLIP logits are used
only to construct seen-class margin rewards, never as policy inputs.  Each
episode starts with one view, performs two actions, and fuses exactly three
views.  The fourth view is not part of the DQN return.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from . import train_ranked_variant as adapter
from .. import multistep_g18_view_policy_v1 as common


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_ROOT = (
    ROOT / "data" / "frame0_body_angle_rank_g1_pseudo_selected_stage_b" / "cache"
)
DEFAULT_TRACK_ROOT = (
    ROOT
    / "data"
    / "frame0_body_angle_rank_g1_pseudo_selected_stage_b"
    / "object_tracks_full"
)
DEFAULT_TRACK_ROOT_V3 = (
    ROOT
    / "data"
    / "frame0_body_angle_rank_g1_pseudo_selected_stage_b"
    / "object_tracks_v3_bbox_size"
)
DEFAULT_OUTPUT_ROOT = ROOT / "work_dir" / "frame0_body_angle_rank_new50_5_ddqn_v1_v6"
DEFAULT_SEEDS = [20260909, 20260910, 20260911]
DEFAULT_UNSEEN = [14, 15, 25, 39, 46]
DEFAULT_PSEUDO = [0, 2, 9, 10, 11, 17, 26, 34, 49, 50]
VARIANTS = (
    "v1",
    "v2_legacy",
    "v3",
    "v4",
    "v5",
    "v6",
    "object_only",
    "geometry_only",
)
NUM_VIEWS = 4


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class TransitionPool:
    state_history: torch.Tensor
    state_count: torch.Tensor
    action: torch.Tensor
    next_history: torch.Tensor
    next_count: torch.Tensor
    reward: torch.Tensor
    done: torch.Tensor
    stage1: torch.Tensor

    @property
    def size(self) -> int:
        return int(self.action.numel())


def _margin(scores: np.ndarray, label: int, bank: np.ndarray) -> float:
    values = scores[bank]
    target_index = int(np.flatnonzero(bank == int(label))[0])
    target = float(values[target_index])
    rival = float(np.max(np.delete(values, target_index)))
    return target - rival


def build_transition_pool(
    raw: Any,
    seen_classes: Sequence[int],
    device: torch.device,
) -> TransitionPool:
    """Enumerate all legal one-step transitions in complete seen episodes."""

    valid = raw.valid.detach().cpu().numpy().astype(bool)
    reachable = raw.reachable.detach().cpu().numpy().astype(bool)
    logits = raw.logits.detach().cpu().numpy().astype(np.float32)
    labels = raw.labels.detach().cpu().numpy().astype(np.int64)
    bank = np.asarray(sorted(int(x) for x in seen_classes), dtype=np.int64)
    eligible = np.isin(labels, bank) & (valid.sum(axis=1) == NUM_VIEWS)

    states: list[list[int]] = []
    counts: list[int] = []
    actions: list[int] = []
    next_states: list[list[int]] = []
    next_counts: list[int] = []
    rewards: list[float] = []
    dones: list[bool] = []
    stages: list[bool] = []

    for episode in np.flatnonzero(eligible):
        label = int(labels[episode])
        available = np.flatnonzero(valid[episode]).tolist()
        for start in available:
            first_actions = [
                int(action)
                for action in available
                if action != start and reachable[episode, start, action]
            ]
            for action1 in first_actions:
                before = _margin(logits[episode, [start]].mean(axis=0), label, bank)
                after = _margin(
                    logits[episode, [start, action1]].mean(axis=0), label, bank
                )
                states.append([start, 0, 0])
                counts.append(1)
                actions.append(action1)
                next_states.append([start, action1, 0])
                next_counts.append(2)
                rewards.append(after - before)
                dones.append(False)
                stages.append(True)

                second_actions = [
                    int(action2)
                    for action2 in available
                    if action2 not in (start, action1)
                    and reachable[episode, action1, action2]
                ]
                before_second = after
                for action2 in second_actions:
                    final_margin = _margin(
                        logits[episode, [start, action1, action2]].mean(axis=0),
                        label,
                        bank,
                    )
                    states.append([start, action1, 0])
                    counts.append(2)
                    actions.append(action2)
                    next_states.append([start, action1, action2])
                    next_counts.append(3)
                    rewards.append(final_margin - before_second)
                    dones.append(True)
                    stages.append(False)

    if not actions:
        raise RuntimeError("the DQN replay pool contains no legal transitions")

    def tensor(value: Any, dtype: torch.dtype) -> torch.Tensor:
        return torch.as_tensor(value, dtype=dtype, device=device)

    pool = TransitionPool(
        state_history=tensor(states, torch.long),
        state_count=tensor(counts, torch.long),
        action=tensor(actions, torch.long),
        next_history=tensor(next_states, torch.long),
        next_count=tensor(next_counts, torch.long),
        reward=tensor(rewards, torch.float32),
        done=tensor(dones, torch.float32),
        stage1=tensor(stages, torch.bool),
    )
    return pool


def _build_state_fn(module: Any) -> Callable[..., dict[str, torch.Tensor]]:
    if hasattr(module, "build_trajectory_state"):
        return module.build_trajectory_state
    if hasattr(module, "build_angle_state"):
        return module.build_angle_state
    raise AttributeError(f"{module.__name__} has no compatible state builder")


def _model_class(module: Any) -> type[torch.nn.Module]:
    if hasattr(module, "AngleObjectTrajectoryPolicy"):
        return module.AngleObjectTrajectoryPolicy
    if hasattr(module, "AngleObjectPolicy"):
        return module.AngleObjectPolicy
    raise AttributeError(f"{module.__name__} has no compatible policy class")


def _load_raw(
    module: Any,
    cache_root: Path,
    track_root: Path,
    split: str,
    device: torch.device,
) -> Any:
    raw = module.GPUCache(cache_root / split, device)
    module.attach_view_valid(raw, cache_root, split)
    attach_tracks = getattr(module, "attach_object_tracks", None)
    if attach_tracks is not None:
        attach_tracks(raw, track_root, split)
    return raw


def _permute_candidates(
    state: dict[str, torch.Tensor],
    action: torch.Tensor | None,
    generator: torch.Generator,
) -> tuple[dict[str, torch.Tensor], torch.Tensor | None]:
    """Randomize candidate row order while preserving the action identity."""

    batch = state["candidate"].shape[0]
    order = torch.rand(
        (batch, NUM_VIEWS), device=state["candidate"].device, generator=generator
    ).argsort(-1)
    result = dict(state)
    result["candidate"] = state["candidate"].gather(
        1, order[..., None].expand(-1, -1, state["candidate"].shape[-1])
    )
    result["mask"] = state["mask"].gather(1, order)
    if action is None:
        return result, None
    remapped = (order == action[:, None]).to(torch.long).argmax(-1)
    return result, remapped


def _masked_q(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return values.masked_fill(~mask.bool(), -1.0e9)


def _rollout(
    model: torch.nn.Module,
    state_fn: Callable[..., dict[str, torch.Tensor]],
    raw: Any,
    episodes: torch.Tensor,
    starts: torch.Tensor,
) -> dict[str, torch.Tensor]:
    model.eval()
    batch = episodes.shape[0]
    one = torch.ones(batch, dtype=torch.long, device=episodes.device)
    state1 = state_fn(raw, episodes, starts[:, None], one)
    keep1 = state1["mask"].any(-1)
    episodes = episodes[keep1]
    starts = starts[keep1]
    state1 = {key: value[keep1] for key, value in state1.items()}
    action1 = _masked_q(model(state1), state1["mask"]).argmax(-1)

    history2 = torch.stack((starts, action1), 1)
    two = torch.full((len(episodes),), 2, dtype=torch.long, device=episodes.device)
    state2 = state_fn(raw, episodes, history2, two)
    keep2 = state2["mask"].any(-1)
    episodes = episodes[keep2]
    starts = starts[keep2]
    action1 = action1[keep2]
    history2 = history2[keep2]
    state2 = {key: value[keep2] for key, value in state2.items()}
    action2 = _masked_q(model(state2), state2["mask"]).argmax(-1)
    return {
        "episodes": episodes,
        "starts": starts,
        "action1": action1,
        "action2": action2,
    }


@torch.inference_mode()
def evaluate_protocol(
    model: torch.nn.Module,
    state_fn: Callable[..., dict[str, torch.Tensor]],
    raw: Any,
    classes: Sequence[int],
    protocol: str,
    seed: int,
) -> dict[str, Any]:
    bank = common.class_bank_tensor(classes, raw.device)
    eligible = torch.isin(raw.labels, bank) & (raw.valid.sum(-1) >= 3)
    episodes = eligible.nonzero().flatten()
    if protocol.startswith("fixed"):
        start = int(protocol.removeprefix("fixed"))
        episodes = (eligible & raw.valid[:, start]).nonzero().flatten()
        starts = torch.full_like(episodes, start)
    elif protocol == "random_start":
        starts = common.choose_random_starts(raw, episodes, seed)
    else:
        raise ValueError(protocol)

    single = common._single_metrics(raw, episodes, starts, bank) if len(episodes) else {"top1": 0.0}
    rollout = _rollout(model, state_fn, raw, episodes, starts)
    policy_episodes = rollout["episodes"]
    policy_paths = torch.stack(
        (rollout["starts"], rollout["action1"], rollout["action2"]), 1
    )
    policy = (
        common.metrics_for_paths(raw, policy_episodes, policy_paths, bank)
        if len(policy_episodes)
        else {"top1": 0.0, "mean_entropy": 0.0, "mean_movement_cost": 0.0, "episodes": 0.0}
    )
    random_exact, _ = common.random_exact_metrics(raw, episodes, starts, bank)
    random_ep, random_paths, _ = common.sample_random_paths(
        raw, episodes, starts, seed + 17
    )
    random_sample = (
        common.metrics_for_paths(raw, random_ep, random_paths, bank)
        if len(random_ep)
        else {"top1": 0.0, "mean_entropy": 0.0, "mean_movement_cost": 0.0, "episodes": 0.0}
    )
    oracle_ep, oracle_path = common.oracle_paths(raw, episodes, starts, bank)
    oracle = (
        common.metrics_for_paths(raw, oracle_ep, oracle_path, bank)
        if len(oracle_ep)
        else {"top1": 0.0, "mean_entropy": 0.0, "mean_movement_cost": 0.0, "episodes": 0.0}
    )
    return {
        "protocol": protocol,
        "class_bank": [int(x) for x in classes],
        "episodes_before_two_move_filter": int(len(episodes)),
        "episodes_policy_completed": int(len(policy_episodes)),
        "completion_rate": float(len(policy_episodes) / max(len(episodes), 1)),
        "metrics": {
            "single_start": single,
            "random_3view_sampled": random_sample,
            "random_3view_exact": random_exact,
            "policy_3view": policy,
            "oracle_3view": oracle,
        },
    }


def _sample_indices(
    pool: TransitionPool,
    batch_size: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Balance one-step and terminal transitions in the replay minibatch."""

    half = batch_size // 2
    stage1_ids = torch.nonzero(pool.stage1, as_tuple=False).flatten()
    stage2_ids = torch.nonzero(~pool.stage1, as_tuple=False).flatten()
    first = stage1_ids[
        torch.randint(len(stage1_ids), (half,), device=pool.action.device, generator=generator)
    ]
    second = stage2_ids[
        torch.randint(
            len(stage2_ids),
            (batch_size - half,),
            device=pool.action.device,
            generator=generator,
        )
    ]
    return torch.cat((first, second), 0)


def _batch_state(
    state_fn: Callable[..., dict[str, torch.Tensor]],
    raw: Any,
    history: torch.Tensor,
    count: torch.Tensor,
    episodes: torch.Tensor,
) -> dict[str, torch.Tensor]:
    return state_fn(raw, episodes, history, count)


def train_seed(
    seed: int,
    module: Any,
    state_fn: Callable[..., dict[str, torch.Tensor]],
    train_raw: Any,
    val_raw: Any,
    pool: TransitionPool,
    seen_classes: Sequence[int],
    pseudo_classes: Sequence[int],
    args: argparse.Namespace,
    output_root: Path,
) -> dict[str, Any]:
    run = output_root / f"seed_{seed}"
    run.mkdir(parents=True, exist_ok=True)
    complete = run / "complete.json"
    if complete.exists():
        return json.loads(complete.read_text(encoding="utf-8"))

    seed_all(seed)
    generator = torch.Generator(device=train_raw.device).manual_seed(seed)
    model_cls = _model_class(module)
    online = model_cls().to(train_raw.device)
    target = model_cls().to(train_raw.device)
    target.load_state_dict(online.state_dict(), strict=True)
    target.eval()
    optimizer = torch.optim.AdamW(
        online.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs, eta_min=args.learning_rate * 0.05
    )
    best_score = -float("inf")
    best_epoch = 0
    updates = 0
    train_log = run / "train.log"
    audit = {
        "algorithm": "masked_offline_double_dqn",
        "seed": seed,
        "replay_transitions": pool.size,
        "stage1_transitions": int(pool.stage1.sum().item()),
        "stage2_terminal_transitions": int((~pool.stage1).sum().item()),
        "gamma": args.gamma,
        "target_update_interval": args.target_update_interval,
        "reward": "seen-bank delta margin after adding the selected view",
        "policy_inputs_excluded": [
            "class label",
            "VPOCLIP logits",
            "future-view features",
        ],
        "terminal": "after exactly three acquired views",
    }
    dump(run / "audit.json", audit)

    for epoch in range(1, args.epochs + 1):
        online.train()
        begun = time.monotonic()
        losses: list[float] = []
        q_values: list[float] = []
        targets: list[float] = []
        for _ in range(args.updates_per_epoch):
            ids = _sample_indices(pool, args.batch_size, generator)
            episodes = pool.episode[ids] if hasattr(pool, "episode") else None
            # Episode IDs are attached below without being part of the dataclass
            # constructor used by older checkpoints.
            if episodes is None:
                raise RuntimeError("DQN pool is missing episode IDs")
            histories = pool.state_history[ids]
            counts = pool.state_count[ids]
            next_histories = pool.next_history[ids]
            next_counts = pool.next_count[ids]
            action = pool.action[ids]
            reward = pool.reward[ids]
            done = pool.done[ids]

            state = _batch_state(state_fn, train_raw, histories, counts, episodes)
            next_state = _batch_state(
                state_fn, train_raw, next_histories, next_counts, episodes
            )
            state, action = _permute_candidates(state, action, generator)
            next_state, _ = _permute_candidates(next_state, None, generator)

            q = online(state).gather(1, action[:, None]).squeeze(1)
            with torch.no_grad():
                next_online = _masked_q(online(next_state), next_state["mask"])
                next_action = next_online.argmax(-1)
                next_target = target(next_state).gather(
                    1, next_action[:, None]
                ).squeeze(1)
                bellman = reward + args.gamma * (1.0 - done) * next_target
            loss = F.smooth_l1_loss(q, bellman)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(online.parameters(), args.grad_clip)
            optimizer.step()
            updates += 1
            if updates % args.target_update_interval == 0:
                target.load_state_dict(online.state_dict(), strict=True)
                target.eval()
            losses.append(float(loss.detach()))
            q_values.append(float(q.detach().mean()))
            targets.append(float(bellman.detach().mean()))
        scheduler.step()
        validation = evaluate_protocol(
            online, state_fn, val_raw, pseudo_classes, "fixed0", seed + 90117 + epoch
        )
        vm = validation["metrics"]
        score = float(
            vm["policy_3view"]["top1"] - vm["random_3view_exact"]["top1"]
        )
        improved = score > best_score
        if improved:
            best_score = score
            best_epoch = epoch
        payload = {
            "online": online.state_dict(),
            "target": target.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "updates": updates,
            "best_score": best_score,
            "best_epoch": best_epoch,
            "seed": seed,
            "algorithm": "masked_offline_double_dqn",
        }
        torch.save(payload, run / "last.pt")
        if improved:
            torch.save(payload, run / "best.pt")
        row = {
            "epoch": epoch,
            "seed": seed,
            "loss": float(np.mean(losses)),
            "mean_q": float(np.mean(q_values)),
            "mean_target": float(np.mean(targets)),
            "seconds": time.monotonic() - begun,
            "updates": updates,
            "validation_selection_score": score,
            "best_score": best_score,
            "best_epoch": best_epoch,
            "validation": vm,
        }
        with train_log.open("a", encoding="utf-8", buffering=1) as handle:
            handle.write(json.dumps(row) + "\n")
        print(
            json.dumps(
                {
                    key: row[key]
                    for key in (
                        "epoch",
                        "loss",
                        "mean_q",
                        "mean_target",
                        "validation_selection_score",
                        "best_score",
                        "best_epoch",
                    )
                }
            ),
            flush=True,
        )

    result = {
        "seed": seed,
        "best_score": best_score,
        "best_epoch": best_epoch,
        "checkpoint": str(run / "best.pt"),
        "output": str(run),
        "replay_transitions": pool.size,
    }
    dump(complete, result)
    return result


def configure(
    args: argparse.Namespace,
) -> tuple[Any, Callable[..., dict[str, torch.Tensor]], Path | None, dict[str, Any]]:
    module, track_root, provenance = adapter.configure_variant(args, args.output_root)
    # The frame-0 runner normally assigns these globals inside v1.main().
    # Assign them explicitly because this script has its own training loop.
    seen = sorted(set(range(55)) - set(args.unseen_classes))
    module.SEEN_CLASSES = seen
    module.PSEUDO_UNSEEN_CLASSES = sorted(args.pseudo_unseen_classes)
    module.TRUE_UNSEEN_CLASSES = sorted(args.unseen_classes)
    state_fn = _build_state_fn(module)
    provenance = {**provenance, "effective_track_root": str(track_root) if track_root is not None else None}
    return module, state_fn, track_root, provenance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--track-root", type=Path, default=DEFAULT_TRACK_ROOT)
    parser.add_argument("--track-root-v3", type=Path, default=DEFAULT_TRACK_ROOT_V3)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--updates-per-epoch", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=0.04)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--target-update-interval", type=int, default=250)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--memory-fraction", type=float, default=0.20)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--unseen-classes", type=int, nargs="+", default=DEFAULT_UNSEEN)
    parser.add_argument("--pseudo-unseen-classes", type=int, nargs="+", default=DEFAULT_PSEUDO)
    args = parser.parse_args()
    args.unseen_classes = sorted(set(int(x) for x in args.unseen_classes))
    args.pseudo_unseen_classes = sorted(set(int(x) for x in args.pseudo_unseen_classes))
    if set(args.unseen_classes) & set(args.pseudo_unseen_classes):
        raise ValueError("unseen and pseudo-unseen classes must be disjoint")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the frame-0 DQN experiment")

    args.output_root = args.output_root / args.variant
    args.output_root.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction, 0)
    device = torch.device("cuda:0")
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    module, state_fn, effective_track_root, provenance = configure(args)
    seen_classes = sorted(set(range(55)) - set(args.unseen_classes))
    metadata = {
        "algorithm": "masked_offline_double_dqn",
        "variant": args.variant,
        "implementation": provenance,
        "cache_root": str(args.cache_root),
        "track_root": str(effective_track_root) if effective_track_root is not None else None,
        "split": {
            "seen_classes": seen_classes,
            "pseudo_unseen_classes": args.pseudo_unseen_classes,
            "true_unseen_classes": args.unseen_classes,
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "updates_per_epoch": args.updates_per_epoch,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "gamma": args.gamma,
            "target_update_interval": args.target_update_interval,
            "grad_clip": args.grad_clip,
            "seeds": args.seeds,
        },
        "policy_contract": {
            "starting_views": 1,
            "selected_actions": 2,
            "fused_views": 3,
            "candidate_actions": NUM_VIEWS,
            "fourth_view": "not executed; no return bootstrap",
        },
        "test_labels_used_for_training": False,
    }
    dump(args.output_root / "config_resolved.json", metadata)

    raw: dict[str, Any] = {}
    for split in ("dqn_train", "val", "seen_test", "test"):
        if effective_track_root is None:
            # v2_legacy has no object-track attachment; the state uses the
            # object map already stored in the cache.
            effective_track_root_for_load = args.track_root
        else:
            effective_track_root_for_load = effective_track_root
        raw[split] = _load_raw(
            module, args.cache_root, effective_track_root_for_load, split, device
        )
    print(
        "FRAME0_DDQN_CACHE_SHAPES",
        json.dumps({key: value.num_episodes for key, value in raw.items()}),
        flush=True,
    )
    pool = build_transition_pool(raw["dqn_train"], seen_classes, device)
    # The dataclass is intentionally kept compact, but episode IDs are needed
    # to reconstruct states from the shared cache during replay.
    # Attach it after construction to preserve a simple serialized contract.
    eligible = (
        torch.isin(raw["dqn_train"].labels, torch.as_tensor(seen_classes, device=device))
        & (raw["dqn_train"].valid.sum(-1) == NUM_VIEWS)
    )
    # Rebuild the same episode ordering used by build_transition_pool.
    episode_ids: list[int] = []
    valid_cpu = raw["dqn_train"].valid.detach().cpu().numpy().astype(bool)
    labels_cpu = raw["dqn_train"].labels.detach().cpu().numpy().astype(np.int64)
    eligible_cpu = np.isin(labels_cpu, np.asarray(seen_classes)) & (valid_cpu.sum(1) == NUM_VIEWS)
    reachable_cpu = raw["dqn_train"].reachable.detach().cpu().numpy().astype(bool)
    for episode in np.flatnonzero(eligible_cpu):
        available = np.flatnonzero(valid_cpu[episode]).tolist()
        for start in available:
            for action1 in available:
                if action1 == start or not reachable_cpu[episode, start, action1]:
                    continue
                episode_ids.append(int(episode))
                for action2 in available:
                    if action2 in (start, action1) or not reachable_cpu[episode, action1, action2]:
                        continue
                    episode_ids.append(int(episode))
    # The nested order above is exactly the transition append order.
    if len(episode_ids) != pool.size:
        raise RuntimeError(
            f"transition/episode alignment mismatch: {len(episode_ids)} vs {pool.size}"
        )
    pool.episode = torch.as_tensor(episode_ids, dtype=torch.long, device=device)  # type: ignore[attr-defined]
    del eligible

    print(
        "FRAME0_DDQN_REPLAY",
        json.dumps(
            {
                "variant": args.variant,
                "transitions": pool.size,
                "stage1": int(pool.stage1.sum().item()),
                "stage2": int((~pool.stage1).sum().item()),
            }
        ),
        flush=True,
    )

    summaries: list[dict[str, Any]] = []
    for seed in args.seeds:
        summaries.append(
            train_seed(
                seed,
                module,
                state_fn,
                raw["dqn_train"],
                raw["val"],
                pool,
                seen_classes,
                args.pseudo_unseen_classes,
                args,
                args.output_root,
            )
        )
        torch.cuda.empty_cache()
    dump(args.output_root / "training_summary.json", summaries)
    print(
        "FRAME0_DDQN_TRAIN_COMPLETE",
        json.dumps({"variant": args.variant, "output": str(args.output_root)}),
        flush=True,
    )


if __name__ == "__main__":
    main()
