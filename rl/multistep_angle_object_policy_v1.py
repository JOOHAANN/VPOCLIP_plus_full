"""Causal multi-step active-view experiment using only angle + object state.

This is an isolated comparison against ``multistep_g18_view_policy_v1``.
The selector receives no video embedding, skeleton, semantic belief, current
VPOCLIP logits, class ID, or future-view feature.  Its state contains only:

    * the sample-specific human/camera relative direction and candidate
      relative-angle geometry;
    * the mean object-detection map from the currently observed views.

The trajectory is:

    one observed view -> predict the second view
    two observed views -> predict the third view
    three observed views -> predict/log the fourth view only

The final HAR score fuses exactly the first three observed-view logits.  The
frozen VPOCLIP logits are used only for offline delta-margin targets and final
evaluation, never as selector inputs.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .multistep_g18_view_policy_v1 import (
    CACHE_ROOT,
    OUTPUT_ROOT as G18_OUTPUT_ROOT,
    NUM_VIEWS,
    SEEN_CLASSES,
    PSEUDO_UNSEEN_CLASSES,
    TRUE_UNSEEN_CLASSES,
    BASE_CHECKPOINT,
    GPUCache,
    class_bank_tensor,
    utility_targets,
    permute_candidates,
    multistep_loss,
    choose_random_starts,
    _path_score,
    _single_metrics,
    random_exact_metrics,
    sample_random_paths,
    oracle_paths,
    metrics_for_paths,
    attach_view_valid,
)


OUTPUT_ROOT = G18_OUTPUT_ROOT.parent / "multistep_angle_object_policy_v1"
OBJECT_DIM = 1800
ANGLE_CANDIDATE_DIM = 10


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def angle_candidate_features(candidate: torch.Tensor) -> torch.Tensor:
    """Encode only sample-specific directions; omit valid/cost fields."""

    camera_angle = candidate[..., :2]
    relative_delta = candidate[..., 2:4]
    current_angle = candidate[..., 4:6]
    camera_harmonic = torch.stack(
        (
            2 * camera_angle[..., 0] * camera_angle[..., 1],
            camera_angle[..., 1].square() - camera_angle[..., 0].square(),
        ),
        -1,
    )
    delta_harmonic = torch.stack(
        (
            2 * relative_delta[..., 0] * relative_delta[..., 1],
            relative_delta[..., 1].square() - relative_delta[..., 0].square(),
        ),
        -1,
    )
    result = torch.cat(
        (camera_angle, relative_delta, current_angle, camera_harmonic, delta_harmonic),
        -1,
    )
    if result.shape[-1] != ANGLE_CANDIDATE_DIM:
        raise AssertionError(result.shape)
    return result


class AngleObjectPolicy(nn.Module):
    """Low-capacity scorer whose observable state is angle + object only."""

    def __init__(self, object_width: int = 32, context_width: int = 48) -> None:
        super().__init__()
        self.object_encoder = nn.Sequential(
            nn.Linear(OBJECT_DIM, object_width),
            nn.LayerNorm(object_width),
            nn.GELU(),
        )
        self.relative_angle_encoder = nn.Sequential(
            nn.Linear(2, 16), nn.LayerNorm(16), nn.GELU()
        )
        self.context = nn.Sequential(
            nn.Linear(object_width + 16, context_width),
            nn.LayerNorm(context_width),
            nn.GELU(),
        )
        self.candidate = nn.Sequential(
            nn.Linear(ANGLE_CANDIDATE_DIM, context_width),
            nn.LayerNorm(context_width),
            nn.GELU(),
        )
        self.context_to_action = nn.Linear(context_width, context_width, bias=False)
        self.residual = nn.Sequential(
            nn.Linear(context_width * 2, 32), nn.GELU(), nn.Linear(32, 1)
        )

    def forward(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        object_context = self.object_encoder(state["object_state"])
        angle_context = self.relative_angle_encoder(state["current_relative_angle"])
        context = self.context(torch.cat((object_context, angle_context), -1))
        candidate = self.candidate(angle_candidate_features(state["candidate"]))
        context_expanded = context[:, None].expand(-1, NUM_VIEWS, -1)
        q = (
            self.context_to_action(context)[:, None, :] * candidate
        ).sum(-1) / math.sqrt(float(candidate.shape[-1]))
        q = q + 0.20 * self.residual(
            torch.cat((context_expanded, candidate), -1)
        ).squeeze(-1)
        return q


def build_angle_state(
    raw: GPUCache,
    episodes: torch.Tensor,
    history: torch.Tensor,
    history_count: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Construct a state without allocating or passing other modalities."""

    episodes = episodes.long()
    history = history.long()
    count = history_count.long().clamp(1, history.shape[1])
    batch = episodes.shape[0]
    positions = torch.arange(history.shape[1], device=episodes.device)[None, :]
    observed = positions < count[:, None]
    observed_float = observed.float()
    denominator = count.float().clamp_min(1.0)[:, None]
    current = history.gather(1, (count - 1)[:, None]).squeeze(1)

    # Object detections are causal sensor content.  The same bounded transform
    # used by the previous selector is retained, but z/skeleton/logits are not
    # even allocated in this state.
    object_map = raw.object_map[episodes[:, None], history]
    object_map = object_map.clamp_min(0).log1p().clamp_max(5.0) / math.log(11.0)
    object_state = (object_map * observed_float[..., None]).sum(1) / denominator

    observed_views = torch.zeros(
        (batch, NUM_VIEWS), dtype=torch.bool, device=episodes.device
    )
    observed_views.scatter_(1, history, observed)
    current_relative_angle = raw.geometry[episodes, current, :2]
    geometry = raw.geometry[episodes, :, :2]
    sin_delta = geometry[..., 0] * current_relative_angle[:, None, 1] - geometry[..., 1] * current_relative_angle[:, None, 0]
    cos_delta = (geometry * current_relative_angle[:, None]).sum(-1)
    valid = (
        raw.valid[episodes]
        & raw.reachable[episodes, current]
        & (~observed_views)
    )
    candidate = torch.cat(
        (
            geometry,
            torch.stack((sin_delta, cos_delta), -1),
            current_relative_angle[:, None].expand(-1, NUM_VIEWS, -1),
            valid[..., None].float(),
        ),
        -1,
    )
    return {
        "object_state": object_state,
        "current_relative_angle": current_relative_angle,
        "candidate": candidate,
        "mask": valid,
    }


@torch.inference_mode()
def policy_rollout_angle(
    model: nn.Module,
    raw: GPUCache,
    episodes: torch.Tensor,
    starts: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Two executed actions plus a non-executed fourth-view prediction."""

    model.eval()
    batch = episodes.shape[0]
    one = torch.ones(batch, dtype=torch.long, device=episodes.device)
    state1 = build_angle_state(raw, episodes, starts[:, None], one)
    keep1 = state1["mask"].any(-1)
    episodes = episodes[keep1]
    starts = starts[keep1]
    state1 = {key: value[keep1] for key, value in state1.items()}
    action1 = model(state1).masked_fill(~state1["mask"], -torch.inf).argmax(-1)

    history2 = torch.stack((starts, action1), 1)
    two = torch.full((episodes.shape[0],), 2, dtype=torch.long, device=episodes.device)
    state2 = build_angle_state(raw, episodes, history2, two)
    keep2 = state2["mask"].any(-1)
    episodes = episodes[keep2]
    starts = starts[keep2]
    action1 = action1[keep2]
    history2 = history2[keep2]
    state2 = {key: value[keep2] for key, value in state2.items()}
    action2 = model(state2).masked_fill(~state2["mask"], -torch.inf).argmax(-1)

    history3 = torch.stack((history2[:, 0], history2[:, 1], action2), 1)
    three = torch.full((episodes.shape[0],), 3, dtype=torch.long, device=episodes.device)
    state3 = build_angle_state(raw, episodes, history3, three)
    q3 = model(state3)
    mask3 = state3["mask"]
    future_present = mask3.any(-1)
    future_action = q3.masked_fill(~mask3, -torch.inf).argmax(-1)
    future_q = q3.gather(1, future_action[:, None]).squeeze(1)
    return {
        "episodes": episodes,
        "starts": starts,
        "action1": action1,
        "action2": action2,
        "future_action": future_action,
        "future_present": future_present,
        "future_q": future_q,
        "mask3": mask3,
    }


def evaluate_angle_protocol(
    model: nn.Module,
    raw: GPUCache,
    classes: Sequence[int],
    protocol: str,
    seed: int,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    bank = class_bank_tensor(classes, raw.device)
    eligible = torch.isin(raw.labels, bank) & (raw.valid.sum(-1) >= 3)
    episodes = eligible.nonzero().flatten()
    if protocol.startswith("fixed"):
        start = int(protocol.removeprefix("fixed"))
        episodes = (eligible & raw.valid[:, start]).nonzero().flatten()
        starts = torch.full_like(episodes, start)
    elif protocol == "random_start":
        starts = choose_random_starts(raw, episodes, seed)
    else:
        raise ValueError(protocol)

    single = _single_metrics(raw, episodes, starts, bank) if len(episodes) else {"top1": 0.0}
    rollout = policy_rollout_angle(model, raw, episodes, starts)
    policy_episodes = rollout["episodes"]
    policy_paths = torch.stack(
        (rollout["starts"], rollout["action1"], rollout["action2"]), 1
    )
    policy = (
        metrics_for_paths(raw, policy_episodes, policy_paths, bank)
        if len(policy_episodes)
        else {"top1": 0.0, "mean_entropy": 0.0, "mean_movement_cost": 0.0, "episodes": 0.0}
    )
    random_exact, random_store = random_exact_metrics(raw, episodes, starts, bank)
    random_ep, random_paths, _ = sample_random_paths(raw, episodes, starts, seed + 17)
    random_sample = (
        metrics_for_paths(raw, random_ep, random_paths, bank)
        if len(random_ep)
        else {"top1": 0.0, "mean_entropy": 0.0, "mean_movement_cost": 0.0, "episodes": 0.0}
    )
    oracle_ep, oracle_path = oracle_paths(raw, episodes, starts, bank)
    oracle = (
        metrics_for_paths(raw, oracle_ep, oracle_path, bank)
        if len(oracle_ep)
        else {"top1": 0.0, "mean_entropy": 0.0, "mean_movement_cost": 0.0, "episodes": 0.0}
    )
    all_valid = raw.valid[episodes]
    all_weights = all_valid.float().unsqueeze(-1)
    all_scores = (raw.logits[episodes] * all_weights).sum(1) / all_weights.sum(1).clamp_min(1.0)
    all_scores = all_scores.index_select(-1, bank)
    all_local = torch.searchsorted(bank, raw.labels[episodes])
    all_top1 = all_scores.argmax(-1).eq(all_local)
    future_legal = rollout["mask3"].gather(1, rollout["future_action"][:, None]).squeeze(1)
    report: dict[str, Any] = {
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
            "all_valid_views": {"top1": float(all_top1.float().mean()) if len(episodes) else 0.0},
        },
        "future_fourth_prediction": {
            "episodes_after_three_observed": int(len(policy_episodes)),
            "remaining_view_present_rate": float(rollout["future_present"].float().mean()) if len(policy_episodes) else 0.0,
            "predicted_action_legal_rate": float(future_legal.float().mean()) if len(policy_episodes) else 0.0,
            "mean_predicted_q": float(rollout["future_q"].mean()) if len(policy_episodes) else 0.0,
        },
        "starts": starts.cpu().tolist(),
        "policy_paths": policy_paths.cpu().tolist(),
        "policy_future_actions": rollout["future_action"].cpu().tolist(),
        "episode_ids": policy_episodes.cpu().tolist(),
    }
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_dir / f"paths_{protocol}.npz",
            episodes=policy_episodes.cpu().numpy(),
            starts=rollout["starts"].cpu().numpy(),
            action1=rollout["action1"].cpu().numpy(),
            action2=rollout["action2"].cpu().numpy(),
            future_action=rollout["future_action"].cpu().numpy(),
            future_present=rollout["future_present"].cpu().numpy(),
            random_exact_paths=random_store.get("paths", np.empty((0, 3), dtype=np.int64)),
            random_exact_probabilities=random_store.get("probabilities", np.empty(0, dtype=np.float32)),
        )
    return report


def train_one_seed(
    seed: int,
    train_raw: GPUCache,
    val_raw: GPUCache,
    device: torch.device,
    epochs: int,
    batch_size: int,
    updates_per_epoch: int,
    learning_rate: float,
    output_root: Path,
) -> dict[str, Any]:
    run = output_root / f"seed_{seed}"
    run.mkdir(parents=True, exist_ok=True)
    complete = run / "complete.json"
    if complete.exists():
        return json.loads(complete.read_text(encoding="utf-8"))

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    generator = torch.Generator(device=device).manual_seed(seed)
    model = AngleObjectPolicy().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.04)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, epochs, eta_min=learning_rate * 0.05
    )
    seen_bank = class_bank_tensor(SEEN_CLASSES, device)
    eligible = (train_raw.valid.sum(-1) == 4) & torch.isin(train_raw.labels, seen_bank)
    episode_pool = eligible.nonzero().flatten()
    if not len(episode_pool):
        raise RuntimeError("no four-view seen training episodes")
    audit = {
        "seed": seed,
        "training_episodes_four_view_seen": int(len(episode_pool)),
        "training_labels": SEEN_CLASSES,
        "pseudo_unseen_for_selection": PSEUDO_UNSEEN_CLASSES,
        "true_unseen_test_only": TRUE_UNSEEN_CLASSES,
        "policy_inputs_only": ["sample-specific relative angle geometry", "mean observed object detection map"],
        "policy_inputs_excluded": ["video z", "skeleton", "semantic belief", "current logits", "class ID", "future-view features"],
        "training_target": "offline seen-bank delta margin after adding candidate",
        "trajectory": "one observed -> second; two observed -> third; three observed -> fourth prediction only",
        "final_fusion": "mean logits of exactly three observed views",
        "movement_cost_in_reward": False,
        "slot_permutation": True,
    }
    dump(run / "audit.json", audit)
    best_score = -float("inf")
    best_epoch = 0
    for epoch in range(1, epochs + 1):
        model.train()
        begun = time.monotonic()
        losses: list[float] = []
        for _ in range(updates_per_epoch):
            ids = torch.randint(len(episode_pool), (batch_size,), device=device, generator=generator)
            episodes = episode_pool[ids]
            permutation = torch.rand((batch_size, NUM_VIEWS), device=device, generator=generator).argsort(-1)
            stage = torch.multinomial(
                torch.as_tensor([0.45, 0.45, 0.10], device=device),
                batch_size,
                replacement=True,
                generator=generator,
            ) + 1
            history = permutation[:, :3]
            state = build_angle_state(train_raw, episodes, history, stage)
            utility = utility_targets(
                train_raw, episodes, history, stage, state["mask"], seen_bank
            )
            state, utility = permute_candidates(state, utility, generator)
            prediction = model(state)
            loss, _ = multistep_loss(prediction, utility, state["mask"], stage)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss seed={seed} epoch={epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        torch.cuda.synchronize(device)
        validation = evaluate_angle_protocol(
            model, val_raw, PSEUDO_UNSEEN_CLASSES, "fixed0", seed + 90117
        )
        vm = validation["metrics"]
        score = float(vm["policy_3view"]["top1"] - vm["random_3view_exact"]["top1"])
        improved = score > best_score
        if improved:
            best_score = score
            best_epoch = epoch
        payload = {
            "online": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "generator": generator.get_state(),
            "epoch": epoch,
            "best_score": best_score,
            "best_epoch": best_epoch,
            "seed": seed,
            "policy_inputs": audit["policy_inputs_only"],
        }
        torch.save(payload, run / "last.pt")
        if improved:
            torch.save(payload, run / "best.pt")
        row = {
            "epoch": epoch,
            "seed": seed,
            "loss": float(np.mean(losses)),
            "seconds": time.monotonic() - begun,
            "samples_per_second": float(batch_size * updates_per_epoch / max(time.monotonic() - begun, 1e-6)),
            "validation_selection_score": score,
            "best_score": best_score,
            "best_epoch": best_epoch,
            "validation": vm,
        }
        with (run / "train.log").open("a", encoding="utf-8", buffering=1) as log:
            log.write(json.dumps(row) + "\n")
        print(json.dumps({k: row[k] for k in ("epoch", "loss", "seconds", "validation_selection_score", "best_score", "best_epoch")}), flush=True)
    result = {
        "seed": seed,
        "best_score": best_score,
        "best_epoch": best_epoch,
        "checkpoint": str(run / "best.pt"),
        "output": str(run),
        "training_episodes": int(len(episode_pool)),
    }
    dump(complete, result)
    return result


def final_evaluation(
    summaries: list[dict[str, Any]],
    val_raw: GPUCache,
    seen_raw: GPUCache,
    test_raw: GPUCache,
    device: torch.device,
    output_root: Path,
) -> None:
    chosen = max(summaries, key=lambda item: float(item["best_score"]))
    model = AngleObjectPolicy().to(device)
    saved = torch.load(Path(chosen["checkpoint"]), map_location=device, weights_only=False)
    model.load_state_dict(saved["online"])
    model.eval()
    selection = {
        "criterion": "validation pseudo-unseen fixed0 3-view gain",
        "chosen": chosen,
        "runs": summaries,
    }
    dump(output_root / "selection.json", selection)
    reports: dict[str, Any] = {}
    for split_name, raw, classes in (
        ("val_pseudo_unseen", val_raw, PSEUDO_UNSEEN_CLASSES),
        ("seen_test", seen_raw, SEEN_CLASSES),
        ("true_unseen_5way", test_raw, TRUE_UNSEEN_CLASSES),
    ):
        split_dir = output_root / split_name
        reports[split_name] = {}
        for protocol in ("fixed0", "fixed1", "fixed2", "fixed3", "random_start"):
            report = evaluate_angle_protocol(model, raw, classes, protocol, 20260910 + len(protocol), split_dir)
            report["selected_seed"] = int(chosen["seed"])
            reports[split_name][protocol] = report
            dump(split_dir / f"evaluation_{protocol}.json", report)
            print("ANGLE_OBJECT_EVALUATION", split_name, protocol, json.dumps(report["metrics"]), flush=True)
    dump(
        output_root / "summary.json",
        {
            "variant": "multistep_angle_object_policy_v1",
            "base_comparison": "multistep_g18_view_policy_v1",
            "selection": selection,
            "policy_inputs": ["sample-specific relative angle geometry", "mean observed object detection map"],
            "policy_inputs_excluded": ["video z", "skeleton", "semantic belief", "current logits", "class ID", "future-view features"],
            "reports": reports,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--updates-per-epoch", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260909, 20260910, 20260911])
    parser.add_argument("--unseen-classes", type=int, nargs="+", default=None)
    parser.add_argument("--pseudo-unseen-classes", type=int, nargs="+", default=None)
    args = parser.parse_args()
    if args.unseen_classes is not None:
        if args.pseudo_unseen_classes is None:
            raise ValueError("--pseudo-unseen-classes is required with --unseen-classes")
        unseen = sorted(set(int(x) for x in args.unseen_classes))
        pseudo = sorted(set(int(x) for x in args.pseudo_unseen_classes))
        seen = sorted(set(range(55)) - set(unseen))
        if not unseen or any(x < 0 or x >= 55 for x in unseen):
            raise ValueError(f"invalid unseen class IDs: {unseen}")
        if not pseudo or not set(pseudo).issubset(set(seen)):
            raise ValueError(f"pseudo-unseen classes must be a non-empty subset of seen: {pseudo}")
        global SEEN_CLASSES, PSEUDO_UNSEEN_CLASSES, TRUE_UNSEEN_CLASSES
        SEEN_CLASSES = seen
        PSEUDO_UNSEEN_CLASSES = pseudo
        TRUE_UNSEEN_CLASSES = unseen
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this experiment")
    device = torch.device("cuda:0")
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    args.output_root.mkdir(parents=True, exist_ok=True)
    dump(
        args.output_root / "config_resolved.json",
        {
            "cache_root": str(args.cache_root),
            "output_root": str(args.output_root),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "updates_per_epoch": args.updates_per_epoch,
            "learning_rate": args.learning_rate,
            "seeds": args.seeds,
            "base_comparison": "multistep_g18_view_policy_v1",
            "seen_classes": SEEN_CLASSES,
            "pseudo_unseen_classes": PSEUDO_UNSEEN_CLASSES,
            "unseen_classes": TRUE_UNSEEN_CLASSES,
        },
    )
    raw = {}
    for split in ("dqn_train", "val", "seen_test", "test"):
        raw[split] = GPUCache(args.cache_root / split, device)
        attach_view_valid(raw[split], args.cache_root, split)
    print("ANGLE_OBJECT_CACHE_SHAPES", json.dumps({key: value.num_episodes for key, value in raw.items()}), flush=True)
    summaries: list[dict[str, Any]] = []
    for seed in args.seeds:
        summaries.append(
            train_one_seed(
                seed,
                raw["dqn_train"],
                raw["val"],
                device,
                args.epochs,
                args.batch_size,
                args.updates_per_epoch,
                args.learning_rate,
                args.output_root,
            )
        )
        torch.cuda.empty_cache()
    dump(args.output_root / "training_summary.json", summaries)
    final_evaluation(summaries, raw["val"], raw["seen_test"], raw["test"], device, args.output_root)
    print("ANGLE_OBJECT_TRAIN_EVAL_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
