"""DDQN whose terminal reward is computed from the fused two-view decision.

The recognizer and the weighted-logit fusion gate are frozen.  For every
candidate pair, the terminal reward is computed from the *fused* class logits,
not from either robot's single-view prediction.  It contains four recognition
and geometry terms:

* negative cross-entropy of the fused two-view decision;
* negative normalized entropy of the fused two-view decision;
* the fused ground-truth class margin;
* the normalized geometric separation of the two target viewpoints;
* a soft penalty for selecting adjacent rank views.

The assignment movement cost is subtracted separately.  Adjacent rank pairs
remain legal by default and receive a configurable soft penalty; an optional
hard mask is retained only for reproducing earlier ablations.  The geometric
separation term still uses the actual sine/cosine view directions.
The same reward is used for Full, Human-only, Object-only, and LSTM variants.

This module deliberately reports three different quantities:

* Bellman loss: the DDQN regression diagnostic during training;
* recognition loss: cross-entropy of the selected fused logits;
* accuracy: the final unseen-class evaluation metric.

Bellman loss is not used as a substitute for test accuracy.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import sequential as seq


FAR_VARIANTS = {
    "full_depth19",
    "human_only",
    "object_only_depth19",
    "full_lstm19",
    "human_only_lstm",
    "object_only_lstm19",
}
MIN_RANK_GAP = int(os.environ.get("FAR_MIN_RANK_GAP", "2"))
HARD_NONADJACENT = os.environ.get("FAR_HARD_NONADJACENT", "0") in {"1", "true", "True"}
ADJACENCY_PENALTY = float(os.environ.get("FAR_ADJACENCY_PENALTY", "0.05"))
MARGIN_WEIGHT = float(os.environ.get("FAR_MARGIN_WEIGHT", "1.0"))
LOSS_WEIGHT = float(os.environ.get("FAR_LOSS_WEIGHT", "1.0"))
ENTROPY_WEIGHT = float(os.environ.get("FAR_ENTROPY_WEIGHT", "0.5"))
SEPARATION_WEIGHT = float(os.environ.get("FAR_SEPARATION_WEIGHT", "0.5"))
MARGIN_SCALE = float(os.environ.get("FAR_MARGIN_SCALE", "2.0"))
LOSS_SCALE = float(os.environ.get("FAR_LOSS_SCALE", "1.0"))
ENTROPY_SCALE = float(os.environ.get("FAR_ENTROPY_SCALE", "1.0"))
MARGIN_TANH = os.environ.get("FAR_MARGIN_TANH", "1") not in {"0", "false", "False"}
MOVEMENT_RATIO_LIMIT = float(os.environ.get("FAR_MOVEMENT_RATIO_LIMIT", "0.5"))
VALIDATION_SELECTION = os.environ.get("FAR_VALIDATION_SELECTION", "reward").lower()

_original_build_state = seq.b.build_state


def _nonadjacent_mask(device: torch.device) -> torch.Tensor:
    ranks = torch.arange(seq.b.NUM_VIEWS, device=device)
    return (ranks[:, None] - ranks[None, :]).abs() >= MIN_RANK_GAP


def _constrain_state(raw, episodes, start_a, start_b, selected, variant):
    state = _original_build_state(raw, episodes, start_a, start_b, selected, variant)
    if (
        variant not in FAR_VARIANTS
        or MIN_RANK_GAP <= 0
        or not HARD_NONADJACENT
    ):
        return state

    pair_mask = _nonadjacent_mask(raw.device)[None]
    rows = torch.arange(len(episodes), device=raw.device)
    _, pair_feasible = seq.b.assignment_matrix(raw, episodes, start_a, start_b)
    present = selected >= 0
    allowed = torch.zeros_like(state["mask"], dtype=torch.bool)

    stage0 = ~present
    if stage0.any():
        stage0_rows = rows[stage0]
        feasible = pair_feasible[stage0_rows] & pair_mask
        allowed[stage0] = feasible.any(dim=-1)

    stage1 = present
    if stage1.any():
        stage1_rows = rows[stage1]
        feasible = pair_feasible[stage1_rows] & pair_mask
        selected_stage1 = selected[stage1]
        allowed[stage1] = feasible[
            torch.arange(len(selected_stage1), device=raw.device), selected_stage1
        ]

    state["mask"] = state["mask"] & allowed
    return state


def _filter_nonadjacent(pool, raw):
    """Apply the optional hard pair constraint to transition targets."""

    if MIN_RANK_GAP <= 0 or not HARD_NONADJACENT:
        return pool

    terminal_pair = (pool.selected - pool.action).abs() >= MIN_RANK_GAP
    stage1 = pool.stage1
    stage1_rows_all = torch.arange(pool.size, device=pool.action.device)[stage1]
    _, pair_feasible = seq.b.assignment_matrix(
        raw,
        pool.episode[stage1],
        pool.start_a[stage1],
        pool.start_b[stage1],
    )
    feasible = pair_feasible & _nonadjacent_mask(raw.device)[None]
    selected_stage1 = pool.action[stage1]
    stage1_keep = feasible[
        torch.arange(len(selected_stage1), device=pool.action.device), selected_stage1
    ].any(dim=-1)

    keep = torch.zeros_like(stage1)
    keep[stage1] = stage1_keep
    keep[~stage1] = terminal_pair[~stage1]

    fields = (
        "episode",
        "start_a",
        "start_b",
        "selected",
        "action",
        "reward",
        "done",
        "stage1",
        "next_episode",
        "next_a",
        "next_b",
        "next_selected",
    )
    for name in fields:
        if hasattr(pool, name):
            setattr(pool, name, getattr(pool, name)[keep])
    return pool


def _normalized_entropy(scores: torch.Tensor) -> torch.Tensor:
    """Categorical entropy normalized to [0, 1] for the declared class bank."""

    probabilities = torch.softmax(scores, dim=-1)
    entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(dim=-1)
    return entropy / math.log(float(scores.shape[-1]))


def _terminal_statistics(
    raw,
    fused_pairs,
    episodes: torch.Tensor,
    target_a: torch.Tensor,
    target_b: torch.Tensor,
    bank_values,
):
    """Compute all terms used by the fused terminal reward and evaluation."""

    if len(episodes) == 0:
        zero = torch.zeros(0, device=raw.device)
        return {
            "fused_loss": zero,
            "best_single_loss": zero,
            "loss_gain": zero,
            "fused_entropy": zero,
            "best_single_entropy": zero,
            "entropy_gain": zero,
            "margin": zero,
            "margin_score": zero,
            "top2_margin": zero,
            "separation": zero,
            "separation_angle_deg": zero,
            "rank_gap": zero,
        }

    pair_index = torch.as_tensor(seq.b.PAIR_INDEX, device=raw.device)
    pair_ids = pair_index[target_a, target_b]
    pair_logits = fused_pairs[episodes, pair_ids]
    bank = torch.as_tensor(bank_values, device=raw.device, dtype=torch.long)
    labels = raw.labels[episodes]
    local = torch.searchsorted(bank, labels)
    pair_scores = pair_logits.index_select(-1, bank)
    fused_loss = F.cross_entropy(pair_scores, local, reduction="none")
    fused_entropy = _normalized_entropy(pair_scores)
    margin = seq.b.pair_margin(pair_logits, labels, bank_values)
    top2_margin = pair_scores.topk(k=min(2, pair_scores.shape[-1]), dim=-1).values
    top2_margin = top2_margin[:, 0] - top2_margin[:, 1]

    single_scores = raw.logits[episodes].index_select(-1, bank)
    repeated_labels = local[:, None].expand(-1, single_scores.shape[1])
    single_loss = F.cross_entropy(
        single_scores.reshape(-1, single_scores.shape[-1]),
        repeated_labels.reshape(-1),
        reduction="none",
    ).reshape(single_scores.shape[:2])
    best_single_loss = single_loss.min(dim=-1).values
    single_entropy = _normalized_entropy(single_scores)
    best_single_entropy = single_entropy.min(dim=-1).values

    geometry = raw.geometry[episodes, :, :2]
    left = geometry[torch.arange(len(episodes), device=raw.device), target_a]
    right = geometry[torch.arange(len(episodes), device=raw.device), target_b]
    cosine = (left * right).sum(dim=-1).clamp(-1.0, 1.0)
    separation_angle = torch.acos(cosine)
    # Chord length on the unit view circle, normalized so opposite views = 1.
    separation = torch.sqrt((2.0 - 2.0 * cosine).clamp_min(0.0)) * 0.5
    rank_gap = (target_a - target_b).abs().float()
    if MARGIN_TANH:
        margin_score = torch.tanh(margin / max(MARGIN_SCALE, 1e-6))
    else:
        margin_score = margin / max(MARGIN_SCALE, 1e-6)
    return {
        "fused_loss": fused_loss,
        "best_single_loss": best_single_loss,
        "loss_gain": (best_single_loss - fused_loss) / max(LOSS_SCALE, 1e-6),
        "fused_entropy": fused_entropy,
        "best_single_entropy": best_single_entropy,
        "entropy_gain": (best_single_entropy - fused_entropy) / max(ENTROPY_SCALE, 1e-6),
        "margin": margin,
        "margin_score": margin_score,
        "top2_margin": top2_margin,
        "separation": separation,
        "separation_angle_deg": separation_angle * (180.0 / math.pi),
        "rank_gap": rank_gap,
    }


def _pair_statistics(raw, fused, rollout, bank_values):
    episodes = rollout["episodes"]
    target_a = rollout["target_a"]
    target_b = rollout["target_b"]
    if len(episodes) == 0:
        zero = torch.zeros(0, device=raw.device)
        return {
            "fused_loss": zero,
            "best_single_loss": zero,
            "loss_gain": zero,
            "fused_entropy": zero,
            "best_single_entropy": zero,
            "entropy_gain": zero,
            "margin": zero,
            "margin_score": zero,
            "top2_margin": zero,
            "separation": zero,
            "separation_angle_deg": zero,
            "rank_gap": zero,
            "movement": zero,
        }

    stats = _terminal_statistics(
        raw, fused, episodes, target_a, target_b, bank_values
    )
    movement_matrix, _ = seq.b.assignment_matrix(
        raw, episodes, rollout["start_a"], rollout["start_b"]
    )
    rows = torch.arange(len(episodes), device=raw.device)
    movement = movement_matrix[rows, target_a, target_b]
    movement = seq.b.finite_float(movement, fill=1.0)
    stats["movement"] = movement
    return stats


def _combined_reward(stats, distance_lambda: float):
    # The Q target is supervised by the *fused* recognition decision.  The
    # single-view quantities are retained for analysis only; they must not
    # define the terminal classification objective.
    fused_loss_reward = -stats["fused_loss"] / max(LOSS_SCALE, 1e-6)
    fused_entropy_reward = -stats["fused_entropy"] / max(ENTROPY_SCALE, 1e-6)
    reward = (
        MARGIN_WEIGHT * stats["margin_score"]
        + LOSS_WEIGHT * fused_loss_reward
        + ENTROPY_WEIGHT * fused_entropy_reward
        + SEPARATION_WEIGHT * stats["separation"]
        - distance_lambda * stats["movement"]
    )
    if not HARD_NONADJACENT and MIN_RANK_GAP > 0:
        adjacent = (stats["rank_gap"] < MIN_RANK_GAP).to(reward.dtype)
        reward = reward - ADJACENCY_PENALTY * adjacent
    return reward


def pool(raw, fused, penalty, device):
    base = seq.pool(raw, fused, penalty, device)
    base = _filter_nonadjacent(base, raw)
    terminal = ~base.stage1
    if terminal.any():
        rollout = {
            "episodes": base.episode[terminal],
            "start_a": base.start_a[terminal],
            "start_b": base.start_b[terminal],
            "target_a": base.selected[terminal],
            "target_b": base.action[terminal],
        }
        stats = _pair_statistics(
            raw, fused, rollout, seq.b.POLICY_TRAIN_CLASSES
        )
        reward = _combined_reward(stats, penalty)
        base.reward = base.reward.clone()
        base.reward[terminal] = reward
    return base


def _evaluate_rollout(raw, fused, rollout, bank, distance_lambda):
    metrics = seq.b.pair_metrics(raw, fused, rollout, bank)
    stats = _pair_statistics(raw, fused, rollout, bank)
    metrics["recognition_loss"] = float(stats["fused_loss"].mean()) if len(stats["fused_loss"]) else 0.0
    metrics["best_single_recognition_loss"] = float(stats["best_single_loss"].mean()) if len(stats["best_single_loss"]) else 0.0
    metrics["recognition_loss_gain"] = float(stats["loss_gain"].mean()) if len(stats["loss_gain"]) else 0.0
    metrics["normalized_entropy"] = float(stats["fused_entropy"].mean()) if len(stats["fused_entropy"]) else 0.0
    metrics["best_single_normalized_entropy"] = float(stats["best_single_entropy"].mean()) if len(stats["best_single_entropy"]) else 0.0
    metrics["entropy_gain"] = float(stats["entropy_gain"].mean()) if len(stats["entropy_gain"]) else 0.0
    metrics["mean_margin"] = float(stats["margin"].mean()) if len(stats["margin"]) else 0.0
    metrics["mean_top2_margin"] = float(stats["top2_margin"].mean()) if len(stats["top2_margin"]) else 0.0
    metrics["mean_separation"] = float(stats["separation"].mean()) if len(stats["separation"]) else 0.0
    metrics["mean_separation_angle_deg"] = float(stats["separation_angle_deg"].mean()) if len(stats["separation_angle_deg"]) else 0.0
    metrics["mean_rank_gap"] = float(stats["rank_gap"].mean()) if len(stats["rank_gap"]) else 0.0
    metrics["mean_reward"] = float(_combined_reward(stats, distance_lambda).mean()) if len(stats["margin"]) else 0.0
    return metrics


@torch.inference_mode()
def replay_rollout(raw, model, variant, seed, bank, nonadjacent=False):
    """Return selected paths instead of only the scalar replay metrics."""

    episodes = seq.b.eligible_episodes(raw, bank).nonzero().flatten()
    starts = torch.cartesian_prod(
        torch.arange(seq.b.NUM_VIEWS, device=raw.device),
        torch.arange(seq.b.NUM_VIEWS, device=raw.device),
    )
    episode_batch = episodes.repeat_interleave(seq.b.NUM_VIEWS ** 2)
    start_a_batch = starts[:, 0].repeat(len(episodes))
    start_b_batch = starts[:, 1].repeat(len(episodes))
    lookup = {}
    if model is not None:
        out = seq.b.dqn_rollout(
            model,
            raw,
            episode_batch,
            start_a_batch,
            start_b_batch,
            variant,
        )
        for values in zip(
            *(out[key].tolist() for key in ("episodes", "start_a", "start_b", "target_a", "target_b"))
        ):
            lookup[values[0], values[1], values[2]] = (values[3], values[4])

    rng = np.random.default_rng(seed)
    rows = {key: [] for key in ("episodes", "start_a", "start_b", "target_a", "target_b")}
    for stream in seq.streams(raw, episodes.tolist(), seed):
        start_a, start_b = rng.choice(seq.b.NUM_VIEWS, 2, replace=False).tolist()
        previous_a = previous_b = None
        for step, episode in enumerate(stream):
            if step:
                cameras = raw.camera_ids[episode]
                start_a = cameras.index(previous_a)
                start_b = cameras.index(previous_b)
            if model is not None:
                target_a, target_b = lookup[episode, start_a, start_b]
            else:
                _, feasible_matrix = seq.b.assignment_matrix(
                    raw,
                    torch.as_tensor([episode], device=raw.device),
                    torch.as_tensor([start_a], device=raw.device),
                    torch.as_tensor([start_b], device=raw.device),
                )
                feasible = feasible_matrix[0].detach().cpu().numpy().astype(bool)
                legal = [
                    pair_id
                    for pair_id, (left, right) in enumerate(seq.b.PAIR_LIST)
                    if feasible[left, right]
                    and (not nonadjacent or abs(left - right) >= MIN_RANK_GAP)
                ]
                if not legal:
                    continue
                if variant == "random_pair" or variant == "random_nonadjacent":
                    pair_id = int(rng.choice(legal))
                    target_a, target_b = seq.b.PAIR_LIST[pair_id]
                else:
                    cameras = sorted(raw.camera_ids[episode])
                    offset = 1 if variant == "cyclic_adjacent_pair" else 2
                    wanted = tuple(
                        sorted(
                            (
                                raw.camera_ids[episode].index(cameras[step % 4]),
                                raw.camera_ids[episode].index(cameras[(step + offset) % 4]),
                            )
                        )
                    )
                    pair_id = int(seq.b.PAIR_INDEX[wanted])
                    if pair_id not in legal:
                        pair_id = legal[(seed + step) % len(legal)]
                    target_a, target_b = seq.b.PAIR_LIST[pair_id]
            rows["episodes"].append(episode)
            rows["start_a"].append(start_a)
            rows["start_b"].append(start_b)
            rows["target_a"].append(target_a)
            rows["target_b"].append(target_b)
            assigned = seq.assign(raw, episode, start_a, start_b, target_a, target_b)
            previous_a = raw.camera_ids[episode][assigned[0]]
            previous_b = raw.camera_ids[episode][assigned[1]]
    return {
        key: torch.as_tensor(value, device=raw.device, dtype=torch.long)
        for key, value in rows.items()
    }


def replay_nonadjacent_random(raw, fused, seed, bank, mode="random_nonadjacent"):
    """Persistent-stream random baseline restricted to non-adjacent pairs."""

    episodes = seq.b.eligible_episodes(raw, bank).nonzero().flatten()
    rng = np.random.default_rng(seed)
    rows = {key: [] for key in ("episodes", "start_a", "start_b", "target_a", "target_b")}
    for stream in seq.streams(raw, episodes.tolist(), seed):
        start_a, start_b = rng.choice(seq.b.NUM_VIEWS, 2, replace=False).tolist()
        previous_a = previous_b = None
        for step, episode in enumerate(stream):
            if step:
                cameras = raw.camera_ids[episode]
                start_a = cameras.index(previous_a)
                start_b = cameras.index(previous_b)
            _, feasible_matrix = seq.b.assignment_matrix(
                raw,
                torch.as_tensor([episode], device=raw.device),
                torch.as_tensor([start_a], device=raw.device),
                torch.as_tensor([start_b], device=raw.device),
            )
            feasible = feasible_matrix[0].detach().cpu().numpy().astype(bool)
            legal = [
                pair_id
                for pair_id, (left, right) in enumerate(seq.b.PAIR_LIST)
                if feasible[left, right] and abs(left - right) >= MIN_RANK_GAP
            ]
            if not legal:
                continue
            if mode == "random_nonadjacent":
                pair_id = int(rng.choice(legal))
            else:
                raise ValueError(mode)
            left, right = seq.b.PAIR_LIST[pair_id]
            rows["episodes"].append(episode)
            rows["start_a"].append(start_a)
            rows["start_b"].append(start_b)
            rows["target_a"].append(left)
            rows["target_b"].append(right)
            assigned = seq.assign(raw, episode, start_a, start_b, left, right)
            previous_a, previous_b = raw.camera_ids[episode][assigned[0]], raw.camera_ids[episode][assigned[1]]
    return {key: torch.as_tensor(value, device=raw.device, dtype=torch.long) for key, value in rows.items()}


def validation(model, raw, fused, episodes, starts_a, starts_b, variant, bank, penalty):
    rollout = replay_rollout(raw, model, variant, 731, bank, nonadjacent=True)
    result = _evaluate_rollout(raw, fused, rollout, bank, penalty)
    random_rollout = replay_rollout(raw, None, "random_pair", 731, bank)
    random_metrics = _evaluate_rollout(raw, fused, random_rollout, bank, penalty)
    random_cost = max(float(random_metrics["mean_movement_cost"]), 1e-8)
    result["movement_ratio_to_random"] = float(result["mean_movement_cost"]) / random_cost
    # A checkpoint must satisfy the movement budget.  The default selection
    # criterion is the RL objective; validation accuracy or fused CE can be
    # requested explicitly without using test labels.
    in_budget = result["movement_ratio_to_random"] <= MOVEMENT_RATIO_LIMIT
    rl_reward = result["mean_reward"]
    if not in_budget:
        selection_score = -result["movement_ratio_to_random"]
    elif VALIDATION_SELECTION == "accuracy":
        selection_score = result["accuracy"]
    elif VALIDATION_SELECTION in {"loss", "recognition_loss", "ce"}:
        selection_score = -result["recognition_loss"]
    else:
        selection_score = rl_reward
    result["rl_reward"] = rl_reward
    result["selection_score"] = selection_score
    result["selection_metric"] = VALIDATION_SELECTION
    result["mean_reward"] = selection_score
    return result


def evaluation(raw, fused, policies, variants, seed, bank):
    methods = {}
    for variant in variants:
        rollout = replay_rollout(raw, policies[variant], variant, seed, bank, nonadjacent=True)
        methods[variant] = _evaluate_rollout(raw, fused, rollout, bank, 0.0)
    random_rollout = replay_rollout(raw, None, "random_pair", seed, bank)
    methods["random_pair"] = _evaluate_rollout(raw, fused, random_rollout, bank, 0.0)
    nonadj_random = replay_rollout(raw, None, "random_nonadjacent", seed, bank, nonadjacent=True)
    methods["random_nonadjacent"] = _evaluate_rollout(raw, fused, nonadj_random, bank, 0.0)
    methods["cyclic_pair"] = _evaluate_rollout(
        raw, fused, replay_rollout(raw, None, "cyclic_pair", seed, bank), bank, 0.0
    )
    methods["cyclic_adjacent_pair"] = _evaluate_rollout(
        raw, fused, replay_rollout(raw, None, "cyclic_adjacent_pair", seed, bank), bank, 0.0
    )
    return {
        "seed": seed,
        "bank": bank,
        "methods": methods,
        "protocol": (
            "subject-wise synthetic sequential replay with soft adjacent-view "
            "DDQN penalty"
        ),
    }


def summarize_seed_results(seed_results):
    methods = sorted({method for row in seed_results for method in row["methods"]})
    summary = {}
    for method in methods:
        rows = [row["methods"][method] for row in seed_results]
        summary[method] = {
            "accuracy": seq.b.ci_summary([float(row["accuracy"]) for row in rows]),
            "mean_movement_cost": seq.b.ci_summary(
                [float(row["mean_movement_cost"]) for row in rows]
            ),
            "mean_movement_angle_deg": seq.b.ci_summary(
                [float(row["mean_movement_angle_deg"]) for row in rows]
            ),
            "recognition_loss": seq.b.ci_summary(
                [float(row["recognition_loss"]) for row in rows]
            ),
            "recognition_loss_gain": seq.b.ci_summary(
                [float(row["recognition_loss_gain"]) for row in rows]
            ),
            "normalized_entropy": seq.b.ci_summary(
                [float(row["normalized_entropy"]) for row in rows]
            ),
            "best_single_normalized_entropy": seq.b.ci_summary(
                [float(row["best_single_normalized_entropy"]) for row in rows]
            ),
            "entropy_gain": seq.b.ci_summary(
                [float(row["entropy_gain"]) for row in rows]
            ),
            "mean_margin": seq.b.ci_summary([float(row["mean_margin"]) for row in rows]),
            "mean_top2_margin": seq.b.ci_summary(
                [float(row["mean_top2_margin"]) for row in rows]
            ),
            "mean_separation": seq.b.ci_summary(
                [float(row["mean_separation"]) for row in rows]
            ),
            "mean_separation_angle_deg": seq.b.ci_summary(
                [float(row["mean_separation_angle_deg"]) for row in rows]
            ),
            "mean_rank_gap": seq.b.ci_summary(
                [float(row["mean_rank_gap"]) for row in rows]
            ),
            "mean_reward": seq.b.ci_summary([float(row["mean_reward"]) for row in rows]),
            "completion_rate": seq.b.ci_summary(
                [float(row["completion_rate"]) for row in rows]
            ),
            "episodes_per_seed": seq.b.ci_summary(
                [float(row["episodes"]) for row in rows]
            ),
        }
    return summary


def _training_loss_summary(root: Path):
    output = {}
    for selection_path in sorted(root.glob("selection_*.json")):
        variant = selection_path.stem.removeprefix("selection_")
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        variant_output = {}
        for run in selection.get("runs", []):
            log_path = root / "models" / variant / f"seed_{run['seed']}" / "train.log"
            if not log_path.exists():
                continue
            rows = [
                json.loads(line)
                for line in log_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            matching = [
                row
                for row in rows
                if int(row.get("epoch", -1)) == int(run["best_epoch"])
            ]
            if matching:
                validation = matching[-1].get("validation", {})
                variant_output[str(run["seed"])] = {
                    "best_epoch": int(run["best_epoch"]),
                    "bellman_loss": float(matching[-1]["loss"]),
                    "validation_recognition_loss": float(
                        validation.get("recognition_loss", 0.0)
                    ),
                    "validation_entropy": float(
                        validation.get("normalized_entropy", 0.0)
                    ),
                    "validation_margin": float(
                        validation.get("mean_margin", 0.0)
                    ),
                    "validation_selection_score": float(
                        validation.get(
                            "selection_score", validation.get("mean_reward", 0.0)
                        )
                    ),
                    "validation_rl_reward": float(
                        validation.get(
                            "rl_reward", validation.get("mean_reward", 0.0)
                        )
                    ),
                }
        output[variant] = variant_output
    return output


def report(root, metadata, summary):
    if HARD_NONADJACENT:
        metadata["pair_constraint"] = (
            f"hard rank gap >= {MIN_RANK_GAP}: adjacent pairs masked"
        )
    else:
        metadata["pair_constraint"] = (
            f"soft adjacent-pair penalty: rank gap < {MIN_RANK_GAP} remains legal"
        )
    metadata["hard_nonadjacent"] = HARD_NONADJACENT
    metadata["adjacency_penalty"] = ADJACENCY_PENALTY
    metadata["checkpoint_selection"] = (
        "validation accuracy subject to movement ratio limit"
        if VALIDATION_SELECTION == "accuracy"
        else "validation fused CE subject to movement ratio limit"
        if VALIDATION_SELECTION in {"loss", "recognition_loss", "ce"}
        else "validation RL reward subject to movement ratio limit"
    )
    metadata["movement_ratio_limit"] = MOVEMENT_RATIO_LIMIT
    metadata["reward"] = (
        "fused two-view reward: negative fused CE + fused entropy penalty + "
        "fused ground-truth margin + target-view separation - movement cost"
    )
    margin_expression = (
        "tanh(fused_ground_truth_margin / margin_scale)"
        if MARGIN_TANH
        else "fused_ground_truth_margin / margin_scale"
    )
    metadata["reward_design"] = {
        "terminal_reward": (
            f"margin_weight * {margin_expression} + "
            "loss_weight * (-fused_pair_CE) / loss_scale + "
            "entropy_weight * (-fused_entropy) / entropy_scale + "
            "separation_weight * normalized_target_chord_separation - "
            "distance_lambda * assignment_cost"
        ),
        "margin_weight": MARGIN_WEIGHT,
        "loss_weight": LOSS_WEIGHT,
        "entropy_weight": ENTROPY_WEIGHT,
        "separation_weight": SEPARATION_WEIGHT,
        "margin_scale": MARGIN_SCALE,
        "margin_transform": "tanh(margin / margin_scale)" if MARGIN_TANH else "margin / margin_scale",
        "loss_scale": LOSS_SCALE,
        "entropy_scale": ENTROPY_SCALE,
        "recognition_loss": "cross-entropy over the seen-class bank during policy training and the declared test bank during evaluation",
        "recognition_source": "weighted fusion of the two selected per-view logits; fused CE is directly used by the terminal Q reward",
        "entropy": "normalized entropy of softmax(fused logits) over the declared bank; lower is better",
        "class_margin": "fused true-class logit minus the largest fused rival logit; higher is better",
        "target_separation": "chord separation from the two body-relative sine/cosine target directions; opposite views equal 1",
        "adjacency": (
            "hard mask"
            if HARD_NONADJACENT
            else f"subtract {ADJACENCY_PENALTY} when target rank gap < {MIN_RANK_GAP}"
        ),
        "checkpoint_selection": metadata["checkpoint_selection"],
        "bellman_loss_note": "logged as a DDQN regression diagnostic; not treated as test accuracy",
    }
    metadata["selected_training_loss"] = _training_loss_summary(root)
    seq.b.dump_json(root / "config_resolved.json", metadata)
    lines = [
        "# Far-view loss-aware DDQN results",
        "",
        f"Subject-wise sequential replay; {next(iter(summary.values()))['accuracy']['n'] if summary else 0} evaluation seeds; movement is an angular proxy.",
        "The policy is trained on seen classes and tested on the true unseen five-way bank.",
        "",
        "|Method|Accuracy|95% CI|Fused CE|Entropy|GT margin|Angular separation|Loss gain|Move cost|RL reward|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method, values in summary.items():
        accuracy = values["accuracy"]
        loss = values["recognition_loss"]
        entropy = values["normalized_entropy"]
        margin = values["mean_margin"]
        angle = values["mean_separation_angle_deg"]
        gain = values["recognition_loss_gain"]
        reward = values["mean_reward"]
        movement = values["mean_movement_cost"]
        separation = values["mean_separation"]
        lines.append(
            f"|{method}|{100 * accuracy['mean']:.2f}%|"
            f"[{100 * accuracy['ci95_low']:.2f}, {100 * accuracy['ci95_high']:.2f}]|"
            f"{loss['mean']:.4f}|{entropy['mean']:.4f}|{margin['mean']:.4f}|"
            f"{angle['mean']:.2f}|{gain['mean']:.4f}|{movement['mean']:.4f}|"
            f"{reward['mean']:.4f}|"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "* `Fused CE` and `Entropy` are lower-is-better; both are computed after weighted fusion of the selected two views.",
            "* The Q-learning terminal reward uses fused CE and fused entropy directly; single-view CE/entropy differences are diagnostics only.",
            "* `GT margin` and `Loss gain` are higher-is-better. `GT margin` uses the fused true class against the strongest fused rival.",
            "* `Angular separation` is the actual body-relative target-view angle in degrees; it is separate from the start-to-target movement cost.",
            "* `RL reward` is the configured training objective. Bellman loss is stored in `config_resolved.json` under `selected_training_loss`; it measures Q-target fitting, not recognition accuracy.",
            f"* Every scalar in the table is aggregated over {next(iter(summary.values()))['accuracy']['n'] if summary else 0} evaluation seeds with a 95% normal CI.",
            "",
        ]
    )
    (root / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    seq.b.load_raw = seq.load
    seq.b.build_state = _constrain_state
    seq.b.build_transition_pool = pool
    seq.b.evaluate_policy_once = validation
    seq.b.test_seed_evaluation = evaluation
    seq.b.summarize_seed_results = summarize_seed_results
    seq.b.write_results_markdown = report
    seq.b.main()
