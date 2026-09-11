"""Two-transition extension of the preserved g18 active-view policy.

The experiment is deliberately isolated from the historical v1--v9 and g18
files.  It keeps the best preserved g18 checkpoint as initialization and
trains the same low-rank skeleton/geometry scorer on causal multi-view
states:

    [v0] -> predict v1
    [v0, v1] -> predict v2
    [v0, v1, v2] -> predict/log only the future v3

Only the first three observations are fused for the final classification.
The fourth view is never read by the policy state or included in the final
score; it is retained only as a causal fourth-view prediction diagnostic.

The training target is an offline seen-class delta-margin utility.  Future
view logits are used only to construct this target, never as model inputs.
The final test uses fixed starts 0/1/2/3 and a deterministic random-start
protocol on both seen-test and true-unseen 5-way data.
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

from .train_rl_fast import GPUCache


ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = ROOT / "data/rl_candidate_rank_v6_new50_5/cache"
OUTPUT_ROOT = ROOT / "work_dir/multistep_g18_view_policy_v1"

BASE_CHECKPOINT = (
    ROOT / "work_dir/active_view_iterative_test/generation_18/"
    "g18_factorized_skeleton/seed_20260911/best.pt"
)

NUM_VIEWS = 4
NUM_CLASSES = 55
POSE_DIM = 442
SKELETON_DIM = 212
Z_DIM = 512

SEEN_CLASSES = [
    1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
    19, 20, 21, 22, 23, 24, 25, 27, 28, 29, 30, 31, 32, 33, 35, 36,
    37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 51, 52, 53,
    54,
]
PSEUDO_UNSEEN_CLASSES = [5, 6, 14, 22, 25, 38, 42, 44, 45, 51]
TRUE_UNSEEN_CLASSES = [0, 2, 26, 34, 50]


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def geometry_features(candidate: torch.Tensor) -> torch.Tensor:
    """The exact g18 candidate harmonic encoding (7 -> 11 fields)."""

    angle, delta = candidate[..., :2], candidate[..., 2:4]
    angle2 = torch.stack(
        (2 * angle[..., 0] * angle[..., 1], angle[..., 1].square() - angle[..., 0].square()),
        -1,
    )
    delta2 = torch.stack(
        (2 * delta[..., 0] * delta[..., 1], delta[..., 1].square() - delta[..., 0].square()),
        -1,
    )
    return torch.cat((candidate, angle2, delta2), -1)


def g18_skeleton_features(pose: torch.Tensor) -> torch.Tensor:
    """Recover the exact 212-D feature transform used by preserved g18.

    The historical generation-17 source was removed during cleanup, so this
    implementation is kept locally and supports arbitrary leading dimensions
    (for example ``[batch, history, 442]``) without changing the 2-D result.
    """

    shape = pose.shape[:-1]
    joints = pose.reshape(-1, 13, 17, 2)
    shoulder = (joints[:, :, 5] + joints[:, :, 6]) * 0.5
    hip = (joints[:, :, 11] + joints[:, :, 12]) * 0.5
    center = (shoulder + hip) * 0.5
    torso = shoulder - hip
    shoulder_width = (joints[:, :, 5] - joints[:, :, 6]).norm(dim=-1, keepdim=True)
    torso_length = torso.norm(dim=-1, keepdim=True)
    scale = (shoulder_width + torso_length).mul(0.5).clamp_min(0.02)

    theta = torch.atan2(torso[..., 1], torso[..., 0])
    c, s = theta.cos(), theta.sin()
    rel = joints - center[:, :, None, :]
    canonical = torch.stack(
        (
            rel[..., 0] * c[:, :, None] + rel[..., 1] * s[:, :, None],
            -rel[..., 0] * s[:, :, None] + rel[..., 1] * c[:, :, None],
        ),
        -1,
    )
    canonical = canonical / scale[:, :, None, :].clamp_min(0.02)
    shape_features = torch.cat(
        (
            canonical.mean(1).flatten(1),
            canonical.std(1).flatten(1),
            (canonical[:, -1] - canonical[:, 0]).flatten(1),
        ),
        -1,
    )
    velocity = canonical[:, 1:] - canonical[:, :-1]
    motion = torch.cat((velocity.mean(1).flatten(1), velocity.std(1).flatten(1)), -1)

    shoulder_axis = joints[:, :, 6] - joints[:, :, 5]
    hip_axis = joints[:, :, 12] - joints[:, :, 11]
    axes = torch.cat((shoulder_axis, hip_axis, torso), -1)
    angles = torch.atan2(axes[..., 1::2], axes[..., 0::2])
    angle_summary = torch.cat(
        (
            angles.sin().mean(1),
            angles.sin().std(1),
            angles.cos().mean(1),
            angles.cos().std(1),
        ),
        -1,
    )
    torso_summary = torch.cat(
        (torso.mean(1), torso.std(1), scale.mean(1), scale.std(1)), -1
    )

    last = angles[:, -1]
    first = angles[:, 0]
    endpoint = torch.cat((last.sin(), last.cos()), -1)
    endpoint_delta = last - first
    endpoint_delta = torch.cat((endpoint_delta.sin(), endpoint_delta.cos()), -1)
    angular_velocity = angles[:, 1:] - angles[:, :-1]
    velocity_orientation = torch.cat(
        (
            angular_velocity.sin().mean(1),
            angular_velocity.sin().std(1),
            angular_velocity.cos().mean(1),
            angular_velocity.cos().std(1),
        ),
        -1,
    )
    features = torch.cat(
        (
            shape_features,
            motion,
            angle_summary,
            torso_summary,
            endpoint,
            endpoint_delta,
            velocity_orientation,
        ),
        -1,
    )
    if features.shape[-1] != SKELETON_DIM:
        raise AssertionError(f"g18 skeleton feature dimension is {features.shape[-1]}")
    return features.reshape(*shape, SKELETON_DIM)


class G18Policy(nn.Module):
    """Exact preserved g18 model contract for checkpoint compatibility."""

    def __init__(self, variant: str = "g18_factorized_skeleton") -> None:
        super().__init__()
        self.variant = variant
        self.skeleton = nn.Sequential(nn.Linear(212, 48), nn.LayerNorm(48), nn.GELU())
        self.direction = nn.Sequential(nn.Linear(2, 8), nn.LayerNorm(8), nn.GELU())
        self.semantic = nn.Sequential(nn.Linear(512, 20), nn.LayerNorm(20), nn.GELU())
        self.stats = nn.Sequential(nn.Linear(7, 8), nn.LayerNorm(8), nn.GELU())
        self.bank_item = nn.Sequential(nn.Linear(514, 20), nn.LayerNorm(20), nn.GELU())
        self.bank_summary = nn.Sequential(nn.Linear(40, 20), nn.LayerNorm(20), nn.GELU())
        self.logits = nn.Sequential(nn.Linear(110, 10), nn.LayerNorm(10), nn.GELU())
        self.z = nn.Sequential(nn.Linear(512, 16), nn.LayerNorm(16), nn.GELU())

        # The historical string contains the letter z in "skeleton", so the
        # preserved g18 checkpoint has the 16-D z branch active.
        context_dim = 48 + 8 + 16
        if "semantic" in variant:
            context_dim += 20 + 8 + 20
        if "logits" in variant:
            context_dim += 10
        self.context = nn.Sequential(nn.Linear(context_dim, 48), nn.LayerNorm(48), nn.GELU())
        self.candidate = nn.Sequential(nn.Linear(15, 48), nn.LayerNorm(48), nn.GELU())
        self.context_to_action = nn.Linear(48, 48, bias=False)
        self.residual = nn.Sequential(nn.Linear(96, 32), nn.GELU(), nn.Linear(32, 1))
        self.transition = nn.Sequential(nn.Linear(96, 64), nn.GELU(), nn.Linear(64, 55))

    def forward(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        parts = [
            self.skeleton(state["skeleton_prior"]),
            self.direction(state["current_body_direction"]),
            self.z(state["z"]),
        ]
        h = self.context(torch.cat(parts, -1))
        source = state["current_body_direction"][:, None, :]
        candidate = state["candidate"]
        candidate_dir, delta = candidate[..., :2], candidate[..., 2:4]
        relation = torch.stack(
            (
                (candidate_dir * source).sum(-1),
                candidate_dir[..., 0] * source[..., 1]
                - candidate_dir[..., 1] * source[..., 0],
                (delta * source).sum(-1),
                delta[..., 0] * source[..., 1]
                - delta[..., 1] * source[..., 0],
            ),
            -1,
        )
        g = self.candidate(torch.cat((geometry_features(candidate), relation), -1))
        hc = h[:, None].expand(-1, NUM_VIEWS, -1)
        latent = self.context_to_action(h)[:, None, :] * g
        q = latent.sum(-1) / math.sqrt(48.0)
        q = q + 0.20 * self.residual(torch.cat((hc, g), -1)).squeeze(-1)
        return q


class MultiStepG18Policy(G18Policy):
    """g18 scorer whose current state is the causal mean of observed views."""

    pass


def load_checkpoint(model: nn.Module, path: Path, device: torch.device) -> dict[str, Any]:
    saved = torch.load(path, map_location=device, weights_only=False)
    online = saved.get("online", saved)
    model.load_state_dict(online, strict=True)
    model.to(device).eval()
    return saved


def precompute_skeleton(raw: GPUCache) -> torch.Tensor:
    with torch.inference_mode():
        return g18_skeleton_features(raw.pose)


def attach_view_valid(raw: GPUCache, cache_root: Path, split: str) -> None:
    """Attach the explicit physical-view availability mask to GPUCache."""

    path = cache_root / split / "view_valid.npy"
    raw.valid = torch.from_numpy(np.load(path, allow_pickle=False)).to(
        device=raw.device, dtype=torch.bool
    )


def build_state(
    raw: GPUCache,
    skeleton: torch.Tensor,
    episodes: torch.Tensor,
    history: torch.Tensor,
    history_count: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Build a causal state from the first 1/2/3 observed physical views."""

    episodes = episodes.long()
    history = history.long()
    count = history_count.long().clamp(1, history.shape[1])
    batch = episodes.shape[0]
    positions = torch.arange(history.shape[1], device=episodes.device)[None, :]
    observed = positions < count[:, None]
    observed_float = observed.float()
    denominator = count.float().clamp_min(1.0)[:, None]
    current_position = (count - 1)[:, None]
    current = history.gather(1, current_position).squeeze(1)

    history_skeleton = skeleton[episodes[:, None], history]
    history_z = raw.z[episodes[:, None], history]
    pooled_skeleton = (history_skeleton * observed_float[..., None]).sum(1) / denominator
    pooled_z = (history_z * observed_float[..., None]).sum(1) / denominator

    observed_views = torch.zeros((batch, NUM_VIEWS), dtype=torch.bool, device=episodes.device)
    observed_views.scatter_(1, history, observed)
    source = raw.geometry[episodes, current, :2]
    geometry = raw.geometry[episodes, :, :2]
    sin_delta = geometry[..., 0] * source[:, None, 1] - geometry[..., 1] * source[:, None, 0]
    cos_delta = (geometry * source[:, None]).sum(-1)
    valid = (
        raw.valid[episodes]
        & raw.reachable[episodes, current]
        & (~observed_views)
    )
    candidate = torch.cat(
        (
            geometry,
            torch.stack((sin_delta, cos_delta), -1),
            source[:, None].expand(-1, NUM_VIEWS, -1),
            valid[..., None].float(),
        ),
        -1,
    )
    return {
        "skeleton_prior": pooled_skeleton,
        "current_body_direction": source,
        "z": pooled_z,
        "candidate": candidate,
        "mask": valid,
    }


def permute_candidates(
    state: dict[str, torch.Tensor], utility: torch.Tensor, generator: torch.Generator
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    batch = state["candidate"].shape[0]
    order = torch.rand(
        (batch, NUM_VIEWS), device=state["candidate"].device, generator=generator
    ).argsort(-1)
    out = dict(state)
    out["candidate"] = state["candidate"].gather(
        1, order[..., None].expand(-1, -1, state["candidate"].shape[-1])
    )
    out["mask"] = state["mask"].gather(1, order)
    return out, utility.gather(1, order)


def class_bank_tensor(classes: Sequence[int], device: torch.device) -> torch.Tensor:
    return torch.as_tensor(sorted(int(c) for c in classes), device=device, dtype=torch.long)


def margin_against_bank(
    scores: torch.Tensor, labels: torch.Tensor, bank: torch.Tensor
) -> torch.Tensor:
    restricted = scores.index_select(-1, bank)
    local = torch.searchsorted(bank, labels)
    target = restricted.gather(-1, local[..., None]).squeeze(-1)
    rival = restricted.clone()
    rival.scatter_(-1, local[..., None], -torch.inf)
    return target - rival.amax(-1)


def utility_targets(
    raw: GPUCache,
    episodes: torch.Tensor,
    history: torch.Tensor,
    history_count: torch.Tensor,
    valid: torch.Tensor,
    bank: torch.Tensor,
) -> torch.Tensor:
    """Offline delta-margin targets; no target tensor enters the model state."""

    count = history_count.long().clamp(1, history.shape[1])
    observed = torch.arange(history.shape[1], device=history.device)[None, :] < count[:, None]
    history_logits = raw.logits[episodes[:, None], history]
    history_sum = (history_logits * observed[..., None]).sum(1)
    before_scores = history_sum / count.float()[:, None]
    before = margin_against_bank(before_scores, raw.labels[episodes], bank)
    fused = (history_sum[:, None, :] + raw.logits[episodes]) / (count.float() + 1.0)[:, None, None]
    after = margin_against_bank(
        fused,
        raw.labels[episodes, None].expand(-1, NUM_VIEWS),
        bank,
    )
    return (after - before[:, None]).masked_fill(~valid, 0.0)


def multistep_loss(
    prediction: torch.Tensor,
    utility: torch.Tensor,
    valid: torch.Tensor,
    history_count: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Soft listwise + pairwise loss, with a small diagnostic-only stage-3 weight."""

    safe_utility = utility.masked_fill(~valid, -1e4)
    safe_prediction = prediction.masked_fill(~valid, -1e4)
    target_probability = F.softmax(safe_utility / 0.20, dim=-1)
    log_probability = F.log_softmax(safe_prediction, dim=-1)
    count = valid.sum(-1).clamp_min(1).float()
    listwise = -(
        target_probability
        * torch.where(valid, log_probability, torch.zeros_like(log_probability))
    ).sum(-1) / count

    pair_total = prediction.new_zeros(())
    pair_weight = prediction.new_zeros(())
    for left in range(NUM_VIEWS):
        for right in range(left + 1, NUM_VIEWS):
            delta = utility[:, left] - utility[:, right]
            pair = valid[:, left] & valid[:, right] & (delta.abs() > 0.02)
            sign = delta.sign()
            required = 0.10 + 0.40 * delta.abs().clamp_max(2.0)
            violation = F.relu(required - sign * (prediction[:, left] - prediction[:, right]))
            weights = pair.float()
            pair_total = pair_total + (violation * weights).sum()
            pair_weight = pair_weight + weights.sum()
    pairwise = pair_total / pair_weight.clamp_min(1.0)
    stage_weight = torch.where(history_count == 3, 0.10, torch.ones_like(history_count, dtype=torch.float32))
    loss = ((listwise + 0.50 * pairwise) * stage_weight).sum() / stage_weight.sum().clamp_min(1.0)
    return loss, {
        "listwise": float(listwise.mean().detach()),
        "pairwise": float(pairwise.detach()),
        "stage3_fraction": float((history_count == 3).float().mean().detach()),
    }


def choose_random_starts(
    raw: GPUCache, episodes: torch.Tensor, seed: int
) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    valid = raw.valid[episodes].cpu().numpy()
    starts = [int(rng.choice(np.flatnonzero(row))) for row in valid]
    return torch.as_tensor(starts, device=episodes.device, dtype=torch.long)


def _path_score(
    raw: GPUCache,
    episodes: torch.Tensor,
    views: torch.Tensor,
    bank: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    scores = raw.logits[episodes[:, None], views].mean(1).index_select(-1, bank)
    labels = raw.labels[episodes]
    local = torch.searchsorted(bank, labels)
    top1 = scores.argmax(-1).eq(local)
    probability = F.softmax(scores, -1)
    entropy = -(probability * probability.clamp_min(1e-12).log()).sum(-1)
    entropy = entropy / math.log(float(len(bank)))
    return top1, entropy, scores


def _single_metrics(
    raw: GPUCache, episodes: torch.Tensor, starts: torch.Tensor, bank: torch.Tensor
) -> dict[str, float]:
    scores = raw.logits[episodes, starts].index_select(-1, bank)
    local = torch.searchsorted(bank, raw.labels[episodes])
    top1 = scores.argmax(-1).eq(local)
    probability = F.softmax(scores, -1)
    entropy = -(probability * probability.clamp_min(1e-12).log()).sum(-1)
    entropy = entropy / math.log(float(len(bank)))
    return {"top1": float(top1.float().mean()), "mean_entropy": float(entropy.mean())}


@torch.inference_mode()
def policy_rollout(
    model: nn.Module,
    raw: GPUCache,
    skeleton: torch.Tensor,
    episodes: torch.Tensor,
    starts: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Run exactly two moves, then make but do not execute the third prediction."""

    model.eval()
    batch = episodes.shape[0]
    one = torch.ones(batch, dtype=torch.long, device=episodes.device)
    state1 = build_state(raw, skeleton, episodes, starts[:, None], one)
    mask1 = state1["mask"]
    keep1 = mask1.any(-1)
    episodes = episodes[keep1]
    starts = starts[keep1]
    state1 = {key: value[keep1] for key, value in state1.items()}
    action1 = model(state1).masked_fill(~state1["mask"], -torch.inf).argmax(-1)

    history2 = torch.stack((starts, action1), 1)
    two = torch.full((episodes.shape[0],), 2, dtype=torch.long, device=episodes.device)
    state2 = build_state(raw, skeleton, episodes, history2, two)
    mask2 = state2["mask"]
    keep2 = mask2.any(-1)
    episodes = episodes[keep2]
    starts = starts[keep2]
    action1 = action1[keep2]
    history2 = history2[keep2]
    state2 = {key: value[keep2] for key, value in state2.items()}
    action2 = model(state2).masked_fill(~state2["mask"], -torch.inf).argmax(-1)

    history3 = torch.stack((history2[:, 0], history2[:, 1], action2), 1)
    three = torch.full((episodes.shape[0],), 3, dtype=torch.long, device=episodes.device)
    state3 = build_state(raw, skeleton, episodes, history3, three)
    mask3 = state3["mask"]
    q3 = model(state3)
    future_action = q3.masked_fill(~mask3, -torch.inf).argmax(-1)
    future_present = mask3.any(-1)
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


@torch.inference_mode()
def legacy_one_move(
    model: nn.Module,
    raw: GPUCache,
    skeleton: torch.Tensor,
    episodes: torch.Tensor,
    starts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    state = build_state(
        raw,
        skeleton,
        episodes,
        starts[:, None],
        torch.ones_like(starts),
    )
    action = model(state).masked_fill(~state["mask"], -torch.inf).argmax(-1)
    return action, state["mask"].any(-1)


def enumerate_two_move_paths(
    raw: GPUCache,
    episode: int,
    start: int,
) -> list[tuple[int, int, float]]:
    valid = raw.valid[episode].cpu().numpy().astype(bool)
    reachable = raw.reachable[episode].cpu().numpy().astype(bool)
    first = [v for v in range(NUM_VIEWS) if valid[v] and reachable[start, v] and v != start]
    paths: list[tuple[int, int, float]] = []
    usable_first: list[tuple[int, list[int]]] = []
    for first_view in first:
        second = [
            v
            for v in range(NUM_VIEWS)
            if valid[v]
            and reachable[first_view, v]
            and v not in {start, first_view}
        ]
        if second:
            usable_first.append((first_view, second))
    if not usable_first:
        return paths
    first_probability = 1.0 / len(usable_first)
    for first_view, second in usable_first:
        second_probability = first_probability / len(second)
        paths.extend((first_view, second_view, second_probability) for second_view in second)
    return paths


def random_exact_metrics(
    raw: GPUCache,
    episodes: torch.Tensor,
    starts: torch.Tensor,
    bank: torch.Tensor,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Exact expectation over random two-step paths, not one sampled path."""

    top1_values: list[float] = []
    entropy_values: list[float] = []
    cost_values: list[float] = []
    path_store: list[list[int]] = []
    probability_store: list[float] = []
    kept_episodes: list[int] = []
    for episode, start in zip(episodes.cpu().tolist(), starts.cpu().tolist()):
        paths = enumerate_two_move_paths(raw, int(episode), int(start))
        if not paths:
            continue
        ep_tensor = torch.full((len(paths),), int(episode), dtype=torch.long, device=raw.device)
        views = torch.as_tensor(
            [[int(start), first, second] for first, second, _ in paths],
            dtype=torch.long,
            device=raw.device,
        )
        top1, entropy, _ = _path_score(raw, ep_tensor, views, bank)
        probabilities = torch.as_tensor([p for _, _, p in paths], device=raw.device)
        top1_values.append(float((top1.float() * probabilities).sum()))
        entropy_values.append(float((entropy * probabilities).sum()))
        costs = raw.cost[ep_tensor, views[:, 0], views[:, 1]] + raw.cost[
            ep_tensor, views[:, 1], views[:, 2]
        ]
        cost_values.append(float((costs.nan_to_num(1.0) * probabilities).sum()))
        kept_episodes.append(int(episode))
        path_store.extend(views.cpu().tolist())
        probability_store.extend(probabilities.cpu().tolist())
    if not top1_values:
        return {"top1": 0.0, "mean_entropy": 0.0, "mean_movement_cost": 0.0, "episodes": 0.0}, {}
    return (
        {
            "top1": float(np.mean(top1_values)),
            "mean_entropy": float(np.mean(entropy_values)),
            "mean_movement_cost": float(np.mean(cost_values)),
            "episodes": float(len(top1_values)),
        },
        {
            "episodes": np.asarray(kept_episodes, dtype=np.int64),
            "paths": np.asarray(path_store, dtype=np.int64),
            "probabilities": np.asarray(probability_store, dtype=np.float32),
        },
    )


def sample_random_paths(
    raw: GPUCache,
    episodes: torch.Tensor,
    starts: torch.Tensor,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    kept: list[int] = []
    paths: list[list[int]] = []
    for episode, start in zip(episodes.cpu().tolist(), starts.cpu().tolist()):
        candidates = enumerate_two_move_paths(raw, int(episode), int(start))
        if not candidates:
            continue
        first_values = sorted(set(int(x[0]) for x in candidates))
        first = int(rng.choice(first_values))
        second_values = [int(second) for f, second, _ in candidates if int(f) == first]
        second = int(rng.choice(second_values))
        kept.append(int(episode))
        paths.append([int(start), first, second])
    return (
        torch.as_tensor(kept, device=episodes.device, dtype=torch.long),
        torch.as_tensor(paths, device=episodes.device, dtype=torch.long),
        torch.as_tensor(starts.cpu().numpy()[: len(kept)], device=episodes.device, dtype=torch.long)
        if kept
        else torch.empty(0, dtype=torch.long, device=episodes.device),
    )


def oracle_paths(
    raw: GPUCache,
    episodes: torch.Tensor,
    starts: torch.Tensor,
    bank: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    selected_episodes: list[int] = []
    selected_paths: list[list[int]] = []
    for episode, start in zip(episodes.cpu().tolist(), starts.cpu().tolist()):
        candidates = enumerate_two_move_paths(raw, int(episode), int(start))
        if not candidates:
            continue
        paths = torch.as_tensor(
            [[int(start), first, second] for first, second, _ in candidates],
            device=raw.device,
            dtype=torch.long,
        )
        ep = torch.full((len(candidates),), int(episode), device=raw.device, dtype=torch.long)
        scores = raw.logits[ep[:, None], paths].mean(1).index_select(-1, bank)
        local = torch.searchsorted(bank, raw.labels[ep])
        target = scores.gather(-1, local[:, None]).squeeze(-1)
        rival = scores.clone()
        rival.scatter_(1, local[:, None], -torch.inf)
        margin = target - rival.amax(-1)
        selected = int(margin.argmax())
        selected_episodes.append(int(episode))
        selected_paths.append(paths[selected].cpu().tolist())
    return (
        torch.as_tensor(selected_episodes, device=episodes.device, dtype=torch.long),
        torch.as_tensor(selected_paths, device=episodes.device, dtype=torch.long),
    )


def metrics_for_paths(
    raw: GPUCache,
    episodes: torch.Tensor,
    paths: torch.Tensor,
    bank: torch.Tensor,
) -> dict[str, float]:
    top1, entropy, _ = _path_score(raw, episodes, paths, bank)
    costs = raw.cost[episodes, paths[:, 0], paths[:, 1]] + raw.cost[
        episodes, paths[:, 1], paths[:, 2]
    ]
    return {
        "top1": float(top1.float().mean()),
        "mean_entropy": float(entropy.mean()),
        "mean_movement_cost": float(costs.nan_to_num(1.0).mean()),
        "episodes": float(len(episodes)),
    }


def evaluate_protocol(
    model: nn.Module,
    base_model: nn.Module,
    raw: GPUCache,
    skeleton: torch.Tensor,
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
        eligible = eligible & raw.valid[:, start]
        episodes = eligible.nonzero().flatten()
        starts = torch.full_like(episodes, start)
    elif protocol == "random_start":
        starts = choose_random_starts(raw, episodes, seed)
    else:
        raise ValueError(f"unknown protocol {protocol}")

    single = _single_metrics(raw, episodes, starts, bank) if len(episodes) else {"top1": 0.0, "mean_entropy": 0.0}
    rollout = policy_rollout(model, raw, skeleton, episodes, starts)
    policy_episodes = rollout["episodes"]
    policy_paths = torch.stack(
        (rollout["starts"], rollout["action1"], rollout["action2"]), 1
    )
    policy_metrics = metrics_for_paths(raw, policy_episodes, policy_paths, bank) if len(policy_episodes) else {
        "top1": 0.0, "mean_entropy": 0.0, "mean_movement_cost": 0.0, "episodes": 0.0
    }

    random_metrics, random_store = random_exact_metrics(raw, episodes, starts, bank)
    random_sample_episodes, random_sample_paths, _ = sample_random_paths(raw, episodes, starts, seed + 17)
    random_sample_metrics = (
        metrics_for_paths(raw, random_sample_episodes, random_sample_paths, bank)
        if len(random_sample_episodes)
        else {"top1": 0.0, "mean_entropy": 0.0, "mean_movement_cost": 0.0, "episodes": 0.0}
    )
    oracle_episode, oracle_path = oracle_paths(raw, episodes, starts, bank)
    oracle_metrics = (
        metrics_for_paths(raw, oracle_episode, oracle_path, bank)
        if len(oracle_episode)
        else {"top1": 0.0, "mean_entropy": 0.0, "mean_movement_cost": 0.0, "episodes": 0.0}
    )

    legacy_action, legacy_keep = legacy_one_move(base_model, raw, skeleton, episodes, starts)
    legacy_episodes = episodes[legacy_keep]
    legacy_paths = torch.stack((starts[legacy_keep], legacy_action[legacy_keep]), 1)
    legacy_scores = raw.logits[legacy_episodes[:, None], legacy_paths].mean(1).index_select(-1, bank)
    legacy_local = torch.searchsorted(bank, raw.labels[legacy_episodes])
    legacy_top1 = legacy_scores.argmax(-1).eq(legacy_local)
    legacy_cost = raw.cost[legacy_episodes, legacy_paths[:, 0], legacy_paths[:, 1]]

    all_valid = raw.valid[episodes]
    all_logits = raw.logits[episodes]
    all_weights = all_valid.float().unsqueeze(-1)
    all_scores = (all_logits * all_weights).sum(1) / all_weights.sum(1).clamp_min(1.0)
    all_scores = all_scores.index_select(-1, bank)
    all_local = torch.searchsorted(bank, raw.labels[episodes])
    all_top1 = all_scores.argmax(-1).eq(all_local)

    future_present = rollout["future_present"]
    future_action = rollout["future_action"]
    future_legal = rollout["mask3"].gather(1, future_action[:, None]).squeeze(1)
    report: dict[str, Any] = {
        "protocol": protocol,
        "class_bank": [int(c) for c in classes],
        "episodes_before_two_move_filter": int(len(episodes)),
        "episodes_policy_completed": int(len(policy_episodes)),
        "completion_rate": float(len(policy_episodes) / max(len(episodes), 1)),
        "metrics": {
            "single_start": single,
            "legacy_g18_one_move": {
                "top1": float(legacy_top1.float().mean()) if len(legacy_episodes) else 0.0,
                "mean_movement_cost": float(legacy_cost.nan_to_num(1.0).mean()) if len(legacy_episodes) else 0.0,
                "episodes": float(len(legacy_episodes)),
            },
            "random_3view_sampled": random_sample_metrics,
            "random_3view_exact": random_metrics,
            "policy_3view": policy_metrics,
            "oracle_3view": oracle_metrics,
            "all_valid_views": {"top1": float(all_top1.float().mean()) if len(episodes) else 0.0},
        },
        "future_fourth_prediction": {
            "episodes_after_three_observed": int(len(policy_episodes)),
            "remaining_view_present_rate": float(future_present.float().mean()) if len(policy_episodes) else 0.0,
            "predicted_action_legal_rate": float(future_legal.float().mean()) if len(policy_episodes) else 0.0,
            "mean_predicted_q": float(rollout["future_q"].mean()) if len(policy_episodes) else 0.0,
        },
        "starts": starts.cpu().tolist(),
        "policy_paths": policy_paths.cpu().tolist(),
        "policy_future_actions": future_action.cpu().tolist(),
        "labels": raw.labels[policy_episodes].cpu().tolist(),
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
            future_action=future_action.cpu().numpy(),
            future_present=future_present.cpu().numpy(),
            random_exact_paths=random_store.get("paths", np.empty((0, 3), dtype=np.int64)),
            random_exact_probabilities=random_store.get("probabilities", np.empty(0, dtype=np.float32)),
        )
    return report


def train_one_seed(
    seed: int,
    train_raw: GPUCache,
    val_raw: GPUCache,
    train_skeleton: torch.Tensor,
    val_skeleton: torch.Tensor,
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

    seed_all(seed)
    generator = torch.Generator(device=device).manual_seed(seed)
    model = MultiStepG18Policy().to(device)
    base_saved = load_checkpoint(model, BASE_CHECKPOINT, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.04)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, epochs, eta_min=learning_rate * 0.05
    )

    eligible = (train_raw.valid.sum(-1) == 4) & torch.isin(
        train_raw.labels, class_bank_tensor(SEEN_CLASSES, device)
    )
    episode_pool = eligible.nonzero().flatten()
    if not len(episode_pool):
        raise RuntimeError("no four-view seen-class training episodes")
    audit = {
        "seed": seed,
        "base_policy": "g18_factorized_skeleton",
        "base_checkpoint": str(BASE_CHECKPOINT),
        "base_checkpoint_epoch": int(base_saved.get("epoch", -1)),
        "training_episodes_four_view_seen": int(len(episode_pool)),
        "training_labels": SEEN_CLASSES,
        "pseudo_unseen_for_selection": PSEUDO_UNSEEN_CLASSES,
        "true_unseen_test_only": TRUE_UNSEEN_CLASSES,
        "causal_state": ["mean observed skeleton", "mean observed z", "current geometry", "candidate geometry"],
        "future_view_inputs_to_policy": False,
        "training_target": "offline delta margin after adding candidate; seen bank only",
        "trajectory": "one observed -> predict second; two observed -> predict third; three observed -> log fourth only",
        "final_fusion": "mean logits of exactly three observed views",
        "stage3_training_weight": 0.10,
        "slot_permutation": True,
        "movement_cost_in_reward": False,
    }
    dump(run / "audit.json", audit)

    # Validation is fixed view0 and pseudo-unseen only; true unseen never
    # participates in checkpoint selection.
    val_model = model
    best_score = -float("inf")
    best_epoch = 0
    start_epoch = 1
    last = run / "last.pt"
    if last.exists():
        saved = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(saved["online"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        if "generator" in saved:
            generator.set_state(saved["generator"])
        best_score = float(saved.get("best_score", -float("inf")))
        best_epoch = int(saved.get("best_epoch", 0))
        start_epoch = int(saved.get("epoch", 0)) + 1

    with (run / "train.log").open("a", encoding="utf-8", buffering=1) as log:
        print("MULTISTEP_TRAIN_START", json.dumps(audit), flush=True)
        for epoch in range(start_epoch, epochs + 1):
            model.train()
            begun = time.monotonic()
            losses: list[float] = []
            listwise_values: list[float] = []
            pairwise_values: list[float] = []
            for _ in range(updates_per_epoch):
                ids = torch.randint(
                    len(episode_pool), (batch_size,), device=device, generator=generator
                )
                episodes = episode_pool[ids]
                permutation = torch.rand(
                    (batch_size, NUM_VIEWS), device=device, generator=generator
                ).argsort(-1)
                stage = torch.multinomial(
                    torch.as_tensor([0.45, 0.45, 0.10], device=device),
                    batch_size,
                    replacement=True,
                    generator=generator,
                ) + 1
                history = permutation[:, :3]
                state = build_state(train_raw, train_skeleton, episodes, history, stage)
                utility = utility_targets(
                    train_raw,
                    episodes,
                    history,
                    stage,
                    state["mask"],
                    class_bank_tensor(SEEN_CLASSES, device),
                )
                state, utility = permute_candidates(state, utility, generator)
                prediction = model(state)
                loss, parts = multistep_loss(prediction, utility, state["mask"], stage)
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"non-finite multistep loss seed={seed} epoch={epoch}"
                    )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                losses.append(float(loss.detach()))
                listwise_values.append(parts["listwise"])
                pairwise_values.append(parts["pairwise"])
            scheduler.step()
            if torch.cuda.is_available():
                torch.cuda.synchronize(device)

            validation = evaluate_protocol(
                val_model,
                val_model,
                val_raw,
                val_skeleton,
                PSEUDO_UNSEEN_CLASSES,
                "fixed0",
                seed + 90117,
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
                "base_checkpoint": str(BASE_CHECKPOINT),
            }
            torch.save(payload, run / "last.pt")
            if improved:
                torch.save(payload, run / "best.pt")
            row = {
                "epoch": epoch,
                "seed": seed,
                "loss": float(np.mean(losses)),
                "listwise": float(np.mean(listwise_values)),
                "pairwise": float(np.mean(pairwise_values)),
                "seconds": time.monotonic() - begun,
                "samples_per_second": float(batch_size * updates_per_epoch / max(time.monotonic() - begun, 1e-6)),
                "validation_selection_score": score,
                "best_score": best_score,
                "best_epoch": best_epoch,
                "validation": vm,
            }
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
    val_skeleton: torch.Tensor,
    seen_skeleton: torch.Tensor,
    test_skeleton: torch.Tensor,
    device: torch.device,
    output_root: Path,
) -> dict[str, Any]:
    chosen = max(summaries, key=lambda item: float(item["best_score"]))
    chosen_seed = int(chosen["seed"])
    chosen_model = MultiStepG18Policy().to(device)
    load_checkpoint(chosen_model, Path(chosen["checkpoint"]), device)
    base_model = G18Policy().to(device)
    load_checkpoint(base_model, BASE_CHECKPOINT, device)
    selection = {"criterion": "validation pseudo-unseen fixed0 3-view gain", "chosen": chosen, "runs": summaries}
    dump(output_root / "selection.json", selection)

    reports: dict[str, Any] = {}
    split_specs: Iterable[tuple[str, GPUCache, torch.Tensor, Sequence[int]]] = (
        ("val_pseudo_unseen", val_raw, val_skeleton, PSEUDO_UNSEEN_CLASSES),
        ("seen_test", seen_raw, seen_skeleton, SEEN_CLASSES),
        ("true_unseen_5way", test_raw, test_skeleton, TRUE_UNSEEN_CLASSES),
    )
    for split_name, raw, skeleton, classes in split_specs:
        split_dir = output_root / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        reports[split_name] = {}
        for protocol in ("fixed0", "fixed1", "fixed2", "fixed3", "random_start"):
            report = evaluate_protocol(
                chosen_model,
                base_model,
                raw,
                skeleton,
                classes,
                protocol,
                20260910 + len(protocol),
                split_dir,
            )
            report["selected_seed"] = chosen_seed
            reports[split_name][protocol] = report
            dump(split_dir / f"evaluation_{protocol}.json", report)
            print(
                "MULTISTEP_EVALUATION",
                split_name,
                protocol,
                json.dumps(report["metrics"]),
                flush=True,
            )
    result = {
        "base_policy": "g18_factorized_skeleton seed_20260911",
        "base_checkpoint": str(BASE_CHECKPOINT),
        "selection": selection,
        "protocols": {
            "fixed0_to_fixed3": "physical initial view fixed to 0, 1, 2, or 3",
            "random_start": "one deterministic uniform valid start per episode",
            "first_transition": "state of one observed view predicts second",
            "second_transition": "mean state of first two observed views predicts third",
            "third_prediction": "state of first three observed views predicts/logs fourth only",
            "final_classification": "mean logits of exactly first three observed views; fourth excluded",
        },
        "reports": reports,
    }
    dump(output_root / "summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--updates-per-epoch", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260909, 20260910, 20260911])
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This experiment is intended to run on CUDA; refusing an accidental CPU run")
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
            "base_checkpoint": str(BASE_CHECKPOINT),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "updates_per_epoch": args.updates_per_epoch,
            "learning_rate": args.learning_rate,
            "seeds": args.seeds,
            "seen_classes": SEEN_CLASSES,
            "pseudo_unseen_classes": PSEUDO_UNSEEN_CLASSES,
            "true_unseen_classes": TRUE_UNSEEN_CLASSES,
        },
    )
    print("MULTISTEP_CACHE_LOAD", json.dumps({"cache_root": str(args.cache_root), "device": str(device)}), flush=True)
    train_raw = GPUCache(args.cache_root / "dqn_train", device)
    val_raw = GPUCache(args.cache_root / "val", device)
    seen_raw = GPUCache(args.cache_root / "seen_test", device)
    test_raw = GPUCache(args.cache_root / "test", device)
    for split, raw in (("dqn_train", train_raw), ("val", val_raw),
                       ("seen_test", seen_raw), ("test", test_raw)):
        attach_view_valid(raw, args.cache_root, split)
    print(
        "MULTISTEP_CACHE_SHAPES",
        json.dumps({
            "dqn_train": train_raw.num_episodes,
            "val": val_raw.num_episodes,
            "seen_test": seen_raw.num_episodes,
            "test": test_raw.num_episodes,
        }),
        flush=True,
    )
    print("MULTISTEP_PRECOMPUTE_SKELETON", flush=True)
    train_skeleton = precompute_skeleton(train_raw)
    val_skeleton = precompute_skeleton(val_raw)
    seen_skeleton = precompute_skeleton(seen_raw)
    test_skeleton = precompute_skeleton(test_raw)
    print("MULTISTEP_SKELETON_READY", flush=True)

    summaries: list[dict[str, Any]] = []
    for seed in args.seeds:
        summaries.append(
            train_one_seed(
                seed,
                train_raw,
                val_raw,
                train_skeleton,
                val_skeleton,
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
    final_evaluation(
        summaries,
        val_raw,
        seen_raw,
        test_raw,
        val_skeleton,
        seen_skeleton,
        test_skeleton,
        device,
        args.output_root,
    )
    print("MULTISTEP_TRAIN_EVAL_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
