"""Environment-conditioned active-view experiment on the latest group1 cache.

This is an isolated extension of the selected object/person trajectory policy.
It preserves the causal state and adds an observed-only occlusion trajectory.
The current machine does not contain raw X3D tensors aligned with every row of
the latest 25-person ``dqn_train`` cache, so the first benchmark is explicitly
marked as a feature-space, view-consistent proxy: clean latest VPOCLIP logits
are class-agnostically degraded by one synchronized virtual occluder per
episode.  Clean pose/object tracks remain available to the policy, as allowed
by the document's minimum viable setting.  No GT label or future occlusion is
given to the policy.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ..multistep_angle_object_trajectory_policy_v1 import (
    AngleObjectTrajectoryPolicy,
    NUM_FRAMES,
    NUM_OBJECT_CLASSES,
    OBJECT_CHANNELS,
    utility_targets,
    multistep_loss,
)
from .run_v9_latest_vpoclip_all_starts import LatestGPUCache


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_ROOT = PROJECT_ROOT / "data/rl_raw_new50_5_group1_stage_b/cache"
TRACK_ROOT = PROJECT_ROOT / "data/trajectory_latest_vpoclip_new50_5/object_tracks_full"
BASE_POLICY = PROJECT_ROOT / (
    "work_dir/multistep_angle_object_trajectory_policy_latest_vpoclip_new50_5/"
    "seed_20260909/best.pt"
)
OUTPUT_ROOT = PROJECT_ROOT / "work_dir/multistep_environment_occlusion_policy_v2"
NUM_VIEWS = 4
NUM_CLASSES = 55
OCC_DIM = 12
UNSEEN = [14, 15, 25, 39, 46]
SEEN = [x for x in range(NUM_CLASSES) if x not in UNSEEN]
PSEUDO = [5, 6, 22, 38, 42, 44, 45, 51, 52, 53]
LEVELS = {"none": 0.0, "mild": 0.30, "medium": 0.50, "heavy": 0.70}
TRAIN_MIX = ("none", "mild", "medium", "heavy")
TRAIN_MIX_PROB = (0.25, 0.25, 0.35, 0.15)


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def attach_object_tracks(raw: LatestGPUCache, split: str) -> None:
    path = TRACK_ROOT / split / "object_position_track.npy"
    if not path.exists():
        raise FileNotFoundError(path)
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    expected = (raw.num_episodes, NUM_VIEWS, NUM_FRAMES, NUM_OBJECT_CLASSES, 4)
    if tuple(array.shape) != expected:
        raise ValueError(f"{path} shape {array.shape} != {expected}")
    raw.object_position_track = torch.from_numpy(np.array(array, copy=True)).to(
        device=raw.device, dtype=torch.float32
    )


def attach_view_valid(raw: LatestGPUCache, split: str) -> None:
    path = CACHE_ROOT / split / "view_valid.npy"
    raw.view_valid = torch.from_numpy(np.array(np.load(path, mmap_mode="r"), copy=True)).to(
        device=raw.device, dtype=torch.bool
    )


def seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def wrap_angle(x: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(x), torch.cos(x))


def sample_layout(raw: LatestGPUCache, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample one synchronized occluder and enforce a clear valid view."""

    geometry = raw.geometry[..., :2].detach().cpu().numpy()
    valid = raw.view_valid.detach().cpu().numpy().astype(bool)
    rng = np.random.default_rng(seed)
    angles = np.arctan2(geometry[..., 0], geometry[..., 1])
    occ = np.empty(raw.num_episodes, dtype=np.float32)
    distance = np.empty(raw.num_episodes, dtype=np.float32)
    style = np.empty(raw.num_episodes, dtype=np.float32)
    sigma = math.radians(45.0)
    for i in range(raw.num_episodes):
        occ[i] = rng.uniform(-math.pi, math.pi)
        distance[i] = rng.uniform(0.28, 0.52)
        style[i] = rng.uniform(0.85, 1.15)
        available = valid[i]
        if available.any():
            for _ in range(50):
                sev = np.exp(-0.5 * wrap_np(angles[i, available] - occ[i]) ** 2 / sigma**2)
                if float(sev.min()) < 0.20:
                    break
                occ[i] = rng.uniform(-math.pi, math.pi)
            else:
                # This is only a recoverability safeguard, not a target-based
                # choice: put the occluder opposite one randomly selected view.
                selected = int(rng.choice(np.flatnonzero(available)))
                occ[i] = float(wrap_np(angles[i, selected] + math.pi))
    device = raw.device
    return (
        torch.as_tensor(occ, device=device, dtype=torch.float32),
        torch.as_tensor(distance, device=device, dtype=torch.float32),
        torch.as_tensor(style, device=device, dtype=torch.float32),
    )


def wrap_np(x: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(x) + math.pi) % (2.0 * math.pi) - math.pi


def make_occlusion(
    raw: LatestGPUCache,
    level: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return [N,V,T,OCC_DIM] observed descriptors and [N,V,55] logits."""

    occ_angle, distance, style = sample_layout(raw, seed)
    theta = torch.atan2(raw.geometry[..., 0], raw.geometry[..., 1])
    delta = wrap_angle(theta - occ_angle[:, None])
    sigma = math.radians(45.0)
    severity = (level * torch.exp(-0.5 * (delta / sigma).square())).clamp(0.0, 0.70)
    side = torch.sign(torch.sin(delta))
    side = torch.where(side == 0, torch.ones_like(side), side)

    pose = raw.pose.reshape(raw.num_episodes, NUM_VIEWS, NUM_FRAMES, 17, 2)
    present = pose.norm(dim=-1) > 0.02
    torso_ids = torch.as_tensor([5, 6, 11, 12], device=raw.device)
    upper_ids = torch.as_tensor([0, 1, 2, 5, 6, 7, 8, 9, 10], device=raw.device)
    torso = present.index_select(-1, torso_ids).float().mean(-1)
    upper = present.index_select(-1, upper_ids).float().mean(-1)
    hands = present.index_select(-1, torch.as_tensor([9, 10], device=raw.device)).float()
    object_track = raw.object_position_track
    object_presence = object_track[..., 0].clamp(0.0, 1.0).mean(-1)
    object_conf = object_track[..., 3].clamp(0.0, 1.0).mean(-1)
    interaction = object_presence * object_conf * hands.mean(-1)
    person = present.float().mean(-1)

    sev = severity[..., None].expand(-1, -1, NUM_FRAMES)
    side_t = side[..., None].expand_as(sev)
    style_t = style[:, None, None].expand_as(sev)
    # Twelve explicit, causal sensor descriptors.  They describe the
    # observed current/previous view only; candidate-view severity is never
    # inserted into the state.
    descriptor = torch.stack(
        (
            0.20 * sev,
            sev * person,
            sev * torso,
            sev * upper,
            sev * hands[..., 0],
            sev * hands[..., 1],
            sev * interaction,
            side_t * 0.20 * sev,
            0.42 * sev,
            style_t * sev,
            style_t * (0.50 + 0.30 * sev),
            (torch.cos(delta) * severity)[..., None].expand_as(sev),
        ),
        -1,
    )

    # Class-agnostic feature-space proxy for masked VPOCLIP.  It lowers the
    # view's class margin and adds deterministic occlusion-dependent nuisance,
    # without looking at the label.  ``level=0`` is exactly the clean cache.
    logits = raw.logits
    center = logits.mean(-1, keepdim=True)
    spread = logits.std(-1, keepdim=True).clamp_min(1e-3)
    generator = torch.Generator(device=raw.device).manual_seed(seed + 17011)
    nuisance = torch.randn(logits.shape, device=raw.device, generator=generator)
    damage = severity[..., None]
    masked_logits = center + (1.0 - 0.70 * damage) * (logits - center)
    masked_logits = masked_logits + (0.10 * damage * spread * nuisance)
    return descriptor, masked_logits


class EnvCache:
    """One level of the synthetic environment over a clean latest cache."""

    def __init__(self, base: LatestGPUCache, occlusion: torch.Tensor, logits: torch.Tensor):
        self.device = base.device
        self.num_episodes = base.num_episodes
        self.logits = logits
        self.labels = base.labels
        self.valid = base.view_valid
        self.reachable = base.reachable
        self.cost = base.cost
        self.geometry = base.geometry
        self.pose = base.pose
        self.object_position_track = base.object_position_track
        self.occlusion_trajectory = occlusion


def angle_candidate_features(candidate: torch.Tensor) -> torch.Tensor:
    camera_angle = candidate[..., :2]
    relative_delta = candidate[..., 2:4]
    current_angle = candidate[..., 4:6]
    camera_harmonic = torch.stack(
        (2 * camera_angle[..., 0] * camera_angle[..., 1],
         camera_angle[..., 1].square() - camera_angle[..., 0].square()), -1
    )
    delta_harmonic = torch.stack(
        (2 * relative_delta[..., 0] * relative_delta[..., 1],
         relative_delta[..., 1].square() - relative_delta[..., 0].square()), -1
    )
    return torch.cat((camera_angle, relative_delta, current_angle, camera_harmonic, delta_harmonic), -1)


def person_trajectory_from_pose(pose: torch.Tensor) -> torch.Tensor:
    points = pose.reshape(*pose.shape[:-1], NUM_FRAMES, 17, 2)
    torso_ids = torch.as_tensor([5, 6, 11, 12], device=pose.device)
    torso = points.index_select(-2, torso_ids)
    present = torso.norm(dim=-1) > 0.02
    denom = present.sum(-1, keepdim=True).clamp_min(1).to(points.dtype)
    center = (torso * present[..., None]).sum(-2) / denom
    center = torch.nan_to_num(center)
    velocity = torch.zeros_like(center)
    velocity[..., 1:, :] = center[..., 1:, :] - center[..., :-1, :]
    return torch.cat((center, velocity), -1)


def build_state(
    raw: EnvCache,
    episodes: torch.Tensor,
    history: torch.Tensor,
    history_count: torch.Tensor,
) -> dict[str, torch.Tensor]:
    episodes = episodes.long()
    history = history.long()
    count = history_count.long().clamp(1, history.shape[1])
    batch = episodes.shape[0]
    positions = torch.arange(history.shape[1], device=episodes.device)[None, :]
    observed = positions < count[:, None]
    observed_f = observed.float()
    denominator = count.float().clamp_min(1.0)[:, None]
    current = history.gather(1, (count - 1)[:, None]).squeeze(1)

    observed_pose = raw.pose[episodes[:, None], history]
    person_per_view = person_trajectory_from_pose(observed_pose)
    person = (person_per_view * observed_f[:, :, None, None]).sum(1) / denominator[:, :, None]
    object_track = raw.object_position_track[episodes[:, None], history]
    person_xy = person_per_view[..., :2]
    object_xy = object_track[..., 1:3]
    relative_xy = object_xy - person_xy[..., None, :]
    object_sensor = torch.cat((object_track, relative_xy), -1)
    object_sensor = (object_sensor * observed_f[:, :, None, None, None]).sum(1) / denominator[:, :, None, None]
    observed_occ = raw.occlusion_trajectory[episodes[:, None], history]
    occlusion = (observed_occ * observed_f[:, :, None, None]).sum(1) / denominator[:, :, None]

    observed_views = torch.zeros((batch, NUM_VIEWS), dtype=torch.bool, device=episodes.device)
    observed_views.scatter_(1, history, observed)
    current_angle = raw.geometry[episodes, current, :2]
    geometry = raw.geometry[episodes, :, :2]
    sin_delta = geometry[..., 0] * current_angle[:, None, 1] - geometry[..., 1] * current_angle[:, None, 0]
    cos_delta = (geometry * current_angle[:, None]).sum(-1)
    valid = raw.valid[episodes] & raw.reachable[episodes, current] & (~observed_views)
    candidate = torch.cat(
        (geometry, torch.stack((sin_delta, cos_delta), -1),
         current_angle[:, None].expand(-1, NUM_VIEWS, -1), valid[..., None].float()), -1
    )
    return {
        "person_trajectory": person,
        "object_trajectory": object_sensor,
        "occlusion_trajectory": occlusion,
        "current_relative_angle": current_angle,
        "candidate": candidate,
        "mask": valid,
    }


class EnvironmentOcclusionPolicy(nn.Module):
    """v1 trajectory scorer plus a low-capacity observed occlusion branch."""

    def __init__(self) -> None:
        super().__init__()
        self.person_temporal = nn.Sequential(nn.Conv1d(4, 16, 3, padding=1), nn.GroupNorm(4, 16), nn.GELU())
        self.person_summary = nn.Sequential(nn.Linear(16 * NUM_FRAMES, 32), nn.LayerNorm(32), nn.GELU())
        self.object_temporal = nn.Sequential(nn.Conv1d(NUM_OBJECT_CLASSES * OBJECT_CHANNELS, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.GELU())
        self.object_summary = nn.Sequential(nn.Linear(64 * NUM_FRAMES, 64), nn.LayerNorm(64), nn.GELU())
        self.occlusion_temporal = nn.Sequential(nn.Conv1d(OCC_DIM, 16, 3, padding=1), nn.GroupNorm(4, 16), nn.GELU())
        self.occlusion_summary = nn.Sequential(nn.Linear(16 * NUM_FRAMES, 32), nn.LayerNorm(32), nn.GELU())
        self.angle = nn.Sequential(nn.Linear(2, 16), nn.LayerNorm(16), nn.GELU())
        self.context = nn.Sequential(nn.Linear(32 + 64 + 32 + 16, 64), nn.LayerNorm(64), nn.GELU())
        self.candidate = nn.Sequential(nn.Linear(10, 64), nn.LayerNorm(64), nn.GELU())
        self.context_to_action = nn.Linear(64, 64, bias=False)
        self.residual = nn.Sequential(nn.Linear(128, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        person = self.person_summary(self.person_temporal(state["person_trajectory"].transpose(1, 2)).flatten(1))
        obj = state["object_trajectory"].reshape(state["object_trajectory"].shape[0], NUM_FRAMES, NUM_OBJECT_CLASSES * OBJECT_CHANNELS)
        obj = self.object_summary(self.object_temporal(obj.transpose(1, 2)).flatten(1))
        occ = self.occlusion_summary(self.occlusion_temporal(state["occlusion_trajectory"].transpose(1, 2)).flatten(1))
        angle = self.angle(state["current_relative_angle"])
        context = self.context(torch.cat((person, obj, occ, angle), -1))
        candidate = self.candidate(angle_candidate_features(state["candidate"]))
        expanded = context[:, None].expand(-1, NUM_VIEWS, -1)
        q = (self.context_to_action(context)[:, None] * candidate).sum(-1) / math.sqrt(64.0)
        return q + 0.20 * self.residual(torch.cat((expanded, candidate), -1)).squeeze(-1)


def initialize_from_v1(model: EnvironmentOcclusionPolicy, checkpoint: Path, device: torch.device) -> None:
    base = AngleObjectTrajectoryPolicy().to(device)
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    base.load_state_dict(saved.get("online", saved), strict=True)
    with torch.no_grad():
        for name, value in base.state_dict().items():
            if name == "context.0.weight":
                model.state_dict()[name][:, : value.shape[1]].copy_(value)
            elif name in model.state_dict() and model.state_dict()[name].shape == value.shape:
                model.state_dict()[name].copy_(value)


def class_bank(classes: Sequence[int], device: torch.device) -> torch.Tensor:
    return torch.as_tensor(sorted(int(x) for x in classes), device=device, dtype=torch.long)


def paths_for(raw: EnvCache, episode: int, start: int) -> list[tuple[int, int]]:
    valid = raw.valid[episode].cpu().numpy().astype(bool)
    reachable = raw.reachable[episode].cpu().numpy().astype(bool)
    first = [x for x in range(NUM_VIEWS) if valid[x] and reachable[start, x] and x != start]
    result = []
    for a in first:
        for b in range(NUM_VIEWS):
            if valid[b] and reachable[a, b] and b not in {start, a}:
                result.append((a, b))
    return result


def path_top1(raw: EnvCache, episodes: torch.Tensor, paths: torch.Tensor, bank: torch.Tensor) -> torch.Tensor:
    score = raw.logits[episodes[:, None], paths].mean(1).index_select(-1, bank)
    local = torch.searchsorted(bank, raw.labels[episodes])
    return score.argmax(-1).eq(local)


def path_cost(raw: EnvCache, episodes: torch.Tensor, paths: torch.Tensor) -> torch.Tensor:
    return (raw.cost[episodes, paths[:, 0], paths[:, 1]] + raw.cost[episodes, paths[:, 1], paths[:, 2]]).nan_to_num(1.0)


@torch.inference_mode()
def rollout(model: nn.Module, raw: EnvCache, episodes: torch.Tensor, starts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    one = torch.ones(len(episodes), dtype=torch.long, device=episodes.device)
    state1 = build_state(raw, episodes, starts[:, None], one)
    a1 = model(state1).masked_fill(~state1["mask"], -torch.inf).argmax(-1)
    history2 = torch.stack((starts, a1), 1)
    state2 = build_state(raw, episodes, history2, torch.full_like(starts, 2))
    a2 = model(state2).masked_fill(~state2["mask"], -torch.inf).argmax(-1)
    return torch.stack((starts, a1, a2), -1), state2["mask"].any(-1)


def random_exact(raw: EnvCache, episodes: torch.Tensor, starts: torch.Tensor, bank: torch.Tensor) -> dict[str, float]:
    values, costs = [], []
    for ep, start in zip(episodes.cpu().tolist(), starts.cpu().tolist()):
        paths = paths_for(raw, int(ep), int(start))
        if not paths:
            continue
        path_tensor = torch.as_tensor([[start, a, b] for a, b in paths], device=raw.device)
        ep_tensor = torch.full((len(paths),), int(ep), device=raw.device, dtype=torch.long)
        values.append(float(path_top1(raw, ep_tensor, path_tensor, bank).float().mean()))
        costs.append(float(path_cost(raw, ep_tensor, path_tensor).mean()))
    return {"top1": float(np.mean(values)) if values else 0.0, "mean_movement_cost": float(np.mean(costs)) if costs else 0.0, "episodes": float(len(values))}


def random_sample(raw: EnvCache, episodes: torch.Tensor, starts: torch.Tensor, bank: torch.Tensor, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    paths, ids = [], []
    for ep, start in zip(episodes.cpu().tolist(), starts.cpu().tolist()):
        options = paths_for(raw, int(ep), int(start))
        if options:
            a, b = options[int(rng.integers(len(options)))]
            ids.append(ep); paths.append([start, a, b])
    if not paths:
        return {"top1": 0.0, "mean_movement_cost": 0.0, "episodes": 0.0}
    ep_tensor = torch.as_tensor(ids, device=raw.device, dtype=torch.long)
    path_tensor = torch.as_tensor(paths, device=raw.device, dtype=torch.long)
    return {"top1": float(path_top1(raw, ep_tensor, path_tensor, bank).float().mean()), "mean_movement_cost": float(path_cost(raw, ep_tensor, path_tensor).mean()), "episodes": float(len(paths))}


def oracle(raw: EnvCache, episodes: torch.Tensor, starts: torch.Tensor, bank: torch.Tensor) -> dict[str, float]:
    values, costs = [], []
    for ep, start in zip(episodes.cpu().tolist(), starts.cpu().tolist()):
        options = paths_for(raw, int(ep), int(start))
        if not options:
            continue
        paths = torch.as_tensor([[start, a, b] for a, b in options], device=raw.device)
        ep_tensor = torch.full((len(options),), int(ep), device=raw.device, dtype=torch.long)
        score = raw.logits[ep_tensor[:, None], paths].mean(1).index_select(-1, bank)
        label = raw.labels[ep]
        target = score[:, torch.searchsorted(bank, label)]
        best = target.argmax()
        values.append(float(score[best].argmax().item() == torch.searchsorted(bank, label).item()))
        costs.append(float(path_cost(raw, ep_tensor[best:best + 1], paths[best:best + 1])[0]))
    return {"top1": float(np.mean(values)) if values else 0.0, "mean_movement_cost": float(np.mean(costs)) if costs else 0.0, "episodes": float(len(values))}


@torch.inference_mode()
def evaluate(model: nn.Module | None, raw: EnvCache, classes: Sequence[int], start: int, seed: int) -> dict[str, Any]:
    bank = class_bank(classes, raw.device)
    keep = torch.isin(raw.labels, bank) & raw.valid[:, start] & (raw.valid.sum(-1) >= 3)
    episodes = keep.nonzero().flatten()
    starts = torch.full_like(episodes, start)
    if not len(episodes):
        return {"episodes": 0}
    single = raw.logits[episodes, starts].index_select(-1, bank).argmax(-1).eq(torch.searchsorted(bank, raw.labels[episodes]))
    report: dict[str, Any] = {
        "episodes": int(len(episodes)),
        "single": {"top1": float(single.float().mean())},
        "random_exact": random_exact(raw, episodes, starts, bank),
        "oracle": oracle(raw, episodes, starts, bank),
        "random_seeds": [random_sample(raw, episodes, starts, bank, seed + i) for i in range(10)],
    }
    random_values = np.asarray([x["top1"] for x in report["random_seeds"]], dtype=np.float64)
    report["random_10_mean_std"] = {"mean": float(random_values.mean()), "std": float(random_values.std(ddof=1))}
    if model is not None:
        paths, complete = rollout(model, raw, episodes, starts)
        if bool(complete.all()):
            selected = path_top1(raw, episodes, paths, bank)
            report["policy"] = {"top1": float(selected.float().mean()), "mean_movement_cost": float(path_cost(raw, episodes, paths).mean())}
        else:
            report["policy"] = {"top1": float(path_top1(raw, episodes[complete], paths[complete], bank).float().mean()), "completion_rate": float(complete.float().mean())}
    return report


def build_levels(raw: LatestGPUCache, split: str, seed: int, output: Path) -> dict[str, EnvCache]:
    output.mkdir(parents=True, exist_ok=True)
    levels: dict[str, EnvCache] = {}
    for name, strength in LEVELS.items():
        occ, logits = make_occlusion(raw, strength, seed)
        np.savez_compressed(output / f"{name}.npz", occlusion=occ.detach().cpu().numpy(), logits=logits.detach().cpu().numpy(), labels=raw.labels.cpu().numpy())
        levels[name] = EnvCache(raw, occ, logits)
        print("OCCLUSION_CACHE_READY", split, name, json.dumps({"episodes": raw.num_episodes, "mean_occlusion": float(occ[..., 0].mean())}), flush=True)
    return levels


def phase0(levels: dict[str, EnvCache], classes: Sequence[int], start: int, seed: int) -> dict[str, Any]:
    result = {}
    for name, raw in levels.items():
        result[name] = evaluate(None, raw, classes, start, seed)
    return result


def train_one(
    seed: int,
    levels: dict[str, EnvCache],
    val_levels: dict[str, EnvCache],
    args: argparse.Namespace,
) -> dict[str, Any]:
    run = args.output_root / f"seed_{seed}"
    run.mkdir(parents=True, exist_ok=True)
    complete = run / "complete.json"
    if complete.exists():
        return json.loads(complete.read_text())
    device = levels["none"].device
    seed_all(seed)
    generator = torch.Generator(device=device).manual_seed(seed)
    model = EnvironmentOcclusionPolicy().to(device)
    initialize_from_v1(model, BASE_POLICY, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.04)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=args.learning_rate * 0.05)
    seen_bank = class_bank(SEEN, device)
    base = levels["none"]
    pool = ((base.valid.sum(-1) == 4) & torch.isin(base.labels, seen_bank)).nonzero().flatten()
    if not len(pool):
        raise RuntimeError("empty seen training pool")
    audit = {
        "seed": seed,
        "training_episodes": int(len(pool)),
        "training_classes": SEEN,
        "true_unseen_5way": UNSEEN,
        "initialization": str(BASE_POLICY),
        "loss": "v1 listwise + 0.5 pairwise, masked-logit delta-margin targets",
        "training_mix": dict(zip(TRAIN_MIX, TRAIN_MIX_PROB)),
        "policy_inputs": ["causal person trajectory", "causal object trajectory", "sample-specific geometry", "observed occlusion trajectory"],
        "policy_excludes": ["GT label", "future candidate occlusion", "future candidate logits", "action class ID"],
        "movement_cost_in_training": False,
        "frame_shape_reference": [640, 480],
        "feature_space_proxy": True,
    }
    dump(run / "audit.json", audit)
    best_score = -float("inf")
    best_epoch = 0
    log_path = run / "train.log"
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        for epoch in range(1, args.epochs + 1):
            begun = time.monotonic()
            losses = []
            chosen_levels = torch.multinomial(torch.as_tensor(TRAIN_MIX_PROB, device=device), args.updates_per_epoch * args.batch_size, replacement=True, generator=generator)
            for update in range(args.updates_per_epoch):
                begin = update * args.batch_size
                end = begin + args.batch_size
                ids = torch.randint(len(pool), (args.batch_size,), device=device, generator=generator)
                episodes = pool[ids]
                level_ids = chosen_levels[begin:end]
                permutation = torch.rand((args.batch_size, NUM_VIEWS), device=device, generator=generator).argsort(-1)
                stages = torch.multinomial(torch.as_tensor([0.45, 0.45, 0.10], device=device), args.batch_size, replacement=True, generator=generator) + 1
                history = permutation[:, :3]
                # Group the batch by level so every state/target pair uses the
                # same synthetic environment, without a hidden future-view leak.
                prediction_parts = []
                for level_id, name in enumerate(TRAIN_MIX):
                    selected = level_ids == level_id
                    if not bool(selected.any()):
                        continue
                    local_ep, local_hist, local_stage = episodes[selected], history[selected], stages[selected]
                    state = build_state(levels[name], local_ep, local_hist, local_stage)
                    target = utility_targets(levels[name], local_ep, local_hist, local_stage, state["mask"], seen_bank)
                    prediction_parts.append((state, target, local_stage))
                optimizer.zero_grad(set_to_none=True)
                batch_loss = torch.zeros((), device=device)
                count = 0
                for state, target, local_stage in prediction_parts:
                    prediction = model(state)
                    loss, _ = multistep_loss(prediction, target, state["mask"], local_stage)
                    batch_loss = batch_loss + loss * len(target)
                    count += len(target)
                batch_loss = batch_loss / max(count, 1)
                if not torch.isfinite(batch_loss):
                    raise FloatingPointError(f"non-finite loss seed={seed} epoch={epoch}")
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                losses.append(float(batch_loss.detach()))
            scheduler.step()
            torch.cuda.synchronize(device)
            validation = evaluate(model, val_levels["medium"], PSEUDO, 0, seed + 7700 + epoch)
            score = float(validation["policy"]["top1"] - validation["random_exact"]["top1"])
            improved = score > best_score
            if improved:
                best_score, best_epoch = score, epoch
            payload = {"online": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "epoch": epoch, "best_score": best_score, "best_epoch": best_epoch, "seed": seed}
            torch.save(payload, run / "last.pt")
            if improved:
                torch.save(payload, run / "best.pt")
            row = {"epoch": epoch, "loss": float(np.mean(losses)), "seconds": time.monotonic() - begun, "training_episodes": int(len(pool)), "selection_medium_fixed0_gain": score, "best_score": best_score, "best_epoch": best_epoch}
            log.write(json.dumps(row) + "\n")
            print("ENV_OCCLUSION_TRAIN", json.dumps(row), flush=True)
    result = {"seed": seed, "best_score": best_score, "best_epoch": best_epoch, "checkpoint": str(run / "best.pt"), "training_episodes": int(len(pool))}
    dump(complete, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--updates-per-epoch", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260909, 20260910, 20260911])
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda:0")
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    args.output_root.mkdir(parents=True, exist_ok=True)
    dump(args.output_root / "config_resolved.json", {
        "cache_root": str(CACHE_ROOT), "track_root": str(TRACK_ROOT), "base_policy": str(BASE_POLICY),
        "epochs": args.epochs, "updates_per_epoch": args.updates_per_epoch, "batch_size": args.batch_size,
        "seeds": args.seeds, "strict_unseen": UNSEEN, "mode": "feature-space view-consistent synthetic occlusion proxy",
        "raw_video_recompute": False, "clean_pose_object_tracks": True,
    })
    base_train = LatestGPUCache(CACHE_ROOT / "dqn_train", device)
    base_val = LatestGPUCache(CACHE_ROOT / "val", device)
    base_test = LatestGPUCache(CACHE_ROOT / "test", device)
    attach_object_tracks(base_train, "dqn_train")
    attach_object_tracks(base_val, "val")
    attach_object_tracks(base_test, "test")
    attach_view_valid(base_train, "dqn_train")
    attach_view_valid(base_val, "val")
    attach_view_valid(base_test, "test")
    print("ENV_OCCLUSION_READY", json.dumps({"train": base_train.num_episodes, "val": base_val.num_episodes, "test": base_test.num_episodes, "base_policy": str(BASE_POLICY), "unseen": UNSEEN}), flush=True)
    train_levels = build_levels(base_train, "dqn_train", 1100, args.output_root / "synthetic_cache" / "dqn_train")
    val_levels = build_levels(base_val, "val", 2200, args.output_root / "synthetic_cache" / "val")
    test_levels = build_levels(base_test, "test", 3300, args.output_root / "synthetic_cache" / "test")
    phase = {
        "seen_validation": phase0(val_levels, PSEUDO, 0, 2200),
        "true_unseen_5way": phase0(test_levels, UNSEEN, 0, 3300),
    }
    dump(args.output_root / "phase0_diagnostic.json", phase)
    print("ENV_OCCLUSION_PHASE0", json.dumps({"validation_medium": phase["seen_validation"]["medium"], "unseen_medium": phase["true_unseen_5way"]["medium"]}), flush=True)
    records = [train_one(seed, train_levels, val_levels, args) for seed in args.seeds]
    chosen = max(records, key=lambda x: float(x["best_score"]))
    selected = EnvironmentOcclusionPolicy().to(device)
    saved = torch.load(chosen["checkpoint"], map_location=device, weights_only=False)
    selected.load_state_dict(saved["online"], strict=True)
    reports: dict[str, Any] = {}
    for level, raw in test_levels.items():
        reports[level] = {}
        for start in range(4):
            reports[level][f"fixed{start}"] = evaluate(selected, raw, UNSEEN, start, 30000 + start)
    # Same selected checkpoint, on clean and medium validation, for selection audit.
    validation_reports = {level: {f"fixed{start}": evaluate(selected, raw, PSEUDO, start, 22000 + start) for start in range(4)} for level, raw in val_levels.items()}
    final = {
        "variant": "multistep_environment_occlusion_policy_v2",
        "recognizer_cache": str(CACHE_ROOT),
        "base_policy": str(BASE_POLICY),
        "strict_unseen_classes": UNSEEN,
        "selected": {"seed": chosen["seed"], "checkpoint": chosen["checkpoint"], "best_epoch": saved["epoch"]},
        "training_runs": records,
        "phase0": phase,
        "validation": validation_reports,
        "true_unseen_5way": reports,
        "random_protocol": "exact expectation plus 10 sampled evaluation seeds",
        "primary_protocol": "fixed view0, true unseen 5-way, medium synthetic occlusion",
        "limitations": ["feature-space occlusion proxy because latest dqn_train lacks aligned raw X3D tensors", "clean pose/object trajectories are retained and explicitly audited", "no PPO; supervised offline ranking only"],
    }
    dump(args.output_root / "summary.json", final)
    print("ENV_OCCLUSION_TRAIN_EVAL_COMPLETE", json.dumps({"output": str(args.output_root), "selected": chosen, "primary": reports["medium"]["fixed0"]}), flush=True)


if __name__ == "__main__":
    main()
