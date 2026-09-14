"""Two-robot depth-aware active-view masked Double-DQN experiment.

The experiment is intentionally self-contained.  It uses the frozen
frame-0 cache and the already selected lightweight weighted-logit fusion gate,
but creates a new policy state, action protocol, reward, checkpoints, and
30-seed report in a separate output directory.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .. import fusion_lightweight_gating_v1 as fusion_gate
from .. import fusion_weighted_common_v1 as fusion_common
from .. import multistep_angle_object_trajectory_policy_v1 as trajectory_api
from .. import multistep_g18_view_policy_v1 as cache_api


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
DEFAULT_DEPTH_ROOT = ROOT / "data" / "new50_5_group1_zsl_depth_first_frame_fp32"
DEFAULT_GATE_CHECKPOINT = (
    ROOT
    / "work_dir"
    / "fusion_lightweight_gating_v1"
    / "models"
    / "seed_20260910"
    / "best.pt"
)
DEFAULT_OUTPUT_ROOT = ROOT / "work_dir" / "dual_robot_depth_ddqn_v1"

NUM_VIEWS = 4
NUM_FRAMES = 13
NUM_OBJECTS = 50
NUM_CLASSES = 55
PAIR_LIST = tuple((i, j) for i in range(NUM_VIEWS) for j in range(i + 1, NUM_VIEWS))
PAIR_INDEX = np.full((NUM_VIEWS, NUM_VIEWS), -1, dtype=np.int64)
for _pair_id, (_left, _right) in enumerate(PAIR_LIST):
    PAIR_INDEX[_left, _right] = _pair_id
    PAIR_INDEX[_right, _left] = _pair_id

TRUE_UNSEEN = [14, 15, 25, 39, 46]
SEEN_CLASSES = sorted(set(range(NUM_CLASSES)) - set(TRUE_UNSEEN))
PSEUDO_UNSEEN = [0, 2, 9, 10, 11, 17, 26, 34, 49, 50]
DEFAULT_TRAIN_SEEDS = [20260909, 20260910, 20260911]
DEFAULT_EVAL_SEEDS = list(range(20260920, 20260950))
DEFAULT_VARIANTS = ["full_depth19", "human_only", "object_only_depth19"]
LSTM_VARIANTS = {"full_lstm19", "human_only_lstm", "object_only_lstm19"}
HUMAN_TRAJECTORY_ONLY_VARIANT = "human_trajectory_only"
HUMAN_ORIENTATION_ONLY_VARIANT = "human_orientation_only"


def variant_uses_human(variant: str) -> bool:
    return variant not in {"object_only_depth19", "object_only_lstm19"}


def variant_uses_object(variant: str) -> bool:
    return variant not in {
        "human_only",
        "human_only_lstm",
        HUMAN_TRAJECTORY_ONLY_VARIANT,
        HUMAN_ORIENTATION_ONLY_VARIANT,
    }


def variant_uses_lstm(variant: str) -> bool:
    return variant in LSTM_VARIANTS


def variant_includes_depth(variant: str) -> bool:
    return variant != "full_pooled16"


def human_feature_channels(variant: str) -> int:
    """Return the number of per-frame channels exposed to the human branch."""

    if variant == HUMAN_TRAJECTORY_ONLY_VARIANT:
        return 4
    if variant == HUMAN_ORIENTATION_ONLY_VARIANT:
        return 2
    return 6


def dump_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def finite_float(value: torch.Tensor, fill: float = 1.0) -> torch.Tensor:
    return torch.nan_to_num(value.float(), nan=fill, posinf=fill, neginf=fill)


def load_depth_lookup(depth_root: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Index extracted per-sample depth rows by the exact RGB sample name."""

    lookup: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if not depth_root.exists():
        raise FileNotFoundError(depth_root)
    for directory in sorted(depth_root.iterdir()):
        if not directory.is_dir():
            continue
        names_path = directory / "sample_names.npy"
        values_path = directory / "depth_rel_fp32.npy"
        valid_path = directory / "depth_valid_uint8.npy"
        if not (names_path.exists() and values_path.exists() and valid_path.exists()):
            continue
        names = np.load(names_path, allow_pickle=True)
        values = np.load(values_path, mmap_mode="r", allow_pickle=False)
        valid = np.load(valid_path, mmap_mode="r", allow_pickle=False)
        if values.ndim != 2 or values.shape[1] != NUM_OBJECTS:
            raise ValueError(f"unexpected depth shape {values.shape} in {directory}")
        if len(names) != len(values) or valid.shape != values.shape:
            raise ValueError(f"depth metadata mismatch in {directory}")
        for index, name in enumerate(names.tolist()):
            key = str(name)
            lookup.setdefault(key, (values[index], valid[index]))
    if not lookup:
        raise RuntimeError(f"no extracted depth rows found under {depth_root}")
    return lookup


def join_depth_rows(
    cache_root: Path,
    split: str,
    depth_lookup: dict[str, tuple[np.ndarray, np.ndarray]],
    expected_episodes: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    metadata = json.loads((cache_root / split / "metadata.json").read_text(encoding="utf-8"))
    episodes = metadata["episodes"]
    if len(episodes) != expected_episodes:
        raise ValueError(f"{split}: metadata has {len(episodes)} episodes, expected {expected_episodes}")
    depth = np.zeros((expected_episodes, NUM_VIEWS, NUM_OBJECTS), dtype=np.float32)
    valid = np.zeros((expected_episodes, NUM_VIEWS, NUM_OBJECTS), dtype=np.uint8)
    missing: list[str] = []
    joined = 0
    for episode_id, episode in enumerate(episodes):
        views = episode.get("views", [])
        if len(views) > NUM_VIEWS:
            raise ValueError(f"{split} episode {episode_id} has {len(views)} views")
        for view_id, view in enumerate(views):
            name = str(view["sample_name"])
            row = depth_lookup.get(name)
            if row is None:
                missing.append(name)
                continue
            # The cache is stored in signed-bearing rank order, while the
            # metadata view list retains source/action order.  Always join by
            # the explicit rank field instead of list position.
            slot = int(view.get("signed_bearing_rank", view_id))
            if not 0 <= slot < NUM_VIEWS:
                raise ValueError(f"invalid signed_bearing_rank={slot} in {split} episode {episode_id}")
            depth[episode_id, slot] = row[0]
            valid[episode_id, slot] = row[1]
            joined += 1
    if missing:
        raise FileNotFoundError(
            f"{split}: {len(missing)} views missing from depth lookup; examples {missing[:3]}"
        )
    audit = {
        "split": split,
        "episodes": expected_episodes,
        "views_joined": joined,
        "depth_rows_available": len(depth_lookup),
        "join_key": "metadata.episodes[*].views[*].sample_name",
        "depth_definition": "first-frame MiDaS normalized relative inverse depth; not metric depth",
    }
    return depth, valid.astype(bool), audit


def load_raw(
    cache_root: Path,
    track_root: Path,
    split: str,
    device: torch.device,
    depth_lookup: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[Any, dict[str, Any]]:
    raw = cache_api.GPUCache(cache_root / split, device)
    cache_api.attach_view_valid(raw, cache_root, split)

    yaw_path = cache_root / split / "body_yaw_deg.npy"
    yaw = np.load(yaw_path, allow_pickle=False)
    if tuple(yaw.shape) != (raw.num_episodes, NUM_VIEWS):
        raise ValueError(f"body yaw shape {yaw.shape} is incompatible with {split}")
    raw.body_yaw_deg = torch.from_numpy(np.array(yaw, copy=True)).to(
        device=device, dtype=torch.float32
    )

    trajectory_api.attach_object_tracks(raw, track_root, split)
    depth, depth_valid, depth_audit = join_depth_rows(
        cache_root, split, depth_lookup, raw.num_episodes
    )
    raw.depth = torch.from_numpy(depth).to(device=device, dtype=torch.float32)
    raw.depth_valid = torch.from_numpy(depth_valid).to(device=device, dtype=torch.bool)
    raw._dual_robot_depth_audit = depth_audit
    return raw, depth_audit


def person_center_velocity(pose: torch.Tensor) -> torch.Tensor:
    """Return center/velocity channels with shape [B,T,4]."""

    points = pose.reshape(*pose.shape[:-1], NUM_FRAMES, 17, 2)
    torso_ids = torch.as_tensor([5, 6, 11, 12], device=pose.device)
    torso = points.index_select(-2, torso_ids)
    present = torso.norm(dim=-1) > 0.02
    denominator = present.sum(-1, keepdim=True).clamp_min(1).to(points.dtype)
    center = (torso * present[..., None]).sum(-2) / denominator
    center = torch.nan_to_num(center)
    velocity = torch.zeros_like(center)
    velocity[..., 1:, :] = center[..., 1:, :] - center[..., :-1, :]
    return torch.cat((center, velocity), dim=-1)


def human_sensor(
    raw: Any,
    episodes: torch.Tensor,
    anchor: torch.Tensor,
    variant: str = "human_only",
) -> torch.Tensor:
    pose = raw.pose[episodes, anchor]
    base = person_center_velocity(pose)
    yaw = torch.deg2rad(torch.nan_to_num(raw.body_yaw_deg[episodes, anchor]))
    yaw_sincos = torch.stack((yaw.sin(), yaw.cos()), dim=-1)
    yaw_sincos = yaw_sincos[:, None, :].expand(-1, NUM_FRAMES, -1)
    if variant == HUMAN_TRAJECTORY_ONLY_VARIANT:
        result = base
    elif variant == HUMAN_ORIENTATION_ONLY_VARIANT:
        result = yaw_sincos
    else:
        result = torch.cat((base, yaw_sincos), dim=-1)
    expected = human_feature_channels(variant)
    if result.shape[-1] != expected:
        raise AssertionError(
            f"human sensor must have {expected} channels for {variant}, got {result.shape}"
        )
    return result


def pooled_object_sensor(
    raw: Any,
    episodes: torch.Tensor,
    anchor: torch.Tensor,
    include_depth: bool,
) -> torch.Tensor:
    """Build permutation-invariant 16- or 19-channel object trajectories."""

    tracks = raw.object_position_track[episodes, anchor]
    # Stored fields: presence, center_x, center_y, confidence.
    person_xy = person_center_velocity(raw.pose[episodes, anchor])[..., :2]
    relative_xy = tracks[..., 1:3] - person_xy[:, :, None, :]

    numeric = [tracks[..., 1:4], relative_xy]
    if include_depth:
        object_depth = raw.depth[episodes, anchor]
        object_valid = raw.depth_valid[episodes, anchor]
        person_depth = object_depth[:, :1]
        person_depth_valid = object_valid[:, :1]
        relative_depth = object_depth - person_depth
        depth_valid = object_valid & person_depth_valid
        relative_depth = torch.where(depth_valid, relative_depth, torch.zeros_like(relative_depth))
        repeated_depth = relative_depth[:, None, :, None].expand(
            -1, NUM_FRAMES, -1, -1
        )
        numeric.append(repeated_depth)

    numeric_values = torch.cat(numeric, dim=-1)
    presence = tracks[..., 0].clamp(0.0, 1.0)
    count = presence.sum(dim=-1, keepdim=True)
    denominator = count.clamp_min(1.0)
    weights = presence.unsqueeze(-1)
    mean = (numeric_values * weights).sum(dim=-2) / denominator
    present_mask = presence.unsqueeze(-1) > 0.0
    minimum = numeric_values.masked_fill(~present_mask, float("inf")).amin(dim=-2)
    maximum = numeric_values.masked_fill(~present_mask, float("-inf")).amax(dim=-2)
    has_object = count > 0.0
    minimum = torch.where(has_object, minimum, torch.zeros_like(minimum))
    maximum = torch.where(has_object, maximum, torch.zeros_like(maximum))
    count_ratio = count / float(NUM_OBJECTS)
    pooled = torch.cat((count_ratio, mean, minimum, maximum), dim=-1)
    expected = 19 if include_depth else 16
    if pooled.shape[-1] != expected:
        raise AssertionError(f"expected pooled object width {expected}, got {pooled.shape}")
    return torch.nan_to_num(pooled)


def relative_pair_angles(
    candidate: torch.Tensor, source: torch.Tensor
) -> torch.Tensor:
    sin_delta = candidate[..., 0] * source[..., 1] - candidate[..., 1] * source[..., 0]
    cos_delta = (candidate * source).sum(dim=-1)
    return torch.stack((sin_delta, cos_delta), dim=-1)


def assignment_matrix(
    raw: Any,
    episodes: torch.Tensor,
    start_a: torch.Tensor,
    start_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return [B,4,4] minimum assignment cost and feasibility."""

    batch = episodes.shape[0]
    cost = torch.full((batch, NUM_VIEWS, NUM_VIEWS), float("inf"), device=raw.device)
    feasible = torch.zeros((batch, NUM_VIEWS, NUM_VIEWS), dtype=torch.bool, device=raw.device)
    rows = torch.arange(batch, device=raw.device)

    for left in range(NUM_VIEWS):
        for right in range(left + 1, NUM_VIEWS):
            a_left = start_a == left
            a_right = start_a == right
            b_left = start_b == left
            b_right = start_b == right
            # A may stay at its own start and B may stay at its own start;
            # cache diagonal reachability is not assumed for this rule.
            reach_a_left = raw.reachable[episodes, start_a, left] | a_left
            reach_a_right = raw.reachable[episodes, start_a, right] | a_right
            reach_b_left = raw.reachable[episodes, start_b, left] | b_left
            reach_b_right = raw.reachable[episodes, start_b, right] | b_right
            c_a_left = finite_float(raw.cost[episodes, start_a, left])
            c_a_right = finite_float(raw.cost[episodes, start_a, right])
            c_b_left = finite_float(raw.cost[episodes, start_b, left])
            c_b_right = finite_float(raw.cost[episodes, start_b, right])
            c_a_left = torch.where(a_left, torch.zeros_like(c_a_left), c_a_left)
            c_a_right = torch.where(a_right, torch.zeros_like(c_a_right), c_a_right)
            c_b_left = torch.where(b_left, torch.zeros_like(c_b_left), c_b_left)
            c_b_right = torch.where(b_right, torch.zeros_like(c_b_right), c_b_right)
            first_valid = reach_a_left & reach_b_right
            second_valid = reach_a_right & reach_b_left
            first_cost = torch.where(
                first_valid, c_a_left + c_b_right, torch.full_like(c_a_left, float("inf"))
            )
            second_cost = torch.where(
                second_valid, c_a_right + c_b_left, torch.full_like(c_a_left, float("inf"))
            )
            pair_cost = torch.minimum(first_cost, second_cost)
            pair_valid = first_valid | second_valid
            cost[rows, left, right] = pair_cost
            cost[rows, right, left] = pair_cost
            feasible[rows, left, right] = pair_valid
            feasible[rows, right, left] = pair_valid
    return cost, feasible


def build_state(
    raw: Any,
    episodes: torch.Tensor,
    start_a: torch.Tensor,
    start_b: torch.Tensor,
    selected: torch.Tensor,
    variant: str,
) -> dict[str, torch.Tensor]:
    """Build the causal state for one of the two DDQN decision stages."""

    episodes = episodes.long()
    start_a = start_a.long()
    start_b = start_b.long()
    selected = selected.long()
    rows = torch.arange(len(episodes), device=raw.device)
    use_human = variant_uses_human(variant)
    use_object = variant_uses_object(variant)
    include_depth = variant_includes_depth(variant)

    state: dict[str, torch.Tensor] = {}
    if use_human:
        state["human"] = human_sensor(raw, episodes, start_a, variant)
    if use_object:
        state["object"] = pooled_object_sensor(
            raw, episodes, start_a, include_depth=include_depth
        )

    geometry = raw.geometry[episodes, :, :2]
    source_a = geometry[rows, start_a]
    source_b = geometry[rows, start_b]
    relative_a = relative_pair_angles(geometry, source_a[:, None, :])
    relative_b = relative_pair_angles(geometry, source_b[:, None, :])

    pair_cost, pair_feasible = assignment_matrix(raw, episodes, start_a, start_b)
    selected_present = selected >= 0
    safe_selected = selected.clamp_min(0)
    selected_geometry = geometry[rows, safe_selected]
    selected_relative = relative_pair_angles(geometry, selected_geometry[:, None, :])
    selected_relative = selected_relative * selected_present[:, None, None].float()

    # At stage 0, the pair-cost feature is the cheapest feasible partner for
    # each candidate.  At stage 1, it is the actual selected-target pair cost.
    row_pair_cost = pair_cost[rows, safe_selected]
    row_pair_feasible = pair_feasible[rows, safe_selected]
    best_partner_cost = pair_cost.masked_fill(
        ~pair_feasible | torch.eye(NUM_VIEWS, device=raw.device, dtype=torch.bool)[None],
        float("inf"),
    ).amin(dim=-1)
    best_partner_feasible = pair_feasible.clone()
    best_partner_feasible = best_partner_feasible & ~torch.eye(
        NUM_VIEWS, device=raw.device, dtype=torch.bool
    )[None]
    best_partner_feasible = best_partner_feasible.any(dim=-1)
    pair_cost_feature = torch.where(
        selected_present[:, None], row_pair_cost, best_partner_cost
    )
    pair_feasible_feature = torch.where(
        selected_present[:, None], row_pair_feasible, best_partner_feasible
    )
    pair_cost_feature = finite_float(pair_cost_feature, fill=1.0).clamp(0.0, 2.0)

    cost_a = finite_float(raw.cost[episodes, start_a]).clamp(0.0, 1.0)
    cost_b = finite_float(raw.cost[episodes, start_b]).clamp(0.0, 1.0)
    cost_a[rows, start_a] = 0.0
    cost_b[rows, start_b] = 0.0
    reach_a = raw.reachable[episodes, start_a].clone()
    reach_b = raw.reachable[episodes, start_b].clone()
    reach_a[rows, start_a] = True
    reach_b[rows, start_b] = True

    selected_flag = F.one_hot(safe_selected, NUM_VIEWS).float()
    selected_flag = selected_flag * selected_present[:, None].float()
    candidate = torch.cat(
        (
            geometry,
            relative_a,
            relative_b,
            cost_a[..., None],
            cost_b[..., None],
            selected_relative,
            pair_cost_feature[..., None],
            pair_feasible_feature.float()[..., None],
            selected_flag[..., None],
        ),
        dim=-1,
    )
    if candidate.shape[-1] != 13:
        raise AssertionError(f"candidate feature width must be 13, got {candidate.shape}")

    valid = raw.valid[episodes]
    if selected_present.any():
        valid = valid.clone()
        valid[rows, safe_selected] = False
    stage0_has_partner = best_partner_feasible
    action_mask = valid & torch.where(
        selected_present[:, None], row_pair_feasible, stage0_has_partner
    )
    state["starts"] = torch.cat((source_a, source_b), dim=-1)
    state["selection"] = torch.cat(
        (
            selected_flag,
            F.one_hot(selected_present.long(), 2).float(),
        ),
        dim=-1,
    )
    state["candidate"] = candidate
    state["mask"] = action_mask
    return state


class PairDuelingQNetwork(nn.Module):
    """Small dueling Q network shared by all three state ablations."""

    def __init__(self, variant: str) -> None:
        super().__init__()
        self.variant = variant
        self.use_human = variant_uses_human(variant)
        self.use_object = variant_uses_object(variant)
        self.use_lstm = variant_uses_lstm(variant)
        object_dim = 16 if variant == "full_pooled16" else 19

        context_dim = 0
        if self.use_human:
            human_dim = human_feature_channels(variant)
            if self.use_lstm:
                self.human_temporal = nn.LSTM(
                    input_size=human_dim, hidden_size=32, num_layers=1, batch_first=True
                )
                self.human_summary = nn.Sequential(
                    nn.Linear(32, 64), nn.LayerNorm(64), nn.GELU()
                )
            else:
                self.human_temporal = nn.Sequential(
                    nn.Conv1d(human_dim, 32, kernel_size=3, padding=1),
                    nn.GroupNorm(8, 32),
                    nn.GELU(),
                )
                self.human_summary = nn.Sequential(
                    nn.Linear(32 * NUM_FRAMES, 64), nn.LayerNorm(64), nn.GELU()
                )
            context_dim += 64
        if self.use_object:
            if self.use_lstm:
                self.object_temporal = nn.LSTM(
                    input_size=object_dim, hidden_size=32, num_layers=1, batch_first=True
                )
                self.object_summary = nn.Sequential(
                    nn.Linear(32, 64), nn.LayerNorm(64), nn.GELU()
                )
            else:
                self.object_temporal = nn.Sequential(
                    nn.Conv1d(object_dim, 32, kernel_size=3, padding=1),
                    nn.GroupNorm(8, 32),
                    nn.GELU(),
                )
                self.object_summary = nn.Sequential(
                    nn.Linear(32 * NUM_FRAMES, 64), nn.LayerNorm(64), nn.GELU()
                )
            context_dim += 64
        self.start_encoder = nn.Sequential(
            nn.Linear(4, 16), nn.LayerNorm(16), nn.GELU()
        )
        self.selection_encoder = nn.Sequential(
            nn.Linear(6, 16), nn.LayerNorm(16), nn.GELU()
        )
        context_dim += 32
        self.context = nn.Sequential(
            nn.Linear(context_dim, 128), nn.LayerNorm(128), nn.GELU()
        )
        self.candidate = nn.Sequential(
            nn.Linear(13, 64), nn.LayerNorm(64), nn.GELU()
        )
        self.value = nn.Sequential(
            nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1)
        )
        self.advantage = nn.Sequential(
            nn.Linear(128 + 64, 64), nn.GELU(), nn.Linear(64, 1)
        )

    def forward(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        context_parts: list[torch.Tensor] = []
        if self.use_human:
            if self.use_lstm:
                _, (human, _) = self.human_temporal(state["human"])
                human = human[-1]
            else:
                human = self.human_temporal(state["human"].transpose(1, 2)).flatten(1)
            context_parts.append(self.human_summary(human))
        if self.use_object:
            if self.use_lstm:
                _, (obj, _) = self.object_temporal(state["object"])
                obj = obj[-1]
            else:
                obj = self.object_temporal(state["object"].transpose(1, 2)).flatten(1)
            context_parts.append(self.object_summary(obj))
        context_parts.append(self.start_encoder(state["starts"]))
        context_parts.append(self.selection_encoder(state["selection"]))
        context = self.context(torch.cat(context_parts, dim=-1))
        candidate = self.candidate(state["candidate"])
        expanded = context[:, None, :].expand(-1, NUM_VIEWS, -1)
        advantage = self.advantage(torch.cat((expanded, candidate), dim=-1)).squeeze(-1)
        value = self.value(context)
        return value + advantage - advantage.mean(dim=-1, keepdim=True)


@dataclass
class TransitionPool:
    episode: torch.Tensor
    start_a: torch.Tensor
    start_b: torch.Tensor
    selected: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    done: torch.Tensor
    stage1: torch.Tensor

    @property
    def size(self) -> int:
        return int(self.action.numel())


def margin_numpy(scores: np.ndarray, label: int, bank: np.ndarray) -> float:
    values = scores[bank]
    target_id = int(np.flatnonzero(bank == int(label))[0])
    target = float(values[target_id])
    rival = float(np.max(np.delete(values, target_id)))
    return target - rival


def pair_fused_logits(
    raw: Any,
    gate: nn.Module,
    chunk_size: int = 4096,
) -> torch.Tensor:
    """Fuse every unordered pair using the existing weighted fusion gate."""

    pair_tensor = torch.as_tensor(PAIR_LIST, device=raw.device, dtype=torch.long)
    output: list[torch.Tensor] = []
    with torch.inference_mode():
        for begin in range(0, raw.num_episodes, chunk_size):
            end = min(begin + chunk_size, raw.num_episodes)
            episode_ids = torch.arange(begin, end, device=raw.device, dtype=torch.long)
            repeated = episode_ids[:, None].expand(-1, len(PAIR_LIST)).reshape(-1)
            paths = pair_tensor[None].expand(end - begin, -1, -1).reshape(-1, 2)
            inputs, components = fusion_common.path_inputs(raw, repeated, paths)
            weights = fusion_common.weights_for_method(
                "lightweight_gating", inputs, components, gate
            )
            logits = raw.logits[repeated[:, None], paths]
            fused = (weights[..., None] * logits).sum(dim=1)
            output.append(fused.reshape(end - begin, len(PAIR_LIST), NUM_CLASSES))
    return torch.cat(output, dim=0)


def pair_margin(
    fused: torch.Tensor, labels: torch.Tensor, bank: Sequence[int]
) -> torch.Tensor:
    bank_tensor = torch.as_tensor(bank, device=fused.device, dtype=torch.long)
    scores = fused.index_select(-1, bank_tensor)
    local = torch.searchsorted(bank_tensor, labels)
    if scores.ndim == 2:
        target = scores.gather(-1, local[:, None]).squeeze(-1)
        rivals = scores.clone()
        rivals.scatter_(-1, local[:, None], -torch.inf)
        return target - rivals.amax(dim=-1)
    target = scores.gather(-1, local[:, None, None].expand(-1, scores.shape[1], 1)).squeeze(-1)
    rivals = scores.clone()
    rivals.scatter_(-1, local[:, None, None].expand(-1, scores.shape[1], 1), -torch.inf)
    return target - rivals.amax(dim=-1)


def build_transition_pool(
    raw: Any,
    fused_pairs: torch.Tensor,
    distance_lambda: float,
    device: torch.device,
) -> TransitionPool:
    """Build all legal two-stage pair-selection transitions on seen episodes."""

    valid = raw.valid.detach().cpu().numpy().astype(bool)
    labels = raw.labels.detach().cpu().numpy().astype(np.int64)
    reachable = raw.reachable.detach().cpu().numpy().astype(bool)
    edge_cost = np.nan_to_num(raw.cost.detach().cpu().numpy(), nan=1.0, posinf=1.0, neginf=1.0)
    margins = pair_margin(fused_pairs, raw.labels, SEEN_CLASSES).detach().cpu().numpy()

    episodes_out: list[int] = []
    a_out: list[int] = []
    b_out: list[int] = []
    selected_out: list[int] = []
    action_out: list[int] = []
    reward_out: list[float] = []
    done_out: list[bool] = []
    stage1_out: list[bool] = []

    for episode in range(raw.num_episodes):
        if labels[episode] not in SEEN_CLASSES or valid[episode].sum() != NUM_VIEWS:
            continue
        for start_a in range(NUM_VIEWS):
            for start_b in range(NUM_VIEWS):
                feasible_pair: dict[tuple[int, int], float] = {}
                for pair_id, (left, right) in enumerate(PAIR_LIST):
                    reach_1 = (reachable[episode, start_a, left] or start_a == left) and (
                        reachable[episode, start_b, right] or start_b == right
                    )
                    reach_2 = (reachable[episode, start_a, right] or start_a == right) and (
                        reachable[episode, start_b, left] or start_b == left
                    )
                    cost_1 = edge_cost[episode, start_a, left] + edge_cost[episode, start_b, right]
                    cost_2 = edge_cost[episode, start_a, right] + edge_cost[episode, start_b, left]
                    if reach_1 or reach_2:
                        feasible_costs = []
                        if reach_1:
                            feasible_costs.append(cost_1)
                        if reach_2:
                            feasible_costs.append(cost_2)
                        feasible_pair[(left, right)] = min(feasible_costs)
                first_actions = sorted({view for pair in feasible_pair for view in pair})
                for first in first_actions:
                    # The first action can be the current position of robot A.
                    episodes_out.append(episode)
                    a_out.append(start_a)
                    b_out.append(start_b)
                    selected_out.append(-1)
                    action_out.append(first)
                    reward_out.append(0.0)
                    done_out.append(False)
                    stage1_out.append(True)
                    for (left, right), movement in feasible_pair.items():
                        if first not in (left, right):
                            continue
                        second = right if first == left else left
                        pair_id = int(PAIR_INDEX[left, right])
                        reward = float(margins[episode, pair_id] - distance_lambda * movement)
                        episodes_out.append(episode)
                        a_out.append(start_a)
                        b_out.append(start_b)
                        selected_out.append(first)
                        action_out.append(second)
                        reward_out.append(reward)
                        done_out.append(True)
                        stage1_out.append(False)

    if not action_out:
        raise RuntimeError("no legal transitions were generated")

    def tensor(values: Iterable[Any], dtype: torch.dtype) -> torch.Tensor:
        return torch.as_tensor(list(values), dtype=dtype, device=device)

    pool = TransitionPool(
        episode=tensor(episodes_out, torch.long),
        start_a=tensor(a_out, torch.long),
        start_b=tensor(b_out, torch.long),
        selected=tensor(selected_out, torch.long),
        action=tensor(action_out, torch.long),
        reward=tensor(reward_out, torch.float32),
        done=tensor(done_out, torch.float32),
        stage1=tensor(stage1_out, torch.bool),
    )
    return pool


def permute_candidates(
    state: dict[str, torch.Tensor],
    action: torch.Tensor,
    generator: torch.Generator,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    batch = state["candidate"].shape[0]
    order = torch.rand(
        (batch, NUM_VIEWS), device=state["candidate"].device, generator=generator
    ).argsort(dim=-1)
    result = dict(state)
    result["candidate"] = state["candidate"].gather(
        1, order[..., None].expand(-1, -1, state["candidate"].shape[-1])
    )
    result["mask"] = state["mask"].gather(1, order)
    remapped = (order == action[:, None]).to(torch.long).argmax(dim=-1)
    return result, remapped


def masked_q(q: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return q.masked_fill(~mask.bool(), -torch.inf)


def sample_pool_ids(
    pool: TransitionPool,
    batch_size: int,
    generator: torch.Generator,
) -> torch.Tensor:
    stage1_ids = torch.nonzero(pool.stage1, as_tuple=False).flatten()
    stage2_ids = torch.nonzero(~pool.stage1, as_tuple=False).flatten()
    first_count = batch_size // 2
    first = stage1_ids[
        torch.randint(len(stage1_ids), (first_count,), device=pool.action.device, generator=generator)
    ]
    second = stage2_ids[
        torch.randint(
            len(stage2_ids),
            (batch_size - first_count,),
            device=pool.action.device,
            generator=generator,
        )
    ]
    return torch.cat((first, second), dim=0)


def eligible_episodes(raw: Any, classes: Sequence[int]) -> torch.Tensor:
    bank = torch.as_tensor(classes, device=raw.device, dtype=torch.long)
    return torch.isin(raw.labels, bank) & (raw.valid.sum(dim=-1) == NUM_VIEWS)


def sample_starts(raw: Any, episodes: torch.Tensor, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    valid = raw.valid[episodes].detach().cpu().numpy().astype(bool)
    starts_a = np.asarray([int(rng.choice(np.flatnonzero(row))) for row in valid], dtype=np.int64)
    starts_b = np.asarray([int(rng.choice(np.flatnonzero(row))) for row in valid], dtype=np.int64)
    return (
        torch.as_tensor(starts_a, dtype=torch.long, device=raw.device),
        torch.as_tensor(starts_b, dtype=torch.long, device=raw.device),
    )


@torch.inference_mode()
def dqn_rollout(
    model: nn.Module,
    raw: Any,
    episodes: torch.Tensor,
    starts_a: torch.Tensor,
    starts_b: torch.Tensor,
    variant: str,
) -> dict[str, torch.Tensor]:
    model.eval()
    selected_none = torch.full_like(starts_a, -1)
    state0 = build_state(raw, episodes, starts_a, starts_b, selected_none, variant)
    keep0 = state0["mask"].any(dim=-1)
    episodes = episodes[keep0]
    starts_a = starts_a[keep0]
    starts_b = starts_b[keep0]
    state0 = {key: value[keep0] for key, value in state0.items()}
    action1 = masked_q(model(state0), state0["mask"]).argmax(dim=-1)

    state1 = build_state(raw, episodes, starts_a, starts_b, action1, variant)
    keep1 = state1["mask"].any(dim=-1)
    episodes = episodes[keep1]
    starts_a = starts_a[keep1]
    starts_b = starts_b[keep1]
    action1 = action1[keep1]
    state1 = {key: value[keep1] for key, value in state1.items()}
    action2 = masked_q(model(state1), state1["mask"]).argmax(dim=-1)
    return {
        "episodes": episodes,
        "start_a": starts_a,
        "start_b": starts_b,
        "target_a": action1,
        "target_b": action2,
    }


def pair_metrics(
    raw: Any,
    fused_pairs: torch.Tensor,
    rollout: dict[str, torch.Tensor],
    bank_values: Sequence[int],
) -> dict[str, float]:
    episodes = rollout["episodes"]
    target_a = rollout["target_a"]
    target_b = rollout["target_b"]
    if len(episodes) == 0:
        return {
            "accuracy": 0.0,
            "mean_movement_cost": 0.0,
            "mean_movement_angle_deg": 0.0,
            "completion_rate": 0.0,
            "episodes": 0,
        }
    pair_ids = torch.as_tensor(PAIR_INDEX, device=raw.device, dtype=torch.long)[target_a, target_b]
    fused = fused_pairs[episodes, pair_ids]
    bank = torch.as_tensor(bank_values, device=raw.device, dtype=torch.long)
    scores = fused.index_select(-1, bank)
    local = torch.searchsorted(bank, raw.labels[episodes])
    correct = scores.argmax(dim=-1).eq(local)
    costs, feasible = assignment_matrix(raw, episodes, rollout["start_a"], rollout["start_b"])
    movement = costs[torch.arange(len(episodes), device=raw.device), target_a, target_b]
    movement = finite_float(movement, fill=1.0)
    return {
        "accuracy": float(correct.float().mean()),
        "mean_movement_cost": float(movement.mean()),
        "mean_movement_angle_deg": float((movement * 180.0).mean()),
        "completion_rate": float(len(episodes) / max(int(rollout.get("episodes_before", len(episodes))), 1)),
        "episodes": int(len(episodes)),
        "feasible_fraction_after_rollout": float(feasible[torch.arange(len(episodes), device=raw.device), target_a, target_b].float().mean()),
    }


def baseline_pair(
    raw: Any,
    episodes: torch.Tensor,
    starts_a: torch.Tensor,
    starts_b: torch.Tensor,
    seed: int,
    mode: str,
) -> dict[str, torch.Tensor]:
    rng = np.random.default_rng(seed)
    kept: list[int] = []
    target_a: list[int] = []
    target_b: list[int] = []
    kept_a: list[int] = []
    kept_b: list[int] = []
    for row, episode in enumerate(episodes.detach().cpu().tolist()):
        start_a = int(starts_a[row])
        start_b = int(starts_b[row])
        costs, feasible_matrix = assignment_matrix(
            raw,
            torch.as_tensor([episode], device=raw.device),
            torch.as_tensor([start_a], device=raw.device),
            torch.as_tensor([start_b], device=raw.device),
        )
        feasible = feasible_matrix[0].detach().cpu().numpy().astype(bool)
        legal = [pair_id for pair_id, (left, right) in enumerate(PAIR_LIST) if feasible[left, right]]
        if not legal:
            continue
        if mode == "random_pair":
            pair_id = int(rng.choice(legal))
        elif mode in {"cyclic_pair", "cyclic_adjacent_pair"}:
            phase = (seed + row) % NUM_VIEWS
            offset = 2 if mode == "cyclic_pair" else 1
            wanted = tuple(sorted((phase, (phase + offset) % NUM_VIEWS)))
            pair_id = int(PAIR_INDEX[wanted])
            if pair_id not in legal:
                pair_id = int(legal[(seed + row) % len(legal)])
        else:
            raise ValueError(mode)
        left, right = PAIR_LIST[pair_id]
        kept.append(int(episode))
        target_a.append(left)
        target_b.append(right)
        kept_a.append(start_a)
        kept_b.append(start_b)
    return {
        "episodes": torch.as_tensor(kept, device=raw.device, dtype=torch.long),
        "start_a": torch.as_tensor(kept_a, device=raw.device, dtype=torch.long),
        "start_b": torch.as_tensor(kept_b, device=raw.device, dtype=torch.long),
        "target_a": torch.as_tensor(target_a, device=raw.device, dtype=torch.long),
        "target_b": torch.as_tensor(target_b, device=raw.device, dtype=torch.long),
        "episodes_before": torch.as_tensor(len(episodes), device=raw.device),
    }


def complete_rollout(rollout: dict[str, torch.Tensor], before: int) -> dict[str, torch.Tensor]:
    output = dict(rollout)
    output["episodes_before"] = torch.as_tensor(before, device=rollout["episodes"].device)
    return output


@torch.inference_mode()
def evaluate_policy_once(
    model: nn.Module,
    raw: Any,
    fused_pairs: torch.Tensor,
    episodes: torch.Tensor,
    starts_a: torch.Tensor,
    starts_b: torch.Tensor,
    variant: str,
    bank: Sequence[int],
    distance_lambda: float,
) -> dict[str, float]:
    rollout = complete_rollout(
        dqn_rollout(model, raw, episodes, starts_a, starts_b, variant), len(episodes)
    )
    metrics = pair_metrics(raw, fused_pairs, rollout, bank)
    selected_margin = pair_margin(
        fused_pairs[rollout["episodes"], torch.as_tensor(PAIR_INDEX, device=raw.device)[rollout["target_a"], rollout["target_b"]]],
        raw.labels[rollout["episodes"]],
        bank,
    )
    movement, _ = assignment_matrix(
        raw, rollout["episodes"], rollout["start_a"], rollout["start_b"]
    )
    chosen_movement = movement[
        torch.arange(len(rollout["episodes"]), device=raw.device),
        rollout["target_a"],
        rollout["target_b"],
    ]
    metrics["mean_reward"] = float((selected_margin - distance_lambda * chosen_movement).mean()) if len(rollout["episodes"]) else 0.0
    return metrics


def train_one_seed(
    variant: str,
    seed: int,
    train_raw: Any,
    val_raw: Any,
    train_pool: TransitionPool,
    val_fused: torch.Tensor,
    args: argparse.Namespace,
    output_root: Path,
) -> dict[str, Any]:
    run = output_root / "models" / variant / f"seed_{seed}"
    run.mkdir(parents=True, exist_ok=True)
    complete_path = run / "complete.json"
    if complete_path.exists() and (run / "best.pt").exists():
        return json.loads(complete_path.read_text(encoding="utf-8"))

    seed_all(seed)
    model = PairDuelingQNetwork(variant).to(train_raw.device)
    target = PairDuelingQNetwork(variant).to(train_raw.device)
    target.load_state_dict(model.state_dict(), strict=True)
    target.eval()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs, eta_min=args.learning_rate * 0.05
    )
    generator = torch.Generator(device=train_raw.device).manual_seed(seed + 17)
    stage1_count = int(train_pool.stage1.sum())
    stage2_count = int((~train_pool.stage1).sum())

    val_episodes = (torch.isin(val_raw.labels, torch.as_tensor(PSEUDO_UNSEEN, device=val_raw.device)) & (val_raw.valid.sum(-1) == NUM_VIEWS)).nonzero().flatten()
    best_score = -float("inf")
    best_epoch = 0
    updates = 0
    log_path = run / "train.log"
    audit = {
        "algorithm": "masked_offline_double_dueling_dqn",
        "variant": variant,
        "seed": seed,
        "replay_transitions": train_pool.size,
        "stage1_transitions": stage1_count,
        "stage2_terminal_transitions": stage2_count,
        "gamma": args.gamma,
        "target_update_interval": args.target_update_interval,
        "distance_lambda": args.distance_lambda,
        "policy_inputs_excluded": ["class label", "VPOCLIP logits", "future-view features"],
        "terminal": "after selecting two distinct target slots",
    }
    dump_json(run / "audit.json", audit)

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses: list[float] = []
        q_values: list[float] = []
        targets: list[float] = []
        started = time.monotonic()
        for _ in range(args.updates_per_epoch):
            ids = sample_pool_ids(train_pool, args.batch_size, generator)
            episodes = train_pool.episode[ids]
            start_a = train_pool.start_a[ids]
            start_b = train_pool.start_b[ids]
            selected = train_pool.selected[ids]
            action = train_pool.action[ids]
            reward = train_pool.reward[ids]
            done = train_pool.done[ids]
            next_selected = action

            state = build_state(raw=train_raw, episodes=episodes, start_a=start_a, start_b=start_b, selected=selected, variant=variant)
            next_state = build_state(raw=train_raw, episodes=episodes, start_a=start_a, start_b=start_b, selected=next_selected, variant=variant)
            if hasattr(train_pool, "next_episode"):
                next_state = build_state(train_raw, train_pool.next_episode[ids],
                    train_pool.next_a[ids], train_pool.next_b[ids],
                    train_pool.next_selected[ids], variant)
            state, remapped_action = permute_candidates(state, action, generator)
            # Candidate permutation is independent at the next state; action is
            # not needed because Double-DQN only takes the masked argmax.
            next_dummy = torch.zeros_like(action)
            next_state, _ = permute_candidates(next_state, next_dummy, generator)

            q = model(state).gather(1, remapped_action[:, None]).squeeze(1)
            with torch.no_grad():
                next_online = masked_q(model(next_state), next_state["mask"])
                next_action = next_online.argmax(dim=-1)
                next_target = target(next_state).gather(1, next_action[:, None]).squeeze(1)
                bellman = reward + args.gamma * (1.0 - done) * next_target
            loss = F.smooth_l1_loss(q, bellman)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            updates += 1
            if updates % args.target_update_interval == 0:
                target.load_state_dict(model.state_dict(), strict=True)
            losses.append(float(loss.detach()))
            q_values.append(float(q.detach().mean()))
            targets.append(float(bellman.detach().mean()))
        scheduler.step()

        if epoch == 1 or epoch % args.validation_interval == 0 or epoch == args.epochs:
            starts_a, starts_b = sample_starts(val_raw, val_episodes, seed + 100000 + epoch)
            validation = evaluate_policy_once(
                model,
                val_raw,
                val_fused,
                val_episodes,
                starts_a,
                starts_b,
                variant,
                PSEUDO_UNSEEN,
                args.distance_lambda,
            )
            score = float(validation["mean_reward"])
        else:
            validation = {"accuracy": 0.0, "mean_reward": best_score, "episodes": 0}
            score = best_score
        improved = score > best_score
        if improved:
            best_score = score
            best_epoch = epoch
        payload = {
            "online": model.state_dict(),
            "target": target.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "updates": updates,
            "best_score": best_score,
            "best_epoch": best_epoch,
            "seed": seed,
            "variant": variant,
            "algorithm": "masked_offline_double_dueling_dqn",
        }
        torch.save(payload, run / "last.pt")
        if improved or not (run / "best.pt").exists():
            torch.save(payload, run / "best.pt")
        row = {
            "epoch": epoch,
            "seed": seed,
            "variant": variant,
            "loss": float(np.mean(losses)),
            "mean_q": float(np.mean(q_values)),
            "mean_target": float(np.mean(targets)),
            "seconds": time.monotonic() - started,
            "updates": updates,
            "validation_selection_reward": score,
            "best_score": best_score,
            "best_epoch": best_epoch,
            "validation": validation,
            "gpu_memory_gb": float(torch.cuda.max_memory_allocated() / 2**30) if torch.cuda.is_available() else 0.0,
        }
        with log_path.open("a", encoding="utf-8", buffering=1) as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print("TRAIN", json.dumps(row, ensure_ascii=False), flush=True)

    result = {
        "variant": variant,
        "seed": seed,
        "best_score": best_score,
        "best_epoch": best_epoch,
        "checkpoint": str(run / "best.pt"),
        "replay_transitions": train_pool.size,
    }
    dump_json(complete_path, result)
    return result


def load_policy(checkpoint: Path, variant: str, device: torch.device) -> nn.Module:
    model = PairDuelingQNetwork(variant).to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["online"], strict=True)
    model.eval()
    return model


def ci_summary(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if len(array) == 0:
        return {"n": 0, "mean": 0.0, "std": 0.0, "ci95": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
    mean = float(array.mean())
    std = float(array.std(ddof=1)) if len(array) > 1 else 0.0
    half = 1.96 * std / math.sqrt(len(array)) if len(array) > 1 else 0.0
    return {
        "n": int(len(array)),
        "mean": mean,
        "std": std,
        "ci95": half,
        "ci95_low": mean - half,
        "ci95_high": mean + half,
    }


def test_seed_evaluation(
    raw: Any,
    fused_pairs: torch.Tensor,
    policies: dict[str, nn.Module],
    variants: Sequence[str],
    seed: int,
    bank: Sequence[int],
) -> dict[str, Any]:
    eligible = eligible_episodes(raw, bank)
    episodes = eligible.nonzero().flatten()
    starts_a, starts_b = sample_starts(raw, episodes, seed)
    results: dict[str, Any] = {}
    for variant in variants:
        rollout = complete_rollout(
            dqn_rollout(policies[variant], raw, episodes, starts_a, starts_b, variant),
            len(episodes),
        )
        results[variant] = pair_metrics(raw, fused_pairs, rollout, bank)
    for baseline in ("random_pair", "cyclic_pair", "cyclic_adjacent_pair"):
        rollout = baseline_pair(raw, episodes, starts_a, starts_b, seed + 7001, baseline)
        results[baseline] = pair_metrics(raw, fused_pairs, rollout, bank)
    return {
        "seed": seed,
        "bank": [int(x) for x in bank],
        "episodes_before": int(len(episodes)),
        "methods": results,
    }


def summarize_seed_results(seed_results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    methods = sorted({method for row in seed_results for method in row["methods"]})
    summary: dict[str, Any] = {}
    for method in methods:
        rows = [row["methods"][method] for row in seed_results]
        summary[method] = {
            "accuracy": ci_summary([float(row["accuracy"]) for row in rows]),
            "mean_movement_cost": ci_summary([float(row["mean_movement_cost"]) for row in rows]),
            "mean_movement_angle_deg": ci_summary([float(row["mean_movement_angle_deg"]) for row in rows]),
            "completion_rate": ci_summary([float(row["completion_rate"]) for row in rows]),
            "episodes_per_seed": ci_summary([float(row["episodes"]) for row in rows]),
        }
    return summary


def write_results_markdown(
    output_root: Path,
    metadata: dict[str, Any],
    summary: dict[str, Any],
) -> None:
    lines = [
        "# Dual-robot depth-aware DDQN results",
        "",
        "Primary test bank: true unseen five-way classes `[14, 15, 25, 39, 46]`.",
        "Each cell is the mean over 30 random-start seed aggregates with a 95% normal CI.",
        "",
        "| method | weighted-fusion accuracy | normalized move cost | angular cost (deg-equivalent) | completion |",
        "|---|---:|---:|---:|---:|",
    ]
    for method, values in summary.items():
        accuracy = values["accuracy"]
        cost = values["mean_movement_cost"]
        degrees = values["mean_movement_angle_deg"]
        completion = values["completion_rate"]
        lines.append(
            f"| `{method}` | {accuracy['mean']:.4f} [{accuracy['ci95_low']:.4f}, {accuracy['ci95_high']:.4f}] "
            f"| {cost['mean']:.4f} [{cost['ci95_low']:.4f}, {cost['ci95_high']:.4f}] "
            f"| {degrees['mean']:.2f} [{degrees['ci95_low']:.2f}, {degrees['ci95_high']:.2f}] "
            f"| {completion['mean']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Protocol notes",
            "",
            "* Robot A starts at a uniformly sampled valid slot and performs one single-view recognition. This view is not fused during selection; its human/object state is the policy input.",
            "* Robot B starts independently at a uniformly sampled valid slot. The selected pair is assigned to the two robots using the cheaper feasible assignment; a selected current slot therefore has zero cost for that robot.",
            "* `cyclic_pair` uses `(phase, (phase+2) mod 4)`; `cyclic_adjacent_pair` is also reported as a sensitivity check.",
            "* Movement values are normalized angular costs from the cache. They are not meters. The degree-equivalent column is `180 * normalized_move_cost`.",
            "* The final output uses the existing lightweight weighted-logit fusion gate over the two selected per-view logits.",
            "",
            "## Resolved configuration",
            "",
            "```json",
            json.dumps(metadata, indent=2, ensure_ascii=False),
            "```",
        ]
    )
    (output_root / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def smoke_test(
    raws: dict[str, Any],
    gate: nn.Module,
    variants: Sequence[str],
) -> dict[str, Any]:
    raw = raws["test"]
    episodes = torch.arange(min(8, raw.num_episodes), device=raw.device)
    starts_a = torch.zeros_like(episodes)
    starts_b = torch.ones_like(episodes)
    result: dict[str, Any] = {}
    for variant in variants:
        selected = torch.full_like(episodes, -1)
        state0 = build_state(raw, episodes, starts_a, starts_b, selected, variant)
        model = PairDuelingQNetwork(variant).to(raw.device)
        q = model(state0)
        result[variant] = {
            "human_shape": list(state0["human"].shape) if "human" in state0 else None,
            "object_shape": list(state0["object"].shape) if "object" in state0 else None,
            "candidate_shape": list(state0["candidate"].shape),
            "q_shape": list(q.shape),
            "legal_actions": int(state0["mask"].sum()),
        }
    pair_logits = pair_fused_logits(raw, gate, chunk_size=64)
    result["pair_fused_shape"] = list(pair_logits.shape)
    result["depth_audit"] = raw._dual_robot_depth_audit
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--track-root", type=Path, default=DEFAULT_TRACK_ROOT)
    parser.add_argument("--depth-root", type=Path, default=DEFAULT_DEPTH_ROOT)
    parser.add_argument("--gate-checkpoint", type=Path, default=DEFAULT_GATE_CHECKPOINT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--train-seeds", type=int, nargs="+", default=DEFAULT_TRAIN_SEEDS)
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=DEFAULT_EVAL_SEEDS)
    parser.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32768)
    parser.add_argument("--updates-per-epoch", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--target-update-interval", type=int, default=250)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--distance-lambda", type=float, default=0.25)
    parser.add_argument("--validation-interval", type=int, default=2)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this experiment")
    device = torch.device("cuda:0")
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    args.output_root.mkdir(parents=True, exist_ok=True)

    depth_lookup = load_depth_lookup(args.depth_root)
    raws: dict[str, Any] = {}
    depth_audits: dict[str, Any] = {}
    for split in ("dqn_train", "val", "test"):
        raws[split], depth_audits[split] = load_raw(
            args.cache_root, args.track_root, split, device, depth_lookup
        )
        print("LOADED", split, raws[split].num_episodes, depth_audits[split], flush=True)

    gate = fusion_gate.load_gate(args.gate_checkpoint, device)
    print("GATE", args.gate_checkpoint, flush=True)
    smoke = smoke_test(raws, gate, args.variants)
    dump_json(args.output_root / "smoke_test.json", smoke)
    print("SMOKE", json.dumps(smoke, ensure_ascii=False), flush=True)
    if args.smoke_only:
        return

    fused_train = pair_fused_logits(raws["dqn_train"], gate)
    fused_val = pair_fused_logits(raws["val"], gate)
    fused_test = pair_fused_logits(raws["test"], gate)
    print("FUSED_PAIR_LOGITS", [list(x.shape) for x in (fused_train, fused_val, fused_test)], flush=True)

    pool = build_transition_pool(
        raws["dqn_train"], fused_train, args.distance_lambda, device
    )
    pool_info = {
        "transitions": pool.size,
        "stage1": int(pool.stage1.sum()),
        "stage2": int((~pool.stage1).sum()),
        "training_episodes_seen_classes_four_valid_views": int(
            ((torch.isin(raws["dqn_train"].labels, torch.as_tensor(SEEN_CLASSES, device=device))) & (raws["dqn_train"].valid.sum(-1) == NUM_VIEWS)).sum()
        ),
    }
    dump_json(args.output_root / "transition_pool.json", pool_info)
    print("POOL", json.dumps(pool_info), flush=True)

    metadata = {
        "experiment": "dual_robot_depth_ddqn_v1",
        "cache_root": str(args.cache_root),
        "track_root": str(args.track_root),
        "depth_root": str(args.depth_root),
        "gate_checkpoint": str(args.gate_checkpoint),
        "device": str(device),
        "training_classes": SEEN_CLASSES,
        "true_unseen_test_classes": TRUE_UNSEEN,
        "pseudo_validation_classes": PSEUDO_UNSEEN,
        "train_seeds": args.train_seeds,
        "eval_seeds": args.eval_seeds,
        "variants": args.variants,
        "human_channels": ["center_x", "center_y", "velocity_x", "velocity_y", "sin_3d_body_yaw", "cos_3d_body_yaw"],
        "human_channels_by_variant": {
            "full_depth19": ["center_x", "center_y", "velocity_x", "velocity_y", "sin_3d_body_yaw", "cos_3d_body_yaw"],
            "human_only": ["center_x", "center_y", "velocity_x", "velocity_y", "sin_3d_body_yaw", "cos_3d_body_yaw"],
            "human_trajectory_only": ["center_x", "center_y", "velocity_x", "velocity_y"],
            "human_orientation_only": ["sin_3d_body_yaw", "cos_3d_body_yaw"],
        },
        "human_yaw_source": "released 3-D skeleton body_yaw_deg.npy; repeated over 13 frames",
        "temporal_input": "13-frame sequence for both human and object branches",
        "temporal_encoder_by_variant": {
            "full_depth19": "Conv1d",
            "human_only": "Conv1d",
            "human_trajectory_only": "Conv1d",
            "human_orientation_only": "Conv1d",
            "object_only_depth19": "Conv1d",
            "full_lstm19": "one-layer LSTM, hidden_size=32",
            "human_only_lstm": "one-layer LSTM, hidden_size=32",
            "object_only_lstm19": "one-layer LSTM, hidden_size=32",
        },
        "object_raw_track": ["presence", "center_x", "center_y", "confidence"],
        "object_pooled_fields": ["center_x", "center_y", "confidence", "relative_x_to_person", "relative_y_to_person", "relative_depth_to_person"],
        "object_pooled_width_with_depth": 19,
        "object_pooled_width_without_depth": 16,
        "object_category_or_slot_seen_by_policy": False,
        "relative_depth_definition": "object MiDaS relative inverse depth minus person-slot depth; used only as object relative z; normalized first-frame depth; not metric",
        "robot_protocol": "robot A random valid start performs single-view recognition; robot B random independent start; select two distinct final targets",
        "action_space": "four target slots with invalid, already-selected, and infeasible pairs masked",
        "assignment_cost": "min(cost(A->i)+cost(B->j), cost(A->j)+cost(B->i))",
        "movement_metric": "normalized angular movement cost from move_cost.npy; degree equivalent equals cost*180; not meters",
        "policy_algorithm": "offline masked Double Dueling DQN",
        "reward": "terminal seen-bank weighted-fusion margin minus distance_lambda times assignment cost",
        "distance_lambda": args.distance_lambda,
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "updates_per_epoch": args.updates_per_epoch,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "gamma": args.gamma,
            "target_update_interval": args.target_update_interval,
        },
        "depth_join_audits": depth_audits,
        "pool": pool_info,
    }
    dump_json(args.output_root / "config_resolved.json", metadata)

    summaries: dict[str, list[dict[str, Any]]] = {}
    if not args.skip_train:
        for variant in args.variants:
            if variant not in {
                "full_depth19",
                "full_depth19_nonadjacent",
                "human_only",
                "human_trajectory_only",
                "human_orientation_only",
                "object_only_depth19",
                "full_pooled16",
                "full_lstm19",
                "human_only_lstm",
                "object_only_lstm19",
            }:
                raise ValueError(f"unknown variant {variant}")
            summaries[variant] = []
            for seed in args.train_seeds:
                result = train_one_seed(
                    variant,
                    int(seed),
                    raws["dqn_train"],
                    raws["val"],
                    pool,
                    fused_val,
                    args,
                    args.output_root,
                )
                summaries[variant].append(result)
            chosen = max(summaries[variant], key=lambda item: float(item["best_score"]))
            dump_json(args.output_root / f"selection_{variant}.json", {"runs": summaries[variant], "chosen": chosen})
    else:
        for variant in args.variants:
            selection_path = args.output_root / f"selection_{variant}.json"
            if not selection_path.exists():
                raise FileNotFoundError(f"--skip-train requested but {selection_path} is missing")
            summaries[variant] = json.loads(selection_path.read_text(encoding="utf-8"))["runs"]

    policies: dict[str, nn.Module] = {}
    selected_checkpoints: dict[str, str] = {}
    for variant in args.variants:
        chosen = max(summaries[variant], key=lambda item: float(item["best_score"]))
        checkpoint = Path(chosen["checkpoint"])
        policies[variant] = load_policy(checkpoint, variant, device)
        selected_checkpoints[variant] = str(checkpoint)

    seed_results: list[dict[str, Any]] = []
    for index, seed in enumerate(args.eval_seeds, start=1):
        row = test_seed_evaluation(
            raws["test"], fused_test, policies, args.variants, int(seed), TRUE_UNSEEN
        )
        seed_results.append(row)
        dump_json(args.output_root / "test" / f"seed_{seed}.json", row)
        print("EVAL", index, "/", len(args.eval_seeds), json.dumps(row, ensure_ascii=False), flush=True)
    summary = summarize_seed_results(seed_results)
    dump_json(args.output_root / "test" / "per_seed.json", seed_results)
    dump_json(args.output_root / "test" / "summary.json", summary)
    metadata["selected_checkpoints"] = selected_checkpoints
    metadata["depth_audits"] = depth_audits
    dump_json(args.output_root / "config_resolved.json", metadata)
    write_results_markdown(args.output_root, metadata, summary)
    dump_json(args.output_root / "summary.json", {"config": metadata, "training": summaries, "test": summary})
    print("COMPLETE", args.output_root / "RESULTS.md", flush=True)


if __name__ == "__main__":
    main()
