"""Experiment 2 diagnostic: view-consistent synthetic occlusion.

The recognizer is frozen.  This first implementation applies one shared
virtual occluder per synchronized episode to all three *pre-VPOCLIP* cached
modalities (X3D spatial maps, pose feature/joint slots, and object RS maps),
then runs the unchanged VPOCLIP adapter.  It is therefore a consistent
feature-space occlusion diagnostic, not a claim of raw-video physical
occlusion.  The limitation is recorded in every output and is intentionally
kept separate from the original cache and checkpoints.

No occluder latent is supplied to a policy.  The latent is used only by this
diagnostic to construct the paired corruption and to compute the privileged
visibility diagnostic.  A later PPO stage, if the validation gate passes,
uses only the observed corrupted state.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .dual_robot_orthogonal_experiment import (
    TARGET_ANGLE_DEGREES,
    choose_methods,
    legal_candidates,
    pair_metrics,
    single_metrics,
    oracle_destination,
)
from .multistep_g18_view_policy_v1 import (
    CACHE_ROOT,
    SEEN_CLASSES,
    PSEUDO_UNSEEN_CLASSES,
    TRUE_UNSEEN_CLASSES,
    GPUCache,
    class_bank_tensor,
    attach_view_valid,
)
from .vpoclip_adapter import VPOCLIPAdapter


ROOT = CACHE_ROOT.parents[2]
OUTPUT_ROOT = ROOT / "work_dir" / "exp02_synthetic_occlusion_rl"
SOURCE_ROOT = ROOT / "data" / "rl_offline_new50_5_nocost"
ALLVIEWS_ROOT = ROOT / "data" / "allviews_cs45_lowview50_5_features"
NUM_VIEWS = 4
NUM_FRAMES = 13
NUM_OBJECT_CLASSES = 50

LEVELS = {"none": 0.0, "mild": 0.30, "medium": 0.50, "heavy": 0.70}


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def normalized_name(value: str | Path) -> str:
    text = str(value).replace("\\", "/")
    marker = text.find("RGB_")
    if marker < 0:
        raise ValueError(f"cannot normalize sample name {value}")
    return text[marker:]


class SourceFeatures:
    """Memory-mapped feature arrays aligned by video sample name."""

    def __init__(self, cache_root: Path, split: str):
        if split == "dqn_train":
            x3d_root = SOURCE_ROOT / "x3d"
            pose_root = SOURCE_ROOT / "pose"
            x3d_file = x3d_root / "train_video.npy"
            x3d_names = SOURCE_ROOT / "rgb_dqn" / "train_sample_names.npy"
            pose_file = pose_root / "dqn_pose_coco17.npy"
            pose_names = pose_root / "dqn_sample_names.npy"
            joint_file = pose_root / "dqn_joint_xy_coco17.npy"
            object_file = SOURCE_ROOT / "object" / "train_object.npy"
        else:
            prefix = "val" if split == "val" else "test"
            x3d_root = ALLVIEWS_ROOT / "x3d"
            pose_root = ALLVIEWS_ROOT / "pose"
            object_root = ALLVIEWS_ROOT / "object"
            x3d_file = x3d_root / f"{prefix}_video.npy"
            x3d_names = pose_root / f"{prefix}_sample_names.npy"
            pose_file = pose_root / f"{prefix}_pose_coco17.npy"
            pose_names = pose_root / f"{prefix}_sample_names.npy"
            joint_file = pose_root / f"{prefix}_joint_xy_coco17.npy"
            object_file = object_root / f"{prefix}_object.npy"
        self.x3d = np.load(x3d_file, mmap_mode="r", allow_pickle=False)
        self.pose = np.load(pose_file, mmap_mode="r", allow_pickle=False)
        self.joint = np.load(joint_file, mmap_mode="r", allow_pickle=False)
        self.object = np.load(object_file, mmap_mode="r", allow_pickle=False)
        self.x3d_index = {
            normalized_name(name): index
            for index, name in enumerate(np.load(x3d_names, allow_pickle=True).astype(str))
        }
        self.pose_index = {
            normalized_name(name): index
            for index, name in enumerate(np.load(pose_names, allow_pickle=True).astype(str))
        }
        metadata = json.loads((cache_root / split / "metadata.json").read_text(encoding="utf-8"))
        rows = []
        for episode in metadata["episodes"]:
            row = []
            for view in episode.get("views", [])[:NUM_VIEWS]:
                name = normalized_name(view["video"])
                if name not in self.x3d_index or name not in self.pose_index:
                    raise KeyError(f"source feature missing {name} for split {split}")
                row.append((self.x3d_index[name], self.pose_index[name]))
            while len(row) < NUM_VIEWS:
                row.append(row[-1] if row else (0, 0))
            rows.append(row)
        self.indices = np.asarray(rows, dtype=np.int64)


def sample_occluders(episode_ids: torch.Tensor, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    values = []
    for episode in episode_ids.detach().cpu().tolist():
        rng = np.random.default_rng(int(seed) + int(episode) * 7919)
        values.append((rng.uniform(-math.pi, math.pi), rng.uniform(0.28, 0.52)))
    array = torch.as_tensor(values, device=episode_ids.device, dtype=torch.float32)
    return array[:, 0], array[:, 1]


def view_occlusion(
    raw: GPUCache,
    episodes: torch.Tensor,
    occluder_angle: torch.Tensor,
    occluder_distance: torch.Tensor,
    level: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return [B,V] severity and signed side from a shared 2-D occluder."""

    geometry = raw.geometry[episodes, :, :2]
    theta = torch.atan2(geometry[..., 0], geometry[..., 1])
    camera = torch.stack((theta.cos(), theta.sin()), -1)
    occluder = torch.stack(
        (occluder_distance.cos() * 0 + (occluder_distance * occluder_angle.cos()),
         occluder_distance * occluder_angle.sin()), -1
    )
    # C is one unit from H=(0,0), and the segment direction is C -> H.
    direction = -camera
    relative = occluder[:, None, :] - camera
    projection = (relative * direction).sum(-1).clamp(0.0, 1.0)
    closest = camera + projection[..., None] * direction
    distance = (occluder[:, None, :] - closest).norm(dim=-1)
    between = ((relative * direction).sum(-1) >= 0.0) & ((relative * direction).sum(-1) <= 1.0)
    base = torch.exp(-distance.square() / (2.0 * 0.15**2)) * between.float()
    side = direction[..., 0] * relative[..., 1] - direction[..., 1] * relative[..., 0]
    side = torch.sign(side)
    side = torch.where(side == 0, torch.ones_like(side), side)
    return (level * base).clamp(0.0, 0.70), side


def mask_rectangles(joint: torch.Tensor, severity: torch.Tensor, side: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Build temporally smooth human-centric rectangles from keypoint boxes."""

    present = joint.norm(dim=-1) > 0.02
    x, y = joint[..., 0], joint[..., 1]
    inf = torch.full_like(x, torch.inf)
    ninf = torch.full_like(x, -torch.inf)
    x1 = torch.where(present, x, inf).amin(-1)
    x2 = torch.where(present, x, ninf).amax(-1)
    y1 = torch.where(present, y, inf).amin(-1)
    y2 = torch.where(present, y, ninf).amax(-1)
    x1 = torch.nan_to_num(x1, nan=-0.35, posinf=-0.35)
    x2 = torch.nan_to_num(x2, nan=0.35, neginf=0.35)
    y1 = torch.nan_to_num(y1, nan=-0.80, posinf=-0.80)
    y2 = torch.nan_to_num(y2, nan=0.80, neginf=0.80)
    width = (x2 - x1).clamp_min(0.25)
    height = (y2 - y1).clamp_min(0.45)
    center_x = 0.5 * (x1 + x2) + side[..., None] * 0.20 * width
    center_y = y1 + 0.42 * height
    mask_width = severity[..., None] * width
    mask_height = (0.50 + 0.30 * severity[..., None]) * height
    left = center_x - 0.5 * mask_width
    right = center_x + 0.5 * mask_width
    top = center_y - 0.5 * mask_height
    bottom = center_y + 0.5 * mask_height
    grid_x = torch.linspace(-1.0, 1.0, 6, device=joint.device)
    grid_y = torch.linspace(-1.0, 1.0, 6, device=joint.device)
    gx, gy = torch.meshgrid(grid_x, grid_y, indexing="xy")
    feature_mask = (
        (gx[None, None, None] >= left[..., None, None])
        & (gx[None, None, None] <= right[..., None, None])
        & (gy[None, None, None] >= top[..., None, None])
        & (gy[None, None, None] <= bottom[..., None, None])
        & (severity[..., None, None, None] > 0.02)
    )
    joint_mask = (
        (x >= left[..., None]) & (x <= right[..., None])
        & (y >= top[..., None]) & (y <= bottom[..., None])
        & (severity[..., None, None] > 0.02)
    )
    return feature_mask, joint_mask


def masked_logits_batch(
    raw: GPUCache,
    source: SourceFeatures,
    adapter: VPOCLIPAdapter,
    episodes: torch.Tensor,
    level: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    ids = episodes.detach().cpu().numpy()
    source_ids = source.indices[ids]
    x3d_ids = torch.as_tensor(source_ids[..., 0], device=raw.device)
    pose_ids = torch.as_tensor(source_ids[..., 1], device=raw.device)
    video = torch.from_numpy(np.array(source.x3d[x3d_ids.cpu().numpy()], copy=True)).to(raw.device, dtype=torch.float32)
    pose = torch.from_numpy(np.array(source.pose[pose_ids.cpu().numpy()], copy=True)).to(raw.device, dtype=torch.float32)
    joint = torch.from_numpy(np.array(source.joint[pose_ids.cpu().numpy()], copy=True)).to(raw.device, dtype=torch.float32)
    obj = torch.from_numpy(np.array(source.object[x3d_ids.cpu().numpy()], copy=True)).to(raw.device, dtype=torch.float32)
    angles, distance = sample_occluders(episodes, seed)
    severity, side = view_occlusion(raw, episodes, angles, distance, level)
    feature_mask, joint_mask = mask_rectangles(joint, severity, side)
    # feature_mask is [B,V,T,6,6].  The X3D feature map uses the same spatial
    # normalized image convention for this controlled feature-space proxy.
    video = video.masked_fill(feature_mask[..., None, :, :], 0.0)
    pose = pose.masked_fill(joint_mask[:, :, None, None], 0.0)
    joint = joint.masked_fill(joint_mask[..., None], 0.0)
    object_mask = feature_mask[:, :, NUM_FRAMES // 2]
    # The stored RS map has y-up rows, while joint/X3D coordinates are y-down;
    # the six sample locations are symmetric, so reversing rows aligns them.
    object_mask = object_mask.flip(-2)
    obj = obj.masked_fill(object_mask[:, :, None], 0.0)
    with torch.inference_mode():
        encoded = adapter.encode_all_views({"video": video, "pose": pose, "object": obj, "joint_xy": joint})
    return encoded["logits"], severity


def raw_proxy(raw: GPUCache, logits: torch.Tensor) -> SimpleNamespace:
    return SimpleNamespace(
        device=raw.device, logits=logits, labels=raw.labels, valid=raw.valid,
        reachable=raw.reachable, cost=raw.cost, geometry=raw.geometry,
    )


def diagnostic_metrics(raw: GPUCache, logits: torch.Tensor, severity: torch.Tensor, episodes: torch.Tensor, classes: Sequence[int], seed: int) -> dict[str, Any]:
    proxy = raw_proxy(raw, logits)
    bank = class_bank_tensor(classes, raw.device)
    anchor = torch.zeros(len(episodes), dtype=torch.long, device=raw.device)
    legal = legal_candidates(proxy, episodes, anchor)
    methods = choose_methods(proxy, episodes, anchor, seed)
    single = single_metrics(proxy, episodes, anchor, bank)
    reports = {"fixed": {k: v for k, v in single.items() if k not in {"correct", "cost_values"}}}
    correctness = {"fixed": single["correct"]}
    for name, dest in methods.items():
        result = pair_metrics(proxy, episodes, anchor, dest, bank)
        reports[name] = {k: v for k, v in result.items() if k not in {"correct", "cost_values"}}
        correctness[name] = result["correct"]
    oracle = oracle_destination(proxy, episodes, anchor, bank)
    oracle_result = pair_metrics(proxy, episodes, anchor, oracle, bank)
    reports["oracle"] = {k: v for k, v in oracle_result.items() if k not in {"correct", "cost_values"}}
    correctness["oracle"] = oracle_result["correct"]
    fixed_correct = correctness["fixed"]
    fused_all = 0.5 * (logits[episodes, 0, None, :] + logits[episodes])
    all_correct = fused_all.index_select(-1, bank).argmax(-1).eq(torch.searchsorted(bank, raw.labels[episodes])[:, None])
    recoverable = (~fixed_correct) & all_correct[:, 1:].any(-1)
    visibility_dest = severity.masked_fill(~legal, torch.inf).argmin(-1)
    visibility_result = pair_metrics(proxy, episodes, anchor, visibility_dest, bank)
    reports["visibility_heuristic"] = {
        k: v for k, v in visibility_result.items() if k not in {"correct", "cost_values"}
    }
    correctness["visibility_heuristic"] = visibility_result["correct"]
    visibility_agreement = visibility_dest.eq(oracle)
    random_expected = (1.0 / legal.sum(-1).clamp_min(1).float()).mean()
    reports["recoverable_ratio"] = float(recoverable.float().mean())
    reports["least_occluded_har_best_agreement"] = float(visibility_agreement.float().mean())
    reports["random_agreement_expected"] = float(random_expected)
    reports["mean_anchor_occlusion"] = float(severity[:, 0].mean())
    reports["mean_best_candidate_occlusion"] = float(severity.masked_fill(~legal, torch.inf).min(-1).values.mean())
    reports["episodes"] = int(len(episodes))
    return reports


def run_split(raw: GPUCache, source: SourceFeatures, adapter: VPOCLIPAdapter, split: str, classes: Sequence[int], batch_size: int, output_root: Path, seed: int) -> dict[str, Any]:
    bank = class_bank_tensor(classes, raw.device)
    eligible = torch.isin(raw.labels, bank) & raw.valid[:, 0] & (raw.valid.sum(-1) >= 2)
    episodes = eligible.nonzero().flatten()
    split_result: dict[str, Any] = {"class_bank": [int(x) for x in classes], "episodes": int(len(episodes)), "levels": {}}
    for level_name, level in LEVELS.items():
        batches = []
        severities = []
        for begin in range(0, len(episodes), batch_size):
            batch = episodes[begin : begin + batch_size]
            batch_logits, batch_severity = masked_logits_batch(raw, source, adapter, batch, level, seed)
            batches.append(batch_logits)
            severities.append(batch_severity)
        logits = torch.cat(batches, 0) if batches else torch.empty((0, NUM_VIEWS, 55), device=raw.device)
        severity = torch.cat(severities, 0) if severities else torch.empty((0, NUM_VIEWS), device=raw.device)
        # ``logits`` is concatenated in eligible-episode order, so its first
        # dimension is local to this split rather than the global cache row
        # IDs.  Restrict the metadata to the same order before evaluating;
        # otherwise a sparse validation subset would index logits with global
        # IDs and eventually trigger a CUDA out-of-bounds assertion.
        local_raw = SimpleNamespace(
            device=raw.device,
            labels=raw.labels[episodes],
            valid=raw.valid[episodes],
            reachable=raw.reachable[episodes],
            cost=raw.cost[episodes],
            geometry=raw.geometry[episodes],
        )
        local_episodes = torch.arange(len(episodes), device=raw.device)
        metrics = diagnostic_metrics(local_raw, logits, severity, local_episodes, classes, seed + 101)
        split_result["levels"][level_name] = metrics
        # Save the causal benchmark tensors for a conditional PPO stage.  The
        # policy will be trained from observed states; latent severity is not
        # included in the state construction, but is retained for audit.
        (output_root / split).mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_root / split / f"states_{level_name}.npz", episodes=episodes.cpu().numpy(), logits=logits.cpu().numpy(), severity=severity.cpu().numpy(), labels=raw.labels[episodes].cpu().numpy(), geometry=raw.geometry[episodes].cpu().numpy(), valid=raw.valid[episodes].cpu().numpy(), reachable=raw.reachable[episodes].cpu().numpy(), cost=raw.cost[episodes].cpu().numpy())
        print("OCCLUSION_DIAGNOSTIC", split, level_name, json.dumps(metrics), flush=True)
    return split_result


def decide_gate(val_result: Mapping[str, Any]) -> dict[str, Any]:
    none = val_result["levels"]["none"]
    medium = val_result["levels"]["medium"]
    heavy = val_result["levels"]["heavy"]
    none_gap = none["oracle"]["top1"] - none["random_pair"]["top1"]
    heavy_gap = heavy["oracle"]["top1"] - heavy["random_pair"]["top1"]
    gate_checks = {
        "oracle_random_gap_expands": heavy_gap >= none_gap + 0.01,
        "recoverable_ratio_increases": heavy["recoverable_ratio"] >= none["recoverable_ratio"] + 0.05,
        "visibility_agreement_above_random": heavy["least_occluded_har_best_agreement"] >= heavy["random_agreement_expected"] + 0.05,
        "medium_or_heavy_orthogonal_not_below_random": max(medium["preferred_orthogonal"]["top1"] - medium["random_pair"]["top1"], heavy["preferred_orthogonal"]["top1"] - heavy["random_pair"]["top1"]) >= -0.005,
    }
    return {"validation_only": True, "checks": gate_checks, "gate_passed": bool(sum(gate_checks.values()) >= 3), "values": {"none_oracle_random_gap": none_gap, "heavy_oracle_random_gap": heavy_gap, "heavy_recoverable_ratio": heavy["recoverable_ratio"], "heavy_visibility_agreement": heavy["least_occluded_har_best_agreement"], "heavy_random_expected_agreement": heavy["random_agreement_expected"]}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260910)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda:0")
    args.output_root.mkdir(parents=True, exist_ok=True)
    dump(args.output_root / "synthetic_occlusion_config.json", {
        "mode": "pre-VPOCLIP tri-modal feature-space consistent proxy",
        "not_raw_video_claim": True,
        "levels": LEVELS,
        "occluder_sampling": "one angle and distance per synchronized episode, shared by all views",
        "camera_geometry": "sample-specific body-relative view_geometry",
        "frame_shape_reference": [640, 480],
        "target_region": "upper body / hands / face-torso interaction region",
        "true_unseen_locked": TRUE_UNSEEN_CLASSES,
        "gate_uses": "val pseudo-unseen only",
    })
    recognizer = json.loads((args.cache_root / "test" / "metadata.json").read_text(encoding="utf-8"))["recognizer"]
    adapter = VPOCLIPAdapter.from_config(recognizer["config"], recognizer["checkpoint"], str(device))
    raw: dict[str, GPUCache] = {}
    sources: dict[str, SourceFeatures] = {}
    for split in ("dqn_train", "val", "test"):
        raw[split] = GPUCache(args.cache_root / split, device)
        attach_view_valid(raw[split], args.cache_root, split)
        sources[split] = SourceFeatures(args.cache_root, split)
    result = {
        "seen_train": run_split(raw["dqn_train"], sources["dqn_train"], adapter, "seen_train", SEEN_CLASSES, args.batch_size, args.output_root / "diagnostics", args.seed),
        "seen_validation": run_split(raw["val"], sources["val"], adapter, "seen_validation", PSEUDO_UNSEEN_CLASSES, args.batch_size, args.output_root / "diagnostics", args.seed + 1),
        "true_unseen_5way": run_split(raw["test"], sources["test"], adapter, "true_unseen_5way", TRUE_UNSEEN_CLASSES, args.batch_size, args.output_root / "diagnostics", args.seed + 2),
    }
    gate = decide_gate(result["seen_validation"])
    dump(args.output_root / "gate.json", gate)
    rows = []
    for split, payload in result.items():
        for level, metrics in payload["levels"].items():
            rows.append({"split": split, "occlusion": level, "fixed": metrics["fixed"]["top1"], "random": metrics["random_pair"]["top1"], "orthogonal": metrics["preferred_orthogonal"]["top1"], "visibility_heuristic": metrics["visibility_heuristic"]["top1"], "oracle": metrics["oracle"]["top1"], "oracle_random_gap": metrics["oracle"]["top1"] - metrics["random_pair"]["top1"], "recoverable_ratio": metrics["recoverable_ratio"], "visibility_agreement": metrics["least_occluded_har_best_agreement"]})
    args.output_root.joinpath("diagnostics").mkdir(parents=True, exist_ok=True)
    with (args.output_root / "diagnostics" / "occlusion_diagnostic.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["split"]); writer.writeheader(); writer.writerows(rows)
    dump(args.output_root / "summary.json", {"experiment": "exp02_synthetic_occlusion", "result": result, "gate": gate})
    print("SYNTHETIC_OCCLUSION_DIAGNOSTIC_COMPLETE", json.dumps(gate), flush=True)


if __name__ == "__main__":
    main()
