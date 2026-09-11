"""Build and evaluate the richer temporal/cross-modal VoI experiment.

The frozen VPOCLIP recognizer is never retrained.  The cache builder reuses
the already extracted X3D/CTR-GCN inputs, creates eight local temporal
segments, and runs the exact frozen VPOCLIP fusion model on each segment.
This avoids a second RGB detector pass while keeping the visual backbone and
the recognizer checkpoint identical to the old RL cache.

The current project cache has no independent RGB/Skeleton/Object logits or
per-joint confidence.  Those feature groups are therefore marked unavailable
and are not silently synthesized.  Temporal uncertainty, pose interaction
trajectories, and available observation-quality features are evaluated
separately and in combination.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from torch import nn

from .diagnose_structured_viewpoint import _paired_ci
from .need_to_move_diagnostics import _current_state_features, _stratified_group_folds
from .structured_sa_experiment import (
    CANDIDATE_DIM,
    CURRENT_VIEW,
    NUM_VIEWS,
    SEEN_CLASSES,
    UNSEEN_CLASSES,
    StructuredScorer,
    candidate_inputs,
    margin_scores,
    prepare_data,
)
from .vpoclip_adapter import VPOCLIPAdapter
from .voi_view_sensitivity_v1 import (
    RidgeRegressor,
    TinyRandomForest,
    VoIMLP,
    fit_mlp_predict,
    json_safe,
)


CANDIDATE_VIEWS = np.asarray([1, 2, 3], dtype=np.int64)
TEMPORAL_SEGMENTS = 8
TEMPORAL_CONTEXT_FRAMES = 5
MLP_EPOCHS = 220
GB_ESTIMATORS = 48
GB_DEPTH = 2
GB_MIN_LEAF = 24
SENSITIVITY_QUANTILES = (0.20, 0.30, 0.40)
OLD_UNSEEN_SPEARMAN = 0.13697413086391255


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class FeatureStore:
    """Index pre-extracted VPOCLIP inputs by complete relative sample name."""

    def __init__(self, root: Path, names_file: Path, video_file: Path,
                 pose_names_file: Path, pose_file: Path, joint_file: Path,
                 object_file: Path) -> None:
        self.root = Path(root)
        self.names = np.load(names_file, allow_pickle=True).astype(str)
        self.video = np.load(video_file, mmap_mode="r", allow_pickle=False)
        pose_names = np.load(pose_names_file, allow_pickle=True).astype(str)
        self.pose = np.load(pose_file, mmap_mode="r", allow_pickle=False)
        self.joint = np.load(joint_file, mmap_mode="r", allow_pickle=False)
        self.object_map = np.load(object_file, mmap_mode="r", allow_pickle=False)
        self.video_index = {name: int(i) for i, name in enumerate(self.names)}
        self.pose_index = {name: int(i) for i, name in enumerate(pose_names)}
        if len(self.video) != len(self.names):
            raise ValueError(f"video/name mismatch: {len(self.video)} vs {len(self.names)}")
        if len(self.object_map) != len(self.names):
            raise ValueError(
                f"object/name mismatch: {len(self.object_map)} vs {len(self.names)}"
            )

    def rows(self, names: Sequence[str]) -> np.ndarray:
        result = []
        for name in names:
            if name not in self.video_index or name not in self.pose_index:
                raise KeyError(f"feature store has no sample {name}")
            result.append((self.video_index[name], self.pose_index[name]))
        return np.asarray(result, dtype=np.int64)

    def load(self, names: Sequence[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        rows = self.rows(names)
        video = np.asarray(self.video[rows[:, 0]], dtype=np.float32)
        pose = np.asarray(self.pose[rows[:, 1]], dtype=np.float32)
        joint = np.asarray(self.joint[rows[:, 1]], dtype=np.float32)
        obj = np.asarray(self.object_map[rows[:, 0]], dtype=np.float32)
        if video.ndim != 5 or video.shape[1:] != (13, 192, 6, 6):
            raise ValueError(f"unexpected X3D shape: {video.shape}")
        if pose.ndim != 5 or pose.shape[1:] != (2, 64, 13, 17):
            raise ValueError(f"unexpected pose feature shape: {pose.shape}")
        if joint.shape[1:] != (13, 17, 2):
            raise ValueError(f"unexpected joint shape: {joint.shape}")
        if obj.shape[1:] != (50, 6, 6):
            raise ValueError(f"unexpected object shape: {obj.shape}")
        return video, pose, joint, obj


def make_store(split: str, base: Path) -> FeatureStore:
    offline = base / "data/rl_offline_new50_5_nocost"
    allviews = base / "data/allviews_cs45_lowview50_5_features"
    if split == "dqn_train":
        return FeatureStore(
            offline,
            offline / "rgb_dqn/train_sample_names.npy",
            offline / "x3d/train_video.npy",
            offline / "pose/dqn_sample_names.npy",
            offline / "pose/dqn_pose_coco17.npy",
            offline / "pose/dqn_joint_xy_coco17.npy",
            offline / "object/train_object.npy",
        )
    return FeatureStore(
        allviews,
        allviews / "pose/test_sample_names.npy",
        allviews / "x3d/test_video.npy",
        allviews / "pose/test_sample_names.npy",
        allviews / "pose/test_pose_coco17.npy",
        allviews / "pose/test_joint_xy_coco17.npy",
        allviews / "object/test_object.npy",
    )


def segment_indices(num_frames: int = 13, segments: int = TEMPORAL_SEGMENTS,
                    context_frames: int = TEMPORAL_CONTEXT_FRAMES) -> np.ndarray:
    if context_frames < 2 or context_frames > num_frames:
        raise ValueError("context_frames must be in [2, num_frames]")
    starts = np.linspace(0, num_frames - context_frames, segments).round().astype(np.int64)
    result = []
    for start in starts:
        result.append(np.linspace(start, start + context_frames - 1, num_frames).round().astype(np.int64))
    return np.asarray(result, dtype=np.int64)


def selected_sample_names(data: Mapping[str, Any], base: Path) -> tuple[list[list[str]], np.ndarray]:
    del base
    rgb_root = Path("/home/youhan/ws/ETRI-Activity3D-RGB")
    metadata = data["raw"]["metadata"]
    source_ids = np.asarray(data["source_episode_ids"], dtype=np.int64)
    all_names: list[list[str]] = []
    for source_id in source_ids:
        row = metadata["episodes"][int(source_id)]
        names = []
        for view in row["views"]:
            path = Path(str(view["video"]))
            try:
                names.append(str(path.relative_to(rgb_root)))
            except ValueError as exc:
                raise ValueError(f"video is outside RGB root: {path}") from exc
        if len(names) != NUM_VIEWS:
            raise ValueError("rich cache requires exactly four physical views")
        all_names.append(names)
    return all_names, source_ids


def softmax_stats(logits: np.ndarray) -> dict[str, np.ndarray]:
    values = logits - logits.max(axis=-1, keepdims=True)
    p = np.exp(np.clip(values, -80.0, 80.0))
    p /= np.maximum(p.sum(axis=-1, keepdims=True), 1e-8)
    ordered = np.sort(p, axis=-1)[..., ::-1]
    entropy = -(p * np.log(np.maximum(p, 1e-8))).sum(axis=-1) / math.log(logits.shape[-1])
    return {
        "probs": p.astype(np.float32),
        "top1_class": logits.argmax(axis=-1).astype(np.int16),
        "top1_prob": ordered[..., 0].astype(np.float32),
        "top2_prob": ordered[..., 1].astype(np.float32),
        "entropy": entropy.astype(np.float32),
        "margin": (ordered[..., 0] - ordered[..., 1]).astype(np.float32),
    }


def build_rich_cache(
    split: str, data: Mapping[str, Any], classes: Sequence[int],
    base: Path, output_root: Path, adapter: VPOCLIPAdapter,
    device: torch.device, batch_size: int,
) -> Path:
    output = output_root / split
    complete = output / "complete.json"
    if complete.exists():
        print(json.dumps({"RICH_CACHE_REUSE": str(output)}), flush=True)
        return output
    output.mkdir(parents=True, exist_ok=True)
    names_by_episode, source_ids = selected_sample_names(data, base)
    n = len(names_by_episode)
    store = make_store(split, base)
    indices = segment_indices()
    np.save(output / "segment_source_indices.npy", indices)
    shape_logits = (n, NUM_VIEWS, TEMPORAL_SEGMENTS, 55)
    shape_view = (n, NUM_VIEWS)
    shape_seg = (n, NUM_VIEWS, TEMPORAL_SEGMENTS)
    arrays: dict[str, Any] = {
        "fused_logits": np.lib.format.open_memmap(output / "fused_logits.npy", mode="w+", dtype=np.float32, shape=shape_logits),
        "fused_probs": np.lib.format.open_memmap(output / "fused_probs.npy", mode="w+", dtype=np.float32, shape=shape_logits),
        "top1_class": np.lib.format.open_memmap(output / "top1_class.npy", mode="w+", dtype=np.int16, shape=shape_seg),
        "top1_prob": np.lib.format.open_memmap(output / "top1_prob.npy", mode="w+", dtype=np.float32, shape=shape_seg),
        "top2_prob": np.lib.format.open_memmap(output / "top2_prob.npy", mode="w+", dtype=np.float32, shape=shape_seg),
        "entropy": np.lib.format.open_memmap(output / "entropy.npy", mode="w+", dtype=np.float32, shape=shape_seg),
        "margin": np.lib.format.open_memmap(output / "margin.npy", mode="w+", dtype=np.float32, shape=shape_seg),
        "pose_sequence": np.lib.format.open_memmap(output / "pose_sequence.npy", mode="w+", dtype=np.float32, shape=(n, NUM_VIEWS, 13, 17, 2)),
        "object_map_static": np.lib.format.open_memmap(output / "object_map_static.npy", mode="w+", dtype=np.float32, shape=(n, NUM_VIEWS, 50, 6, 6)),
        "image_quality": np.lib.format.open_memmap(output / "image_quality.npy", mode="w+", dtype=np.float32, shape=(n, NUM_VIEWS, 4)),
        "source_feature_rows": np.lib.format.open_memmap(output / "source_feature_rows.npy", mode="w+", dtype=np.int64, shape=(n, NUM_VIEWS)),
    }
    rgb_root = Path("/home/youhan/ws/ETRI-Activity3D-RGB")
    parity_done = False
    try:
        for start in range(0, n, batch_size):
            stop = min(n, start + batch_size)
            batch_names = [name for names in names_by_episode[start:stop] for name in names]
            video, pose, joint, obj = store.load(batch_names)
            b = stop - start
            # [B*V, T, ...] -> [B,V,S,T,...] -> flattened virtual views.
            video = video.reshape(b, NUM_VIEWS, 13, 192, 6, 6)
            pose = pose.reshape(b, NUM_VIEWS, 2, 64, 13, 17)
            joint = joint.reshape(b, NUM_VIEWS, 13, 17, 2)
            obj = obj.reshape(b, NUM_VIEWS, 50, 6, 6)
            video_seg = video[:, :, indices, :, :, :]
            video_seg = video_seg.reshape(b, NUM_VIEWS, TEMPORAL_SEGMENTS, 13, 192, 6, 6)
            pose_seg = pose[:, :, :, :, indices, :]
            pose_seg = pose_seg.transpose(0, 1, 4, 2, 3, 5, 6)
            pose_seg = pose_seg.reshape(b, NUM_VIEWS, TEMPORAL_SEGMENTS, 2, 64, 13, 17)
            joint_seg = joint[:, :, indices, :, :]
            joint_seg = joint_seg.reshape(b, NUM_VIEWS, TEMPORAL_SEGMENTS, 13, 17, 2)
            object_seg = np.repeat(obj[:, :, None], TEMPORAL_SEGMENTS, axis=2)
            virtual_views = b * NUM_VIEWS * TEMPORAL_SEGMENTS
            with torch.inference_mode():
                encoded = adapter.encode_all_views({
                    "video": torch.from_numpy(video_seg.reshape(1, virtual_views, 13, 192, 6, 6)).to(device),
                    "pose": torch.from_numpy(pose_seg.reshape(1, virtual_views, 2, 64, 13, 17)).to(device),
                    "object": torch.from_numpy(object_seg.reshape(1, virtual_views, 50, 6, 6)).to(device),
                    "joint_xy": torch.from_numpy(joint_seg.reshape(1, virtual_views, 13, 17, 2)).to(device),
                })
            logits = encoded["logits"][0].cpu().numpy().reshape(b, NUM_VIEWS, TEMPORAL_SEGMENTS, 55)
            stats = softmax_stats(logits)
            arrays["fused_logits"][start:stop] = logits
            arrays["fused_probs"][start:stop] = stats["probs"]
            for key in ("top1_class", "top1_prob", "top2_prob", "entropy", "margin"):
                arrays[key][start:stop] = stats[key]
            arrays["pose_sequence"][start:stop] = joint.reshape(b, NUM_VIEWS, 13, 17, 2)
            arrays["object_map_static"][start:stop] = obj
            arrays["image_quality"][start:stop] = np.asarray(data["raw"]["quality"][start:stop], dtype=np.float32)
            # The feature-store row ID is useful for parity/audit; names are
            # saved in metadata because train and test stores have separate IDs.
            rows = store.rows(batch_names)[:, 0].reshape(b, NUM_VIEWS)
            arrays["source_feature_rows"][start:stop] = rows
            if not parity_done:
                full_logits = np.asarray(data["raw"]["logits"][start], dtype=np.float32)
                # Re-run the exact full 13-frame input for one episode.  The
                # result must match the preserved old cache bit-for-bit up to
                # normal floating-point tolerance.
                with torch.inference_mode():
                    full = adapter.encode_all_views({
                        "video": torch.from_numpy(video[:1]).to(device),
                        "pose": torch.from_numpy(pose[:1]).to(device),
                        "object": torch.from_numpy(obj[:1]).to(device),
                        "joint_xy": torch.from_numpy(joint[:1]).to(device),
                    })
                actual = full["logits"][0].cpu().numpy()
                parity = {
                    "max_abs_logit_error": float(np.max(np.abs(actual - full_logits))),
                    "mean_abs_logit_error": float(np.mean(np.abs(actual - full_logits))),
                    "same_frozen_vpoclip_checkpoint": True,
                    "full_input_shape": list(video[:1].shape),
                }
                (output / "forward_parity.json").write_text(json.dumps(parity, indent=2), encoding="utf-8")
                if parity["max_abs_logit_error"] > 1e-3:
                    raise RuntimeError(f"VPOCLIP full-input parity failed: {parity}")
                parity_done = True
            for value in arrays.values():
                value.flush()
            print(f"RICH CACHE {split} {stop}/{n}", flush=True)
    finally:
        for value in arrays.values():
            del value
    metadata = {
        "format": "richer_active_view_cache_v1",
        "split": split,
        "episodes": int(n),
        "classes": list(classes),
        "num_views": NUM_VIEWS,
        "temporal_segments": TEMPORAL_SEGMENTS,
        "segment_source_indices": indices.tolist(),
        "temporal_backbone_note": "X3D/CTR-GCN inputs are pre-extracted full-clip features; local segment re-windowing is applied before frozen fusion",
        "sample_names": names_by_episode,
        "source_episode_ids": source_ids.tolist(),
        "input_resolution": [640, 480],
        "recognizer": {
            "config": str(base / "logs/allviews_lowview50_5_20260907/final_aug_entropy_single_h2.yaml"),
            "checkpoint": str(base / "work_dir/allviews_lowview50_5_final_aug_entropy_single_h2/allviews_lowview50_5_20260907/last_model.pth"),
            "frozen": True,
        },
        "available": [
            "fused_segment_logits", "fused_segment_probabilities",
            "current_pose_sequence", "static_object_map", "image_quality",
        ],
        "unavailable_not_fabricated": [
            "independent_rgb_logits", "independent_skeleton_logits", "independent_object_logits",
            "per_joint_keypoint_confidence", "per-segment_object_detection_and_bbox_tracks",
        ],
    }
    (output / "metadata.json").write_text(json.dumps(json_safe(metadata), indent=2), encoding="utf-8")
    (output / "complete.json").write_text(json.dumps({"complete": True, "episodes": n, "temporal_segments": TEMPORAL_SEGMENTS}, indent=2), encoding="utf-8")
    return output


def load_rich_cache(path: Path) -> dict[str, Any]:
    return {
        "root": str(path),
        "fused_logits": np.load(path / "fused_logits.npy", mmap_mode="r"),
        "top1_class": np.load(path / "top1_class.npy", mmap_mode="r"),
        "top1_prob": np.load(path / "top1_prob.npy", mmap_mode="r"),
        "top2_prob": np.load(path / "top2_prob.npy", mmap_mode="r"),
        "entropy": np.load(path / "entropy.npy", mmap_mode="r"),
        "margin": np.load(path / "margin.npy", mmap_mode="r"),
        "pose_sequence": np.load(path / "pose_sequence.npy", mmap_mode="r"),
        "object_map_static": np.load(path / "object_map_static.npy", mmap_mode="r"),
        "image_quality": np.load(path / "image_quality.npy", mmap_mode="r"),
        "metadata": json.loads((path / "metadata.json").read_text(encoding="utf-8")),
    }


def targets_from_old_cache(data: Mapping[str, Any], classes: Sequence[int]) -> dict[str, np.ndarray]:
    raw = data["raw"]
    n = len(data["episodes"])
    logits = np.asarray(raw["logits"][:n], dtype=np.float32)
    labels = np.asarray(raw["labels"][:n], dtype=np.int64)
    current = logits[:, CURRENT_VIEW]
    fused = 0.5 * (current[:, None] + logits)
    before = margin_scores(current, labels, classes)
    after = margin_scores(fused, labels[:, None], classes)
    utility = after - before[:, None]
    _, _, info = candidate_inputs(raw, np.arange(n), data["features"])
    valid = np.asarray(info["valid"], dtype=bool)
    valid[:, CURRENT_VIEW] = False
    cand = np.where(valid[:, CANDIDATE_VIEWS], utility[:, CANDIDATE_VIEWS], np.nan)
    best = np.nanmax(cand, axis=1)
    worst = np.nanmin(cand, axis=1)
    mean = np.nanmean(cand, axis=1)
    bank = np.asarray(sorted(set(int(x) for x in classes)), dtype=np.int64)
    local = np.searchsorted(bank, labels)
    current_correct = current[:, bank].argmax(-1) == local
    candidate_correct = fused[:, :, bank].argmax(-1) == local[:, None]
    recoverable = (~current_correct) & candidate_correct[:, CANDIDATE_VIEWS].any(axis=1)
    return {
        "utility": utility.astype(np.float32),
        "candidate_utility": cand.astype(np.float32),
        "candidate_valid": valid[:, CANDIDATE_VIEWS],
        "voi_random": (best - mean).astype(np.float32),
        "voi_current": best.astype(np.float32),
        "view_sensitivity": (best - worst).astype(np.float32),
        "current_correct": current_correct,
        "candidate_correct": candidate_correct[:, CANDIDATE_VIEWS],
        "recoverable": recoverable,
    }


def temporal_uncertainty_features(rich: Mapping[str, Any]) -> tuple[np.ndarray, list[str]]:
    logits = np.asarray(rich["fused_logits"][:, CURRENT_VIEW], dtype=np.float32)
    stats = softmax_stats(logits)
    entropy = stats["entropy"]
    top1 = stats["top1_prob"]
    margin = stats["margin"]
    classes = stats["top1_class"]
    logit_variance = logits.var(axis=1)
    switches = (classes[:, 1:] != classes[:, :-1]).mean(axis=1)
    values = np.stack([
        entropy.mean(1), entropy.std(1), entropy.max(1), entropy.min(1),
        top1.mean(1), top1.std(1), top1.min(1),
        margin.mean(1), margin.std(1), margin.min(1),
        logit_variance.mean(1), logit_variance.max(1), switches,
    ], axis=1).astype(np.float32)
    names = [
        "temporal_entropy_mean", "temporal_entropy_std", "temporal_entropy_max", "temporal_entropy_min",
        "temporal_top1_mean", "temporal_top1_std", "temporal_top1_min",
        "temporal_margin_mean", "temporal_margin_std", "temporal_margin_min",
        "temporal_logit_variance_mean", "temporal_logit_variance_max", "top1_switch_rate",
    ]
    return values, names


def object_centers(object_map: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    maps = np.maximum(np.asarray(object_map, dtype=np.float32), 0.0)
    total = maps.sum(axis=(1, 2, 3))
    grid_x = np.linspace(-1.0, 1.0, 6, dtype=np.float32)[None, None, None, :]
    grid_y = np.linspace(1.0, -1.0, 6, dtype=np.float32)[None, None, :, None]
    x = (maps * grid_x).sum(axis=(1, 2, 3)) / np.maximum(total, 1e-6)
    y = (maps * grid_y).sum(axis=(1, 2, 3)) / np.maximum(total, 1e-6)
    return np.stack([x, y], axis=-1), total > 1e-6


def trajectory_stats(values: np.ndarray, prefix: str) -> tuple[list[np.ndarray], list[str]]:
    values = np.asarray(values, dtype=np.float32)
    diff = np.diff(values, axis=1)
    accel = np.diff(values, n=2, axis=1)
    arrays = [values.mean(1), values.std(1), values.min(1), values.max(1),
              np.ptp(values, axis=1), diff.mean(1), diff.std(1),
              np.abs(accel).mean(1), np.abs(accel).std(1)]
    suffixes = ["mean", "std", "min", "max", "range", "velocity_mean",
                "velocity_std", "acceleration_mean", "acceleration_std"]
    return arrays, [f"{prefix}_{suffix}" for suffix in suffixes]


def temporal_interaction_features(rich: Mapping[str, Any]) -> tuple[np.ndarray, list[str]]:
    pose = np.asarray(rich["pose_sequence"][:, CURRENT_VIEW], dtype=np.float32)
    points = np.nan_to_num(pose).reshape(len(pose), 13, 17, 2)
    present = np.linalg.norm(points, axis=-1) > 0.02
    head = points[:, :, :5].mean(axis=2)
    left_hand, right_hand = points[:, :, 9], points[:, :, 10]
    shoulder = 0.5 * (points[:, :, 5] + points[:, :, 6])
    hip = 0.5 * (points[:, :, 11] + points[:, :, 12])
    torso = 0.5 * (shoulder + hip)
    object_center, object_valid = object_centers(np.asarray(rich["object_map_static"][:, CURRENT_VIEW]))
    object_center = object_center[:, None, :]
    trajectories = {
        "hand_face_distance": np.minimum(
            np.linalg.norm(left_hand - head, axis=-1),
            np.linalg.norm(right_hand - head, axis=-1),
        ),
        "hand_torso_distance": np.minimum(
            np.linalg.norm(left_hand - torso, axis=-1),
            np.linalg.norm(right_hand - torso, axis=-1),
        ),
        "left_right_hand_distance": np.linalg.norm(left_hand - right_hand, axis=-1),
        # The object map is static in the current source cache.  This is a
        # static-object interaction trajectory, not a claimed object track.
        "hand_object_distance_static": np.minimum(
            np.linalg.norm(left_hand - object_center, axis=-1),
            np.linalg.norm(right_hand - object_center, axis=-1),
        ),
    }
    parts: list[np.ndarray] = []
    names: list[str] = []
    for prefix, values in trajectories.items():
        arrays, feature_names = trajectory_stats(values, prefix)
        parts.extend(arrays)
        names.extend(feature_names)
    # Presence dynamics are useful and are genuinely available from joint xy.
    missing_ratio = 1.0 - present.mean(axis=-1)
    arrays, feature_names = trajectory_stats(missing_ratio, "joint_missing_ratio")
    parts.extend(arrays)
    names.extend(feature_names)
    parts.extend([object_valid.astype(np.float32), (~object_valid).astype(np.float32)])
    names.extend(["object_static_valid", "object_static_missing_rate"])
    return np.stack(parts, axis=1).astype(np.float32), names


def observation_quality_features(rich: Mapping[str, Any]) -> tuple[np.ndarray, list[str]]:
    pose = np.asarray(rich["pose_sequence"][:, CURRENT_VIEW], dtype=np.float32)
    points = np.nan_to_num(pose)
    present = np.linalg.norm(points, axis=-1) > 0.02
    missing = 1.0 - present.mean(axis=-1)
    hand_present = present[:, :, [9, 10]].mean(axis=(1, 2))
    torso_present = present[:, :, [5, 6, 11, 12]].mean(axis=(1, 2))
    quality = np.asarray(rich["image_quality"][:, CURRENT_VIEW], dtype=np.float32)
    object_map = np.asarray(rich["object_map_static"][:, CURRENT_VIEW], dtype=np.float32)
    _, object_valid = object_centers(object_map)
    values = np.stack([
        present.mean(axis=(1, 2)), present.std(axis=(1, 2)),
        missing.mean(axis=1), missing.std(axis=1), missing.max(axis=1),
        hand_present, torso_present,
        quality[:, 0], quality[:, 1], quality[:, 2], quality[:, 3],
        object_valid.astype(np.float32), (~object_valid).astype(np.float32),
    ], axis=1).astype(np.float32)
    names = [
        "joint_presence_mean", "joint_presence_std", "missing_joint_ratio_mean",
        "missing_joint_ratio_std", "missing_joint_ratio_max", "hand_presence",
        "torso_presence", "image_quality_presence", "image_quality_mean_score",
        "image_quality_max_score", "geometry_confidence", "object_static_valid",
        "object_static_missing",
    ]
    return values, names


def add_feature_group(groups: dict[str, np.ndarray], names: dict[str, list[str]],
                     key: str, value: tuple[np.ndarray, list[str]]) -> None:
    groups[key], names[key] = value


def r2_score(y: np.ndarray, pred: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    return float(1.0 - np.sum((y - pred) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-12))


def scalar_metrics(y: np.ndarray, prediction: np.ndarray,
                   sensitivity: np.ndarray, recoverable: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(prediction, dtype=np.float64)
    finite = np.isfinite(y) & np.isfinite(p)
    y, p = y[finite], p[finite]
    result: dict[str, Any] = {
        "n": int(len(y)), "r2": r2_score(y, p),
        "mae": float(np.mean(np.abs(y - p))),
        "target_mean": float(y.mean()), "prediction_mean": float(p.mean()),
    }
    if len(y) > 1 and y.std() > 1e-12 and p.std() > 1e-12:
        result["pearson"] = float(pearsonr(y, p).statistic)
        result["spearman"] = float(spearmanr(y, p).statistic)
    else:
        result["pearson"] = None
        result["spearman"] = None
    order = np.argsort(-p)
    result["top_fraction"] = {}
    for fraction in (0.10, 0.20, 0.30):
        count = max(1, int(math.ceil(len(y) * fraction)))
        index = order[:count]
        # The caller passes the unfiltered arrays in normal use; if there are
        # nonfinite predictions, the top-subset report is still deterministic.
        result["top_fraction"][str(int(fraction * 100))] = {
            "count": int(count),
            "true_mean_voi": float(y[index].mean()),
            "true_median_voi": float(np.median(y[index])),
            "true_mean_view_sensitivity": float(np.asarray(sensitivity)[finite][index].mean()),
            "recoverable_ratio": float(np.asarray(recoverable)[finite][index].mean()),
        }
    return result


class TinyGradientBoosting:
    """Small dependency-free gradient-boosted tree regressor."""

    def __init__(self, n_estimators: int = GB_ESTIMATORS, depth: int = GB_DEPTH,
                 min_leaf: int = GB_MIN_LEAF, learning_rate: float = 0.05,
                 seed: int = 0) -> None:
        self.n_estimators = n_estimators
        self.depth = depth
        self.min_leaf = min_leaf
        self.learning_rate = learning_rate
        self.seed = seed
        self.base = 0.0
        self.trees: list[Any] = []

    def fit(self, x: np.ndarray, y: np.ndarray) -> "TinyGradientBoosting":
        self.base = float(y.mean())
        prediction = np.full(len(y), self.base, dtype=np.float32)
        self.trees = []
        for index in range(self.n_estimators):
            residual = (y - prediction).astype(np.float32)
            helper = TinyRandomForest(
                n_estimators=1, max_depth=self.depth, min_leaf=self.min_leaf,
                max_features=max(3, int(math.sqrt(x.shape[1]))), seed=self.seed + index,
            ).fit(x, residual)
            tree = helper.trees[0]
            self.trees.append(tree)
            prediction += self.learning_rate * helper.predict(x)
        return self

    @staticmethod
    def _predict_tree(tree: Any, row: np.ndarray) -> float:
        while tree[0] == "node":
            tree = tree[3] if row[tree[1]] <= tree[2] else tree[4]
        return float(tree[1])

    def predict(self, x: np.ndarray) -> np.ndarray:
        result = np.full(len(x), self.base, dtype=np.float32)
        for tree in self.trees:
            result += self.learning_rate * np.asarray(
                [self._predict_tree(tree, row) for row in x], dtype=np.float32
            )
        return result


def fit_predictor(name: str, x_train: np.ndarray, y_train: np.ndarray,
                  x_query: np.ndarray, device: torch.device, seed: int) -> tuple[Any, np.ndarray, Any]:
    if name == "ridge":
        model = RidgeRegressor(alpha=10.0).fit(x_train, y_train)
        return model, model.predict(x_query), None
    if name == "random_forest":
        model = TinyRandomForest(seed=seed, n_estimators=32, max_depth=7, min_leaf=20).fit(x_train, y_train)
        return model, model.predict(x_query), None
    if name == "gradient_boosting":
        model = TinyGradientBoosting(seed=seed).fit(x_train, y_train)
        return model, model.predict(x_query), None
    if name == "mlp":
        model, prediction, normalization = fit_mlp_predict(
            x_train, y_train, x_query, device, seed, epochs=MLP_EPOCHS
        )
        return model, prediction, normalization
    raise ValueError(name)


def model_predict(name: str, model: Any, x: np.ndarray, normalization: Any,
                  device: torch.device) -> np.ndarray:
    if name != "mlp":
        return model.predict(x)
    mean = np.asarray(normalization["mean"], dtype=np.float32)
    std = np.asarray(normalization["std"], dtype=np.float32)
    y_mean = float(np.asarray(normalization["y_mean"]))
    y_std = float(np.asarray(normalization["y_std"]))
    model.eval()
    with torch.inference_mode():
        value = model(torch.from_numpy(((x - mean) / std).astype(np.float32)).to(device))
    return (value.cpu().numpy() * y_std + y_mean).astype(np.float32)


def fit_all_feature_models(
    feature_sets: Mapping[str, np.ndarray], feature_names: Mapping[str, list[str]],
    y: np.ndarray, sensitivity: np.ndarray, recoverable: np.ndarray,
    groups: np.ndarray, device: torch.device, seed: int,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    high = (sensitivity >= np.quantile(sensitivity, 0.70)).astype(np.int64)
    folds = _stratified_group_folds(high, groups, n_splits=5, seed=seed)
    model_names = ("ridge", "random_forest", "gradient_boosting", "mlp")
    reports: dict[str, Any] = {}
    final: dict[str, Any] = {}
    for feature_key, x in feature_sets.items():
        oof: dict[str, np.ndarray] = {
            name: np.full(len(y), np.nan, dtype=np.float32) for name in model_names
        }
        for fold, (train_idx, val_idx) in enumerate(folds):
            for model_index, model_name in enumerate(model_names):
                _, pred, _ = fit_predictor(
                    model_name, x[train_idx], y[train_idx], x[val_idx], device,
                    seed + fold * 100 + model_index,
                )
                oof[model_name][val_idx] = pred
        reports[feature_key] = {
            "feature_dim": int(x.shape[1]), "feature_names": feature_names[feature_key],
            "oof_subject_holdout": {
                name: scalar_metrics(y, pred, sensitivity, recoverable)
                for name, pred in oof.items()
            },
            "test": {},
        }
        fitted: dict[str, tuple[Any, Any]] = {}
        for model_index, model_name in enumerate(model_names):
            model, _, normalization = fit_predictor(
                model_name, x, y, x[:1], device, seed + 1000 + model_index
            )
            fitted[model_name] = (model, normalization)
        final[feature_key] = fitted
    return reports, final, [{"fold": int(i), "validation_subjects": sorted(set(groups[idx].tolist())),
                             "validation_episodes": int(len(idx))} for i, (_, idx) in enumerate(folds)]


def add_test_predictions(
    reports: dict[str, Any], final: Mapping[str, Any], feature_sets: Mapping[str, np.ndarray],
    test_sets: Mapping[str, Mapping[str, np.ndarray]], test_names: Sequence[str],
    device: torch.device,
) -> None:
    for feature_key, models in final.items():
        reports[feature_key]["test"] = {}
        for split_name, payload in test_sets.items():
            x = payload["features"][feature_key]
            y, sensitivity, recoverable = payload["y"], payload["sensitivity"], payload["recoverable"]
            reports[feature_key]["test"][split_name] = {}
            for model_name, (model, normalization) in models.items():
                pred = model_predict(model_name, model, x, normalization, device)
                reports[feature_key]["test"][split_name][model_name] = scalar_metrics(
                    y, pred, sensitivity, recoverable
                )


def permutation_importance(model: TinyRandomForest, x: np.ndarray, y: np.ndarray,
                           names: Sequence[str], seed: int) -> list[dict[str, Any]]:
    baseline = r2_score(y, model.predict(x))
    rng = np.random.default_rng(seed)
    result = []
    for feature in range(x.shape[1]):
        permuted = np.array(x, copy=True)
        permuted[:, feature] = rng.permutation(permuted[:, feature])
        result.append({
            "feature": names[feature],
            "r2_drop": float(baseline - r2_score(y, model.predict(permuted))),
        })
    return sorted(result, key=lambda row: row["r2_drop"], reverse=True)


class RicherRanker(nn.Module):
    def __init__(self, context_dim: int, candidate_dim: int = CANDIDATE_DIM, hidden: int = 128) -> None:
        super().__init__()
        self.context = nn.Sequential(nn.Linear(context_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())
        self.candidate = nn.Sequential(nn.Linear(candidate_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())
        self.head = nn.Sequential(nn.Linear(hidden * 3, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, context: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
        c = self.context(context)[:, None, :]
        a = self.candidate(candidate)
        return self.head(torch.cat([c.expand_as(a), a, c * a], dim=-1)).squeeze(-1)


def train_ranker_variant(
    train_data: Mapping[str, Any], train_targets: Mapping[str, np.ndarray],
    rich_train: Mapping[str, Any], context_features: np.ndarray, quantile: float,
    device: torch.device, seed: int, output: Path,
) -> dict[str, Any]:
    context, candidate, info = candidate_inputs(
        train_data["raw"], train_data["episodes"], train_data["features"]
    )
    # Replace the old compact context with the richer current-view features.
    del context
    _, candidate, info = candidate_inputs(train_data["raw"], train_data["episodes"], train_data["features"])
    utility = np.asarray(train_targets["utility"], dtype=np.float32).copy()
    valid = np.asarray(info["valid"], dtype=bool)
    utility[~valid] = 0.0
    sensitivity = np.asarray(train_targets["view_sensitivity"], dtype=np.float32)
    cutoff = float(np.quantile(sensitivity, quantile))
    selected = sensitivity >= cutoff
    x = torch.from_numpy(context_features[selected]).to(device)
    c = torch.from_numpy(candidate[selected]).to(device)
    y = torch.from_numpy(utility[selected]).to(device)
    valid_t = torch.from_numpy(valid[selected]).to(device)
    weights = torch.from_numpy((sensitivity[selected] / max(float(sensitivity[selected].mean()), 1e-5)).astype(np.float32)).to(device)
    model = RicherRanker(context_features.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=2e-2)
    for _ in range(140):
        pred = model(x, c)
        mse_each = ((pred - y) ** 2)[valid_t].reshape(-1)
        # The episode weight is repeated by the mask's candidate entries.
        weight_each = weights[:, None].expand_as(valid_t)[valid_t]
        mse = (mse_each * weight_each).sum() / weight_each.sum().clamp_min(1e-6)
        pairs = []
        for left in CANDIDATE_VIEWS:
            for right in CANDIDATE_VIEWS:
                if left >= right:
                    continue
                ok = valid_t[:, left] & valid_t[:, right]
                delta = y[:, left] - y[:, right]
                use = ok & (delta.abs() > 0.02)
                if use.any():
                    pairs.append(torch.relu(0.10 - delta[use].sign() * (pred[use, left] - pred[use, right])))
        pair = torch.cat(pairs).mean() if pairs else mse.new_zeros(())
        loss = mse + 0.5 * pair
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    saved = output / f"richer_ranker_top{int(quantile * 100)}.pt"
    torch.save({"model": model.state_dict(), "hidden": 128, "context_dim": int(context_features.shape[1]), "quantile": quantile, "cutoff": cutoff, "selected": int(selected.sum()), "future_inputs": False}, saved)
    return {"model": model, "candidate": candidate, "cutoff": cutoff, "selected": selected, "checkpoint": saved}


def evaluate_ranker(
    ranker: Mapping[str, Any], data: Mapping[str, Any], classes: Sequence[int],
    device: torch.device, targets: Mapping[str, np.ndarray], seed: int,
) -> dict[str, Any]:
    _, candidate, info = candidate_inputs(data["raw"], data["episodes"], data["features"])
    rich_model = ranker["model"]
    rich_model.eval()
    # The training context is saved separately by the caller and attached here.
    context = ranker["context_features"]
    with torch.inference_mode():
        score = rich_model(torch.from_numpy(context).to(device), torch.from_numpy(candidate).to(device))
        valid = torch.from_numpy(info["valid"]).to(device)
        action = score.masked_fill(~valid, -torch.inf).argmax(-1).cpu().numpy().astype(np.int64)
    raw = data["raw"]
    n = len(data["episodes"])
    logits = np.asarray(raw["logits"][:n], dtype=np.float32)
    labels = np.asarray(raw["labels"][:n], dtype=np.int64)
    bank = np.asarray(sorted(set(int(x) for x in classes)), dtype=np.int64)
    local = np.searchsorted(bank, labels)
    current = logits[:, 0]
    fused = 0.5 * (current[:, None] + logits)
    current_ok = current[:, bank].argmax(-1) == local
    candidate_ok = fused[..., bank].argmax(-1) == local[:, None]
    policy_ok = candidate_ok[np.arange(n), action]
    random_expected = candidate_ok[:, 1:4].mean(1)
    fixed_ok = candidate_ok[:, 1]
    delta = policy_ok.astype(np.float32) - current_ok.astype(np.float32)
    policy_minus_random = policy_ok.astype(np.float32) - random_expected.astype(np.float32)
    true_utility = np.asarray(targets["utility"], dtype=np.float32)
    selected_utility = true_utility[np.arange(n), action]
    best_utility = np.max(np.where(info["valid"], true_utility, -np.inf), axis=1)
    return {
        "episodes": int(n), "single_view0_top1": float(current_ok.mean()),
        "random_expected_top1": float(random_expected.mean()), "fixed_view1_top1": float(fixed_ok.mean()),
        "policy_top1": float(policy_ok.mean()), "policy_delta_vs_single": float(delta.mean()),
        "policy_minus_random": float(policy_minus_random.mean()),
        "policy_minus_random_ci95": _paired_ci(policy_minus_random, seed),
        "oracle_top1": float(candidate_ok[np.arange(n), 1 + np.where(candidate_ok[:, 1:4].any(1), candidate_ok[:, 1:4].argmax(1), 0)].mean()),
        "oracle_regret": float((best_utility - selected_utility).mean()),
        "selected_utility": float(selected_utility.mean()),
        "policy_action_histogram": np.bincount(action, minlength=NUM_VIEWS).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-cache", type=Path, default=Path("data/rl_candidate_rank_v6_new50_5/cache"))
    parser.add_argument("--output", type=Path, default=Path("work_dir/richer_cache_temporal_crossmodal_voi_v1"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--cache-batch-size", type=int, default=8)
    args = parser.parse_args()
    base = Path(__file__).resolve().parents[1]
    old_cache = args.old_cache if args.old_cache.is_absolute() else base / args.old_cache
    output = args.output if args.output.is_absolute() else base / args.output
    rich_root = output / "cache"
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    seed_all(args.seed)
    print(json.dumps({
        "RICHER_CACHE_TEMPORAL_CROSSMODAL_VOI_START": True,
        "old_cache": str(old_cache), "output": str(output), "device": str(device),
        "T": TEMPORAL_SEGMENTS, "segment_context_frames": TEMPORAL_CONTEXT_FRAMES,
        "split_protocol": "seen dqn_train -> seen_test; strict true unseen test",
        "future_candidate_inputs_to_predictor": False,
    }), flush=True)
    train_data = prepare_data(old_cache, "dqn_train", SEEN_CLASSES)
    seen_data = prepare_data(old_cache, "seen_test", SEEN_CLASSES)
    unseen_data = prepare_data(old_cache, "test", UNSEEN_CLASSES)
    data_map = {"dqn_train": (train_data, SEEN_CLASSES), "seen_test": (seen_data, SEEN_CLASSES), "true_unseen": (unseen_data, UNSEEN_CLASSES)}
    config = base / "logs/allviews_lowview50_5_20260907/final_aug_entropy_single_h2.yaml"
    checkpoint = base / "work_dir/allviews_lowview50_5_final_aug_entropy_single_h2/allviews_lowview50_5_20260907/last_model.pth"
    adapter = VPOCLIPAdapter.from_config(str(config), str(checkpoint), str(device))
    rich_paths: dict[str, Path] = {}
    for split, (data, classes) in [("dqn_train", (train_data, SEEN_CLASSES)), ("seen_test", (seen_data, SEEN_CLASSES)), ("test", (unseen_data, UNSEEN_CLASSES))]:
        rich_paths[split] = build_rich_cache(split, data, classes, base, rich_root, adapter, device, args.cache_batch_size)
    del adapter
    rich = {"dqn_train": load_rich_cache(rich_paths["dqn_train"]), "seen_test": load_rich_cache(rich_paths["seen_test"]), "test": load_rich_cache(rich_paths["test"])}
    targets = {key: targets_from_old_cache(data, classes) for key, (data, classes) in data_map.items()}
    feature_groups: dict[str, np.ndarray] = {}
    feature_names: dict[str, list[str]] = {}
    add_feature_group(feature_groups, feature_names, "temporal_uncertainty", temporal_uncertainty_features(rich["dqn_train"]))
    add_feature_group(feature_groups, feature_names, "temporal_interaction", temporal_interaction_features(rich["dqn_train"]))
    add_feature_group(feature_groups, feature_names, "observation_quality", observation_quality_features(rich["dqn_train"]))
    feature_groups["uncertainty_plus_interaction"] = np.concatenate([feature_groups["temporal_uncertainty"], feature_groups["temporal_interaction"]], axis=1)
    feature_names["uncertainty_plus_interaction"] = feature_names["temporal_uncertainty"] + feature_names["temporal_interaction"]
    feature_groups["available_full"] = np.concatenate([feature_groups["temporal_uncertainty"], feature_groups["temporal_interaction"], feature_groups["observation_quality"]], axis=1)
    feature_names["available_full"] = feature_names["temporal_uncertainty"] + feature_names["temporal_interaction"] + feature_names["observation_quality"]
    old_x, old_info = _current_state_features(train_data)
    feature_groups["old_cache_baseline"] = np.asarray(old_x, dtype=np.float32)
    feature_names["old_cache_baseline"] = [f"old_{x}" for x in old_info.get("causal_feature_groups", [])] + [f"old_feature_{i}" for i in range(max(0, old_x.shape[1] - len(old_info.get("causal_feature_groups", []))))]
    # Ensure exact feature names for the 30-dimensional legacy vector.
    if len(feature_names["old_cache_baseline"]) != old_x.shape[1]:
        feature_names["old_cache_baseline"] = [f"old_feature_{i}" for i in range(old_x.shape[1])]
    test_feature_sets: dict[str, dict[str, np.ndarray]] = {}
    for split, data_key in [("seen_test", "seen_test"), ("true_unseen", "test")]:
        r = rich[data_key]
        a, _ = temporal_uncertainty_features(r); c, _ = temporal_interaction_features(r); q, _ = observation_quality_features(r)
        old, _ = _current_state_features(data_map[split][0])
        test_feature_sets[split] = {
            "temporal_uncertainty": a, "temporal_interaction": c, "observation_quality": q,
            "uncertainty_plus_interaction": np.concatenate([a, c], axis=1),
            "available_full": np.concatenate([a, c, q], axis=1),
            "old_cache_baseline": old,
        }
    groups = np.asarray([str(train_data["raw"]["metadata"]["episodes"][int(s)].get("subject", "unknown")) for s in train_data["source_episode_ids"]])
    reports, final_models, folds = fit_all_feature_models(
        feature_groups, feature_names, targets["dqn_train"]["voi_random"], targets["dqn_train"]["view_sensitivity"], targets["dqn_train"]["recoverable"], groups, device, args.seed
    )
    test_payload = {
        split: {"features": test_feature_sets[split], "y": targets[data_key]["voi_random"], "sensitivity": targets[data_key]["view_sensitivity"], "recoverable": targets[data_key]["recoverable"]}
        for split, data_key in [("seen_test", "seen_test"), ("true_unseen", "true_unseen")]
    }
    add_test_predictions(reports, final_models, feature_groups, test_payload, ["seen_test", "true_unseen"], device)
    # Feature importance is reported for the full available RF fitted on all
    # seen training episodes and is not used to tune the unseen result.
    rf_model, _ = final_models["available_full"]["random_forest"]
    reports["available_full"]["random_forest_feature_importance"] = permutation_importance(
        rf_model, feature_groups["available_full"], targets["dqn_train"]["voi_random"], feature_names["available_full"], args.seed
    )
    voi_summary = {
        "old_baseline_reference": {"unseen_spearman": OLD_UNSEEN_SPEARMAN},
        "feature_ablation_status": {
            "temporal_uncertainty": "run", "cross_modal_disagreement": "unavailable",
            "temporal_interaction": "run", "observation_quality": "run",
            "uncertainty_plus_disagreement": "unavailable",
            "disagreement_plus_interaction": "unavailable",
            "uncertainty_plus_interaction": "run", "all_features": "partial_available_full",
        },
        "folds": folds, "reports": reports,
    }
    (output / "voi_predictor_metrics.json").write_text(json.dumps(json_safe(voi_summary), indent=2), encoding="utf-8")
    with (output / "voi_predictor_models.pkl").open("wb") as handle:
        pickle.dump(final_models, handle)

    # Gate A is pre-registered against the full available causal feature set
    # and RF, rather than selecting a model using the unseen score.
    full_unseen_rf = reports["available_full"]["test"]["true_unseen"]["random_forest"]
    gate_a = bool(full_unseen_rf.get("spearman") is not None and full_unseen_rf["spearman"] >= 0.20)
    ranker_report: dict[str, Any] = {
        "gate_a": {
            "old_unseen_spearman": OLD_UNSEEN_SPEARMAN,
            "available_full_rf_unseen_spearman": full_unseen_rf.get("spearman"),
            "threshold": 0.20, "passed": gate_a,
        },
        "variants": {},
    }
    if gate_a:
        rich_train_targets = targets["dqn_train"]
        for quantile in SENSITIVITY_QUANTILES:
            ranker = train_ranker_variant(train_data, rich_train_targets, rich["dqn_train"], feature_groups["available_full"], quantile, device, args.seed + int(quantile * 100), output)
            for split, data_key, classes in [("seen_test", "seen_test", SEEN_CLASSES), ("true_unseen", "test", UNSEEN_CLASSES)]:
                xq, _, _, _ = temporal_uncertainty_features(rich[data_key])  # ensure rich data was loaded
                # Recompute the same full available context for the query.
                tq, _ = temporal_interaction_features(rich[data_key]); qq, _ = observation_quality_features(rich[data_key])
                ranker["context_features"] = np.concatenate([xq, tq, qq], axis=1).astype(np.float32)
                result = evaluate_ranker(ranker, data_map[split][0], classes, device, targets[data_key], args.seed + len(split))
                ranker_report["variants"].setdefault(str(int(quantile * 100)), {})[split] = result
    else:
        ranker_report["skip_reason"] = "Gate A failed: richer available causal features did not reach unseen Spearman 0.20; no ranker/PPO was trained."
    (output / "ranker_metrics.json").write_text(json.dumps(json_safe(ranker_report), indent=2), encoding="utf-8")
    summary = {
        "version": "richer_cache_temporal_crossmodal_voi_v1",
        "cache": {
            "temporal_segments": TEMPORAL_SEGMENTS, "context_frames": TEMPORAL_CONTEXT_FRAMES,
            "paths": {key: str(value) for key, value in rich_paths.items()},
        },
        "splits": {name: {"episodes": int(len(data["episodes"])), "classes": list(classes)} for name, (data, classes) in data_map.items()},
        "label_definition": "same as previous: fused GT margin minus view0 GT margin; VoI=max(U)-mean(U); VS=max(U)-min(U)",
        "predictor": voi_summary,
        "ranker": ranker_report,
        "limitations": {
            "independent_rgb_skeleton_object_logits": "unavailable in source model/cache; not fabricated",
            "per_joint_confidence": "unavailable in source pose cache; presence proxy only",
            "object_temporal_tracks": "unavailable; static object map is repeated only as model compatibility input and masked in feature interpretation",
        },
    }
    (output / "summary.json").write_text(json.dumps(json_safe(summary), indent=2), encoding="utf-8")
    print(json.dumps({
        "RICHER_CACHE_TEMPORAL_CROSSMODAL_VOI_COMPLETE": True,
        "output": str(output), "gate_a": ranker_report["gate_a"],
        "unseen_full_rf": full_unseen_rf, "ranker": ranker_report,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
