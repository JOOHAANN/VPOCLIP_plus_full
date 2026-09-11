"""Shared implementation for semantic-conditioned next-best-view training.

The module intentionally keeps the policy causal.  Future-view logits are used
only by :func:`build_split_cache` to create offline utility labels; the model
itself receives the current view and known candidate geometry only.
"""

from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


NUM_VIEWS = 4
POSE_DIM = 442
OBJECT_DIM = 1800
Z_DIM = 512
EVIDENCE_DIM = 8
CANDIDATE_DIM = 8
FUTURE_LOGIT_DIM = 55
BODY_ORIENTATION_DIM = 28
CAMERA_POSE_DIM = 7
TOP_SEMANTIC_PROTOTYPES = 2
TOP_SEMANTIC_PROBABILITY_DIM = 5

# Keep the causal interface explicit. Offline utility, labels, and
# future-view margins may exist in an expanded cache, but they must never be
# passed to the selector during either training or evaluation.
CAUSAL_INPUT_NAMES = (
    "z", "semantic", "uncertainty", "evidence", "pose", "object_map",
    "quality", "candidate", "valid",
)


def causal_batch(batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Return only features available before moving to a candidate view."""

    missing = [name for name in CAUSAL_INPUT_NAMES if name not in batch]
    if missing:
        raise KeyError(f"missing causal selector inputs: {missing}")
    out = {name: batch[name] for name in CAUSAL_INPUT_NAMES}
    # Task-bank-conditioned summaries are optional so the v1--v7 caches and
    # checkpoints remain loadable without rebuilding them.
    for name in (
        "current_logits",
        "bank_current_top5",
        "bank_semantic",
        "bank_evidence",
        "bank_top_semantic",
        "bank_top_probability",
        "candidate_geometry_confidence",
        "candidate_slot",
    ):
        if name in batch:
            out[name] = batch[name]
    return out


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _save_array(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, array)


def _softmax_numpy(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(np.clip(shifted, -80.0, 80.0))
    return exp / np.maximum(exp.sum(axis=axis, keepdims=True), 1e-12)


def semantic_belief(
    z: np.ndarray,
    prototypes: np.ndarray,
    top_k: int = 5,
    temperature: float = 0.07,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return top-K text belief, normalized entropy, and compact evidence.

    ``z`` and text prototypes are normalized before calculating the belief.
    The top-K probabilities are renormalized for the semantic vector as
    required by the implementation guide.  Class IDs are deliberately not
    returned as policy inputs.
    """

    if z.ndim != 2 or z.shape[-1] != Z_DIM:
        raise ValueError(f"z must have shape [N,{Z_DIM}], got {z.shape}")
    if prototypes.ndim != 2 or prototypes.shape[-1] != Z_DIM:
        raise ValueError(f"prototypes must have shape [C,{Z_DIM}], got {prototypes.shape}")
    if not 1 <= top_k <= prototypes.shape[0]:
        raise ValueError("top_k must be within the text bank size")
    z_norm = z / np.maximum(np.linalg.norm(z, axis=-1, keepdims=True), 1e-8)
    t_norm = prototypes / np.maximum(np.linalg.norm(prototypes, axis=-1, keepdims=True), 1e-8)
    similarity = z_norm @ t_norm.T
    probability = _softmax_numpy(similarity / max(float(temperature), 1e-4))
    order = np.argsort(-probability, axis=-1)[:, :top_k]
    top_probability = np.take_along_axis(probability, order, axis=-1)
    top_probability = top_probability / np.maximum(top_probability.sum(-1, keepdims=True), 1e-12)
    semantic = np.einsum("nk,nkd->nd", top_probability, t_norm[order])
    semantic = semantic / np.maximum(np.linalg.norm(semantic, axis=-1, keepdims=True), 1e-8)
    entropy = -(probability * np.log(np.maximum(probability, 1e-12))).sum(-1)
    entropy /= math.log(float(probability.shape[-1]))
    top_gap = top_probability[:, 0] - top_probability[:, 1] if top_k >= 2 else top_probability[:, 0]
    evidence = np.concatenate(
        (top_probability, entropy[:, None], top_gap[:, None], probability.max(-1, keepdims=True)),
        axis=-1,
    ).astype(np.float32)
    if evidence.shape[-1] != EVIDENCE_DIM:
        raise AssertionError(f"unexpected evidence dimension {evidence.shape[-1]}")
    return semantic.astype(np.float32), entropy.astype(np.float32), evidence


def semantic_belief_per_row_bank(
    z: np.ndarray,
    prototypes: np.ndarray,
    row_bank: np.ndarray,
    top_k: int = 5,
    temperature: float = 0.07,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the same belief summary when every row has its own class bank.

    Dynamic 5-way training tasks need the semantic summary to describe the
    exact task bank used by the offline utility target.  The class IDs are
    still not exposed to the selector: only the prototype-weighted semantic
    vector and compact evidence are returned.
    """

    if z.ndim != 2 or z.shape[-1] != Z_DIM:
        raise ValueError(f"z must have shape [N,{Z_DIM}], got {z.shape}")
    if prototypes.ndim != 2 or prototypes.shape[-1] != Z_DIM:
        raise ValueError(f"prototypes must have shape [C,{Z_DIM}], got {prototypes.shape}")
    if row_bank.ndim != 2 or row_bank.shape[0] != z.shape[0]:
        raise ValueError("row_bank must have shape [N,K] matching z")
    if not 1 <= top_k <= row_bank.shape[1]:
        raise ValueError("top_k must be within each row bank size")
    if np.any((row_bank < 0) | (row_bank >= prototypes.shape[0])):
        raise ValueError("row_bank contains an invalid prototype index")
    z_norm = z / np.maximum(np.linalg.norm(z, axis=-1, keepdims=True), 1e-8)
    t_norm = prototypes / np.maximum(np.linalg.norm(prototypes, axis=-1, keepdims=True), 1e-8)
    row_prototypes = t_norm[row_bank]
    similarity = np.einsum("nd,nkd->nk", z_norm, row_prototypes)
    probability = _softmax_numpy(similarity / max(float(temperature), 1e-4))
    order = np.argsort(-probability, axis=-1)[:, :top_k]
    top_probability = np.take_along_axis(probability, order, axis=-1)
    top_probability = top_probability / np.maximum(top_probability.sum(-1, keepdims=True), 1e-12)
    semantic = np.einsum(
        "nk,nkd->nd",
        top_probability,
        np.take_along_axis(row_prototypes, order[..., None], axis=1),
    )
    semantic = semantic / np.maximum(np.linalg.norm(semantic, axis=-1, keepdims=True), 1e-8)
    entropy = -(probability * np.log(np.maximum(probability, 1e-12))).sum(-1)
    entropy /= math.log(float(probability.shape[-1]))
    top_gap = top_probability[:, 0] - top_probability[:, 1] if top_k >= 2 else top_probability[:, 0]
    evidence = np.concatenate(
        (top_probability, entropy[:, None], top_gap[:, None], probability.max(-1, keepdims=True)),
        axis=-1,
    ).astype(np.float32)
    if evidence.shape[-1] != EVIDENCE_DIM:
        raise AssertionError(f"unexpected evidence dimension {evidence.shape}")
    return semantic.astype(np.float32), entropy.astype(np.float32), evidence


def semantic_top_prototypes(
    z: np.ndarray,
    prototypes: np.ndarray,
    top_k: int = 5,
    temperature: float = 0.07,
    top_n: int = TOP_SEMANTIC_PROTOTYPES,
    output_slots: int | None = None,
) -> np.ndarray:
    """Return the highest-belief prototype vectors in a fixed-size format.

    ``semantic_belief`` intentionally compresses a class bank to one weighted
    vector.  That is useful as a smooth prior, but can erase which semantic
    direction was dominant.  This causal companion keeps the top two
    prototype vectors (without class IDs), allowing the selector to use the
    structure of the text bank while remaining valid for unseen classes.
    """

    if z.ndim != 2 or z.shape[-1] != Z_DIM:
        raise ValueError(f"z must have shape [N,{Z_DIM}], got {z.shape}")
    if prototypes.ndim != 2 or prototypes.shape[-1] != Z_DIM:
        raise ValueError(f"prototypes must have shape [C,{Z_DIM}], got {prototypes.shape}")
    if output_slots is None:
        output_slots = TOP_SEMANTIC_PROTOTYPES
    if not 1 <= top_n <= output_slots or top_n > prototypes.shape[0]:
        raise ValueError("top_n must fit output_slots and the class bank")
    z_norm = z / np.maximum(np.linalg.norm(z, axis=-1, keepdims=True), 1e-8)
    t_norm = prototypes / np.maximum(np.linalg.norm(prototypes, axis=-1, keepdims=True), 1e-8)
    similarity = z_norm @ t_norm.T
    # ``top_k`` controls the belief calculation convention; top prototypes
    # are selected from the same temperature-scaled posterior.
    probability = _softmax_numpy(similarity / max(float(temperature), 1e-4))
    order = np.argsort(-probability, axis=-1)[:, :top_n]
    selected = t_norm[order]
    if top_n < output_slots:
        selected = np.pad(
            selected,
            ((0, 0), (0, output_slots - top_n), (0, 0)),
            mode="constant",
        )
    return selected.reshape(len(z), output_slots * Z_DIM).astype(np.float32)


def semantic_top_prototypes_per_row_bank(
    z: np.ndarray,
    prototypes: np.ndarray,
    row_bank: np.ndarray,
    top_k: int = 5,
    temperature: float = 0.07,
    top_n: int = TOP_SEMANTIC_PROTOTYPES,
    output_slots: int | None = None,
) -> np.ndarray:
    """Top prototype vectors for a per-row class bank, with no class IDs."""

    if z.ndim != 2 or z.shape[-1] != Z_DIM:
        raise ValueError(f"z must have shape [N,{Z_DIM}], got {z.shape}")
    if prototypes.ndim != 2 or prototypes.shape[-1] != Z_DIM:
        raise ValueError(f"prototypes must have shape [C,{Z_DIM}], got {prototypes.shape}")
    if row_bank.ndim != 2 or row_bank.shape[0] != z.shape[0]:
        raise ValueError("row_bank must have shape [N,K] matching z")
    if output_slots is None:
        output_slots = TOP_SEMANTIC_PROTOTYPES
    if not 1 <= top_n <= output_slots or top_n > row_bank.shape[1]:
        raise ValueError("top_n must fit output_slots and each row bank")
    if np.any((row_bank < 0) | (row_bank >= prototypes.shape[0])):
        raise ValueError("row_bank contains an invalid prototype index")
    z_norm = z / np.maximum(np.linalg.norm(z, axis=-1, keepdims=True), 1e-8)
    t_norm = prototypes / np.maximum(np.linalg.norm(prototypes, axis=-1, keepdims=True), 1e-8)
    row_prototypes = t_norm[row_bank]
    similarity = np.einsum("nd,nkd->nk", z_norm, row_prototypes)
    probability = _softmax_numpy(similarity / max(float(temperature), 1e-4))
    order = np.argsort(-probability, axis=-1)[:, :top_n]
    selected = np.take_along_axis(row_prototypes, order[..., None], axis=1)
    if top_n < output_slots:
        selected = np.pad(
            selected,
            ((0, 0), (0, output_slots - top_n), (0, 0)),
            mode="constant",
        )
    return selected.reshape(len(z), output_slots * Z_DIM).astype(np.float32)


def semantic_top_probabilities(
    z: np.ndarray,
    prototypes: np.ndarray,
    top_k: int = 5,
    temperature: float = 0.07,
    output_dim: int = TOP_SEMANTIC_PROBABILITY_DIM,
) -> np.ndarray:
    """Return sorted top semantic probabilities without exposing class IDs.

    The weighted semantic vector is intentionally smooth, but it can map
    distinct posterior distributions to nearly the same vector.  Keeping the
    top probabilities gives the causal selector an uncertainty/ambiguity
    signal while the corresponding prototype directions remain in
    ``bank_top_semantic``.  Values are renormalized over the retained top-K
    entries and padded with zeros only when a smaller bank is used.
    """

    if z.ndim != 2 or z.shape[-1] != Z_DIM:
        raise ValueError(f"z must have shape [N,{Z_DIM}], got {z.shape}")
    if prototypes.ndim != 2 or prototypes.shape[-1] != Z_DIM:
        raise ValueError(f"prototypes must have shape [C,{Z_DIM}], got {prototypes.shape}")
    if not 1 <= output_dim:
        raise ValueError("output_dim must be positive")
    if not 1 <= top_k <= prototypes.shape[0]:
        raise ValueError("top_k must be within the text bank size")
    z_norm = z / np.maximum(np.linalg.norm(z, axis=-1, keepdims=True), 1e-8)
    t_norm = prototypes / np.maximum(np.linalg.norm(prototypes, axis=-1, keepdims=True), 1e-8)
    probability = _softmax_numpy((z_norm @ t_norm.T) / max(float(temperature), 1e-4))
    keep = min(int(top_k), int(output_dim))
    order = np.argsort(-probability, axis=-1)[:, :keep]
    selected = np.take_along_axis(probability, order, axis=-1)
    selected /= np.maximum(selected.sum(-1, keepdims=True), 1e-12)
    if keep < output_dim:
        selected = np.pad(
            selected,
            ((0, 0), (0, output_dim - keep)),
            mode="constant",
        )
    return selected.astype(np.float32)


def semantic_top_probabilities_per_row_bank(
    z: np.ndarray,
    prototypes: np.ndarray,
    row_bank: np.ndarray,
    top_k: int = 5,
    temperature: float = 0.07,
    output_dim: int = TOP_SEMANTIC_PROBABILITY_DIM,
) -> np.ndarray:
    """Top semantic probabilities for a per-row bank, with no class IDs."""

    if z.ndim != 2 or z.shape[-1] != Z_DIM:
        raise ValueError(f"z must have shape [N,{Z_DIM}], got {z.shape}")
    if prototypes.ndim != 2 or prototypes.shape[-1] != Z_DIM:
        raise ValueError(f"prototypes must have shape [C,{Z_DIM}], got {prototypes.shape}")
    if row_bank.ndim != 2 or row_bank.shape[0] != z.shape[0]:
        raise ValueError("row_bank must have shape [N,K] matching z")
    if not 1 <= output_dim:
        raise ValueError("output_dim must be positive")
    if not 1 <= top_k <= row_bank.shape[1]:
        raise ValueError("top_k must be within each row bank")
    if np.any((row_bank < 0) | (row_bank >= prototypes.shape[0])):
        raise ValueError("row_bank contains an invalid prototype index")
    z_norm = z / np.maximum(np.linalg.norm(z, axis=-1, keepdims=True), 1e-8)
    t_norm = prototypes / np.maximum(np.linalg.norm(prototypes, axis=-1, keepdims=True), 1e-8)
    row_prototypes = t_norm[row_bank]
    probability = _softmax_numpy(
        np.einsum("nd,nkd->nk", z_norm, row_prototypes)
        / max(float(temperature), 1e-4)
    )
    keep = min(int(top_k), int(output_dim))
    order = np.argsort(-probability, axis=-1)[:, :keep]
    selected = np.take_along_axis(probability, order, axis=-1)
    selected /= np.maximum(selected.sum(-1, keepdims=True), 1e-12)
    if keep < output_dim:
        selected = np.pad(
            selected,
            ((0, 0), (0, output_dim - keep)),
            mode="constant",
        )
    return selected.astype(np.float32)


def _margin(scores: np.ndarray, local_target: np.ndarray) -> np.ndarray:
    """Per-row GT-vs-best-negative margin for ``scores[N,C]``."""

    target = scores[np.arange(len(scores)), local_target]
    negative = scores.copy()
    negative[np.arange(len(scores)), local_target] = -np.inf
    return target - negative.max(-1)


def _load_recording_angle_quality(path: str | Path | None) -> dict[str, float]:
    """Load reliability scores for exact recording/camera angle entries.

    The score is an optional causal sensor.  It does not change the bearing
    or the physical action-slot mapping; it only tells a selector how much it
    should trust an estimated bearing for this recording.
    """

    if path is None:
        return {}
    angle_path = Path(path)
    if not angle_path.exists():
        raise FileNotFoundError(f"angle quality CSV not found: {angle_path}")
    scores = {
        "reference": 1.00,
        "high": 0.85,
        "medium": 0.60,
        "low": 0.30,
        "imputed": 0.10,
    }
    result: dict[str, float] = {}
    with angle_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = str(row.get("sample_id", "")).strip()
            confidence = str(row.get("confidence", "")).strip().lower()
            if key and confidence in scores:
                result[key] = float(scores[confidence])
    return result


def build_split_cache(
    raw_root: Path,
    output_root: Path,
    split: str,
    class_bank: Sequence[int],
    allowed_labels: Sequence[int] | None = None,
    top_k: int = 5,
    semantic_temperature: float = 0.07,
    tie_threshold: float = 0.02,
    task_bank_size: int | None = None,
    task_seed: int = 20260909,
    utility_class_bank: Sequence[int] | None = None,
    top_prototype_count: int = TOP_SEMANTIC_PROTOTYPES,
    angle_quality_csv: str | Path | None = None,
) -> dict[str, Any]:
    """Expand a four-view cache into causal current/candidate NBV samples.

    One record is emitted for every valid current view.  Its four candidate
    slots remain aligned with the original cache, with the current slot and
    padded slots masked out.  The utility is the marginal fused-class margin
    gain, not the quality of a candidate in isolation.
    """

    raw_root = Path(raw_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    z_all = np.load(raw_root / "z.npy", mmap_mode="r", allow_pickle=False)
    logits_all = np.load(raw_root / "logits.npy", mmap_mode="r", allow_pickle=False)
    pose_all = np.load(raw_root / "pose.npy", mmap_mode="r", allow_pickle=False)
    object_all = np.load(raw_root / "object_map.npy", mmap_mode="r", allow_pickle=False)
    quality_all = np.load(raw_root / "image_quality.npy", mmap_mode="r", allow_pickle=False)
    geometry_all = np.load(raw_root / "view_geometry.npy", mmap_mode="r", allow_pickle=False)
    reachable_all = np.load(raw_root / "reachable.npy", mmap_mode="r", allow_pickle=False)
    cost_all = np.load(raw_root / "move_cost.npy", mmap_mode="r", allow_pickle=False)
    valid_all = np.load(raw_root / "view_valid.npy", mmap_mode="r", allow_pickle=False)
    labels_path = raw_root / "target_columns.npy"
    labels_all = np.load(labels_path if labels_path.exists() else raw_root / "labels.npy", allow_pickle=False)
    prototype_path = raw_root / "text_prototypes.npy"
    if not prototype_path.exists():
        raise FileNotFoundError(f"missing text prototypes: {prototype_path}")
    prototypes = np.load(prototype_path, allow_pickle=False)

    # Geometry confidence is an available pose/map-side sensor, not a future
    # image feature.  The raw tensors intentionally keep their historical
    # shape, so read the per-view confidence from the raw metadata and expose
    # it as a separate candidate-side field for newer selector variants.
    raw_metadata_path = raw_root / "metadata.json"
    raw_metadata = json.loads(raw_metadata_path.read_text(encoding="utf-8"))
    raw_episodes = raw_metadata.get("episodes", [])
    camera_to_slot = {"C001": 0, "C003": 1, "C005": 2, "C007": 3}
    # ``view_valid`` is padded for incomplete recordings.  Keep the physical
    # camera identity separately from the compact row position: if C003 is
    # missing, the second legal row is C005, not a camera-slot-1 sample.
    camera_slot_all = np.zeros((len(raw_episodes), NUM_VIEWS), dtype=np.int64)
    for episode_index, episode in enumerate(raw_episodes):
        for view_index, view in enumerate(episode.get("views", [])[:NUM_VIEWS]):
            camera = str(view.get("camera", ""))
            camera_slot_all[episode_index, view_index] = camera_to_slot.get(
                camera, min(view_index, NUM_VIEWS - 1)
            )
    angle_quality = _load_recording_angle_quality(angle_quality_csv)
    geometry_confidence_all = np.zeros((len(raw_episodes), NUM_VIEWS), dtype=np.float32)
    for episode_index, episode in enumerate(raw_episodes):
        for view_index, view in enumerate(episode.get("views", [])[:NUM_VIEWS]):
            value = view.get("geometry_confidence", 0.0)
            try:
                value = float(value)
            except (TypeError, ValueError):
                value = 0.0
            geometry_confidence_all[episode_index, view_index] = float(
                np.clip(value if np.isfinite(value) else 0.0, 0.0, 1.0)
            )
            if (
                angle_quality
                and str(view.get("geometry_source", ""))
                == "csv_optical_axis_relative_yaw_fallback"
            ):
                source_base = str(view.get("source_base", episode.get("base_sample", "")))
                camera = str(view.get("camera", ""))
                quality = angle_quality.get(f"{source_base}_{camera}")
                if quality is not None:
                    geometry_confidence_all[episode_index, view_index] = float(quality)
    if len(geometry_confidence_all) != int(geometry_all.shape[0]):
        raise ValueError(
            f"raw metadata/tensor episode mismatch: metadata={len(geometry_confidence_all)} "
            f"geometry={geometry_all.shape[0]}"
        )

    bank = np.asarray(sorted({int(c) for c in class_bank}), dtype=np.int64)
    if len(bank) < 2:
        raise ValueError("a class bank needs at least two classes")
    if not 1 <= int(top_prototype_count) <= int(top_k):
        raise ValueError("top_prototype_count must be between 1 and top_k")
    top_prototype_count = int(top_prototype_count)
    if np.any((bank < 0) | (bank >= logits_all.shape[-1])):
        raise ValueError(f"class bank outside logits range: {bank.tolist()}")
    utility_bank = np.asarray(
        sorted({int(c) for c in (bank if utility_class_bank is None else utility_class_bank)}),
        dtype=np.int64,
    )
    if len(utility_bank) < 2:
        raise ValueError("a utility class bank needs at least two classes")
    if np.any((utility_bank < 0) | (utility_bank >= logits_all.shape[-1])):
        raise ValueError(f"utility class bank outside logits range: {utility_bank.tolist()}")
    episode_ids, start_views = np.nonzero(valid_all)
    episode_ids = episode_ids.astype(np.int64)
    start_views = start_views.astype(np.int64)
    if allowed_labels is not None:
        keep = np.isin(labels_all[episode_ids], np.asarray(list(allowed_labels), dtype=np.int64))
        episode_ids, start_views = episode_ids[keep], start_views[keep]
    n = len(episode_ids)
    if n == 0:
        raise ValueError(f"no valid current views in {raw_root}")

    current_z = np.asarray(z_all[episode_ids, start_views], dtype=np.float32)
    semantic, uncertainty, evidence = semantic_belief(
        current_z, prototypes, top_k=top_k, temperature=semantic_temperature
    )
    bank_semantic, _, bank_evidence = semantic_belief(
        current_z,
        prototypes[bank],
        top_k=min(int(top_k), len(bank)),
        temperature=semantic_temperature,
    )
    bank_top_semantic = semantic_top_prototypes(
        current_z,
        prototypes[bank],
        top_k=min(int(top_k), len(bank)),
        temperature=semantic_temperature,
        top_n=min(top_prototype_count, len(bank)),
        output_slots=top_prototype_count,
    )
    bank_top_probability = semantic_top_probabilities(
        current_z,
        prototypes[bank],
        top_k=min(int(top_k), len(bank)),
        temperature=semantic_temperature,
        output_dim=top_prototype_count,
    )
    current_pose = np.asarray(pose_all[episode_ids, start_views], dtype=np.float32)
    current_object = np.asarray(object_all[episode_ids, start_views], dtype=np.float32)
    current_quality = np.asarray(quality_all[episode_ids, start_views], dtype=np.float32)

    candidate_valid = np.asarray(valid_all[episode_ids], dtype=bool)
    candidate_valid &= np.asarray(reachable_all[episode_ids, start_views], dtype=bool)
    candidate_valid[np.arange(n), start_views] = False

    candidate_abs = np.asarray(geometry_all[episode_ids, :, :2], dtype=np.float32)
    source_abs = np.asarray(geometry_all[episode_ids, start_views, :2], dtype=np.float32)
    relative = np.stack(
        (
            candidate_abs[..., 0] * source_abs[:, None, 1]
            - candidate_abs[..., 1] * source_abs[:, None, 0],
            candidate_abs[..., 0] * source_abs[:, None, 0]
            + candidate_abs[..., 1] * source_abs[:, None, 1],
        ),
        axis=-1,
    )
    candidate_cost = np.asarray(cost_all[episode_ids, start_views], dtype=np.float32)
    candidate_geometry_confidence = np.asarray(
        geometry_confidence_all[episode_ids], dtype=np.float32
    )
    candidate_geometry_confidence[~candidate_valid] = 0.0
    candidate_slot = np.asarray(camera_slot_all[episode_ids], dtype=np.int64).copy()
    candidate_features = np.concatenate(
        (
            candidate_abs,
            relative,
            np.broadcast_to(source_abs[:, None, :], (n, NUM_VIEWS, 2)),
            candidate_cost[..., None],
            candidate_valid[..., None].astype(np.float32),
        ),
        axis=-1,
    ).astype(np.float32)
    if candidate_features.shape[-1] != CANDIDATE_DIM:
        raise AssertionError(candidate_features.shape)

    labels = np.asarray(labels_all[episode_ids], dtype=np.int64)
    target_bank_size = None if task_bank_size in (None, 0) else int(task_bank_size)
    if target_bank_size is not None:
        if split != "train":
            raise ValueError("task_bank_size is supported only for the training split")
        if not 2 <= target_bank_size <= len(bank):
            raise ValueError("task_bank_size must be between 2 and the training pool size")
        if not np.isin(labels, utility_bank).all():
            raise ValueError(
                f"{split}: labels outside training utility class pool {utility_bank.tolist()}"
            )
        # Match the eventual 5-way unseen protocol while retaining only seen
        # classes in training.  The bank is sampled per recording episode, so
        # all current-view contexts from one clip share the same negatives.
        rng = np.random.default_rng(int(task_seed))
        episode_bank: dict[int, np.ndarray] = {}
        row_bank = np.empty((n, target_bank_size), dtype=np.int64)
        for row, episode_id in enumerate(episode_ids.tolist()):
            label = int(labels[row])
            if episode_id not in episode_bank:
                others = utility_bank[utility_bank != label]
                negatives = rng.choice(others, size=target_bank_size - 1, replace=False)
                episode_bank[episode_id] = np.concatenate((np.asarray([label]), negatives)).astype(np.int64)
            row_bank[row] = episode_bank[episode_id]
        # The policy-side task summary must describe the same sampled bank as
        # the target.  Earlier dynamic caches used the full training bank for
        # this field, which made the state and target disagree.  This branch
        # is only active for newly built dynamic caches; static v1--v3/v6--v9
        # caches retain their existing files and behavior.
        bank_semantic, _, bank_evidence = semantic_belief_per_row_bank(
            current_z,
            prototypes,
            row_bank,
            top_k=min(int(top_k), target_bank_size),
            temperature=semantic_temperature,
        )
        bank_top_semantic = semantic_top_prototypes_per_row_bank(
            current_z,
            prototypes,
            row_bank,
            top_k=min(int(top_k), target_bank_size),
            temperature=semantic_temperature,
            top_n=min(top_prototype_count, target_bank_size),
            output_slots=top_prototype_count,
        )
        bank_top_probability = semantic_top_probabilities_per_row_bank(
            current_z,
            prototypes,
            row_bank,
            top_k=min(int(top_k), target_bank_size),
            temperature=semantic_temperature,
            output_dim=top_prototype_count,
        )
        local_target = np.zeros(n, dtype=np.int64)
        current_logits = np.asarray(logits_all[episode_ids, start_views], dtype=np.float32)
        current_scores = np.take_along_axis(current_logits, row_bank, axis=1)
        future_logits = np.asarray(logits_all[episode_ids], dtype=np.float32)
        row_bank_expanded = np.broadcast_to(row_bank[:, None, :], (n, NUM_VIEWS, target_bank_size))
        all_scores = np.take_along_axis(future_logits, row_bank_expanded, axis=2)
    else:
        local_target = np.searchsorted(utility_bank, labels)
        if np.any(local_target >= len(utility_bank)) or np.any(
            utility_bank[np.minimum(local_target, len(utility_bank) - 1)] != labels
        ):
            raise ValueError(f"{split}: labels outside utility class bank {utility_bank.tolist()}")
        current_scores = np.asarray(
            logits_all[episode_ids, start_views][:, utility_bank], dtype=np.float32
        )
        all_scores = np.asarray(logits_all[episode_ids][:, :, utility_bank], dtype=np.float32)
    # Keep a fixed-size, class-ID-free summary of current evidence inside the
    # exact utility bank used for this split.  For training this is the
    # sampled five-way bank; for evaluation it is the split's known bank.  The
    # sorted values make the sensor comparable when the bank contains a
    # different set of class IDs, without exposing the label or class names.
    bank_current_top5 = np.sort(np.asarray(current_scores, dtype=np.float32), axis=1)[:, ::-1]
    if bank_current_top5.shape[1] < 5:
        bank_current_top5 = np.pad(
            bank_current_top5,
            ((0, 0), (0, 5 - bank_current_top5.shape[1])),
            constant_values=-32.0,
        )
    bank_current_top5 = bank_current_top5[:, :5].astype(np.float32)
    before_margin = _margin(current_scores, local_target)
    fused_scores = 0.5 * (current_scores[:, None, :] + all_scores)
    after_margin = np.empty((n, NUM_VIEWS), dtype=np.float32)
    for view in range(NUM_VIEWS):
        after_margin[:, view] = _margin(fused_scores[:, view], local_target)
    utility = after_margin - before_margin[:, None]
    utility[~candidate_valid] = 0.0
    utility_valid = utility[candidate_valid]
    oracle_action = np.full(n, -1, dtype=np.int64)
    has_candidate = candidate_valid.any(-1)
    masked_utility = utility.copy()
    masked_utility[~candidate_valid] = -np.inf
    oracle_action[has_candidate] = masked_utility[has_candidate].argmax(-1)
    candidate_count = candidate_valid.sum(-1).astype(np.int64)
    state_weight = np.where(candidate_count <= 1, 0.1, 1.0).astype(np.float32)
    state_weight[~has_candidate] = 0.0

    arrays = {
        "episode_ids": episode_ids,
        "start_views": start_views,
        "labels": labels,
        "z": current_z,
        "semantic": semantic,
        "uncertainty": uncertainty[:, None].astype(np.float32),
        "evidence": evidence,
        "bank_semantic": bank_semantic,
        "bank_evidence": bank_evidence,
        "bank_top_semantic": bank_top_semantic,
        "bank_top_probability": bank_top_probability,
        # The current-view recognizer scores are causal state.  Keep them in
        # every split so the selector has the same input contract at train,
        # validation, and real unseen test time.
        "current_logits": np.asarray(
            logits_all[episode_ids, start_views], dtype=np.float32
        ),
        "bank_current_top5": bank_current_top5,
        "pose": current_pose,
        "object_map": current_object,
        "quality": current_quality,
        "candidate": candidate_features,
        "candidate_geometry_confidence": candidate_geometry_confidence,
        "candidate_slot": candidate_slot,
        "valid": candidate_valid,
        "utility": utility.astype(np.float32),
        "before_margin": before_margin.astype(np.float32),
        "after_margin": after_margin.astype(np.float32),
        "oracle_action": oracle_action,
        "state_weight": state_weight,
        "candidate_count": candidate_count,
    }
    # These arrays are deliberately marked as training/evaluation bookkeeping,
    # not selector inputs.  The auxiliary transition objective can use the
    # privileged future logits during training, while ``causal_batch`` below
    # excludes them from every policy forward pass.
    if split == "train":
        arrays["future_logits"] = np.asarray(
            logits_all[episode_ids], dtype=np.float32
        )
        if target_bank_size is not None:
            arrays["task_bank"] = row_bank
    for name, array in arrays.items():
        _save_array(output_root / f"{name}.npy", array)
    metadata = {
        "format": "sc_nbv_cache_v1",
        "split": split,
        "source_cache": str(raw_root),
        "episodes": int(len(np.unique(episode_ids))),
        "contexts": int(n),
        "class_bank": bank.tolist(),
        "utility_class_bank": utility_bank.tolist(),
        "task_bank_size": target_bank_size,
        "task_seed": int(task_seed) if target_bank_size is not None else None,
        "allowed_labels": None if allowed_labels is None else sorted({int(c) for c in allowed_labels}),
        "views": NUM_VIEWS,
        "candidate_feature_dim": CANDIDATE_DIM,
        "candidate_geometry_confidence": (
            "per-candidate geometry confidence from body-relative resolver; "
            "optional recording-angle reliability override; causal sensor"
        ),
        "candidate_slot": (
            "physical camera slot C001=0, C003=1, C005=2, C007=3; "
            "padded rows use 0 and are invalid"
        ),
        "angle_quality_csv": None if angle_quality_csv is None else str(angle_quality_csv),
        "angle_quality_mapping": (
            "reference=1.00, high=0.85, medium=0.60, low=0.30, imputed=0.10"
            if angle_quality_csv is not None else None
        ),
        "evidence_dim": EVIDENCE_DIM,
        "bank_evidence_dim": EVIDENCE_DIM,
        "top_k": int(top_k),
        "top_prototype_count": int(top_prototype_count),
        "semantic_temperature": float(semantic_temperature),
        "utility": "delta_margin = margin(fuse(current,candidate),GT)-margin(current,GT)",
        "tie_threshold": float(tie_threshold),
        "policy_inputs": [
            "current z",
            "current-view VPOCLIP logits (causal; no GT)",
            "sorted top-5 current logits inside the active class bank (causal; no class IDs)",
            "top-k weighted text semantic belief",
            "current uncertainty and label-free evidence",
            "task-bank-conditioned semantic belief and evidence",
            "top-1/top-2 task-bank prototype vectors",
            "current pose/skeleton",
            "current object feature",
            "current quality",
            "candidate continuous geometry and movement cost",
        ],
        "future_fields_used_only_for_label": [
            "candidate-view VPOCLIP logits",
            "ground-truth action label",
        ],
        "training_only_auxiliary_fields": [
            "future_logits",
        ] + (["task_bank"] if target_bank_size is not None else []),
        "view_count_histogram": {
            str(int(k)): int(v) for k, v in zip(*np.unique(valid_all[episode_ids].sum(-1), return_counts=True))
        },
        "candidate_count_histogram": {
            str(int(k)): int(v) for k, v in zip(*np.unique(candidate_count, return_counts=True))
        },
        "informative_pair_contexts": int(
            np.sum(
                np.where(
                    candidate_valid.sum(-1) > 1,
                    np.nanmax(np.where(candidate_valid, utility, np.nan), axis=-1)
                    - np.nanmin(np.where(candidate_valid, utility, np.nan), axis=-1)
                    > float(tie_threshold),
                    False,
                )
            )
        ),
        "utility_stats_valid": {
            "mean": float(utility_valid.mean()) if len(utility_valid) else 0.0,
            "std": float(utility_valid.std()) if len(utility_valid) else 0.0,
            "min": float(utility_valid.min()) if len(utility_valid) else 0.0,
            "max": float(utility_valid.max()) if len(utility_valid) else 0.0,
        },
    }
    (output_root / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def _tensor_array(root: Path, name: str, device: torch.device, dtype: torch.dtype | None = None) -> torch.Tensor:
    array = np.load(root / f"{name}.npy", allow_pickle=False)
    tensor = torch.from_numpy(np.array(array, copy=True))
    return tensor.to(device=device, dtype=dtype) if dtype is not None else tensor.to(device=device)


class SCNBVData:
    """GPU-resident expanded context cache used by training/evaluation."""

    INPUT_NAMES = (
        "episode_ids", "start_views", "labels", "z", "semantic", "uncertainty", "evidence",
        "pose", "object_map", "quality", "candidate", "valid", "utility", "before_margin",
        "after_margin", "oracle_action", "state_weight", "candidate_count",
    )
    OPTIONAL_NAMES = (
        "current_logits", "future_logits", "task_bank", "bank_semantic", "bank_evidence",
        "bank_top_semantic",
        "bank_top_probability",
        "bank_current_top5",
        "candidate_geometry_confidence",
        "candidate_slot",
    )

    def __init__(self, root: Path, device: torch.device) -> None:
        self.root = Path(root)
        metadata = json.loads((self.root / "metadata.json").read_text(encoding="utf-8"))
        self.metadata = metadata
        self.device = device
        self.data = {}
        long_names = {"episode_ids", "start_views", "labels", "oracle_action", "candidate_count"}
        bool_names = {"valid"}
        for name in self.INPUT_NAMES:
            dtype = torch.long if name in long_names else torch.bool if name in bool_names else torch.float32
            self.data[name] = _tensor_array(self.root, name, device, dtype)
        optional_long = {"task_bank", "candidate_slot"}
        for name in self.OPTIONAL_NAMES:
            path = self.root / f"{name}.npy"
            if not path.exists():
                continue
            dtype = torch.long if name in optional_long else torch.float32
            self.data[name] = _tensor_array(self.root, name, device, dtype)
        prototype_path = self.root / "text_prototypes.npy"
        self.prototypes = (
            _tensor_array(self.root, "text_prototypes", device, torch.float32)
            if prototype_path.exists()
            else None
        )
        self.num_contexts = int(self.data["z"].shape[0])
        self.num_episodes = int(metadata["episodes"])
        self.row_lookup = torch.full((self._raw_episode_count(), NUM_VIEWS), -1, dtype=torch.long, device=device)
        self.row_lookup[self.data["episode_ids"], self.data["start_views"]] = torch.arange(
            self.num_contexts, device=device
        )

    def _raw_episode_count(self) -> int:
        # The maximum ID is sufficient for lookup; evaluation indexes only IDs
        # from the corresponding raw cache and expands the table if necessary.
        return int(self.data["episode_ids"].max().item()) + 1

    def rows_for(self, episodes: torch.Tensor, starts: torch.Tensor) -> torch.Tensor:
        if int(episodes.max().item()) >= self.row_lookup.shape[0]:
            raise IndexError("context cache does not contain requested episode")
        rows = self.row_lookup[episodes.long(), starts.long()]
        if (rows < 0).any():
            raise ValueError("requested current view is absent from SC-NBV context cache")
        return rows

    def batch(self, rows: torch.Tensor) -> dict[str, torch.Tensor]:
        return {name: value[rows] for name, value in self.data.items()}


class RawViewData:
    """GPU-resident original four-view cache for final classification metrics."""

    def __init__(self, root: Path, device: torch.device) -> None:
        self.root = Path(root)
        self.device = device
        self.z = _tensor_array(root, "z", device, torch.float32)
        self.logits = _tensor_array(root, "logits", device, torch.float32)
        self.pose = _tensor_array(root, "pose", device, torch.float32)
        self.object_map = _tensor_array(root, "object_map", device, torch.float32)
        self.quality = _tensor_array(root, "image_quality", device, torch.float32)
        self.geometry = _tensor_array(root, "view_geometry", device, torch.float32)
        self.reachable = _tensor_array(root, "reachable", device, torch.bool)
        self.cost = _tensor_array(root, "move_cost", device, torch.float32)
        self.valid = _tensor_array(root, "view_valid", device, torch.bool)
        label_name = "target_columns" if (Path(root) / "target_columns.npy").exists() else "labels"
        self.labels = _tensor_array(root, label_name, device, torch.long)
        self.prototypes = _tensor_array(root, "text_prototypes", device, torch.float32)
        self.num_episodes = int(self.labels.shape[0])


def normalize_model_inputs(
    batch: Mapping[str, torch.Tensor],
    normalize_z: bool = True,
    normalize_semantic: bool = True,
) -> dict[str, torch.Tensor]:
    """Apply the same bounded preprocessing to every model variant."""

    out = dict(batch)
    out["z"] = out["z"].float()
    if normalize_z:
        out["z"] = F.normalize(out["z"], dim=-1)
    out["semantic"] = out["semantic"].float()
    if normalize_semantic:
        out["semantic"] = F.normalize(out["semantic"], dim=-1)
    if "bank_top_semantic" in out:
        top = out["bank_top_semantic"].float()
        if top.shape[-1] % Z_DIM != 0:
            raise ValueError("bank_top_semantic must contain whole prototype vectors")
        top = top.reshape(*top.shape[:-1], -1, Z_DIM)
        out["bank_top_semantic"] = F.normalize(top, dim=-1).reshape(*top.shape[:-2], -1)
    if "bank_top_probability" in out:
        probability = out["bank_top_probability"].float().clamp_min(0.0)
        out["bank_top_probability"] = probability / probability.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)
    if "current_logits" in out:
        # Scores from the current view are causal evidence.  They are not
        # future-view logits and do not contain the ground-truth label.
        out["current_logits"] = out["current_logits"].float().clamp(-32.0, 32.0)
    if "bank_current_top5" in out:
        out["bank_current_top5"] = out["bank_current_top5"].float().clamp(-32.0, 32.0)
    out["pose"] = out["pose"].float().clamp(-2.0, 2.0)
    out["object_map"] = out["object_map"].float().clamp_min(0).log1p().clamp_max(5.0) / math.log(11.0)
    out["quality"] = out["quality"].float().clamp(0.0, 1.0)
    out["candidate"] = out["candidate"].float()
    if "candidate_geometry_confidence" in out:
        out["candidate_geometry_confidence"] = out[
            "candidate_geometry_confidence"
        ].float().clamp(0.0, 1.0)
    out["evidence"] = out["evidence"].float().clamp(0.0, 1.0)
    return out


def canonical_skeleton_features(pose: torch.Tensor) -> torch.Tensor:
    """Convert the cached 13-frame COCO-17 pose into 212 causal features.

    The raw pose is camera-dependent.  We remove translation with a torso
    center and normalize by torso width, then retain both static body shape
    and temporal motion.  The fixed 212-dimensional representation is small
    enough to regularize the policy while retaining the body-orientation and
    motion cues that were useful in the preserved g18 experiment.

    Layout: mean/std/min/max normalized joint positions (136), mean/std joint
    velocity (68), and eight global motion/orientation statistics (8).
    """

    if pose.shape[-1] != POSE_DIM:
        raise ValueError(f"pose must end in {POSE_DIM}, got {tuple(pose.shape)}")
    shape = pose.shape[:-1]
    points = torch.nan_to_num(pose.float()).reshape(*shape, 13, 17, 2)
    # COCO-17: left/right shoulder = 5/6, left/right hip = 11/12.
    shoulder_mid = 0.5 * (points[..., 5, :] + points[..., 6, :])
    hip_mid = 0.5 * (points[..., 11, :] + points[..., 12, :])
    root = 0.5 * (shoulder_mid + hip_mid)
    shoulder_width = torch.linalg.vector_norm(points[..., 5, :] - points[..., 6, :], dim=-1)
    hip_width = torch.linalg.vector_norm(points[..., 11, :] - points[..., 12, :], dim=-1)
    scale = 0.5 * (shoulder_width.median(dim=-1).values + hip_width.median(dim=-1).values)
    # A conservative fallback also handles a missing/degenerate skeleton.
    scale = scale.clamp_min(0.05).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    normalized = ((points - root.unsqueeze(-2)) / scale).clamp(-5.0, 5.0)
    flat = normalized.reshape(*shape, 13, 34)
    position_stats = torch.cat(
        (
            flat.mean(dim=-2),
            flat.std(dim=-2, unbiased=False),
            flat.amin(dim=-2),
            flat.amax(dim=-2),
        ),
        dim=-1,
    )
    velocity = normalized[..., 1:, :, :] - normalized[..., :-1, :, :]
    velocity_flat = velocity.reshape(*shape, 12, 34)
    velocity_stats = torch.cat(
        (
            velocity_flat.mean(dim=-2),
            velocity_flat.std(dim=-2, unbiased=False),
        ),
        dim=-1,
    )
    root_track = normalized[..., :, 0, :]
    root_velocity = root_track[..., 1:, :] - root_track[..., :-1, :]
    root_velocity_mean = root_velocity.mean(dim=-2)
    root_velocity_std = root_velocity.std(dim=-2, unbiased=False)
    torso = points[..., 6, :] - points[..., 5, :]
    torso_angle = torch.atan2(torso[..., 1], torso[..., 0])
    angle_sin = torch.sin(torso_angle)
    angle_cos = torch.cos(torso_angle)
    angle_delta = torch.atan2(
        torch.sin(torso_angle[..., 1:] - torso_angle[..., :-1]),
        torch.cos(torso_angle[..., 1:] - torso_angle[..., :-1]),
    )
    global_stats = torch.cat(
        (
            root_velocity_mean,
            root_velocity_std,
            angle_sin.mean(dim=-1, keepdim=True),
            angle_cos.mean(dim=-1, keepdim=True),
            angle_delta.mean(dim=-1, keepdim=True),
            angle_delta.std(dim=-1, unbiased=False, keepdim=True),
        ),
        dim=-1,
    )
    result = torch.cat((position_stats, velocity_stats, global_stats), dim=-1)
    if result.shape[-1] != 212:
        raise AssertionError(f"unexpected canonical skeleton dimension: {result.shape}")
    return torch.nan_to_num(result).clamp(-8.0, 8.0)


def temporal_skeleton_sequence(
    pose: torch.Tensor,
    skeleton_mode: str = "canonical",
) -> torch.Tensor:
    """Return the current 13-frame skeleton without collapsing time.

    The compact 212-D skeleton statistics are useful as a regularizer, but
    they remove the ordering of the observation frames.  This representation
    keeps a small ``[13,34]`` sequence for a causal temporal encoder.  The
    canonical mode uses the same torso-centering and scale normalization as
    :func:`canonical_skeleton_features`; ``legacy_raw`` is an explicit raw
    compatibility option.
    """

    if pose.shape[-1] != POSE_DIM:
        raise ValueError(f"pose must end in {POSE_DIM}, got {tuple(pose.shape)}")
    if skeleton_mode not in {"canonical", "legacy_raw"}:
        raise ValueError(f"unknown skeleton mode: {skeleton_mode}")
    shape = pose.shape[:-1]
    points = torch.nan_to_num(pose.float()).reshape(*shape, 13, 17, 2)
    if skeleton_mode == "canonical":
        shoulder_mid = 0.5 * (points[..., 5, :] + points[..., 6, :])
        hip_mid = 0.5 * (points[..., 11, :] + points[..., 12, :])
        root = 0.5 * (shoulder_mid + hip_mid)
        shoulder_width = torch.linalg.vector_norm(
            points[..., 5, :] - points[..., 6, :], dim=-1
        )
        hip_width = torch.linalg.vector_norm(
            points[..., 11, :] - points[..., 12, :], dim=-1
        )
        scale = 0.5 * (
            shoulder_width.median(dim=-1).values + hip_width.median(dim=-1).values
        )
        scale = scale.clamp_min(0.05).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        points = ((points - root.unsqueeze(-2)) / scale).clamp(-5.0, 5.0)
    return torch.nan_to_num(points.reshape(*shape, 13, 34)).clamp(-8.0, 8.0)


def legacy_raw_skeleton_features(pose: torch.Tensor) -> torch.Tensor:
    """Build the historical 212-D skeleton statistics without body scaling.

    The preserved g18 checkpoints were trained with a skeleton transform that
    is no longer available in the trimmed source tree.  This compatibility
    representation keeps the same dimensional contract as the current
    encoder, but does not subtract a torso centre or divide by body width.
    It is intentionally opt-in so the v1--v5 behaviour remains unchanged.
    """

    if pose.shape[-1] != POSE_DIM:
        raise ValueError(f"pose must end in {POSE_DIM}, got {tuple(pose.shape)}")
    shape = pose.shape[:-1]
    points = torch.nan_to_num(pose.float()).reshape(*shape, 13, 17, 2)
    flat = points.reshape(*shape, 13, 34)
    position_stats = torch.cat(
        (
            flat.mean(dim=-2),
            flat.std(dim=-2, unbiased=False),
            flat.amin(dim=-2),
            flat.amax(dim=-2),
        ),
        dim=-1,
    )
    velocity = points[..., 1:, :, :] - points[..., :-1, :, :]
    velocity_flat = velocity.reshape(*shape, 12, 34)
    velocity_stats = torch.cat(
        (
            velocity_flat.mean(dim=-2),
            velocity_flat.std(dim=-2, unbiased=False),
        ),
        dim=-1,
    )
    root_track = points[..., :, 0, :]
    root_velocity = root_track[..., 1:, :] - root_track[..., :-1, :]
    root_velocity_mean = root_velocity.mean(dim=-2)
    root_velocity_std = root_velocity.std(dim=-2, unbiased=False)
    shoulder = points[..., 6, :] - points[..., 5, :]
    torso_angle = torch.atan2(shoulder[..., 1], shoulder[..., 0])
    angle_delta = torch.atan2(
        torch.sin(torso_angle[..., 1:] - torso_angle[..., :-1]),
        torch.cos(torso_angle[..., 1:] - torso_angle[..., :-1]),
    )
    global_stats = torch.cat(
        (
            root_velocity_mean,
            root_velocity_std,
            torch.sin(torso_angle).mean(dim=-1, keepdim=True),
            torch.cos(torso_angle).mean(dim=-1, keepdim=True),
            angle_delta.mean(dim=-1, keepdim=True),
            angle_delta.std(dim=-1, unbiased=False, keepdim=True),
        ),
        dim=-1,
    )
    result = torch.cat((position_stats, velocity_stats, global_stats), dim=-1)
    if result.shape[-1] != 212:
        raise AssertionError(f"unexpected legacy skeleton dimension: {result.shape}")
    return torch.nan_to_num(result).clamp(-8.0, 8.0)


def direct_body_orientation_features(pose: torch.Tensor) -> torch.Tensor:
    """Return explicit, label-free body-orientation and motion sensors.

    The pose cache is a 13-frame COCO-17 sequence.  This branch exposes the
    quantities that are otherwise hidden inside the skeleton encoder: shoulder
    and hip directions, their relative twist, body scale, root motion, and
    joint motion.  All statistics are over the current observation window;
    no candidate-view tensor is consulted.
    """

    if pose.shape[-1] != POSE_DIM:
        raise ValueError(f"pose must end in {POSE_DIM}, got {tuple(pose.shape)}")
    shape = pose.shape[:-1]
    points = torch.nan_to_num(pose.float()).reshape(*shape, 13, 17, 2)
    shoulder_vec = points[..., 6, :] - points[..., 5, :]
    hip_vec = points[..., 12, :] - points[..., 11, :]
    torso_vec = 0.5 * (points[..., 5, :] + points[..., 6, :]) - 0.5 * (
        points[..., 11, :] + points[..., 12, :]
    )

    def angle_stats(vector: torch.Tensor) -> torch.Tensor:
        angle = torch.atan2(vector[..., 1], vector[..., 0])
        sincos = torch.stack((torch.sin(angle), torch.cos(angle)), dim=-1)
        return torch.cat(
            (
                sincos.mean(dim=-2),
                sincos.std(dim=-2, unbiased=False),
            ),
            dim=-1,
        )

    shoulder_angle = torch.atan2(shoulder_vec[..., 1], shoulder_vec[..., 0])
    hip_angle = torch.atan2(hip_vec[..., 1], hip_vec[..., 0])
    relative_angle = torch.atan2(
        torch.sin(shoulder_angle - hip_angle),
        torch.cos(shoulder_angle - hip_angle),
    )
    relative_stats = torch.stack(
        (
            torch.sin(relative_angle).mean(dim=-1),
            torch.cos(relative_angle).mean(dim=-1),
            torch.sin(relative_angle).std(dim=-1, unbiased=False),
            torch.cos(relative_angle).std(dim=-1, unbiased=False),
        ),
        dim=-1,
    )
    widths = torch.stack(
        (
            torch.linalg.vector_norm(shoulder_vec, dim=-1),
            torch.linalg.vector_norm(hip_vec, dim=-1),
            torch.linalg.vector_norm(torso_vec, dim=-1),
        ),
        dim=-1,
    )
    width_stats = torch.cat(
        (widths.mean(dim=-2), widths.std(dim=-2, unbiased=False)), dim=-1
    )
    root = 0.5 * (points[..., 5, :] + points[..., 6, :])
    root_velocity = root[..., 1:, :] - root[..., :-1, :]
    root_stats = torch.cat(
        (root_velocity.mean(dim=-2), root_velocity.std(dim=-2, unbiased=False)),
        dim=-1,
    )
    joint_velocity = points[..., 1:, :, :] - points[..., :-1, :, :]
    joint_speed = torch.linalg.vector_norm(joint_velocity, dim=-1).mean(dim=-1)
    speed_stats = torch.stack(
        (
            joint_speed.mean(dim=-1),
            joint_speed.std(dim=-1, unbiased=False),
        ),
        dim=-1,
    )
    result = torch.cat(
        (
            angle_stats(shoulder_vec),
            angle_stats(hip_vec),
            angle_stats(torso_vec),
            relative_stats,
            width_stats,
            root_stats,
            speed_stats,
        ),
        dim=-1,
    )
    if result.shape[-1] != BODY_ORIENTATION_DIM:
        raise AssertionError(f"unexpected body sensor dimension: {result.shape}")
    return torch.nan_to_num(result).clamp(-8.0, 8.0)


def direct_camera_pose_features(
    candidate: torch.Tensor,
    source: torch.Tensor,
) -> torch.Tensor:
    """Summarize current camera pose/navigation sensors invariant to slot order."""

    valid = candidate[..., 7].bool()
    cost = candidate[..., 6].float().masked_fill(~valid, 0.0)
    count = valid.sum(-1).clamp_min(1).float()
    cost_mean = cost.sum(-1) / count
    centered = cost - cost_mean[..., None]
    cost_std = (centered.square() * valid.float()).sum(-1).div(count).sqrt()
    masked_cost = cost.masked_fill(~valid, torch.inf)
    min_cost = masked_cost.amin(-1).masked_fill(~valid.any(-1), 0.0)
    max_cost = cost.masked_fill(~valid, -torch.inf).amax(-1).masked_fill(~valid.any(-1), 0.0)
    valid_fraction = valid.float().mean(-1)
    result = torch.cat(
        (
            source.float(),
            cost_mean[..., None],
            cost_std[..., None],
            min_cost[..., None],
            max_cost[..., None],
            valid_fraction[..., None],
        ),
        dim=-1,
    )
    if result.shape[-1] != CAMERA_POSE_DIM:
        raise AssertionError(f"unexpected camera sensor dimension: {result.shape}")
    return torch.nan_to_num(result).clamp(-8.0, 8.0)


def fourier_candidate_features(candidate: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    """Build slot-independent periodic geometry and source-relative terms."""

    angle = candidate[..., :2]
    delta = candidate[..., 2:4]
    angle2 = torch.stack(
        (2.0 * angle[..., 0] * angle[..., 1], angle[..., 1].square() - angle[..., 0].square()),
        dim=-1,
    )
    delta2 = torch.stack(
        (2.0 * delta[..., 0] * delta[..., 1], delta[..., 1].square() - delta[..., 0].square()),
        dim=-1,
    )
    candidate_dir = angle
    relation = torch.stack(
        (
            (candidate_dir * source[:, None, :]).sum(-1),
            candidate_dir[..., 0] * source[:, None, 1]
            - candidate_dir[..., 1] * source[:, None, 0],
            (delta * source[:, None, :]).sum(-1),
            delta[..., 0] * source[:, None, 1]
            - delta[..., 1] * source[:, None, 0],
        ),
        dim=-1,
    )
    # candidate (8) + second harmonics (4) + relative relation (4).
    return torch.cat((candidate, angle2, delta2, relation), dim=-1)


def g18_candidate_features(candidate: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    """Reproduce the preserved g18 candidate encoding from the new cache.

    The historical g18 model used seven candidate fields: direction, the
    source-relative direction, source direction, and reachability.  The
    current SC-NBV cache additionally stores movement cost, so this adapter
    deliberately drops only that field and keeps the remaining six fields in
    their original order.  It is an isolated compatibility path; the default
    encoding used by v1--v5 is unchanged.
    """

    if candidate.shape[-1] != CANDIDATE_DIM:
        raise ValueError(
            f"g18 candidate adapter expects {CANDIDATE_DIM} fields, got "
            f"{candidate.shape[-1]}"
        )
    # Current cache layout: [direction(2), relative(2), source(2), cost, valid].
    legacy = torch.cat((candidate[..., :6], candidate[..., 7:8]), dim=-1)
    angle = legacy[..., :2]
    delta = legacy[..., 2:4]
    angle2 = torch.stack(
        (2.0 * angle[..., 0] * angle[..., 1], angle[..., 1].square() - angle[..., 0].square()),
        dim=-1,
    )
    delta2 = torch.stack(
        (2.0 * delta[..., 0] * delta[..., 1], delta[..., 1].square() - delta[..., 0].square()),
        dim=-1,
    )
    relation = torch.stack(
        (
            (angle * source[:, None, :]).sum(-1),
            angle[..., 0] * source[:, None, 1] - angle[..., 1] * source[:, None, 0],
            (delta * source[:, None, :]).sum(-1),
            delta[..., 0] * source[:, None, 1] - delta[..., 1] * source[:, None, 0],
        ),
        dim=-1,
    )
    # legacy candidate (7) + two Fourier terms for each direction (4) +
    # source-relative relation (4) = 15, matching g18 exactly.
    return torch.cat((legacy, angle2, delta2, relation), dim=-1)


def g19_candidate_features(
    candidate: torch.Tensor,
    source: torch.Tensor,
    geometry_confidence: torch.Tensor,
) -> torch.Tensor:
    """Add causal per-candidate geometry confidence to the g18 encoding.

    The confidence is produced by the body-relative pose resolver from the
    current map/skeleton geometry.  It is not a future-view visual feature.
    Keeping it outside the historical eight-field candidate tensor makes the
    old cache/checkpoint contracts unchanged while allowing a new variant to
    learn when a bearing estimate is reliable.
    """

    if geometry_confidence.ndim != candidate.ndim - 1:
        raise ValueError(
            "candidate geometry confidence must have shape [batch, views]"
        )
    if geometry_confidence.shape[:2] != candidate.shape[:2]:
        raise ValueError("candidate/confidence shape mismatch")
    legacy = g18_candidate_features(candidate, source)
    return torch.cat((legacy, geometry_confidence.float().clamp(0.0, 1.0)[..., None]), dim=-1)


def g20_candidate_features(
    candidate: torch.Tensor,
    source: torch.Tensor,
    geometry_confidence: torch.Tensor,
) -> torch.Tensor:
    """Add a per-observation bearing rank to the continuous g19 geometry.

    Camera rows intentionally remain in their raw camera-ID order.  This
    scalar is recomputed from each sample's estimated body-relative bearing
    and therefore does not assume that a camera index means left, right, or
    any other fixed direction.  It is useful when the same continuous angles
    are noisy but their within-sample ordering is reliable.
    """

    if candidate.shape[-1] != CANDIDATE_DIM:
        raise ValueError(
            f"g20 candidate encoding expects {CANDIDATE_DIM} fields, got "
            f"{candidate.shape[-1]}"
        )
    legacy = g19_candidate_features(candidate, source, geometry_confidence)
    bearing = torch.atan2(candidate[..., 0], candidate[..., 1])
    valid = candidate[..., 7] > 0.5
    # For each candidate i, count valid candidates j with a smaller signed
    # bearing.  The current slot is already invalidated in candidate[...,7],
    # so this is a rank among legal move candidates for that state.
    lower = (
        valid[:, None, :]
        & (bearing[:, None, :] < bearing[:, :, None])
    )
    rank = lower.sum(dim=-1).float()
    denominator = (valid.sum(dim=-1, keepdim=True).float() - 1.0).clamp_min(1.0)
    rank = (rank / denominator).clamp(0.0, 1.0)
    return torch.cat((legacy, rank[..., None]), dim=-1)


def g21_candidate_features(
    candidate: torch.Tensor,
    source: torch.Tensor,
    geometry_confidence: torch.Tensor,
    candidate_slot: torch.Tensor,
) -> torch.Tensor:
    """Add camera identity as calibration metadata, never as a direction.

    ``candidate_slot`` identifies the physical camera channel (the original
    C001/C003/C005/C007 row), while g20 still carries the per-sample
    body-relative bearing and rank.  The two notions are deliberately kept
    separate: a slot one-hot cannot silently turn into a global left/right
    convention.  Candidate permutation augmentation moves this field with
    the candidate, preserving equivariance.
    """

    if candidate_slot.ndim != candidate.ndim - 1:
        raise ValueError("candidate_slot must have shape [batch, views]")
    if candidate_slot.shape[:2] != candidate.shape[:2]:
        raise ValueError("candidate/slot shape mismatch")
    geometry = g20_candidate_features(candidate, source, geometry_confidence)
    slot = candidate_slot.long()
    if (slot < 0).any() or (slot >= NUM_VIEWS).any():
        raise ValueError("candidate_slot contains an invalid camera slot")
    slot_one_hot = F.one_hot(slot, num_classes=NUM_VIEWS).float()
    return torch.cat((geometry, slot_one_hot), dim=-1)


def g22_candidate_features(
    candidate: torch.Tensor,
    source: torch.Tensor,
    geometry_confidence: torch.Tensor,
) -> torch.Tensor:
    """Add direction-set geometry without using a camera-ID convention.

    The raw candidate rows are intentionally kept in their original camera
    channel order.  A row's meaning is supplied by its estimated
    body-relative bearing, not by its integer slot.  In addition to the
    continuous g20 features, this encoding exposes the signed angle and
    angular distance to the neighbouring legal candidates.  The latter is a
    set property and makes the scorer aware that a ``middle`` candidate in
    one recording can be an extreme candidate in another recording.

    All operations are shared over candidates and equivariant to a row
    permutation.  No future-view feature or class ID is introduced.
    """

    if candidate.shape[-1] != CANDIDATE_DIM:
        raise ValueError(
            f"g22 candidate encoding expects {CANDIDATE_DIM} fields, got "
            f"{candidate.shape[-1]}"
        )
    geometry = g20_candidate_features(candidate, source, geometry_confidence)
    angle = torch.atan2(candidate[..., 0], candidate[..., 1])
    relative = torch.atan2(candidate[..., 2], candidate[..., 3])
    valid = candidate[..., 7] > 0.5
    valid_float = valid.float()
    count = valid_float.sum(dim=-1, keepdim=True).clamp_min(1.0)

    # Signed scalar angles complement sin/cos near the directions where a
    # small MLP otherwise has to learn an inverse atan2 boundary.
    signed = torch.stack(
        (angle / math.pi, relative / math.pi,
         angle.abs() / math.pi, relative.abs() / math.pi),
        dim=-1,
    )

    # Mean direction of the currently legal candidate set.  Dot/cross
    # products are invariant to candidate row order and describe whether a
    # candidate is central or peripheral within this particular layout.
    set_vector = (candidate[..., :2] * valid_float[..., None]).sum(dim=-2) / count
    set_vector = F.normalize(set_vector, dim=-1)
    set_relation = torch.stack(
        (
            (candidate[..., :2] * set_vector[:, None, :]).sum(-1),
            candidate[..., 0] * set_vector[:, None, 1]
            - candidate[..., 1] * set_vector[:, None, 0],
        ),
        dim=-1,
    )

    # For every valid candidate, find its clockwise and counter-clockwise
    # neighbour gaps in the *sample-specific* signed-bearing layout.  Invalid
    # and padded rows receive zeros.  The circular differences are in
    # radians and are normalized by pi, so the representation is bounded.
    angle_i = angle[:, :, None]
    angle_j = angle[:, None, :]
    signed_delta = torch.atan2(
        torch.sin(angle_j - angle_i), torch.cos(angle_j - angle_i)
    )
    positive = signed_delta > 1e-6
    negative = signed_delta < -1e-6
    positive_gap = signed_delta.masked_fill(~(positive & valid[:, None, :]), torch.inf)
    negative_gap = (-signed_delta).masked_fill(~(negative & valid[:, None, :]), torch.inf)
    right_gap = positive_gap.amin(dim=-1)
    left_gap = negative_gap.amin(dim=-1)
    right_gap = right_gap.masked_fill(~torch.isfinite(right_gap), 0.0) / math.pi
    left_gap = left_gap.masked_fill(~torch.isfinite(left_gap), 0.0) / math.pi
    nearest_gap = torch.minimum(left_gap, right_gap)
    neighbour = torch.stack((left_gap, right_gap, nearest_gap), dim=-1)
    extra = torch.cat((signed, set_relation, neighbour), dim=-1)
    extra = extra * valid_float[..., None]
    return torch.cat((geometry, extra), dim=-1)


def g23_candidate_features(
    candidate: torch.Tensor,
    source: torch.Tensor,
    geometry_confidence: torch.Tensor,
) -> torch.Tensor:
    """Encode the candidate set in a frame relative to the current view.

    Camera IDs and the C001 reference are useful for lookup, but neither is
    a transferable direction label.  ``g23`` therefore drops the absolute
    candidate/source bearing and keeps only the candidate-to-current bearing,
    its within-state rank, set relation, neighbour gaps, movement cost, and
    geometry reliability.  Every operation is shared over candidate rows, so
    the representation is both row-permutation equivariant and invariant to
    a common rotation of the recording's camera frame.

    The raw cache is still kept in physical C001/C003/C005/C007 order.  This
    function changes only what the selector sees; action lookup remains tied
    to the physical camera slot.
    """

    if candidate.shape[-1] != CANDIDATE_DIM:
        raise ValueError(
            f"g23 candidate encoding expects {CANDIDATE_DIM} fields, got "
            f"{candidate.shape[-1]}"
        )
    if geometry_confidence.ndim != candidate.ndim - 1:
        raise ValueError("candidate geometry confidence must have shape [batch, views]")
    if geometry_confidence.shape[:2] != candidate.shape[:2]:
        raise ValueError("candidate/confidence shape mismatch")

    relative = candidate[..., 2:4]
    valid = candidate[..., 7] > 0.5
    valid_float = valid.float()
    count = valid_float.sum(dim=-1, keepdim=True).clamp_min(1.0)
    bearing = torch.atan2(relative[..., 0], relative[..., 1])

    # Signed relative bearing and its second Fourier harmonic avoid asking a
    # small MLP to learn either the circular boundary or an inverse atan2.
    signed = bearing / math.pi
    direction = torch.cat(
        (
            relative,
            candidate[..., 6:7].float(),
            candidate[..., 7:8].float(),
            signed[..., None],
            signed.abs()[..., None],
            torch.stack(
                (
                    2.0 * relative[..., 0] * relative[..., 1],
                    relative[..., 1].square() - relative[..., 0].square(),
                ),
                dim=-1,
            ),
        ),
        dim=-1,
    )

    # Rank is recomputed from the sample-specific relative bearings.  The
    # current row is invalid and therefore cannot affect the legal ordering.
    lower = valid[:, None, :] & (bearing[:, None, :] < bearing[:, :, None])
    rank = lower.sum(dim=-1).float()
    denominator = (valid.sum(dim=-1, keepdim=True).float() - 1.0).clamp_min(1.0)
    rank = (rank / denominator).clamp(0.0, 1.0)

    set_vector = (relative * valid_float[..., None]).sum(dim=-2) / count
    set_vector = F.normalize(set_vector, dim=-1)
    set_relation = torch.stack(
        (
            (relative * set_vector[:, None, :]).sum(-1),
            relative[..., 0] * set_vector[:, None, 1]
            - relative[..., 1] * set_vector[:, None, 0],
        ),
        dim=-1,
    )

    # The closest legal neighbour on either side describes the irregular
    # spacing of this particular recording without assigning meaning to a
    # camera number.
    angle_i = bearing[:, :, None]
    angle_j = bearing[:, None, :]
    signed_delta = torch.atan2(
        torch.sin(angle_j - angle_i), torch.cos(angle_j - angle_i)
    )
    positive = signed_delta > 1e-6
    negative = signed_delta < -1e-6
    positive_gap = signed_delta.masked_fill(
        ~(positive & valid[:, None, :]), torch.inf
    )
    negative_gap = (-signed_delta).masked_fill(
        ~(negative & valid[:, None, :]), torch.inf
    )
    right_gap = positive_gap.amin(dim=-1)
    left_gap = negative_gap.amin(dim=-1)
    right_gap = right_gap.masked_fill(~torch.isfinite(right_gap), 0.0) / math.pi
    left_gap = left_gap.masked_fill(~torch.isfinite(left_gap), 0.0) / math.pi
    nearest_gap = torch.minimum(left_gap, right_gap)
    neighbour = torch.stack((left_gap, right_gap, nearest_gap), dim=-1)

    # Centering cost makes the relative movement preference available while
    # removing a nuisance recording-wide scale/offset.  The uncentered cost
    # remains in ``direction`` for compatibility with the navigation metric.
    cost = candidate[..., 6].float() * valid_float
    cost_mean = cost.sum(dim=-1, keepdim=True) / count
    centered = cost - cost_mean
    centered_cost = centered[..., None]
    cost_std = (centered.square() * valid_float).sum(dim=-1, keepdim=True).div(count).sqrt()
    cost_std = cost_std[:, None, :].expand(-1, candidate.shape[1], -1)
    confidence = geometry_confidence.float().clamp(0.0, 1.0)[..., None]
    valid_fraction = (valid_float.mean(dim=-1, keepdim=True))[:, None, :].expand(
        -1, candidate.shape[1], -1
    )

    features = torch.cat(
        (
            direction,
            rank[..., None],
            set_relation,
            neighbour,
            centered_cost,
            cost_std,
            confidence,
            valid_fraction,
        ),
        dim=-1,
    )
    # Invalid/padded rows cannot be selected, but zeroing derived quantities
    # keeps their representation well behaved under candidate permutations.
    features = features * valid_float[..., None]
    # Retain the explicit validity bit after masking so the scorer can still
    # distinguish an invalid row from a valid zero-valued candidate.
    features[..., 3] = candidate[..., 7].float()
    if features.shape[-1] != 18:
        raise AssertionError(f"unexpected g23 feature dimension: {features.shape}")
    return features


def g24_candidate_features(
    candidate: torch.Tensor,
    source: torch.Tensor,
    geometry_confidence: torch.Tensor,
    candidate_slot: torch.Tensor,
) -> torch.Tensor:
    """Add physical-camera calibration metadata to the g23 representation.

    The one-hot field identifies the hardware channel only; it is explicitly
    not interpreted as a left/right or angular label.  It can capture a
    repeatable camera-specific appearance/calibration bias while the actual
    direction remains the sample-specific relative geometry in g23.
    """

    if candidate_slot.ndim != candidate.ndim - 1:
        raise ValueError("candidate_slot must have shape [batch, views]")
    if candidate_slot.shape[:2] != candidate.shape[:2]:
        raise ValueError("candidate/slot shape mismatch")
    slot = candidate_slot.long()
    if (slot < 0).any() or (slot >= NUM_VIEWS).any():
        raise ValueError("candidate_slot contains an invalid camera slot")
    geometry = g23_candidate_features(candidate, source, geometry_confidence)
    slot_one_hot = F.one_hot(slot, num_classes=NUM_VIEWS).float()
    result = torch.cat((geometry, slot_one_hot), dim=-1)
    if result.shape[-1] != 22:
        raise AssertionError(f"unexpected g24 feature dimension: {result.shape}")
    return result


def g25_candidate_features(
    candidate: torch.Tensor,
    source: torch.Tensor,
    geometry_confidence: torch.Tensor,
) -> torch.Tensor:
    """Add an explicit per-recording angular-rank encoding to g23.

    ``g23`` already recomputes the signed-bearing rank from the candidate
    geometry.  This one-hot field lets a small selector distinguish the
    first/second/third legal angular position without learning an ordinal
    threshold from one scalar.  The rank is recomputed independently for
    every recording; it is never the C001/C003/C005/C007 hardware index.
    Invalid/padded rows remain all-zero, and every operation is shared over
    rows so candidate-row permutation equivariance is preserved.
    """

    geometry = g23_candidate_features(candidate, source, geometry_confidence)
    if geometry.shape[-1] != 18:
        raise AssertionError(f"unexpected g23 feature dimension: {geometry.shape}")
    valid = candidate[..., 7] > 0.5
    bearing = torch.atan2(candidate[..., 2], candidate[..., 3])
    lower = valid[:, None, :] & (bearing[:, None, :] < bearing[:, :, None])
    rank = lower.sum(dim=-1).clamp(min=0, max=NUM_VIEWS - 1)
    rank_one_hot = F.one_hot(rank.long(), num_classes=NUM_VIEWS).float()
    rank_one_hot = rank_one_hot * valid.float()[..., None]
    result = torch.cat((geometry, rank_one_hot), dim=-1)
    if result.shape[-1] != 22:
        raise AssertionError(f"unexpected g25 feature dimension: {result.shape}")
    return result


def _mlp(in_dim: int, hidden: int, out_dim: int, dropout: float = 0.0) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU()]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    layers.extend((nn.Linear(hidden, out_dim), nn.GELU()))
    return nn.Sequential(*layers)


class SemanticConditionedViewUtilityNet(nn.Module):
    """Causal scalar utility function Q(current evidence, candidate action)."""

    def __init__(
        self,
        width: int = 128,
        hidden: int = 256,
        include_object: bool = False,
        include_evidence: bool = True,
        semantic_dropout: float = 0.15,
        head_dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.include_object = bool(include_object)
        self.include_evidence = bool(include_evidence)
        self.semantic_dropout = float(semantic_dropout)
        self.visual = _mlp(Z_DIM, width, width)
        self.semantic = _mlp(Z_DIM, width, width)
        self.pose = _mlp(POSE_DIM, max(64, width // 2), max(64, width // 2))
        self.object = _mlp(OBJECT_DIM, max(64, width // 2), max(64, width // 2)) if include_object else None
        self.evidence = _mlp(EVIDENCE_DIM, max(32, width // 4), max(32, width // 4)) if include_evidence else None
        self.uncertainty = _mlp(1, max(16, width // 8), max(16, width // 8))
        self.quality = _mlp(4, max(16, width // 8), max(16, width // 8))
        pose_dim = max(64, width // 2)
        object_dim = max(64, width // 2) if include_object else 0
        evidence_dim = max(32, width // 4) if include_evidence else 0
        uncertainty_dim = max(16, width // 8)
        quality_dim = max(16, width // 8)
        context_input = width + width + width + pose_dim + object_dim + evidence_dim + uncertainty_dim + quality_dim
        self.context = nn.Sequential(
            nn.Linear(context_input, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Dropout(head_dropout), nn.Linear(hidden, hidden), nn.GELU(),
        )
        geometry_width = max(32, width // 2)
        self.geometry = _mlp(CANDIDATE_DIM, max(32, width // 2), geometry_width)
        self.context_to_geometry = nn.Linear(hidden, geometry_width, bias=False)
        self.head = nn.Sequential(
            nn.Linear(hidden + geometry_width * 2, max(64, hidden // 2)),
            nn.LayerNorm(max(64, hidden // 2)), nn.GELU(),
            nn.Dropout(head_dropout), nn.Linear(max(64, hidden // 2), 1),
        )

    def forward(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        batch = normalize_model_inputs(batch)
        visual = self.visual(batch["z"])
        semantic = self.semantic(F.dropout(batch["semantic"], p=self.semantic_dropout, training=self.training))
        parts = [visual, semantic, visual * semantic, self.pose(batch["pose"])]
        if self.object is not None:
            parts.append(self.object(batch["object_map"]))
        if self.evidence is not None:
            parts.append(self.evidence(batch["evidence"]))
        parts.append(self.uncertainty(batch["uncertainty"]))
        parts.append(self.quality(batch["quality"]))
        context = self.context(torch.cat(parts, dim=-1))
        candidate = self.geometry(batch["candidate"])
        context_expanded = context[:, None, :].expand(-1, candidate.shape[1], -1)
        interaction = self.context_to_geometry(context)[:, None, :] * candidate
        return self.head(torch.cat((context_expanded, candidate, interaction), dim=-1)).squeeze(-1)


class FactorizedSkeletonUtilityNet(nn.Module):
    """Low-capacity causal selector with an explicit skeleton prior.

    This is a new architecture used by later SC-NBV generations.  It keeps
    the preserved g18 idea (small skeleton encoder + low-rank context/action
    product) but uses the current SC-NBV cache interface.  ``sensor_features``
    can add current, label-free VPOCLIP/object/quality evidence; it never
    includes a future-view tensor.  ``return_aux`` exposes only an auxiliary
    future-logit *prediction* head for the training loss, not for action
    selection.
    """

    def __init__(
        self,
        sensor_features: Sequence[str] = (),
        auxiliary_transition: bool = False,
        context_width: int = 48,
        candidate_width: int = 48,
        candidate_encoding: str = "current",
        normalize_z: bool = True,
        normalize_semantic: bool = True,
        skeleton_mode: str = "canonical",
        direct_sensor_fusion: bool = False,
        sensor_width: int = 16,
        sensor_interaction: str = "none",
        interaction_scale: float = 0.25,
        skeleton_encoder: str = "summary",
        semantic_dropout: float = 0.0,
        geometry_spread_prior: float = 0.0,
        learned_score_scale: float = 1.0,
        include_absolute_direction: bool = True,
        prototype_prior: bool = False,
        prototype_slots: int = TOP_SEMANTIC_PROTOTYPES,
        prototype_prior_scale: float = 0.25,
        transition_policy: bool = False,
        transition_policy_scale: float = 0.25,
    ) -> None:
        super().__init__()
        self.sensor_features = frozenset(str(name) for name in sensor_features)
        self.auxiliary_transition = bool(auxiliary_transition)
        self.context_width = int(context_width)
        self.candidate_width = int(candidate_width)
        self.candidate_encoding = str(candidate_encoding)
        self.normalize_z = bool(normalize_z)
        self.normalize_semantic = bool(normalize_semantic)
        self.skeleton_mode = str(skeleton_mode)
        self.direct_sensor_fusion = bool(direct_sensor_fusion)
        self.sensor_width = int(sensor_width)
        self.sensor_interaction = str(sensor_interaction)
        self.interaction_scale = float(interaction_scale)
        self.skeleton_encoder = str(skeleton_encoder)
        self.semantic_dropout = float(semantic_dropout)
        self.geometry_spread_prior = float(geometry_spread_prior)
        self.learned_score_scale = float(learned_score_scale)
        self.include_absolute_direction = bool(include_absolute_direction)
        self.prototype_prior = bool(prototype_prior)
        self.prototype_slots = int(prototype_slots)
        self.prototype_prior_scale = float(prototype_prior_scale)
        self.transition_policy = bool(transition_policy)
        self.transition_policy_scale = float(transition_policy_scale)
        if self.candidate_encoding not in {"current", "g18", "g19", "g20", "g21", "g22", "g23", "g24", "g25"}:
            raise ValueError(f"unknown candidate encoding: {self.candidate_encoding}")
        if self.skeleton_mode not in {"canonical", "legacy_raw"}:
            raise ValueError(f"unknown skeleton mode: {self.skeleton_mode}")
        if self.sensor_width < 8:
            raise ValueError("sensor_width must be at least 8")
        if not 1 <= self.prototype_slots <= 5:
            raise ValueError("prototype_slots must be between 1 and 5")
        if self.prototype_prior_scale < 0.0:
            raise ValueError("prototype_prior_scale must be non-negative")
        if self.transition_policy_scale < 0.0:
            raise ValueError("transition_policy_scale must be non-negative")
        if self.sensor_interaction not in {"none", "per_modality"}:
            raise ValueError(f"unknown sensor interaction: {self.sensor_interaction}")
        if not 0.0 <= self.semantic_dropout < 1.0:
            raise ValueError("semantic_dropout must be in [0, 1)")
        if self.geometry_spread_prior < 0.0:
            raise ValueError("geometry_spread_prior must be non-negative")
        if self.learned_score_scale <= 0.0:
            raise ValueError("learned_score_scale must be positive")
        if self.skeleton_encoder not in {"summary", "temporal"}:
            raise ValueError(f"unknown skeleton encoder: {self.skeleton_encoder}")
        if self.skeleton_encoder == "temporal":
            self.skeleton = None
            self.temporal_skeleton = nn.Sequential(
                nn.Conv1d(34, 64, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv1d(64, 64, kernel_size=3, padding=1),
                nn.GELU(),
            )
            self.temporal_skeleton_projection = nn.Sequential(
                nn.Linear(128, 48), nn.LayerNorm(48), nn.GELU()
            )
        else:
            self.skeleton = nn.Sequential(
                nn.Linear(212, 48), nn.LayerNorm(48), nn.GELU()
            )
            self.temporal_skeleton = None
            self.temporal_skeleton_projection = None
        self.direction = nn.Sequential(
            nn.Linear(2, 8), nn.LayerNorm(8), nn.GELU()
        )
        context_dim = 48 + (8 if self.include_absolute_direction else 0)
        if self.direct_sensor_fusion:
            self.body_orientation = nn.Sequential(
                nn.Linear(BODY_ORIENTATION_DIM, self.sensor_width),
                nn.LayerNorm(self.sensor_width), nn.GELU(),
            )
            self.camera_pose = nn.Sequential(
                nn.Linear(CAMERA_POSE_DIM, self.sensor_width),
                nn.LayerNorm(self.sensor_width), nn.GELU(),
            )
            context_dim += 2 * self.sensor_width
        else:
            self.body_orientation = None
            self.camera_pose = None
        if "z" in self.sensor_features:
            self.z_sensor = nn.Sequential(
                nn.Linear(Z_DIM, self.sensor_width), nn.LayerNorm(self.sensor_width), nn.GELU()
            )
            context_dim += self.sensor_width
        else:
            self.z_sensor = None
        if "current_logits" in self.sensor_features:
            self.current_logits_sensor = nn.Sequential(
                nn.Linear(FUTURE_LOGIT_DIM, self.sensor_width),
                nn.LayerNorm(self.sensor_width), nn.GELU(),
            )
            context_dim += self.sensor_width
        else:
            self.current_logits_sensor = None
        if "bank_current_top5" in self.sensor_features:
            self.bank_current_top5_sensor = nn.Sequential(
                nn.Linear(5, self.sensor_width),
                nn.LayerNorm(self.sensor_width), nn.GELU(),
            )
            context_dim += self.sensor_width
        else:
            self.bank_current_top5_sensor = None
        if "semantic" in self.sensor_features:
            self.semantic_sensor = nn.Sequential(
                nn.Linear(Z_DIM, self.sensor_width), nn.LayerNorm(self.sensor_width), nn.GELU()
            )
            context_dim += self.sensor_width
        else:
            self.semantic_sensor = None
        if "bank_semantic" in self.sensor_features:
            self.bank_semantic_sensor = nn.Sequential(
                nn.Linear(Z_DIM, self.sensor_width), nn.LayerNorm(self.sensor_width), nn.GELU()
            )
            context_dim += self.sensor_width
        else:
            self.bank_semantic_sensor = None
        if "bank_top_semantic" in self.sensor_features:
            self.bank_top_semantic_sensor = nn.Sequential(
                nn.Linear(self.prototype_slots * Z_DIM, self.sensor_width),
                nn.LayerNorm(self.sensor_width), nn.GELU(),
            )
            context_dim += self.sensor_width
        else:
            self.bank_top_semantic_sensor = None
        if "object" in self.sensor_features:
            self.object_sensor = nn.Sequential(
                nn.Linear(OBJECT_DIM, self.sensor_width), nn.LayerNorm(self.sensor_width), nn.GELU()
            )
            context_dim += self.sensor_width
        else:
            self.object_sensor = None
        if "evidence" in self.sensor_features:
            self.evidence_sensor = nn.Sequential(
                nn.Linear(EVIDENCE_DIM, self.sensor_width), nn.LayerNorm(self.sensor_width), nn.GELU()
            )
            context_dim += self.sensor_width
        else:
            self.evidence_sensor = None
        if "bank_evidence" in self.sensor_features:
            self.bank_evidence_sensor = nn.Sequential(
                nn.Linear(EVIDENCE_DIM, self.sensor_width), nn.LayerNorm(self.sensor_width), nn.GELU()
            )
            context_dim += self.sensor_width
        else:
            self.bank_evidence_sensor = None
        if "bank_top_probability" in self.sensor_features:
            self.bank_top_probability_sensor = nn.Sequential(
                nn.Linear(self.prototype_slots, self.sensor_width),
                nn.LayerNorm(self.sensor_width), nn.GELU(),
            )
            context_dim += self.sensor_width
        else:
            self.bank_top_probability_sensor = None
        if "quality" in self.sensor_features:
            self.quality_sensor = nn.Sequential(
                nn.Linear(4, self.sensor_width), nn.LayerNorm(self.sensor_width), nn.GELU()
            )
            context_dim += self.sensor_width
        else:
            self.quality_sensor = None
        if "uncertainty" in self.sensor_features:
            self.uncertainty_sensor = nn.Sequential(
                nn.Linear(1, self.sensor_width), nn.LayerNorm(self.sensor_width), nn.GELU()
            )
            context_dim += self.sensor_width
        else:
            self.uncertainty_sensor = None

        self.context = nn.Sequential(
            nn.Linear(context_dim, self.context_width),
            nn.LayerNorm(self.context_width),
            nn.GELU(),
        )
        candidate_input_dim = {
            "current": 16,
            "g18": 15,
            "g19": 16,
            "g20": 17,
            "g21": 21,
            "g22": 26,
            "g23": 18,
            "g24": 22,
            "g25": 22,
        }[self.candidate_encoding]
        # The current cache has 8 raw geometry features + 4 second harmonics
        # + 4 source-relative dot/cross relations.  The compatibility path
        # above reduces this to g18's 15 fields.  No camera-slot one-hot is
        # used in either path.
        self.candidate = nn.Sequential(
            nn.Linear(candidate_input_dim, self.candidate_width),
            nn.LayerNorm(self.candidate_width),
            nn.GELU(),
        )
        self.context_to_action = nn.Linear(
            self.context_width, self.candidate_width, bias=False
        )
        self.residual = nn.Sequential(
            nn.Linear(self.context_width + self.candidate_width, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        # Optional explicit candidate-conditioned modality products.  The
        # original factorized path only formed one product after all sensors
        # had been concatenated.  Keeping one product per causal modality
        # preserves which sensor supplied the evidence and remains a compact,
        # permutation-equivariant scorer over candidate slots.
        interaction_dims: dict[str, int] = {"skeleton": 48}
        if self.include_absolute_direction:
            interaction_dims["direction"] = 8
        if self.body_orientation is not None:
            interaction_dims.update({
                "body_orientation": self.sensor_width,
                "camera_pose": self.sensor_width,
            })
        for name in (
            "z", "semantic", "bank_semantic", "bank_top_semantic", "object",
            "evidence", "bank_evidence", "bank_top_probability", "current_logits",
            "bank_current_top5",
            "quality", "uncertainty",
        ):
            if name in self.sensor_features:
                interaction_dims[name] = self.sensor_width
        self.interaction_names = tuple(interaction_dims)
        if self.sensor_interaction == "per_modality":
            self.modality_to_candidate = nn.ModuleDict({
                name: nn.Linear(dim, self.candidate_width, bias=False)
                for name, dim in interaction_dims.items()
            })
            self.modality_mix = nn.Linear(len(self.interaction_names), 1, bias=False)
            nn.init.constant_(self.modality_mix.weight, 1.0 / max(len(self.interaction_names), 1))
        else:
            self.modality_to_candidate = None
            self.modality_mix = None
        if self.prototype_prior:
            # Encode each retained text prototype separately, then average
            # its candidate score with the current semantic posterior.  This
            # preserves class-conditioned information that a single pooled
            # semantic vector can erase, while staying equivariant over the
            # candidate-slot permutation.
            self.prototype_prior_encoder = nn.Sequential(
                nn.Linear(Z_DIM, self.candidate_width),
                nn.LayerNorm(self.candidate_width),
                nn.GELU(),
            )
        else:
            self.prototype_prior_encoder = None
        if self.auxiliary_transition or self.transition_policy:
            self.transition = nn.Sequential(
                nn.Linear(self.context_width + self.candidate_width, 64),
                nn.GELU(),
                nn.Linear(64, FUTURE_LOGIT_DIM),
            )
        else:
            self.transition = None

    def forward(
        self,
        batch: Mapping[str, torch.Tensor],
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        batch = normalize_model_inputs(
            batch,
            normalize_z=self.normalize_z,
            normalize_semantic=self.normalize_semantic,
        )
        if self.skeleton_encoder == "temporal":
            skeleton_sequence = temporal_skeleton_sequence(batch["pose"], self.skeleton_mode)
            temporal = self.temporal_skeleton(skeleton_sequence.transpose(-1, -2))
            temporal_summary = torch.cat((temporal.mean(dim=-1), temporal.amax(dim=-1)), dim=-1)
            skeleton = self.temporal_skeleton_projection(temporal_summary)
        elif self.skeleton_mode == "legacy_raw":
            skeleton_input = legacy_raw_skeleton_features(batch["pose"])
            skeleton = self.skeleton(skeleton_input)
        else:
            skeleton_input = canonical_skeleton_features(batch["pose"])
            skeleton = self.skeleton(skeleton_input)
        source = batch["candidate"][:, 0, 4:6]
        direction = self.direction(source)
        named_parts: list[tuple[str, torch.Tensor]] = [("skeleton", skeleton)]
        if self.include_absolute_direction:
            named_parts.append(("direction", direction))
        if self.body_orientation is not None:
            named_parts.append((
                "body_orientation",
                self.body_orientation(direct_body_orientation_features(batch["pose"])),
            ))
            named_parts.append((
                "camera_pose",
                self.camera_pose(direct_camera_pose_features(batch["candidate"], source)),
            ))
        if self.z_sensor is not None:
            named_parts.append(("z", self.z_sensor(batch["z"])))
        if self.current_logits_sensor is not None:
            if "current_logits" not in batch:
                raise KeyError("current_logits is required by this model variant")
            named_parts.append((
                "current_logits",
                self.current_logits_sensor(batch["current_logits"]),
            ))
        if self.bank_current_top5_sensor is not None:
            if "bank_current_top5" not in batch:
                raise KeyError("bank_current_top5 is required by this model variant")
            named_parts.append((
                "bank_current_top5",
                self.bank_current_top5_sensor(batch["bank_current_top5"]),
            ))
        if self.semantic_sensor is not None:
            semantic_input = F.dropout(
                batch["semantic"], p=self.semantic_dropout, training=self.training
            )
            named_parts.append(("semantic", self.semantic_sensor(semantic_input)))
        if self.bank_semantic_sensor is not None:
            if "bank_semantic" not in batch:
                raise KeyError("bank_semantic is required by this model variant")
            bank_semantic_input = F.dropout(
                batch["bank_semantic"], p=self.semantic_dropout, training=self.training
            )
            named_parts.append(("bank_semantic", self.bank_semantic_sensor(bank_semantic_input)))
        if self.bank_top_semantic_sensor is not None:
            if "bank_top_semantic" not in batch:
                raise KeyError("bank_top_semantic is required by this model variant")
            bank_top_semantic_input = F.dropout(
                batch["bank_top_semantic"], p=self.semantic_dropout, training=self.training
            )
            named_parts.append((
                "bank_top_semantic",
                self.bank_top_semantic_sensor(bank_top_semantic_input),
            ))
        if self.object_sensor is not None:
            named_parts.append(("object", self.object_sensor(batch["object_map"])))
        if self.evidence_sensor is not None:
            named_parts.append(("evidence", self.evidence_sensor(batch["evidence"])))
        if self.bank_evidence_sensor is not None:
            if "bank_evidence" not in batch:
                raise KeyError("bank_evidence is required by this model variant")
            named_parts.append(("bank_evidence", self.bank_evidence_sensor(batch["bank_evidence"])))
        if self.bank_top_probability_sensor is not None:
            if "bank_top_probability" not in batch:
                raise KeyError("bank_top_probability is required by this model variant")
            named_parts.append((
                "bank_top_probability",
                self.bank_top_probability_sensor(batch["bank_top_probability"]),
            ))
        if self.quality_sensor is not None:
            named_parts.append(("quality", self.quality_sensor(batch["quality"])))
        if self.uncertainty_sensor is not None:
            named_parts.append(("uncertainty", self.uncertainty_sensor(batch["uncertainty"])))
        context = self.context(torch.cat([value for _, value in named_parts], dim=-1))
        if self.candidate_encoding == "g18":
            geometry = g18_candidate_features(batch["candidate"], source)
        elif self.candidate_encoding in {"g19", "g20", "g21", "g22", "g23", "g24", "g25"}:
            if "candidate_geometry_confidence" not in batch:
                raise KeyError(
                    f"candidate_geometry_confidence is required by {self.candidate_encoding} "
                    "candidate encoding"
                )
            if self.candidate_encoding == "g19":
                geometry = g19_candidate_features(
                    batch["candidate"],
                    source,
                    batch["candidate_geometry_confidence"],
                )
            elif self.candidate_encoding == "g20":
                geometry = g20_candidate_features(
                    batch["candidate"],
                    source,
                    batch["candidate_geometry_confidence"],
                )
            elif self.candidate_encoding == "g21":
                if "candidate_slot" not in batch:
                    raise KeyError("candidate_slot is required by g21 candidate encoding")
                geometry = g21_candidate_features(
                    batch["candidate"],
                    source,
                    batch["candidate_geometry_confidence"],
                    batch["candidate_slot"],
                )
            elif self.candidate_encoding == "g23":
                geometry = g23_candidate_features(
                    batch["candidate"],
                    source,
                    batch["candidate_geometry_confidence"],
                )
            elif self.candidate_encoding == "g24":
                if "candidate_slot" not in batch:
                    raise KeyError("candidate_slot is required by g24 candidate encoding")
                geometry = g24_candidate_features(
                    batch["candidate"],
                    source,
                    batch["candidate_geometry_confidence"],
                    batch["candidate_slot"],
                )
            elif self.candidate_encoding == "g25":
                geometry = g25_candidate_features(
                    batch["candidate"],
                    source,
                    batch["candidate_geometry_confidence"],
                )
            else:
                geometry = g22_candidate_features(
                    batch["candidate"],
                    source,
                    batch["candidate_geometry_confidence"],
                )
        else:
            geometry = fourier_candidate_features(batch["candidate"], source)
        candidate = self.candidate(geometry)
        context_expanded = context[:, None, :].expand(-1, candidate.shape[1], -1)
        latent = self.context_to_action(context)[:, None, :] * candidate
        learned_q = latent.sum(-1) / math.sqrt(float(self.candidate_width))
        learned_q = learned_q + 0.20 * self.residual(
            torch.cat((context_expanded, candidate), dim=-1)
        ).squeeze(-1)
        if self.modality_mix is not None:
            modality_scores = []
            for name, value in named_parts:
                projected = self.modality_to_candidate[name](value)
                modality_scores.append(
                    (projected[:, None, :] * candidate).sum(-1)
                    / math.sqrt(float(self.candidate_width))
                )
            modality_scores = torch.stack(modality_scores, dim=-1)
            learned_q = learned_q + self.interaction_scale * self.modality_mix(modality_scores).squeeze(-1)
        if self.prototype_prior_encoder is not None:
            if "bank_top_semantic" not in batch or "bank_top_probability" not in batch:
                raise KeyError(
                    "prototype_prior requires bank_top_semantic and bank_top_probability"
                )
            top = F.dropout(
                batch["bank_top_semantic"], p=self.semantic_dropout, training=self.training
            )
            expected = self.prototype_slots * Z_DIM
            if top.shape[-1] != expected:
                raise ValueError(
                    f"prototype prior expects {expected} semantic values, got {top.shape[-1]}"
                )
            weights = batch["bank_top_probability"]
            if weights.shape[-1] != self.prototype_slots:
                raise ValueError(
                    f"prototype prior expects {self.prototype_slots} probabilities, "
                    f"got {weights.shape[-1]}"
                )
            top = F.normalize(top.reshape(top.shape[0], self.prototype_slots, Z_DIM), dim=-1)
            weights = weights.clamp_min(0.0)
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            prototype = self.prototype_prior_encoder(top)
            prototype_scores = torch.einsum(
                "bkw,bvw->bvk", prototype, candidate
            ) / math.sqrt(float(self.candidate_width))
            learned_q = learned_q + self.prototype_prior_scale * (
                prototype_scores * weights[:, None, :]
            ).sum(-1)
        transition_prediction = None
        if self.transition is not None:
            joint = torch.cat((context_expanded, candidate), dim=-1)
            transition_prediction = self.transition(joint)
        if self.transition_policy:
            if transition_prediction is None:
                raise RuntimeError("transition_policy requires a transition head")
            if "current_logits" not in batch:
                raise KeyError("current_logits is required by transition_policy")
            # The train-only auxiliary target is (future-current)/10.  Convert
            # the causal prediction back to recognizer-logit units and score
            # the predicted fused state by its top-class margin.  No target
            # label is consulted at inference time.
            predicted_future = (
                batch["current_logits"][:, None, :]
                + 10.0 * transition_prediction
            )
            predicted_fused = 0.5 * (
                batch["current_logits"][:, None, :] + predicted_future
            )
            top_two = predicted_fused.topk(2, dim=-1).values
            predicted_margin = (top_two[..., 0] - top_two[..., 1]) / 10.0
            learned_q = learned_q + self.transition_policy_scale * predicted_margin
        # Keep the learned correction explicitly bounded relative to the
        # geometry anchor when requested.  The default scale is one, so all
        # historical checkpoints and configurations retain their original
        # computation.  This is still one deterministic Q function: there is
        # no policy/random gate or fallback branch.
        q = self.learned_score_scale * learned_q
        if self.geometry_spread_prior > 0.0:
            # The movement cost is the normalized absolute angular
            # separation between the current and candidate cameras.  It is
            # computed from the per-recording geometry table, never from a
            # camera-ID ordering convention.
            valid = batch["candidate"][..., 7].float()
            cost = batch["candidate"][..., 6].float() * valid
            count = valid.sum(-1, keepdim=True).clamp_min(1.0)
            mean_cost = cost.sum(-1, keepdim=True) / count
            q = q + self.geometry_spread_prior * (cost - mean_cost) * valid
        if not return_aux:
            return q
        if transition_prediction is None:
            raise RuntimeError("return_aux requested without a transition head")
        return q, transition_prediction


def make_selector_model(config: Mapping[str, Any]) -> nn.Module:
    """Construct a selector while retaining the original SC-NBV model path."""

    architecture = str(config.get("architecture", "semantic_mlp"))
    if architecture == "factorized_skeleton":
        return FactorizedSkeletonUtilityNet(
            sensor_features=config.get("sensor_features", ()),
            auxiliary_transition=bool(config.get("auxiliary_transition", False)),
            context_width=int(config.get("context_width", 48)),
            candidate_width=int(config.get("candidate_width", 48)),
            candidate_encoding=str(config.get("candidate_encoding", "current")),
            normalize_z=bool(config.get("normalize_z", True)),
            normalize_semantic=bool(config.get("normalize_semantic", True)),
            skeleton_mode=str(config.get("skeleton_mode", "canonical")),
            direct_sensor_fusion=bool(config.get("direct_sensor_fusion", False)),
            sensor_width=int(config.get("sensor_width", 16)),
            sensor_interaction=str(config.get("sensor_interaction", "none")),
            interaction_scale=float(config.get("interaction_scale", 0.25)),
            skeleton_encoder=str(config.get("skeleton_encoder", "summary")),
            semantic_dropout=float(config.get("semantic_dropout", 0.0)),
            geometry_spread_prior=float(config.get("geometry_spread_prior", 0.0)),
            learned_score_scale=float(config.get("learned_score_scale", 1.0)),
            include_absolute_direction=bool(config.get("include_absolute_direction", True)),
            prototype_prior=bool(config.get("prototype_prior", False)),
            prototype_slots=int(config.get("prototype_slots", TOP_SEMANTIC_PROTOTYPES)),
            prototype_prior_scale=float(config.get("prototype_prior_scale", 0.25)),
            transition_policy=bool(config.get("transition_policy", False)),
            transition_policy_scale=float(config.get("transition_policy_scale", 0.25)),
        )
    return SemanticConditionedViewUtilityNet(
        width=int(config["width"]),
        hidden=int(config["hidden"]),
        include_object=bool(config.get("include_object", False)),
        include_evidence=bool(config.get("include_evidence", True)),
        semantic_dropout=float(config.get("semantic_dropout", 0.15)),
        head_dropout=float(config.get("head_dropout", 0.10)),
    )


def permute_candidates(
    batch: Mapping[str, torch.Tensor],
    generator: torch.Generator,
) -> dict[str, torch.Tensor]:
    """Candidate-slot augmentation; labels and masks stay aligned."""

    n = batch["candidate"].shape[0]
    order = torch.rand((n, NUM_VIEWS), device=batch["candidate"].device, generator=generator).argsort(-1)
    out = dict(batch)
    for name in (
        "candidate",
        "candidate_geometry_confidence",
        "valid",
        "utility",
        "after_margin",
        "future_logits",
        "candidate_slot",
    ):
        if name not in batch:
            continue
        value = batch[name]
        out[name] = value.gather(1, order[..., None].expand(-1, -1, value.shape[-1])) if value.ndim == 3 else value.gather(1, order)
    return out


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (values * weights).sum() / weights.sum().clamp_min(1e-6)


def pairwise_ranking_loss(
    prediction: torch.Tensor,
    utility: torch.Tensor,
    valid: torch.Tensor,
    state_weight: torch.Tensor,
    tie_threshold: float = 0.02,
    base_margin: float = 0.05,
    utility_margin_scale: float = 0.10,
) -> torch.Tensor:
    """Margin ranking over informative candidate pairs only."""

    total = prediction.new_zeros(())
    weight_total = prediction.new_zeros(())
    for left in range(NUM_VIEWS):
        for right in range(left + 1, NUM_VIEWS):
            du = utility[:, left] - utility[:, right]
            pair = valid[:, left] & valid[:, right] & (du.abs() > float(tie_threshold))
            if not pair.any():
                continue
            sign = du.sign()
            margin = float(base_margin) + float(utility_margin_scale) * du.abs().clamp_max(2.0)
            violation = F.relu(margin - sign * (prediction[:, left] - prediction[:, right]))
            weights = state_weight * pair.float()
            total = total + (violation * weights).sum()
            weight_total = weight_total + weights.sum()
    return total / weight_total.clamp_min(1e-6)


def sc_nbv_loss(
    prediction: torch.Tensor,
    utility: torch.Tensor,
    valid: torch.Tensor,
    state_weight: torch.Tensor,
    utility_mean: float,
    utility_std: float,
    rank_lambda: float = 0.5,
    tie_threshold: float = 0.02,
    base_margin: float = 0.05,
    utility_margin_scale: float = 0.10,
    centered_targets: bool = False,
    loss_mode: str = "huber",
    listwise_temperature: float = 0.20,
    oracle_ce_weight: float = 0.0,
    oracle_ce_min_gap: float = 0.05,
) -> tuple[torch.Tensor, dict[str, float]]:
    if loss_mode not in {"huber", "listwise"}:
        raise ValueError(f"unknown SC-NBV loss mode: {loss_mode}")
    if centered_targets:
        count = valid.sum(-1).clamp_min(1).float()
        target_mean = (utility * valid.float()).sum(-1, keepdim=True) / count[:, None]
        target = utility - target_mean
        # The prediction is only used for within-state argmax, so the
        # corresponding per-state centering removes a nuisance absolute-scale
        # target and matches the factorized g18 objective.
        pred_mean = (prediction * valid.float()).sum(-1, keepdim=True) / count[:, None]
        prediction_for_loss = prediction - pred_mean
    else:
        target = (utility - float(utility_mean)) / max(float(utility_std), 1e-4)
        prediction_for_loss = prediction
    valid_weight = valid.float() * state_weight[:, None]
    if loss_mode == "listwise":
        # A soft target distributes probability over genuinely good views;
        # this avoids inventing a hard label when two viewpoints are nearly
        # tied, while still training every valid candidate in the state.
        temperature = max(float(listwise_temperature), 1e-4)
        target_probability = F.softmax(
            utility.masked_fill(~valid, -torch.inf) / temperature, dim=-1
        )
        log_probability = F.log_softmax(
            prediction.masked_fill(~valid, -torch.inf), dim=-1
        )
        per_state = -(
            target_probability * torch.where(valid, log_probability, torch.zeros_like(log_probability))
        ).sum(-1)
        huber = _weighted_mean(per_state, state_weight)
    else:
        error = F.smooth_l1_loss(prediction_for_loss, target, reduction="none")
        huber = _weighted_mean(error, valid_weight)
    ranking = pairwise_ranking_loss(
        prediction, target, valid,
        state_weight, tie_threshold=tie_threshold, base_margin=base_margin,
        utility_margin_scale=utility_margin_scale,
    )
    oracle_ce = prediction.new_zeros(())
    if float(oracle_ce_weight) > 0.0:
        # Use a hard oracle action only when the best candidate is separated
        # from the runner-up by a real utility gap.  Near ties continue to be
        # handled by the soft/listwise target and pairwise loss.
        valid_count = valid.sum(-1)
        ranked_utility = utility.masked_fill(~valid, -torch.inf).topk(
            min(2, utility.shape[-1]), dim=-1
        ).values
        if utility.shape[-1] == 1:
            gap = torch.full_like(valid_count.float(), float("inf"))
        else:
            gap = ranked_utility[:, 0] - ranked_utility[:, 1]
        confident = (valid_count > 1) & (gap > float(oracle_ce_min_gap))
        if confident.any():
            oracle_action = masked_argmax(utility, valid)
            per_state = F.cross_entropy(
                prediction[confident], oracle_action[confident], reduction="none"
            )
            oracle_ce = _weighted_mean(per_state, state_weight[confident])
    loss = huber + float(rank_lambda) * ranking + float(oracle_ce_weight) * oracle_ce
    return loss, {
        "huber": float(huber.detach()),
        "ranking": float(ranking.detach()),
        "listwise": float(huber.detach()) if loss_mode == "listwise" else 0.0,
        "oracle_ce": float(oracle_ce.detach()),
    }


def masked_argmax(scores: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    return scores.masked_fill(~valid, -torch.inf).argmax(-1)


def _classification_metrics(
    bank_scores: torch.Tensor,
    labels: torch.Tensor,
    bank: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # ``evaluate_policy`` has already restricted logits to ``bank`` before
    # calling this helper.  Indexing again with global class IDs would use,
    # for example, class id 51 on a ten-column tensor and cause a CUDA assert.
    scores = bank_scores
    local_labels = torch.searchsorted(bank, labels)
    top1 = scores.argmax(-1).eq(local_labels)
    topk = scores.topk(min(3, scores.shape[-1]), -1).indices.eq(local_labels[:, None]).any(-1)
    return top1, topk, scores


@torch.inference_mode()
def evaluate_policy(
    model: nn.Module,
    contexts: SCNBVData,
    raw: RawViewData,
    class_bank: Sequence[int],
    protocol: str = "fixed0",
    seed: int = 20260909,
    batch_size: int = 2048,
) -> dict[str, Any]:
    """Evaluate fixed/random/farthest/SC-NBV/oracle on one split."""

    device = raw.device
    bank = torch.as_tensor(sorted({int(c) for c in class_bank}), device=device, dtype=torch.long)
    eligible = torch.isin(raw.labels, bank)
    episode_ids = eligible.nonzero().flatten()
    n = int(episode_ids.numel())
    if n == 0:
        raise ValueError("no evaluation episodes in requested class bank")
    valid_np = raw.valid[episode_ids].cpu().numpy()
    rng = np.random.default_rng(seed)
    starts_np = []
    for row in valid_np:
        ids = np.flatnonzero(row)
        if len(ids) == 0:
            raise ValueError("episode has no valid view")
        starts_np.append(int(0 if protocol == "fixed0" and row[0] else (rng.choice(ids) if protocol == "random_start" else ids[0])))
    starts = torch.as_tensor(starts_np, device=device, dtype=torch.long)
    rows = contexts.rows_for(episode_ids, starts)
    context_batch = contexts.batch(rows)
    valid_candidates = raw.valid[episode_ids] & raw.reachable[episode_ids, starts]
    valid_candidates[torch.arange(n, device=device), starts] = False
    choiceable = valid_candidates.sum(-1) > 0
    if not choiceable.all():
        keep = choiceable
        episode_ids, starts, rows, valid_candidates = episode_ids[keep], starts[keep], rows[keep], valid_candidates[keep]
        context_batch = {key: value[keep] for key, value in context_batch.items()}
        n = int(episode_ids.numel())
    model.eval()
    q_parts = []
    for begin in range(0, n, batch_size):
        batch = {key: value[begin:begin + batch_size] for key, value in context_batch.items()}
        q_parts.append(model(causal_batch(batch)))
    policy_q = torch.cat(q_parts, dim=0)
    policy_action = masked_argmax(policy_q, valid_candidates)
    random_action = torch.empty(n, dtype=torch.long, device=device)
    for i, choices in enumerate(valid_candidates.cpu().numpy()):
        random_action[i] = int(rng.choice(np.flatnonzero(choices)))
    fixed_action = torch.where(valid_candidates[:, 1], torch.ones_like(starts), valid_candidates.float().argmax(-1))
    farthest_action = raw.cost[episode_ids, starts].masked_fill(~valid_candidates, -torch.inf).argmax(-1)
    current_logits = raw.logits[episode_ids, starts]
    fused = 0.5 * (current_logits[:, None, :] + raw.logits[episode_ids])
    bank_fused = fused.index_select(-1, bank)
    local_labels = torch.searchsorted(bank, raw.labels[episode_ids])
    before = _margin_torch(current_logits.index_select(-1, bank), local_labels)
    after = _margin_torch(bank_fused, local_labels[:, None].expand(-1, NUM_VIEWS))
    utility = after - before[:, None]
    utility = utility.masked_fill(~valid_candidates, -torch.inf)
    oracle_action = utility.argmax(-1)
    choices = {
        "single": starts,
        "fixed": fixed_action,
        "random2": random_action,
        "farthest": farthest_action,
        "sc_nbv": policy_action,
        "oracle": oracle_action,
    }
    metrics: dict[str, Any] = {}
    for name, action in choices.items():
        if name == "single":
            selected_scores = current_logits.index_select(-1, bank)
            move = torch.zeros(n, device=device)
            selected_utility = torch.zeros(n, device=device)
        else:
            selected_scores = bank_fused[torch.arange(n, device=device), action]
            move = raw.cost[episode_ids, starts, action]
            selected_utility = utility[torch.arange(n, device=device), action]
        top1, top3, scores = _classification_metrics(
            selected_scores if selected_scores.ndim == 2 else selected_scores[:, None, :],
            raw.labels[episode_ids], bank,
        )
        if selected_scores.ndim == 3:
            top1, top3 = top1[:, 0], top3[:, 0]
            scores = scores[:, 0]
        probability = F.softmax(scores, dim=-1)
        entropy = -(probability * probability.clamp_min(1e-8).log()).sum(-1)
        entropy /= math.log(float(len(class_bank)))
        metrics[name] = {
            "top1": float(top1.float().mean()),
            "top3": float(top3.float().mean()),
            "mean_margin_gain": float(selected_utility.mean()),
            "mean_action_entropy": float(entropy.mean()),
            "mean_movement_distance": float(move.mean()),
            "oracle_hit_top1": float(action.eq(oracle_action).float().mean()),
            "oracle_hit_top3": float(_topk_action_hit(action, utility, valid_candidates, 3).float().mean()),
            "mean_oracle_regret": float((utility.max(-1).values - selected_utility).mean()),
        }
    metrics["random_expected"] = {
        "top1": float(
            _candidate_top1_mean(bank_fused, raw.labels[episode_ids], bank, valid_candidates).mean()
        ),
        "mean_margin_gain": float(
            (utility.masked_fill(~valid_candidates, 0.0).sum(-1) / valid_candidates.sum(-1).float()).mean()
        ),
    }
    return {
        "protocol": protocol,
        "episodes": n,
        "candidate_count_mean": float(valid_candidates.sum(-1).float().mean()),
        "class_bank": [int(x) for x in class_bank],
        "metrics": metrics,
        "initial_views": starts.cpu().tolist(),
        "selected": {key: value.cpu().tolist() for key, value in choices.items()},
        "labels": raw.labels[episode_ids].cpu().tolist(),
        "episode_ids": episode_ids.cpu().tolist(),
    }


def _margin_torch(scores: torch.Tensor, local_target: torch.Tensor) -> torch.Tensor:
    if scores.ndim == 2:
        target = scores.gather(-1, local_target[:, None]).squeeze(-1)
        negative = scores.clone()
        negative.scatter_(1, local_target[:, None], -torch.inf)
        return target - negative.max(-1).values
    target = scores.gather(-1, local_target[..., None]).squeeze(-1)
    negative = scores.clone()
    negative.scatter_(-1, local_target[..., None], -torch.inf)
    return target - negative.max(-1).values


def _topk_action_hit(action: torch.Tensor, utility: torch.Tensor, valid: torch.Tensor, k: int) -> torch.Tensor:
    ranked = utility.masked_fill(~valid, -torch.inf).topk(min(k, utility.shape[-1]), -1).indices
    return ranked.eq(action[:, None]).any(-1)


def _candidate_top1_mean(
    bank_fused: torch.Tensor,
    labels: torch.Tensor,
    bank: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    # ``evaluate_policy`` passes logits already restricted to ``bank`` here.
    # Re-indexing with global class IDs would be invalid for a 5/10-way bank.
    scores = bank_fused
    local = torch.searchsorted(bank, labels)
    correct = scores.argmax(-1).eq(local[:, None]) & valid
    return correct.float().sum(-1) / valid.sum(-1).float().clamp_min(1.0)


def validation_score(report: Mapping[str, Any], criterion: str = "top1_gain") -> float:
    """Return a test-independent pseudo-unseen model-selection score.

    Top-1 is retained for compatibility with the first run, but it can be
    saturated when the recognizer is already nearly perfect on the proxy
    classes.  Margin gain and oracle regret remain sensitive to ranking
    quality in that situation.
    """

    m = report["metrics"]
    if criterion == "margin_gain":
        return float(m["sc_nbv"]["mean_margin_gain"] - m["random_expected"]["mean_margin_gain"])
    if criterion == "oracle_regret":
        return float(-m["sc_nbv"]["mean_oracle_regret"])
    if criterion == "oracle_hit":
        return float(m["sc_nbv"]["oracle_hit_top1"])
    if criterion != "top1_gain":
        raise ValueError(f"unknown validation criterion: {criterion}")
    return float(m["sc_nbv"]["top1"] - m["random_expected"]["top1"])
