"""Shared implementation for the two simple weighted-logit experiments.

This file is new and isolated.  It reads the frozen active-view cache and
never changes the historical RL, cache, checkpoint, or evaluation code.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from . import multistep_g18_view_policy_v1 as cache_api


TRUE_UNSEEN = [14, 15, 25, 39, 46]
SEEN_CLASSES = sorted(set(range(55)) - set(TRUE_UNSEEN))
PSEUDO_UNSEEN = [0, 2, 9, 10, 11, 17, 26, 34, 49, 50]
TRAIN_SEEDS = [20260909, 20260910, 20260911]
EVAL_SEEDS = list(range(20260920, 20260950))
NUM_VIEWS = 4
NUM_CLASSES = 55
BOOTSTRAP_SAMPLES = 5000
METHODS = ("equal_mean", "lightweight_gating", "rule_cls", "rule_geo", "rule_combined")


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def load_raw(cache_root: Path, split: str, device: torch.device) -> Any:
    raw = cache_api.GPUCache(cache_root / split, device)
    cache_api.attach_view_valid(raw, cache_root, split)
    # The preserved GPUCache intentionally does not expose metadata.  Keep a
    # private copy in this new experiment only, so manifests remain auditable
    # without modifying the historical cache implementation.
    metadata_path = cache_root / split / "metadata.json"
    raw._weighted_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return raw


def class_bank(classes: Sequence[int], device: torch.device) -> torch.Tensor:
    return torch.as_tensor(sorted(int(x) for x in classes), device=device, dtype=torch.long)


def _wrap_angle(value: torch.Tensor) -> torch.Tensor:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def _orientation_score(geometry: torch.Tensor) -> torch.Tensor:
    """Map sample-specific body-relative camera angle to the requested rule."""

    angle = torch.acos(geometry[..., 1].clamp(-1.0, 1.0)) * 180.0 / math.pi
    score = torch.full_like(angle, 0.2)
    front = angle < 30.0
    middle = (angle >= 30.0) & (angle <= 90.0)
    rear_middle = (angle > 90.0) & (angle <= 150.0)
    score[front] = 0.8 + 0.2 * angle[front] / 30.0
    score[middle] = 1.0
    score[rear_middle] = 1.0 - 0.6 * (angle[rear_middle] - 90.0) / 60.0
    return score.clamp(0.0, 1.0)


def attach_sensor_features(raw: Any) -> dict[str, torch.Tensor]:
    """Build only fields that exist in the frozen cache.

    The cache has 13-frame pose, per-view logits, image_quality and 2-D
    body-relative azimuth.  It has no elevation or per-frame logits, so those
    proposed fields are intentionally not synthesized here.
    """

    if hasattr(raw, "_weighted_sensor_features"):
        return raw._weighted_sensor_features
    points = raw.pose.reshape(raw.num_episodes, NUM_VIEWS, 13, 17, 2).float()
    present = points.norm(dim=-1) > 0.02
    presence_t = present.float().mean(-1)  # [N,V,T]
    presence_mean = presence_t.mean(-1)
    presence_std = presence_t.std(-1, unbiased=False)
    presence_switch = (presence_t[..., 1:] > 0.5).ne(presence_t[..., :-1] > 0.5).float().mean(-1)

    x, y = points[..., 0], points[..., 1]
    edge_distance = torch.minimum(
        torch.minimum(x + 1.0, 1.0 - x), torch.minimum(y + 1.0, 1.0 - y)
    )
    edge_factor = (edge_distance / 0.20).clamp(0.0, 1.0)
    edge_t = (present.float() * edge_factor).mean(-1)
    edge_mean = edge_t.mean(-1)
    edge_std = edge_t.std(-1, unbiased=False)

    probabilities = torch.softmax(raw.logits.float(), dim=-1)
    entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(-1)
    entropy = entropy / math.log(float(NUM_CLASSES))
    top = probabilities.topk(2, dim=-1).values
    margin = (top[..., 0] - top[..., 1]).clamp(0.0, 1.0)
    max_probability = probabilities.amax(-1)
    geometry = raw.geometry[..., :2].float()
    confidence = raw.quality[..., 3].float().clamp(0.0, 1.0)

    # 11 fields per view.  No elevation channels are added because the cache
    # stores only [sin(azimuth), cos(azimuth)] in view_geometry[..., :2].
    base = torch.stack(
        (
            entropy,
            margin,
            max_probability,
            presence_mean,
            presence_std,
            presence_switch,
            edge_mean,
            edge_std,
            geometry[..., 0],
            geometry[..., 1],
            confidence,
        ),
        dim=-1,
    )
    raw._weighted_sensor_features = {
        "base": base,
        "probabilities": probabilities,
        "geometry": geometry,
        "presence_mean": presence_mean,
        "orientation": _orientation_score(geometry),
    }
    return raw._weighted_sensor_features


def path_inputs(
    raw: Any,
    episodes: torch.Tensor,
    paths: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return [N,K,13] gating inputs and selected sensor components."""

    features = attach_sensor_features(raw)
    base = features["base"][episodes[:, None], paths]
    geometry = features["geometry"][episodes[:, None], paths]
    probability = features["probabilities"][episodes[:, None], paths]
    k = paths.shape[1]

    if k > 1:
        dot = (geometry[:, :, None, :] * geometry[:, None, :, :]).sum(-1).clamp(-1.0, 1.0)
        distance = torch.acos(dot)
        eye = torch.eye(k, device=raw.device, dtype=torch.bool)[None]
        diversity = distance.masked_fill(eye, 0.0).sum(-1) / float(k - 1)

        p = probability.clamp_min(1e-8)
        log_p = p.log()
        mean_p = p.mean(1, keepdim=True).clamp_min(1e-8)
        js_each = 0.5 * (p * (log_p - mean_p.log())).sum(-1)
        # Pairwise JS, written without a Python loop.
        p_i = p[:, :, None, :]
        p_j = p[:, None, :, :]
        m = (0.5 * (p_i + p_j)).clamp_min(1e-8)
        js = 0.5 * (
            (p_i * (p_i.log() - m.log())).sum(-1)
            + (p_j * (p_j.log() - m.log())).sum(-1)
        )
        js = js.masked_fill(eye, 0.0).sum(-1) / float(k - 1)
    else:
        diversity = torch.zeros(base.shape[:2], device=raw.device)
        js = torch.zeros(base.shape[:2], device=raw.device)

    inputs = torch.cat((base, diversity[..., None], js[..., None]), dim=-1)
    return inputs, {
        "base": base,
        "geometry": geometry,
        "diversity": diversity,
        "js": js,
        "orientation": _orientation_score(geometry),
    }


def weights_for_method(
    method: str,
    inputs: torch.Tensor,
    components: dict[str, torch.Tensor],
    gate: torch.nn.Module | None,
) -> torch.Tensor:
    if method == "equal_mean":
        return torch.full(
            inputs.shape[:2], 1.0 / float(inputs.shape[1]), device=inputs.device
        )
    if method == "lightweight_gating":
        if gate is None:
            raise ValueError("lightweight_gating requires a trained gate")
        return torch.softmax(gate(inputs).squeeze(-1), dim=-1)

    base = components["base"]
    entropy = base[..., 0].clamp(0.0, 1.0)
    margin = base[..., 1].clamp(0.0, 1.0)
    visibility = base[..., 3].clamp(0.0, 1.0)
    orientation = components["orientation"]
    if method == "rule_cls":
        score = 0.5 * (1.0 - entropy) + 0.5 * margin
    elif method == "rule_geo":
        score = 0.5 * visibility + 0.5 * orientation
    elif method == "rule_combined":
        score = 0.30 * (1.0 - entropy) + 0.30 * margin + 0.20 * visibility + 0.20 * orientation
    else:
        raise ValueError(method)
    denominator = score.sum(-1, keepdim=True)
    uniform = torch.full_like(score, 1.0 / float(score.shape[1]))
    return torch.where(denominator > 1e-8, score / denominator.clamp_min(1e-8), uniform)


@torch.no_grad()
def path_metrics(
    raw: Any,
    episodes: torch.Tensor,
    paths: torch.Tensor,
    bank: torch.Tensor,
    method: str,
    gate: torch.nn.Module | None,
) -> dict[str, torch.Tensor]:
    inputs, components = path_inputs(raw, episodes, paths)
    weights = weights_for_method(method, inputs, components, gate)
    logits = raw.logits[episodes[:, None], paths]
    fused = (weights[..., None] * logits).sum(1)
    scores = fused.index_select(-1, bank)
    local = torch.searchsorted(bank, raw.labels[episodes])
    correct = scores.argmax(-1).eq(local)
    probability = torch.softmax(scores, dim=-1)
    entropy = -(probability * probability.clamp_min(1e-8).log()).sum(-1)
    entropy = entropy / math.log(float(len(bank)))
    if paths.shape[1] > 1:
        cost = sum(raw.cost[episodes, paths[:, i], paths[:, i + 1]] for i in range(paths.shape[1] - 1))
    else:
        cost = torch.zeros(len(episodes), device=raw.device)
    return {
        "correct": correct,
        "entropy": entropy,
        "cost": cost.nan_to_num(1.0),
        "weights": weights,
        "prediction": scores.argmax(-1),
        "paths": paths,
        "episodes": episodes,
    }


def path_candidates(
    raw: Any,
    episodes: torch.Tensor,
    starts: torch.Tensor,
    set_name: str,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build Random-2/3, Preferred-Orthogonal-2/3, or All-valid paths."""

    valid = raw.valid.detach().cpu().numpy().astype(bool)
    reachable = raw.reachable.detach().cpu().numpy().astype(bool)
    geometry = raw.geometry[..., :2].detach().cpu().numpy()
    cost = raw.cost.detach().cpu().numpy()
    k = 2 if set_name.endswith("-2") else 3 if set_name.endswith("-3") else None
    mode = set_name.split("-")[0]
    rng = np.random.default_rng(seed)
    kept: list[int] = []
    paths: list[list[int]] = []
    for local, (episode, start) in enumerate(zip(episodes.detach().cpu().tolist(), starts.detach().cpu().tolist())):
        start = int(start)
        legal = np.flatnonzero(valid[int(episode)] & reachable[int(episode), start]).astype(int)
        legal = legal[legal != start]
        if mode == "All":
            # "All-valid" is intentionally restricted to samples for which
            # all four slots are valid.  This keeps the path rectangular and
            # makes the protocol auditable; partially observed samples remain
            # covered by Random-2/3 and Preferred-Orthogonal-2/3.
            if not valid[int(episode), start] or len(legal) != NUM_VIEWS - 1:
                continue
            path = [start, *legal.tolist()]
        else:
            assert k is not None
            if len(legal) < k - 1:
                continue
            if mode == "Random":
                selected = rng.choice(legal, size=k - 1, replace=False).astype(int).tolist()
                path = [start, *selected]
            elif mode == "Preferred":
                selected: list[int] = []
                last = start
                for _ in range(k - 1):
                    candidates = [int(x) for x in legal if int(x) not in selected]
                    if not candidates:
                        break
                    source = geometry[int(episode), last]
                    candidate_geometry = geometry[int(episode), candidates]
                    separation = np.arccos(np.clip((candidate_geometry * source[None]).sum(-1), -1.0, 1.0))
                    ranking = [
                        (-abs(float(angle) - math.pi / 2.0), -float(cost[int(episode), last, candidate]), -candidate)
                        for angle, candidate in zip(separation, candidates)
                    ]
                    chosen = candidates[max(range(len(candidates)), key=lambda i: ranking[i])]
                    selected.append(chosen)
                    last = chosen
                if len(selected) != k - 1:
                    continue
                path = [start, *selected]
            else:
                raise ValueError(set_name)
        kept.append(int(episode))
        paths.append(path)
    if not kept:
        return (
            torch.empty(0, dtype=torch.long, device=raw.device),
            torch.empty((0, 1), dtype=torch.long, device=raw.device),
        )
    return (
        torch.as_tensor(kept, dtype=torch.long, device=raw.device),
        torch.as_tensor(paths, dtype=torch.long, device=raw.device),
    )


def random_starts(raw: Any, episodes: torch.Tensor, seed: int) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    valid = raw.valid[episodes].detach().cpu().numpy().astype(bool)
    return torch.as_tensor(
        [int(rng.choice(np.flatnonzero(row))) for row in valid],
        dtype=torch.long,
        device=raw.device,
    )


def eligible_episodes(raw: Any, classes: Sequence[int], minimum_views: int = 1) -> torch.Tensor:
    bank = class_bank(classes, raw.device)
    return (
        torch.isin(raw.labels, bank) & (raw.valid.sum(-1) >= int(minimum_views))
    ).nonzero().flatten()


def bootstrap_ci(values: np.ndarray, seed: int, samples: int = BOOTSTRAP_SAMPLES) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return [0.0, 0.0]
    rng = np.random.default_rng(seed)
    estimates: list[np.ndarray] = []
    for start in range(0, samples, 250):
        count = min(250, samples - start)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        estimates.append(values[indices].mean(axis=1))
    combined = np.concatenate(estimates)
    return [float(np.percentile(combined, 2.5)), float(np.percentile(combined, 97.5))]


def summarize_method(
    method_values: np.ndarray,
    mean_values: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    sample_method = np.asarray(method_values, dtype=np.float64).mean(0)
    sample_mean = np.asarray(mean_values, dtype=np.float64).mean(0)
    delta = sample_method - sample_mean
    return {
        "top1": float(sample_method.mean()) if len(sample_method) else 0.0,
        "top1_ci95": bootstrap_ci(sample_method, seed),
        "delta_vs_equal_mean": float(delta.mean()) if len(delta) else 0.0,
        "delta_vs_equal_mean_ci95": bootstrap_ci(delta, seed + 1),
        "episodes": int(len(sample_method)),
    }


def sample_manifest(raw: Any) -> list[dict[str, Any]]:
    metadata = getattr(raw, "_weighted_metadata", {})
    rows = metadata.get("episodes", [])
    result = []
    for index, row in enumerate(rows):
        if isinstance(row, dict):
            result.append({
                "episode_index": index,
                "base_sample": row.get("base_sample", row.get("sample_name", str(index))),
                "label": int(raw.labels[index].detach().cpu()),
                "group": row.get("group"),
            })
        else:
            result.append({"episode_index": index, "base_sample": str(row), "label": int(raw.labels[index].detach().cpu())})
    return result


def evaluate_protocol(
    raw: Any,
    classes: Sequence[int],
    protocol: str,
    gate: torch.nn.Module | None,
    eval_seeds: Sequence[int],
    output_root: Path,
    minimum_views: int = 1,
) -> dict[str, Any]:
    """Evaluate all five fusion methods under one fixed/random-start protocol."""

    methods = list(METHODS)
    base_episodes = eligible_episodes(raw, classes, minimum_views=minimum_views)
    bank = class_bank(classes, raw.device)
    sensor_features = attach_sensor_features(raw)
    results: dict[str, Any] = {}
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = sample_manifest(raw)
    dump(output_root / "sample_manifest.json", manifest)
    for set_name in ("Random-2", "Preferred-Orthogonal-2", "Random-3", "Preferred-Orthogonal-3", "All-valid"):
        per_method: dict[str, list[np.ndarray]] = {method: [] for method in methods}
        per_method_entropy: dict[str, list[np.ndarray]] = {method: [] for method in methods}
        per_method_cost: dict[str, list[np.ndarray]] = {method: [] for method in methods}
        saved_paths: list[np.ndarray] = []
        saved_weights: dict[str, list[np.ndarray]] = {method: [] for method in methods}
        saved_predictions: dict[str, list[np.ndarray]] = {method: [] for method in methods}
        saved_episodes: list[np.ndarray] = []
        saved_entropy: list[np.ndarray] = []
        saved_margin: list[np.ndarray] = []
        saved_visibility: list[np.ndarray] = []
        saved_orientation: list[np.ndarray] = []
        for eval_seed in eval_seeds:
            if protocol.startswith("fixed"):
                start = int(protocol.removeprefix("fixed"))
                start_mask = raw.valid[:, start]
                episodes = base_episodes[start_mask[base_episodes]]
                starts = torch.full_like(episodes, start)
            elif protocol == "random_start":
                episodes = base_episodes
                starts = random_starts(raw, episodes, eval_seed)
            else:
                raise ValueError(protocol)
            path_episodes, paths = path_candidates(raw, episodes, starts, set_name, eval_seed + 37)
            if not len(path_episodes):
                continue
            saved_episodes.append(path_episodes.detach().cpu().numpy())
            saved_paths.append(paths.detach().cpu().numpy())
            selected_base = sensor_features["base"][path_episodes[:, None], paths]
            saved_entropy.append(selected_base[..., 0].detach().cpu().numpy())
            saved_margin.append(selected_base[..., 1].detach().cpu().numpy())
            saved_visibility.append(selected_base[..., 3].detach().cpu().numpy())
            saved_orientation.append(
                sensor_features["orientation"][path_episodes[:, None], paths]
                .detach().cpu().numpy()
            )
            for method in methods:
                metrics = path_metrics(raw, path_episodes, paths, bank, method, gate)
                per_method[method].append(metrics["correct"].detach().cpu().numpy().astype(np.float32))
                per_method_entropy[method].append(metrics["entropy"].detach().cpu().numpy())
                per_method_cost[method].append(metrics["cost"].detach().cpu().numpy())
                saved_weights[method].append(metrics["weights"].detach().cpu().numpy().astype(np.float32))
                saved_predictions[method].append(metrics["prediction"].detach().cpu().numpy())
        if not per_method["equal_mean"]:
            continue
        # Every method sees the same path and episode set for a given seed.
        # Restrict to the common sample intersection before paired bootstrap.
        common = set(saved_episodes[0].tolist())
        for ids in saved_episodes[1:]:
            common.intersection_update(ids.tolist())
        common_ids = np.asarray(sorted(common), dtype=np.int64)
        method_arrays: dict[str, np.ndarray] = {}
        mean_arrays: dict[str, np.ndarray] = {}
        for method in methods:
            rows = []
            means = []
            for seed_index, (ids, values, entropy_values, cost_values) in enumerate(
                zip(saved_episodes, per_method[method], per_method_entropy[method], per_method_cost[method])
            ):
                position = {int(x): i for i, x in enumerate(ids.tolist())}
                keep = np.asarray([position[int(x)] for x in common_ids], dtype=np.int64)
                rows.append(values[keep])
                means.append(values[keep])
            method_arrays[method] = np.stack(rows, axis=0)
            mean_arrays[method] = method_arrays["equal_mean"] if method != "equal_mean" else method_arrays[method]
        summary_methods = {
            method: summarize_method(method_arrays[method], method_arrays["equal_mean"], 20261000 + len(protocol) + len(set_name) + index)
            for index, method in enumerate(methods)
        }
        records_dir = output_root / "details"
        records_dir.mkdir(parents=True, exist_ok=True)
        # Compact NPZ keeps every sample's path, final weights and prediction
        # without producing hundreds of thousands of JSON lines.
        npz_payload: dict[str, Any] = {
            "episode_indices": common_ids,
            "sample_ids": np.asarray([manifest[int(x)]["base_sample"] for x in common_ids]),
            "labels": raw.labels[torch.as_tensor(common_ids, device=raw.device)].detach().cpu().numpy(),
            "eval_seeds": np.asarray(list(eval_seeds), dtype=np.int64),
        }
        # Re-align the saved arrays to the common episode set.
        for method in methods:
            weight_rows = []
            correct_rows = []
            for ids, weights, values in zip(saved_episodes, saved_weights[method], per_method[method]):
                position = {int(x): i for i, x in enumerate(ids.tolist())}
                keep = np.asarray([position[int(x)] for x in common_ids], dtype=np.int64)
                padded = np.zeros((len(keep), NUM_VIEWS), dtype=np.float32)
                padded[:, : weights.shape[1]] = weights[keep]
                weight_rows.append(padded)
                correct_rows.append(values[keep])
            npz_payload[f"weights_{method}"] = np.stack(weight_rows, axis=0)
            npz_payload[f"correct_{method}"] = np.stack(correct_rows, axis=0)
            prediction_rows = []
            for ids, predictions in zip(saved_episodes, saved_predictions[method]):
                position = {int(x): i for i, x in enumerate(ids.tolist())}
                keep = np.asarray([position[int(x)] for x in common_ids], dtype=np.int64)
                prediction_rows.append(predictions[keep])
            npz_payload[f"prediction_{method}"] = np.stack(prediction_rows, axis=0)
        aligned_paths = []
        for ids, paths in zip(saved_episodes, saved_paths):
            position = {int(x): i for i, x in enumerate(ids.tolist())}
            keep = np.asarray([position[int(x)] for x in common_ids], dtype=np.int64)
            padded = np.full((len(keep), NUM_VIEWS), -1, dtype=np.int64)
            padded[:, : paths.shape[1]] = paths[keep]
            aligned_paths.append(padded)
        npz_payload["paths"] = np.stack(aligned_paths, axis=0)
        for field_name, rows in (
            ("entropy", saved_entropy),
            ("margin", saved_margin),
            ("visibility", saved_visibility),
            ("orientation", saved_orientation),
        ):
            aligned = []
            for ids, values in zip(saved_episodes, rows):
                position = {int(x): i for i, x in enumerate(ids.tolist())}
                keep = np.asarray([position[int(x)] for x in common_ids], dtype=np.int64)
                aligned.append(values[keep])
            npz_payload[field_name] = np.stack(aligned, axis=0)
        detail_name = f"{protocol}_{set_name.lower().replace('-', '_')}.npz"
        np.savez_compressed(records_dir / detail_name, **npz_payload)
        results[set_name] = {
            "methods": summary_methods,
            "episodes": int(len(common_ids)),
            "eval_seeds": list(eval_seeds),
            "details": str(records_dir / detail_name),
            "paired_bootstrap": {
                "samples": BOOTSTRAP_SAMPLES,
                "unit": "sample; random-seed results averaged per sample before resampling",
                "confidence": 0.95,
            },
        }
    return results


def evaluate_all(
    cache_root: Path,
    output_root: Path,
    gate: torch.nn.Module | None,
    eval_seeds: Sequence[int] = EVAL_SEEDS,
) -> dict[str, Any]:
    device = torch.device("cuda:0")
    results: dict[str, Any] = {}
    for split_name, classes in (("seen", SEEN_CLASSES), ("true_unseen_5way", TRUE_UNSEEN)):
        split = "seen_test" if split_name == "seen" else "test"
        raw = load_raw(cache_root, split, device)
        split_root = output_root / split_name
        results[split_name] = {}
        for protocol in ("fixed0", "fixed1", "fixed2", "fixed3", "random_start"):
            results[split_name][protocol] = evaluate_protocol(
                raw, classes, protocol, gate, eval_seeds, split_root / protocol
            )
        dump(split_root / "manifest.json", {
            "split": split,
            "classes": list(classes),
            "episodes_total": raw.num_episodes,
            "eval_seeds": list(eval_seeds),
        })
        del raw
        torch.cuda.empty_cache()
    return results
