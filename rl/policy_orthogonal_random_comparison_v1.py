"""Compare learned policies with preferred-orthogonal and random paths.

This is an isolated evaluation-only experiment.  It reads the preserved
trajectory-policy checkpoints and frozen VPOCLIP caches, and writes every
result below a new work_dir root.  No historical checkpoint, cache, log, or
anchor result is modified.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from . import fusion_weighted_common_v1 as common
from . import weighted_fusion_policy_variants_v1 as variants


ROOT = variants.ROOT
DEVICE = variants.DEVICE
OUTPUT_ROOT = ROOT / "work_dir/policy_orthogonal_random_comparison_v1"
EVAL_SEEDS = variants.EVAL_SEEDS
METHODS = variants.ALL_METHODS
PATH_TYPES = ("policy", "preferred_orthogonal", "random")
NEW_VARIANTS = variants.TRAJECTORY_VARIANTS
NEW_TRUE_UNSEEN = variants.NEW_TRUE_UNSEEN
NEW_SEEN = variants.NEW_SEEN
OLD_TRUE_UNSEEN = variants.OLD_TRUE_UNSEEN
OLD_SEEN = variants.OLD_SEEN


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def bootstrap_ci(values: np.ndarray, seed: int, samples: int = 5000) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return [0.0, 0.0]
    rng = np.random.default_rng(seed)
    chunks = []
    for start in range(0, samples, 250):
        count = min(250, samples - start)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        chunks.append(values[indices].mean(-1))
    boot = np.concatenate(chunks)
    return [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]


def align_arrays(
    rows: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    """Align (episode_ids, metric) rows and average over evaluation seeds."""

    if not rows:
        return np.empty(0, dtype=np.int64), np.empty((0, 0), dtype=np.float32)
    common_ids = None
    for ids, _ in rows:
        current = set(int(x) for x in ids.tolist())
        common_ids = current if common_ids is None else common_ids.intersection(current)
    ids = np.asarray(sorted(common_ids or []), dtype=np.int64)
    aligned = []
    for row_ids, values in rows:
        position = {int(x): i for i, x in enumerate(row_ids.tolist())}
        take = np.asarray([position[int(x)] for x in ids], dtype=np.int64)
        aligned.append(values[take])
    return ids, np.stack(aligned, axis=0)


def paired_delta(
    left: list[tuple[np.ndarray, np.ndarray]],
    right: list[tuple[np.ndarray, np.ndarray]],
    seed: int,
) -> dict[str, Any]:
    """Paired bootstrap of left minus right, averaged over 30 eval seeds."""

    left_ids, left_values = align_arrays(left)
    right_ids, right_values = align_arrays(right)
    common_ids = np.intersect1d(left_ids, right_ids)
    if not len(common_ids):
        return {"delta": 0.0, "ci95": [0.0, 0.0], "episodes": 0}
    left_pos = {int(x): i for i, x in enumerate(left_ids.tolist())}
    right_pos = {int(x): i for i, x in enumerate(right_ids.tolist())}
    left_take = np.asarray([left_pos[int(x)] for x in common_ids], dtype=np.int64)
    right_take = np.asarray([right_pos[int(x)] for x in common_ids], dtype=np.int64)
    per_sample = left_values[:, left_take].mean(0) - right_values[:, right_take].mean(0)
    return {
        "delta": float(per_sample.mean()),
        "ci95": bootstrap_ci(per_sample, seed),
        "episodes": int(len(common_ids)),
        "unit": "sample; 30 evaluation-seed results averaged per sample before resampling",
    }


def path_for_type(
    path_type: str,
    model: torch.nn.Module,
    raw: Any,
    episodes: torch.Tensor,
    starts: torch.Tensor,
    moves: int,
    classes: Sequence[int],
    eval_seed: int,
    v7_mode: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if path_type == "policy":
        if v7_mode:
            return variants.v7_policy_path(model, raw, episodes, starts, classes)
        return variants.trajectory_policy_path(model, raw, episodes, starts, moves)
    if path_type == "preferred_orthogonal":
        return common.path_candidates(
            raw,
            episodes,
            starts,
            f"Preferred-Orthogonal-{moves + 1}",
            eval_seed + 173,
        )
    if path_type == "random":
        return common.path_candidates(
            raw,
            episodes,
            starts,
            f"Random-{moves + 1}",
            eval_seed + 173,
        )
    raise ValueError(path_type)


def save_details(
    path: Path,
    records: list[dict[str, Any]],
    common_ids: np.ndarray,
) -> None:
    payload: dict[str, Any] = {"episode_indices": common_ids}
    aligned_paths = []
    for record in records:
        position = {int(x): i for i, x in enumerate(record["ids"].tolist())}
        take = np.asarray([position[int(x)] for x in common_ids], dtype=np.int64)
        aligned_paths.append(record["paths"][take])
    payload["paths"] = np.stack(aligned_paths, axis=0)
    for method in METHODS:
        prediction_rows = []
        weight_rows = []
        for record in records:
            position = {int(x): i for i, x in enumerate(record["ids"].tolist())}
            take = np.asarray([position[int(x)] for x in common_ids], dtype=np.int64)
            prediction_rows.append(record["predictions"][method][take])
            weight_rows.append(record["weights"][method][take])
        payload[f"prediction_{method}"] = np.stack(prediction_rows, axis=0)
        payload[f"weights_{method}"] = np.stack(weight_rows, axis=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


@torch.inference_mode()
def evaluate_protocol(
    model: torch.nn.Module,
    raw: Any,
    classes: Sequence[int],
    gate: torch.nn.Module,
    protocol: str,
    moves: int,
    v7_mode: bool,
    output_root: Path,
) -> dict[str, Any]:
    minimum_views = moves + 1
    base_episodes = common.eligible_episodes(raw, classes, minimum_views=minimum_views)
    metric_rows: dict[str, dict[str, list[tuple[np.ndarray, np.ndarray]]]] = {
        path_type: {method: [] for method in METHODS} for path_type in PATH_TYPES
    }
    records: dict[str, list[dict[str, Any]]] = {path_type: [] for path_type in PATH_TYPES}

    for eval_seed in EVAL_SEEDS:
        if protocol.startswith("fixed"):
            start = int(protocol.removeprefix("fixed"))
            episodes = base_episodes[raw.valid[base_episodes, start]]
            starts = torch.full_like(episodes, start)
        elif protocol == "random_start":
            episodes = base_episodes
            starts = common.random_starts(raw, episodes, eval_seed)
        else:
            raise ValueError(protocol)

        bank = common.class_bank(classes, raw.device)
        for path_type in PATH_TYPES:
            path_episodes, paths = path_for_type(
                path_type,
                model,
                raw,
                episodes,
                starts,
                moves,
                classes,
                eval_seed,
                v7_mode,
            )
            if not len(path_episodes):
                continue
            ids = path_episodes.detach().cpu().numpy()
            record: dict[str, Any] = {
                "ids": ids,
                "paths": paths.detach().cpu().numpy(),
                "predictions": {},
                "weights": {},
            }
            for method in METHODS:
                metrics = common.path_metrics(raw, path_episodes, paths, bank, method, gate)
                correct = metrics["correct"].detach().cpu().numpy().astype(np.float32)
                metric_rows[path_type][method].append((ids, correct))
                record["predictions"][method] = metrics["prediction"].detach().cpu().numpy()
                record["weights"][method] = metrics["weights"].detach().cpu().numpy().astype(np.float32)
            records[path_type].append(record)

    result: dict[str, Any] = {
        "protocol": protocol,
        "moves": moves,
        "fusion_views": moves + 1,
        "eval_seeds": list(EVAL_SEEDS),
        "path_types": {},
        "paired_vs_random": {},
    }
    for path_index, path_type in enumerate(PATH_TYPES):
        rows = metric_rows[path_type]
        if not rows["equal_mean"]:
            result["path_types"][path_type] = {"methods": {}, "episodes": 0}
            continue
        summaries, common_ids = variants.summarize_rows(
            rows,
            620000 + path_index * 1000 + moves * 100,
        )
        detail_path = output_root / "details" / f"{protocol}_moves_{moves}_{path_type}.npz"
        save_details(detail_path, records[path_type], common_ids)
        result["path_types"][path_type] = {
            "methods": summaries,
            "episodes": int(len(common_ids)),
            "details": str(detail_path),
        }

    for left in ("policy", "preferred_orthogonal"):
        result["paired_vs_random"][left] = {}
        for method_index, method in enumerate(METHODS):
            result["paired_vs_random"][left][method] = paired_delta(
                metric_rows[left][method],
                metric_rows["random"][method],
                700000 + method_index * 101 + moves * 1000,
            )
    return result


def evaluate_new_variants(output_root: Path, gate: torch.nn.Module) -> None:
    raws = {
        "seen": variants.load_split_cache(variants.NEW_CACHE, "seen_test"),
        "true_unseen_5way": variants.load_split_cache(variants.NEW_CACHE, "test"),
    }
    for name in NEW_VARIANTS:
        model, provenance = variants.load_trajectory_model(name)
        _, attach_tracks = variants.configure_trajectory_variant(name)
        for split_name, raw in raws.items():
            attach_tracks(raw, variants.NEW_TRACKS, "seen_test" if split_name == "seen" else "test")
        variant_root = output_root / "new_split_271" / name
        dump(
            variant_root / "config_resolved.json",
            {
                "variant": name,
                "dataset_group": "new_split_271",
                "true_unseen_classes": NEW_TRUE_UNSEEN,
                "policy_provenance": provenance,
                "path_types": list(PATH_TYPES),
                "fusion_methods": list(METHODS),
                "eval_seeds": EVAL_SEEDS,
                "recognizer_frozen": True,
            },
        )
        for split_name, raw in raws.items():
            classes = NEW_SEEN if split_name == "seen" else NEW_TRUE_UNSEEN
            for protocol in ("fixed0", "fixed1", "fixed2", "fixed3", "random_start"):
                for moves in (1, 2):
                    result = evaluate_protocol(
                        model,
                        raw,
                        classes,
                        gate,
                        protocol,
                        moves,
                        False,
                        variant_root / split_name / protocol,
                    )
                    dump(variant_root / split_name / protocol / f"moves_{moves}.json", result)
        del model
        torch.cuda.empty_cache()
    del raws
    torch.cuda.empty_cache()


def evaluate_v7(output_root: Path, gate: torch.nn.Module) -> None:
    raw_seen = variants.load_split_cache(variants.V7_CACHE, "seen_test")
    raw_test = variants.load_split_cache(variants.V7_CACHE, "test")
    model, provenance = variants.load_v7_model()
    root = output_root / "v7_old_split_414"
    dump(
        root / "config_resolved.json",
        {
            "variant": "v7",
            "dataset_group": "old_split_414",
            "true_unseen_classes": OLD_TRUE_UNSEEN,
            "policy_provenance": provenance,
            "path_types": list(PATH_TYPES),
            "fusion_methods": list(METHODS),
            "eval_seeds": EVAL_SEEDS,
            "recognizer_frozen": True,
        },
    )
    for split_name, raw in (("seen", raw_seen), ("true_unseen_5way", raw_test)):
        classes = OLD_SEEN if split_name == "seen" else OLD_TRUE_UNSEEN
        for protocol in ("fixed0", "fixed1", "fixed2", "fixed3", "random_start"):
            result = evaluate_protocol(
                model,
                raw,
                classes,
                gate,
                protocol,
                1,
                True,
                root / split_name / protocol,
            )
            dump(root / split_name / protocol / "moves_1.json", result)
    del raw_seen, raw_test, model
    torch.cuda.empty_cache()


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this evaluation")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    new_gate = variants.load_gate(variants.NEW_GATE_CHECKPOINT)

    old_gate_summary = (
        ROOT / "work_dir/weighted_fusion_policy_variant_comparison_v1/gates/v7_old_split/summary.json"
    )
    if not old_gate_summary.exists():
        old_gate_summary = OUTPUT_ROOT / "gates/v7_old_split/summary.json"
        gate_summary = variants.train_old_cache_gate(
            variants.V7_CACHE,
            old_gate_summary.parent,
            OLD_SEEN,
            variants.read_v7_pseudo_classes(),
        )
    else:
        gate_summary = json.loads(old_gate_summary.read_text(encoding="utf-8"))
    old_gate = variants.load_gate(Path(gate_summary["selected_checkpoint"]))

    evaluate_new_variants(OUTPUT_ROOT, new_gate)
    evaluate_v7(OUTPUT_ROOT, old_gate)
    dump(
        OUTPUT_ROOT / "summary.json",
        {
            "experiment": "policy_vs_preferred_orthogonal_vs_random_v1",
            "path_types": list(PATH_TYPES),
            "trajectory_variants": list(NEW_VARIANTS),
            "v7_included_separately": True,
            "new_split_true_unseen": NEW_TRUE_UNSEEN,
            "old_v7_split_true_unseen": OLD_TRUE_UNSEEN,
            "eval_seeds": EVAL_SEEDS,
            "fusion_methods": list(METHODS),
            "new_gate_checkpoint": str(variants.NEW_GATE_CHECKPOINT),
            "v7_gate_checkpoint": str(gate_summary["selected_checkpoint"]),
            "no_historical_outputs_modified": True,
        },
    )
    print("POLICY_ORTHOGONAL_RANDOM_COMPARISON_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
