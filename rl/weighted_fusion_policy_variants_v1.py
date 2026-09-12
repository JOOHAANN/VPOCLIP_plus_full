"""Apply the simple weighted-logit fusion methods to the preserved policies.

This experiment is deliberately separate from the historical trajectory and
candidate-rank outputs.  It does not retrain v4/v5/v6/object-only/geometry-
only/v7 and does not alter their checkpoints.  It only changes the final
fusion of the views selected by each preserved policy:

    equal_mean, lightweight_gating, rule_cls, rule_geo, rule_combined

The trajectory variants and v7 use different 50/5 protocols in this
workspace, so they are evaluated in separate result groups rather than being
pooled into one misleading number.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch

from . import causal_rank_v6 as rank_v6
from . import causal_rank_v7 as rank_v7
from . import fusion_lightweight_gating_v1 as gate_impl
from . import fusion_weighted_common_v1 as common
from . import multistep_angle_object_trajectory_ablation as ablation
from . import multistep_angle_object_trajectory_policy_v1 as trajectory_base
from . import multistep_angle_object_trajectory_policy_v4 as v4
from . import multistep_angle_object_trajectory_policy_v5_no_category_id as v5
from . import multistep_angle_object_trajectory_policy_v6_no_object as v6


ROOT = Path(__file__).resolve().parents[1]
DEVICE = torch.device("cuda:0")
EVAL_SEEDS = list(range(20260920, 20260950))
TRAIN_SEEDS = [20260909, 20260910, 20260911]
TRAJECTORY_VARIANTS = ("v4", "v5", "v6", "object_only", "geometry_only")
ALL_METHODS = common.METHODS
NEW_TRUE_UNSEEN = [14, 15, 25, 39, 46]
NEW_SEEN = sorted(set(range(55)) - set(NEW_TRUE_UNSEEN))
OLD_TRUE_UNSEEN = [0, 2, 26, 34, 50]
OLD_SEEN = sorted(set(range(55)) - set(OLD_TRUE_UNSEEN))

NEW_CACHE = ROOT / "data/rl_raw_new50_5_group1_g1_pseudo_selected_stage_b/cache"
NEW_TRACKS = ROOT / "data/trajectory_g1_pseudo_selected_stage_b/object_tracks_full"
NEW_POLICY_ROOT = ROOT / "work_dir/trajectory_g1_pseudo_selected_stage_b_2d"
NEW_GATE_CHECKPOINT = (
    ROOT / "work_dir/fusion_lightweight_gating_v1/models/seed_20260910/best.pt"
)
V7_CACHE = ROOT / "data/rl_candidate_rank_v6_new50_5/cache"
V7_POLICY = (
    ROOT / "work_dir/active_view_candidate_rank_v7_new50_5/semantic_20260909/best.pt"
)
V7_CONFIG = ROOT / "logs/rl_candidate_rank_v6_new50_5/config.yaml"
ORIGINAL_TRAJECTORY_ATTACH = trajectory_base.attach_object_tracks


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def configure_trajectory_variant(name: str) -> tuple[type[torch.nn.Module], Callable[..., Any]]:
    """Set the preserved v1 state-builder dispatch and return model/attach fn."""

    if name == "v4":
        trajectory_base.build_trajectory_state = v4.build_trajectory_state_v4
        trajectory_base.AngleObjectTrajectoryPolicy = v4.BodyOrientationTrajectoryPolicy
        trajectory_base.attach_object_tracks = ORIGINAL_TRAJECTORY_ATTACH
        return v4.BodyOrientationTrajectoryPolicy, ORIGINAL_TRAJECTORY_ATTACH
    if name == "v5":
        trajectory_base.build_trajectory_state = v5.build_trajectory_state_v5
        trajectory_base.AngleObjectTrajectoryPolicy = v5.CategoryFreeObjectTrajectoryPolicy
        trajectory_base.attach_object_tracks = ORIGINAL_TRAJECTORY_ATTACH
        return v5.CategoryFreeObjectTrajectoryPolicy, ORIGINAL_TRAJECTORY_ATTACH
    if name == "v6":
        trajectory_base.build_trajectory_state = v6.build_trajectory_state_v5
        trajectory_base.AngleObjectTrajectoryPolicy = v6.PersonGeometryTrajectoryPolicy
        trajectory_base.attach_object_tracks = v6.attach_no_object_tracks
        return v6.PersonGeometryTrajectoryPolicy, v6.attach_no_object_tracks
    if name == "object_only":
        trajectory_base.build_trajectory_state = ablation.build_object_only_state
        trajectory_base.AngleObjectTrajectoryPolicy = ablation.ObjectOnlyTrajectoryPolicy
        trajectory_base.attach_object_tracks = ORIGINAL_TRAJECTORY_ATTACH
        return ablation.ObjectOnlyTrajectoryPolicy, ORIGINAL_TRAJECTORY_ATTACH
    if name == "geometry_only":
        trajectory_base.build_trajectory_state = ablation.build_geometry_only_state
        trajectory_base.AngleObjectTrajectoryPolicy = ablation.GeometryOnlyPolicy
        trajectory_base.attach_object_tracks = ablation.attach_no_object_tracks
        return ablation.GeometryOnlyPolicy, ablation.attach_no_object_tracks
    raise ValueError(name)


def load_trajectory_model(name: str) -> tuple[torch.nn.Module, dict[str, Any]]:
    model_cls, _ = configure_trajectory_variant(name)
    selection = json.loads((NEW_POLICY_ROOT / name / "selection.json").read_text())
    checkpoint = Path(selection["chosen"]["checkpoint"])
    model = model_cls().to(DEVICE)
    payload = torch.load(checkpoint, map_location=DEVICE, weights_only=False)
    model.load_state_dict(payload["online"], strict=True)
    model.eval()
    return model, {
        "checkpoint": str(checkpoint),
        "selected_seed": int(selection["chosen"]["seed"]),
        "selection_criterion": selection.get("criterion"),
    }


@torch.inference_mode()
def trajectory_policy_path(
    model: torch.nn.Module,
    raw: Any,
    episodes: torch.Tensor,
    starts: torch.Tensor,
    moves: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Roll a preserved causal trajectory policy for one or two moves."""

    if moves not in (1, 2):
        raise ValueError(moves)
    current_episodes = episodes
    current_starts = starts
    history = starts[:, None]
    actions: list[torch.Tensor] = []
    for stage in range(1, moves + 1):
        state = trajectory_base.build_trajectory_state(
            raw,
            current_episodes,
            history,
            torch.full_like(current_episodes, stage),
        )
        keep = state["mask"].any(-1)
        if not bool(keep.all()):
            current_episodes = current_episodes[keep]
            current_starts = current_starts[keep]
            history = history[keep]
            actions = [previous[keep] for previous in actions]
            state = {key: value[keep] for key, value in state.items()}
        if not len(current_episodes):
            return (
                torch.empty(0, dtype=torch.long, device=raw.device),
                torch.empty((0, moves + 1), dtype=torch.long, device=raw.device),
            )
        action = model(state).masked_fill(~state["mask"], -torch.inf).argmax(-1)
        actions.append(action)
        history = torch.cat((history, action[:, None]), dim=1)
    return current_episodes, torch.cat((current_starts[:, None], torch.stack(actions, dim=1)), dim=1)


def load_v7_model() -> tuple[torch.nn.Module, dict[str, Any]]:
    model = rank_v7.Ranker(semantic=True).to(DEVICE)
    payload = torch.load(V7_POLICY, map_location=DEVICE, weights_only=False)
    model.load_state_dict(payload["online"], strict=True)
    model.eval()
    return model, {
        "checkpoint": str(V7_POLICY),
        "selected_variant": "semantic",
        "selected_seed": 20260909,
    }


@torch.inference_mode()
def v7_policy_path(
    model: torch.nn.Module,
    raw: Any,
    episodes: torch.Tensor,
    starts: torch.Tensor,
    bank_classes: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    bank = rank_v6.class_mask(bank_classes, len(episodes), raw.device)
    state = rank_v6.causal_state(raw, episodes, starts, bank)
    keep = state["mask"].any(-1)
    episodes = episodes[keep]
    starts = starts[keep]
    state = {key: value[keep] for key, value in state.items()}
    if not len(episodes):
        return (
            torch.empty(0, dtype=torch.long, device=raw.device),
            torch.empty((0, 2), dtype=torch.long, device=raw.device),
        )
    action = model(state).masked_fill(~state["mask"], -torch.inf).argmax(-1)
    return episodes, torch.stack((starts, action), dim=1)


def train_old_cache_gate(
    cache_root: Path,
    output_root: Path,
    seen_classes: Sequence[int],
    pseudo_classes: Sequence[int],
) -> dict[str, Any]:
    """Train the same tiny gate on v7's recognizer/cache, separately."""

    summary_path = output_root / "summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text())
    train_raw = common.load_raw(cache_root, "dqn_train", DEVICE)
    val_raw = common.load_raw(cache_root, "val", DEVICE)
    runs: list[dict[str, Any]] = []
    for seed in TRAIN_SEEDS:
        run = output_root / f"seed_{seed}"
        run.mkdir(parents=True, exist_ok=True)
        complete = run / "complete.json"
        if complete.exists():
            runs.append(json.loads(complete.read_text()))
            continue
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed_all(seed)
        gate = gate_impl.LightweightGate(13).to(DEVICE)
        optimizer = torch.optim.AdamW(gate.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 40, eta_min=5e-5)
        generator = torch.Generator(device=DEVICE).manual_seed(seed)
        pool = common.eligible_episodes(train_raw, seen_classes, minimum_views=4)
        bank = common.class_bank(seen_classes, DEVICE)
        best_score = -float("inf")
        best_epoch = 0
        for epoch in range(1, 41):
            gate.train()
            losses: list[float] = []
            for _ in range(64):
                ids = torch.randint(len(pool), (16384,), device=DEVICE, generator=generator)
                episodes = pool[ids]
                order = torch.rand((len(episodes), 4), device=DEVICE, generator=generator).argsort(-1)
                k = int(torch.randint(2, 5, (), device=DEVICE, generator=generator))
                paths = order[:, :k]
                inputs, _ = common.path_inputs(train_raw, episodes, paths)
                weights = torch.softmax(gate(inputs).squeeze(-1), dim=-1)
                fused = (weights[..., None] * train_raw.logits[episodes[:, None], paths]).sum(1)
                scores = fused.index_select(-1, bank)
                labels = torch.searchsorted(bank, train_raw.labels[episodes])
                loss = torch.nn.functional.cross_entropy(scores, labels)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(gate.parameters(), 5.0)
                optimizer.step()
                losses.append(float(loss.detach()))
            scheduler.step()
            gate.eval()
            validation: list[torch.Tensor] = []
            pseudo_pool = common.eligible_episodes(val_raw, pseudo_classes, minimum_views=2)
            for set_name in ("Random-2", "Preferred-Orthogonal-2", "Random-3", "Preferred-Orthogonal-3"):
                k = 2 if set_name.endswith("-2") else 3
                eligible = pseudo_pool[val_raw.valid[pseudo_pool].sum(-1) >= k]
                starts = torch.zeros_like(eligible)
                ep, paths = common.path_candidates(val_raw, eligible, starts, set_name, seed + epoch)
                if len(ep):
                    metric = common.path_metrics(
                        val_raw, ep, paths, common.class_bank(pseudo_classes, DEVICE),
                        "lightweight_gating", gate,
                    )
                    validation.append(metric["correct"].float().mean())
            score = float(torch.stack(validation).mean()) if validation else 0.0
            payload = {
                "model": gate.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "seed": seed,
                "best_score": best_score,
                "best_epoch": best_epoch,
            }
            torch.save(payload, run / "last.pt")
            if score > best_score:
                best_score, best_epoch = score, epoch
                payload["best_score"] = best_score
                payload["best_epoch"] = best_epoch
                torch.save(payload, run / "best.pt")
            with (run / "train.log").open("a", encoding="utf-8", buffering=1) as handle:
                handle.write(json.dumps({
                    "epoch": epoch, "seed": seed, "loss": float(np.mean(losses)),
                    "validation_score": score, "best_score": best_score,
                    "best_epoch": best_epoch,
                }) + "\n")
        result = {
            "seed": seed,
            "checkpoint": str(run / "best.pt"),
            "best_score": best_score,
            "best_epoch": best_epoch,
            "training_episodes": int(len(pool)),
        }
        dump(complete, result)
        runs.append(result)
    chosen = max(runs, key=lambda row: float(row["best_score"]))
    summary = {
        "experiment": "variant_specific_gate_for_v7_cache",
        "cache_root": str(cache_root),
        "training_classes": list(seen_classes),
        "pseudo_unseen_selection_classes": list(pseudo_classes),
        "train_seeds": TRAIN_SEEDS,
        "selected_checkpoint": chosen["checkpoint"],
        "selection_runs": runs,
    }
    dump(summary_path, summary)
    return summary


def load_gate(checkpoint: Path) -> torch.nn.Module:
    gate = gate_impl.LightweightGate(13).to(DEVICE)
    payload = torch.load(checkpoint, map_location=DEVICE, weights_only=False)
    gate.load_state_dict(payload["model"], strict=True)
    gate.eval()
    return gate


def bootstrap_ci(values: np.ndarray, seed: int, samples: int = 5000) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return [0.0, 0.0]
    rng = np.random.default_rng(seed)
    chunks = []
    for start in range(0, samples, 250):
        count = min(250, samples - start)
        index = rng.integers(0, len(values), size=(count, len(values)))
        chunks.append(values[index].mean(-1))
    boot = np.concatenate(chunks)
    return [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]


def summarize_rows(
    rows: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    seed: int,
) -> tuple[dict[str, Any], np.ndarray]:
    """Align episodes across eval seeds and summarize paired deltas."""

    common_ids = None
    for ids, _ in rows["equal_mean"]:
        current = set(int(x) for x in ids.tolist())
        common_ids = current if common_ids is None else common_ids.intersection(current)
    common = np.asarray(sorted(common_ids or []), dtype=np.int64)
    arrays: dict[str, np.ndarray] = {}
    for method, values in rows.items():
        aligned = []
        for ids, metric in values:
            position = {int(x): i for i, x in enumerate(ids.tolist())}
            take = np.asarray([position[int(x)] for x in common], dtype=np.int64)
            aligned.append(metric[take])
        arrays[method] = np.stack(aligned, axis=0) if aligned else np.empty((0, len(common)))
    equal = arrays["equal_mean"]
    result: dict[str, Any] = {}
    for index, method in enumerate(ALL_METHODS):
        per_sample = arrays[method].mean(0) if len(arrays[method]) else np.zeros(len(common))
        per_equal = equal.mean(0) if len(equal) else np.zeros(len(common))
        delta = per_sample - per_equal
        result[method] = {
            "top1": float(per_sample.mean()) if len(per_sample) else 0.0,
            "top1_ci95": bootstrap_ci(per_sample, seed + index),
            "delta_vs_equal_mean": float(delta.mean()) if len(delta) else 0.0,
            "delta_vs_equal_mean_ci95": bootstrap_ci(delta, seed + 100 + index),
            "episodes": int(len(common)),
        }
    return result, common


def save_detail(
    path: Path,
    raw: Any,
    gate: torch.nn.Module,
    classes: Sequence[int],
    episode_rows: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    path_rows: dict[str, list[np.ndarray]],
    weight_rows: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    sample_ids: np.ndarray,
) -> None:
    """Store selected paths, weights and predictions for later audit."""

    # This detail writer is intentionally compact.  The summary contains the
    # paired bootstrap statistics; the NPZ stores exact per-seed paths and
    # per-method final weights/predictions.
    common_ids = sample_ids.astype(np.int64)
    payload: dict[str, Any] = {"episode_indices": common_ids}
    for method, values in weight_rows.items():
        aligned_weights = []
        aligned_predictions = []
        for ids, weights, predictions in values:
            position = {int(x): i for i, x in enumerate(ids.tolist())}
            take = np.asarray([position[int(x)] for x in common_ids], dtype=np.int64)
            padded = np.zeros((len(take), weights.shape[1]), dtype=np.float32)
            padded[:] = weights[take]
            aligned_weights.append(padded)
            aligned_predictions.append(predictions[take])
        payload[f"weights_{method}"] = np.stack(aligned_weights, axis=0)
        payload[f"prediction_{method}"] = np.stack(aligned_predictions, axis=0)
    aligned_paths = []
    for ids, paths in path_rows["policy"]:
        position = {int(x): i for i, x in enumerate(ids.tolist())}
        take = np.asarray([position[int(x)] for x in common_ids], dtype=np.int64)
        aligned_paths.append(paths[take])
    payload["policy_paths"] = np.stack(aligned_paths, axis=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


@torch.inference_mode()
def evaluate_one_protocol(
    model_name: str,
    model: torch.nn.Module,
    raw: Any,
    classes: Sequence[int],
    gate: torch.nn.Module,
    protocol: str,
    moves: int,
    eval_seeds: Sequence[int],
    output_root: Path,
    v7_mode: bool = False,
) -> dict[str, Any]:
    minimum_views = moves + 1
    base_episodes = common.eligible_episodes(raw, classes, minimum_views=minimum_views)
    per_path_method: dict[str, dict[str, list[tuple[np.ndarray, np.ndarray]]]] = {
        "policy": {method: [] for method in ALL_METHODS},
        "random": {method: [] for method in ALL_METHODS},
    }
    saved_paths: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {"policy": [], "random": []}
    saved_weights: dict[str, dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]]] = {
        "policy": {method: [] for method in ALL_METHODS},
        "random": {method: [] for method in ALL_METHODS},
    }
    for eval_seed in eval_seeds:
        if protocol.startswith("fixed"):
            start = int(protocol.removeprefix("fixed"))
            episodes = base_episodes[raw.valid[base_episodes, start]]
            starts = torch.full_like(episodes, start)
        elif protocol == "random_start":
            episodes = base_episodes
            starts = common.random_starts(raw, episodes, eval_seed)
        else:
            raise ValueError(protocol)

        if v7_mode:
            policy_episodes, policy_paths = v7_policy_path(
                model, raw, episodes, starts, classes
            )
        else:
            policy_episodes, policy_paths = trajectory_policy_path(
                model, raw, episodes, starts, moves
            )
        random_episodes, random_paths = common.path_candidates(
            raw, episodes, starts, f"Random-{moves + 1}", eval_seed + 173
        )
        for path_type, path_episodes, paths in (
            ("policy", policy_episodes, policy_paths),
            ("random", random_episodes, random_paths),
        ):
            if not len(path_episodes):
                continue
            ids = path_episodes.detach().cpu().numpy()
            saved_paths[path_type].append((ids, paths.detach().cpu().numpy()))
            bank = common.class_bank(classes, raw.device)
            for method in ALL_METHODS:
                metrics = common.path_metrics(raw, path_episodes, paths, bank, method, gate)
                correct = metrics["correct"].detach().cpu().numpy().astype(np.float32)
                per_path_method[path_type][method].append((ids, correct))
                saved_weights[path_type][method].append(
                    (
                        ids,
                        metrics["weights"].detach().cpu().numpy().astype(np.float32),
                        metrics["prediction"].detach().cpu().numpy(),
                    )
                )

    result: dict[str, Any] = {
        "protocol": protocol,
        "moves": moves,
        "fusion_views": moves + 1,
        "eval_seeds": list(eval_seeds),
        "path_types": {},
    }
    for path_index, path_type in enumerate(("policy", "random")):
        method_rows = per_path_method[path_type]
        if not method_rows["equal_mean"]:
            result["path_types"][path_type] = {"methods": {}, "episodes": 0}
            continue
        summaries, common_ids = summarize_rows(method_rows, 420000 + path_index + moves * 100)
        result["path_types"][path_type] = {"methods": summaries, "episodes": int(len(common_ids))}
        # Save exact details for the policy path and also for the random path.
        details = {}
        for method in ALL_METHODS:
            values = []
            for ids, weights, predictions in saved_weights[path_type][method]:
                position = {int(x): i for i, x in enumerate(ids.tolist())}
                take = np.asarray([position[int(x)] for x in common_ids], dtype=np.int64)
                values.append((ids, weights, predictions))
            details[method] = values
        detail_path = output_root / "details" / f"{protocol}_moves_{moves}_{path_type}.npz"
        # write path/weights without forcing the two path types to share IDs
        payload: dict[str, Any] = {"episode_indices": common_ids}
        for method, values in details.items():
            aligned_weights = []
            aligned_predictions = []
            for ids, weights, predictions in values:
                position = {int(x): i for i, x in enumerate(ids.tolist())}
                take = np.asarray([position[int(x)] for x in common_ids], dtype=np.int64)
                aligned_weights.append(weights[take])
                aligned_predictions.append(predictions[take])
            payload[f"weights_{method}"] = np.stack(aligned_weights, axis=0)
            payload[f"prediction_{method}"] = np.stack(aligned_predictions, axis=0)
        aligned_paths = []
        for ids, paths in saved_paths[path_type]:
            position = {int(x): i for i, x in enumerate(ids.tolist())}
            take = np.asarray([position[int(x)] for x in common_ids], dtype=np.int64)
            aligned_paths.append(paths[take])
        payload["paths"] = np.stack(aligned_paths, axis=0)
        detail_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(detail_path, **payload)
        result["path_types"][path_type]["details"] = str(detail_path)
    return result


def load_split_cache(cache_root: Path, split: str) -> Any:
    return common.load_raw(cache_root, split, DEVICE)


def evaluate_trajectory_variants(output_root: Path, gate: torch.nn.Module) -> None:
    raws = {
        "seen": load_split_cache(NEW_CACHE, "seen_test"),
        "true_unseen_5way": load_split_cache(NEW_CACHE, "test"),
    }
    for variant in TRAJECTORY_VARIANTS:
        model, provenance = load_trajectory_model(variant)
        _, attach_tracks = configure_trajectory_variant(variant)
        for split_name, raw in raws.items():
            attach_tracks(raw, NEW_TRACKS, "seen_test" if split_name == "seen" else "test")
        variant_root = output_root / "new_split_271" / variant
        metadata = {
            "variant": variant,
            "dataset_group": "new_split_271",
            "true_unseen_classes": NEW_TRUE_UNSEEN,
            "policy_provenance": provenance,
            "fusion_methods": list(ALL_METHODS),
            "eval_seeds": EVAL_SEEDS,
            "recognizer_frozen": True,
            "weighted_fusion_applied_after_policy_path": True,
        }
        dump(variant_root / "config_resolved.json", metadata)
        for split_name, raw in raws.items():
            for protocol in ("fixed0", "fixed1", "fixed2", "fixed3", "random_start"):
                for moves in (1, 2):
                    result = evaluate_one_protocol(
                        variant, model, raw,
                        NEW_SEEN if split_name == "seen" else NEW_TRUE_UNSEEN,
                        gate, protocol, moves, EVAL_SEEDS,
                        variant_root / split_name / protocol,
                    )
                    dump(variant_root / split_name / protocol / f"moves_{moves}.json", result)
        del model
        torch.cuda.empty_cache()
    del raws
    torch.cuda.empty_cache()


def read_v7_pseudo_classes() -> list[int]:
    # Avoid adding a YAML dependency to the result schema; this list is the
    # policy-held-out seen-class list recorded in the preserved config.
    text = V7_CONFIG.read_text(encoding="utf-8")
    marker = "  policy_holdout_classes:\n"
    start = text.index(marker) + len(marker)
    values = []
    for line in text[start:].splitlines():
        if not line.startswith("  - "):
            break
        values.append(int(line.strip()[2:]))
    return values


def evaluate_v7(output_root: Path, gate: torch.nn.Module) -> None:
    raws = {
        "seen": load_split_cache(V7_CACHE, "seen_test"),
        "true_unseen_5way": load_split_cache(V7_CACHE, "test"),
    }
    model, provenance = load_v7_model()
    root = output_root / "v7_old_split_414"
    metadata = {
        "variant": "v7",
        "dataset_group": "old_split_414",
        "true_unseen_classes": OLD_TRUE_UNSEEN,
        "policy_provenance": provenance,
        "fusion_methods": list(ALL_METHODS),
        "eval_seeds": EVAL_SEEDS,
        "recognizer_frozen": True,
        "weighted_fusion_applied_after_policy_path": True,
        "policy_gate_cache": str(V7_CACHE),
    }
    dump(root / "config_resolved.json", metadata)
    for split_name, raw in raws.items():
        classes = OLD_SEEN if split_name == "seen" else OLD_TRUE_UNSEEN
        for protocol in ("fixed0", "fixed1", "fixed2", "fixed3", "random_start"):
            result = evaluate_one_protocol(
                "v7", model, raw, classes, gate, protocol, 1, EVAL_SEEDS,
                root / split_name / protocol, v7_mode=True,
            )
            dump(root / split_name / protocol / "moves_1.json", result)
    del raws, model
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root", type=Path,
        default=ROOT / "work_dir/weighted_fusion_policy_variant_comparison_v1",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    args.output_root.mkdir(parents=True, exist_ok=True)

    if not NEW_GATE_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"new-split gate checkpoint missing: {NEW_GATE_CHECKPOINT}; "
            "run fusion_lightweight_gating_v1 first"
        )
    new_gate = load_gate(NEW_GATE_CHECKPOINT)
    evaluate_trajectory_variants(args.output_root, new_gate)
    del new_gate
    torch.cuda.empty_cache()

    old_gate_summary = train_old_cache_gate(
        V7_CACHE, args.output_root / "gates/v7_old_split",
        OLD_SEEN, read_v7_pseudo_classes(),
    )
    old_gate = load_gate(Path(old_gate_summary["selected_checkpoint"]))
    evaluate_v7(args.output_root, old_gate)
    dump(args.output_root / "summary.json", {
        "experiment": "weighted_fusion_on_preserved_policy_variants_v1",
        "trajectory_variants": list(TRAJECTORY_VARIANTS),
        "v7_included_separately": True,
        "new_split_true_unseen": NEW_TRUE_UNSEEN,
        "old_v7_split_true_unseen": OLD_TRUE_UNSEEN,
        "eval_seeds": EVAL_SEEDS,
        "methods": list(ALL_METHODS),
        "no_historical_outputs_modified": True,
        "new_split_gate_checkpoint": str(NEW_GATE_CHECKPOINT),
        "v7_gate_summary": str(args.output_root / "gates/v7_old_split/summary.json"),
    })
    print("WEIGHTED_FUSION_POLICY_VARIANT_COMPARISON_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
