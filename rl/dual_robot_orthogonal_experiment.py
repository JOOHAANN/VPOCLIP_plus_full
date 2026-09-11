"""Experiment 1: class-agnostic dual-robot geometric cooperation.

This is an offline dual-agent sensing proxy using the synchronized ETRI
camera views.  It does not train a viewpoint policy.  All pair decisions use
only the sample-specific relative camera geometry and deterministic tie
breaks.  VPOCLIP logits are frozen and are used only for HAR evaluation and
the explicitly privileged Oracle-2 baseline.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from .multistep_g18_view_policy_v1 import (
    CACHE_ROOT,
    NUM_VIEWS,
    SEEN_CLASSES,
    PSEUDO_UNSEEN_CLASSES,
    TRUE_UNSEEN_CLASSES,
    GPUCache,
    class_bank_tensor,
    attach_view_valid,
)


ROOT = CACHE_ROOT.parents[2]
OUTPUT_ROOT = ROOT / "work_dir" / "exp01_dual_robot_orthogonal"
TARGET_ANGLE_DEGREES = 90.0
BOOTSTRAP_SAMPLES = 5000


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def write_camera_config(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# Fallback pseudo-angle convention only; the experiment uses each\n"
        "# sample's body-relative view_geometry when it is available.\n"
        "# These are not calibrated world poses.\n"
        "view0: 0\nview1: 90\nview2: 180\nview3: 270\n"
        "angle_source: sample_specific_view_geometry\n",
        encoding="utf-8",
    )


def angular_separation(geometry: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
    """Return [N,V] circular separation in radians from anchor direction."""

    dot = (geometry * geometry.gather(1, anchor[:, None, None].expand(-1, 1, 2)).squeeze(1)[:, None]).sum(-1)
    return torch.acos(dot.clamp(-1.0, 1.0))


def legal_candidates(raw: GPUCache, episodes: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
    valid = raw.valid[episodes]
    reachable = raw.reachable[episodes, anchor]
    legal = valid & reachable
    legal.scatter_(1, anchor[:, None], False)
    return legal


def choose_methods(
    raw: GPUCache,
    episodes: torch.Tensor,
    anchor: torch.Tensor,
    seed: int,
) -> dict[str, torch.Tensor]:
    geometry = raw.geometry[episodes, :, :2]
    legal = legal_candidates(raw, episodes, anchor)
    separation = angular_separation(geometry, anchor)
    cost = raw.cost[episodes, anchor]
    # deterministic tie-break: lower movement cost, then lower physical slot
    # index through argmax's first-index behavior.
    random_scores = torch.rand(legal.shape, device=raw.device, generator=torch.Generator(device=raw.device).manual_seed(seed))
    random_choice = random_scores.masked_fill(~legal, -torch.inf).argmax(-1)
    slot_priority = (torch.arange(NUM_VIEWS, device=raw.device) == 1).float()[None, :] * 10.0
    fixed_scores = (slot_priority - cost - 1e-6 * torch.arange(NUM_VIEWS, device=raw.device)[None, :]).masked_fill(~legal, -torch.inf)
    fixed_choice = fixed_scores.argmax(-1)
    max_choice = (separation - 1e-6 * cost).masked_fill(~legal, -torch.inf).argmax(-1)
    orthogonal_choice = (
        -(separation - math.radians(TARGET_ANGLE_DEGREES)).abs() - 1e-6 * cost
    ).masked_fill(~legal, -torch.inf).argmax(-1)
    # Oracle-2 is filled from HAR scores in evaluate_bank.
    return {
        "random_pair": random_choice,
        "fixed_pair": fixed_choice,
        "max_separation": max_choice,
        "preferred_orthogonal": orthogonal_choice,
    }


def restricted_accuracy(scores: torch.Tensor, labels: torch.Tensor, bank: torch.Tensor) -> torch.Tensor:
    local = torch.searchsorted(bank, labels)
    return scores.index_select(-1, bank).argmax(-1).eq(local)


def pair_metrics(
    raw: GPUCache,
    episodes: torch.Tensor,
    anchor: torch.Tensor,
    destination: torch.Tensor,
    bank: torch.Tensor,
) -> dict[str, Any]:
    fused = 0.5 * (raw.logits[episodes, anchor] + raw.logits[episodes, destination])
    scores = fused.index_select(-1, bank)
    probabilities = torch.softmax(scores, -1)
    entropy = -(probabilities.clamp_min(1e-8).log() * probabilities).sum(-1)
    correct = restricted_accuracy(fused, raw.labels[episodes], bank)
    cost = raw.cost[episodes, anchor, destination]
    return {
        "top1": float(correct.float().mean()),
        "mean_entropy": float(entropy.mean()),
        "mean_movement_cost": float(cost.mean()),
        "episodes": float(len(episodes)),
        "correct": correct,
        "cost_values": cost,
    }


def single_metrics(raw: GPUCache, episodes: torch.Tensor, anchor: torch.Tensor, bank: torch.Tensor) -> dict[str, Any]:
    logits = raw.logits[episodes, anchor]
    scores = logits.index_select(-1, bank)
    probabilities = torch.softmax(scores, -1)
    entropy = -(probabilities.clamp_min(1e-8).log() * probabilities).sum(-1)
    correct = restricted_accuracy(logits, raw.labels[episodes], bank)
    return {
        "top1": float(correct.float().mean()),
        "mean_entropy": float(entropy.mean()),
        "mean_movement_cost": 0.0,
        "episodes": float(len(episodes)),
        "correct": correct,
        "cost_values": torch.zeros_like(correct, dtype=torch.float32),
    }


def all_valid_metrics(raw: GPUCache, episodes: torch.Tensor, bank: torch.Tensor) -> dict[str, Any]:
    weights = raw.valid[episodes].float().unsqueeze(-1)
    logits = (raw.logits[episodes] * weights).sum(1) / weights.sum(1).clamp_min(1.0)
    correct = restricted_accuracy(logits, raw.labels[episodes], bank)
    return {"top1": float(correct.float().mean()), "episodes": float(len(episodes)), "correct": correct}


def oracle_destination(raw: GPUCache, episodes: torch.Tensor, anchor: torch.Tensor, bank: torch.Tensor) -> torch.Tensor:
    legal = legal_candidates(raw, episodes, anchor)
    anchor_logits = raw.logits[episodes, anchor]
    fused = 0.5 * (anchor_logits[:, None, :] + raw.logits[episodes])
    scores = fused.index_select(-1, bank)
    local = torch.searchsorted(bank, raw.labels[episodes])
    target = scores.gather(-1, local[:, None, None].expand(-1, NUM_VIEWS, 1)).squeeze(-1)
    rival = scores.clone()
    rival.scatter_(-1, local[:, None, None].expand(-1, NUM_VIEWS, 1), -torch.inf)
    margin = target - rival.amax(-1)
    margin = margin.masked_fill(~legal, -torch.inf)
    return margin.argmax(-1)


def bootstrap_delta(preferred: torch.Tensor, baseline: torch.Tensor, seed: int) -> dict[str, Any]:
    difference = (preferred.float() - baseline.float()).detach().cpu().numpy()
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(difference), size=(BOOTSTRAP_SAMPLES, len(difference)))
    means = difference[indices].mean(axis=1)
    return {
        "n_episodes": int(len(difference)),
        "observed_accuracy_difference": float(difference.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "positive_episode_fraction": float((difference > 0).mean()),
    }


def evaluate_split(raw: GPUCache, split: str, classes: Sequence[int], output_root: Path, seed: int) -> dict[str, Any]:
    bank = class_bank_tensor(classes, raw.device)
    eligible = torch.isin(raw.labels, bank) & raw.valid[:, 0] & (raw.valid.sum(-1) >= 2)
    episodes = eligible.nonzero().flatten()
    anchor = torch.zeros_like(episodes)
    methods = choose_methods(raw, episodes, anchor, seed)
    single = single_metrics(raw, episodes, anchor, bank)
    reports: dict[str, Any] = {"single_fixed": single}
    correctness: dict[str, torch.Tensor] = {"single_fixed": single["correct"]}
    for name, destination in methods.items():
        result = pair_metrics(raw, episodes, anchor, destination, bank)
        correctness[name] = result["correct"]
        reports[name] = result
    oracle = oracle_destination(raw, episodes, anchor, bank)
    oracle_result = pair_metrics(raw, episodes, anchor, oracle, bank)
    correctness["oracle_2"] = oracle_result["correct"]
    reports["oracle_2"] = oracle_result
    reports["all_valid_views"] = all_valid_metrics(raw, episodes, bank)
    pair_bootstrap = {
        "preferred_orthogonal_vs_random_pair": bootstrap_delta(
            correctness["preferred_orthogonal"], correctness["random_pair"], seed + 11
        ),
        "preferred_orthogonal_vs_fixed_pair": bootstrap_delta(
            correctness["preferred_orthogonal"], correctness["fixed_pair"], seed + 13
        ),
    }
    dump(output_root / split / "bootstrap_results.json", pair_bootstrap)
    rows = []
    for row_index, episode in enumerate(episodes.cpu().tolist()):
        row = {"episode_id": int(episode), "label": int(raw.labels[episode]), "anchor": 0}
        for name, destination in methods.items():
            row[f"{name}_destination"] = int(destination[row_index])
            row[f"{name}_correct"] = int(correctness[name][row_index])
        row["oracle_2_destination"] = int(oracle[row_index])
        row["oracle_2_correct"] = int(correctness["oracle_2"][row_index])
        rows.append(row)
    output_root.joinpath(split).mkdir(parents=True, exist_ok=True)
    with (output_root / split / "per_sample_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["episode_id"])
        writer.writeheader(); writer.writerows(rows)
    per_class = {}
    labels = raw.labels[episodes].cpu().numpy()
    for class_id in classes:
        mask = labels == int(class_id)
        if not mask.any():
            continue
        per_class[str(class_id)] = {
            name: float(correctness[name].cpu().numpy()[mask].mean()) for name in correctness
        }
    dump(output_root / split / "per_class_results.json", per_class)
    per_class_rows = []
    for class_id, class_result in per_class.items():
        per_class_rows.append({"class_id": int(class_id), **class_result})
    with (output_root / split / "per_class_results.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["class_id"] + list(next(iter(per_class.values())).keys()) if per_class else ["class_id"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_class_rows)
    clean_reports = {}
    for name, result in reports.items():
        clean_reports[name] = {key: value for key, value in result.items() if key not in {"correct", "cost_values"}}
    return {
        "split": split,
        "class_bank": [int(x) for x in classes],
        "episodes": int(len(episodes)),
        "reports": clean_reports,
        "bootstrap": pair_bootstrap,
        "per_class": per_class,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=20260910)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda:0")
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_camera_config(args.output_root / "config" / "camera_view_angles.yaml")
    (args.output_root / "config" / "config.yaml").write_text(
        "angle_source: sample_specific_view_geometry\npreferred_target_degrees: 90\n"
        "true_unseen: [0, 2, 26, 34, 50]\nbootstrap_samples: 5000\n"
        "description: dual-agent synchronized multi-view simulation using ETRI cameras\n",
        encoding="utf-8",
    )
    raw: dict[str, GPUCache] = {}
    for split in ("dqn_train", "val", "seen_test", "test"):
        raw[split] = GPUCache(args.cache_root / split, device)
        attach_view_valid(raw[split], args.cache_root, split)
    results = {}
    for split, classes in (
        ("seen_train", SEEN_CLASSES),
        ("seen_validation", PSEUDO_UNSEEN_CLASSES),
        ("seen_test", SEEN_CLASSES),
        ("true_unseen_5way", TRUE_UNSEEN_CLASSES),
    ):
        source = raw["dqn_train" if split == "seen_train" else "val" if split == "seen_validation" else "seen_test" if split == "seen_test" else "test"]
        results[split] = evaluate_split(source, split, classes, args.output_root / "results", args.seed + len(split))
        print("DUAL_ROBOT_SPLIT", split, json.dumps(results[split]["reports"]), flush=True)
    summary = {
        "experiment": "exp01_dual_robot_preferred_orthogonal",
        "simulation_note": "dual-robot sensing proxy / synchronized multi-view simulation using ETRI cameras; not physical robot deployment",
        "angle_source": "sample-specific body-relative view_geometry; fallback pseudo-angle config is not calibrated world pose",
        "anchor_main": "fixed view0",
        "preferred_target_degrees": 90.0,
        "decision_uses": ["sample-specific relative angle", "reachability", "movement cost only for deterministic ties"],
        "oracle_uses_gt": True,
        "true_unseen": TRUE_UNSEEN_CLASSES,
        "results": results,
    }
    dump(args.output_root / "summary.json", summary)
    print("DUAL_ROBOT_EXPERIMENT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
