"""Run Sequential Explore-Then-Decide separately for physical starts 1/2/3.

The main sequential experiment is defined with slot 0 as the start.  This
driver creates an equivalent view-permuted copy for each physical start, so
the existing implementation can be reused without changing or overwriting
the original start-0 experiment.  A permutation is applied to every
view-indexed tensor and to both axes of transition cost/reachability.

Each start is trained independently under the same active-module 40/10/5
protocol.  The frozen recognizer remains the existing 50-seen-class
checkpoint; no true-unseen result is used for model or strategy selection.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .sequential_explore_then_decide_v1 import (
    EPSILON_GAIN,
    TRAIN_CLASSES,
    PSEUDO_CLASSES,
    SEEN_CLASSES,
    UNSEEN_CLASSES,
    NUM_VIEWS,
    adaptive_paths,
    baseline_report,
    evaluate_paths,
    fit_stop_models,
    first_state,
    geometry_arrays,
    json_safe,
    prepare_variable_data,
    exploratory_actions,
    ranker_report,
    sequential_targets,
    train_candidate_ranker,
    train_sequential_dqn,
    evaluate_sequential_dqn,
)


def permute_data(data: Mapping[str, Any], start_view: int) -> dict[str, Any]:
    """Make physical ``start_view`` appear as logical slot 0."""

    if start_view not in range(NUM_VIEWS):
        raise ValueError(start_view)
    permutation = np.asarray([start_view] + [v for v in range(NUM_VIEWS) if v != start_view], dtype=np.int64)
    raw = dict(data["raw"])
    for key in ("z", "logits", "pose", "object_map", "quality", "geometry",
                "valid", "geometry_confidence"):
        value = raw.get(key)
        if isinstance(value, np.ndarray) and value.ndim >= 2 and value.shape[1] == NUM_VIEWS:
            raw[key] = np.asarray(value[:, permutation])
    for key in ("cost", "reachable"):
        value = raw.get(key)
        if isinstance(value, np.ndarray) and value.ndim == 3 and value.shape[1:] == (NUM_VIEWS, NUM_VIEWS):
            raw[key] = np.asarray(value[:, permutation][:, :, permutation])
    features = {}
    for key, value in data["features"].items():
        value = np.asarray(value)
        if value.ndim >= 2 and value.shape[1] == NUM_VIEWS:
            features[key] = np.asarray(value[:, permutation])
        else:
            features[key] = value
    # A physical start is only meaningful when that camera really exists and
    # at least one different reachable view remains.  This removes the
    # documented 2/3-view anomalies from only the affected start sweep; it
    # does not fabricate missing views or silently map them to another slot.
    valid_start = np.asarray(raw["valid"][:, 0], dtype=bool)
    reachable_from_start = np.asarray(raw["valid"], dtype=bool) & np.asarray(raw["reachable"][:, 0], dtype=bool)
    keep = valid_start & (reachable_from_start.sum(axis=1) >= 2)
    n = len(data["episodes"])
    for key, value in list(raw.items()):
        if isinstance(value, np.ndarray) and value.ndim > 0 and len(value) == n:
            raw[key] = np.asarray(value[keep])
    for key, value in list(features.items()):
        if isinstance(value, np.ndarray) and value.ndim > 0 and len(value) == n:
            features[key] = np.asarray(value[keep])
    return {"raw": raw, "features": features,
            "episodes": np.asarray(data["episodes"])[keep],
            "source_episode_ids": np.asarray(data["source_episode_ids"])[keep],
            "classes": list(data["classes"]), "split": data["split"],
            "physical_start_view": int(start_view), "view_permutation": permutation.tolist(),
            "start_filter_kept": int(keep.sum()), "start_filter_removed": int((~keep).sum())}


def strategy_metrics(data: Mapping[str, Any], classes: list[int]) -> tuple[dict[str, Any], str]:
    names = ("max_angular_diversity", "preferred_orthogonal", "diversity_minus_cost")
    rows = {}
    for name in names:
        action = exploratory_actions(data, name)
        rows[name] = evaluate_paths(data, classes,
                                    [[0, int(a)] for a in action], name + "_validation")
    selected = max(rows, key=lambda k: (rows[k]["top1"] - 0.01 * rows[k]["average_movement_cost"], rows[k]["top1"]))
    return rows, selected


def compact_baselines(data: Mapping[str, Any], classes: list[int], selected: str) -> dict[str, Any]:
    report = baseline_report(data, classes, selected)
    return {"selected_first_strategy": selected, "methods": report["methods"]}


def run_one_start(base_data: Mapping[str, Mapping[str, Any]], start_view: int,
                  device: torch.device, seed: int, output: Path) -> dict[str, Any]:
    data = {name: permute_data(value, start_view) for name, value in base_data.items()}
    pseudo_strategy, selected_strategy = strategy_metrics(data["pseudo"], PSEUDO_CLASSES)
    first = {name: exploratory_actions(value, selected_strategy)
             for name, value in data.items()}
    baselines = {
        name: compact_baselines(value, classes, selected_strategy)
        for name, (value, classes) in {
            "pseudo_unseen_10": (data["pseudo"], PSEUDO_CLASSES),
            "seen_test_50": (data["seen"], SEEN_CLASSES),
            "true_unseen_5": (data["unseen"], UNSEEN_CLASSES),
        }.items()
    }

    train_x, feature_names, _ = first_state(data["train"], first["train"])
    train_target = sequential_targets(data["train"], TRAIN_CLASSES, first["train"])
    groups = np.asarray([
        str(data["train"]["raw"]["metadata"]["episodes"][int(x)].get("subject", "unknown"))
        for x in data["train"]["source_episode_ids"]
    ])
    models, oof, folds = fit_stop_models(train_x, train_target["continue"], groups, device, seed)
    query = {}
    targets = {}
    for name, key, classes in (("pseudo", "pseudo", PSEUDO_CLASSES),
                               ("seen", "seen", SEEN_CLASSES),
                               ("unseen", "unseen", UNSEEN_CLASSES)):
        query[name], _, _ = first_state(data[key], first[key])
        targets[name] = sequential_targets(data[key], classes, first[key])

    from .sequential_explore_then_decide_v1 import predict_classifier, auc_metrics, best_threshold
    predictions = {}
    predictor_splits = {}
    for name in ("pseudo", "seen", "unseen"):
        predictions[name] = {}
        predictor_splits[name] = {}
        for model_name in ("logistic", "random_forest", "gradient_boosting", "mlp"):
            probability = predict_classifier(model_name, models[model_name], query[name], device)
            predictions[name][model_name] = probability
            predictor_splits[name][model_name] = {
                **auc_metrics(targets[name]["continue"], probability),
                "best_threshold": best_threshold(targets[name]["continue"], probability),
            }
    selected_model = max(
        predictor_splits["pseudo"],
        key=lambda k: ((predictor_splits["pseudo"][k]["auroc"] or -1.0),
                       (predictor_splits["pseudo"][k]["auprc"] or -1.0)),
    )
    selected_threshold = float(predictor_splits["pseudo"][selected_model]["best_threshold"]["threshold"])
    val_metric = predictor_splits["pseudo"][selected_model]
    prevalence = float(targets["pseudo"]["continue"].mean())
    auprc_clear = bool(val_metric["auprc"] is not None
                       and val_metric["auprc"] >= prevalence * 1.15
                       and val_metric["auprc"] >= prevalence + 0.03)
    gate_a = bool((val_metric["auroc"] is not None and val_metric["auroc"] >= 0.70) or auprc_clear)
    gates: dict[str, Any] = {
        "gate_a": {
            "selected_model_by_validation": selected_model,
            "selected_threshold": selected_threshold,
            "validation_auroc": val_metric["auroc"],
            "validation_auprc": val_metric["auprc"],
            "validation_positive_prevalence": prevalence,
            "passed": gate_a,
        },
        "gate_b": None,
    }
    adaptive: dict[str, Any] = {}
    ranker = None
    if gate_a:
        ranker = train_candidate_ranker(data["train"], TRAIN_CLASSES, first["train"], device, seed + 500, output)
        for name, key, classes in (("pseudo_unseen_10", "pseudo", PSEUDO_CLASSES),
                                   ("seen_test_50", "seen", SEEN_CLASSES),
                                   ("true_unseen_5", "unseen", UNSEEN_CLASSES)):
            gates.setdefault("ranker", {})[name] = ranker_report(
                ranker, data[key], classes, first[key], device, seed + len(name)
            )
        val_rank = gates["ranker"]["pseudo_unseen_10"]
        gate_b = bool(val_rank.get("gate_b_passed", False))
        gates["gate_b"] = {"passed": gate_b,
                           "criterion": "validation delta > 0.01 and paired CI lower bound > 0"}
        for name, key, classes in (("train_40", "train", TRAIN_CLASSES),
                                   ("pseudo_unseen_10", "pseudo", PSEUDO_CLASSES),
                                   ("seen_test_50", "seen", SEEN_CLASSES),
                                   ("true_unseen_5", "unseen", UNSEEN_CLASSES)):
            p = predictions.get(key, {}).get(selected_model)
            if p is None:
                x, _, _ = first_state(data[key], first[key])
                p = predict_classifier(selected_model, models[selected_model], x, device)
            paths, stop = adaptive_paths(data[key], first[key], p, selected_threshold,
                                         ranker if gate_b else None, device)
            adaptive[name] = evaluate_paths(data[key], classes, paths,
                                             "adaptive_explore_then_decide")
            adaptive[name]["stop_rate"] = float(stop.mean())
            adaptive[name]["ranker_used"] = gate_b
        if gate_b:
            q_info = train_sequential_dqn(data["train"], TRAIN_CLASSES, first["train"], device, seed + 700, output)
            for name, key, classes in (("pseudo_unseen_10", "pseudo", PSEUDO_CLASSES),
                                       ("seen_test_50", "seen", SEEN_CLASSES),
                                       ("true_unseen_5", "unseen", UNSEEN_CLASSES)):
                adaptive[name]["sequential_dqn"] = evaluate_sequential_dqn(
                    q_info, data[key], classes, first[key], device
                )
    else:
        gates["skip_reason"] = "Gate A failed; no remaining-view ranker or sequential RL trained."
        adaptive = {"status": "not_run_gate_a_failed"}
    split_summary = {}
    for name, value in data.items():
        info = geometry_arrays(value)
        split_summary[name] = {
            "episodes": int(len(value["episodes"])),
            "valid_view_count_histogram": {str(k): int((info["valid"].sum(1) == k).sum()) for k in range(1, NUM_VIEWS + 1)},
        }
    result = {
        "physical_start_view": start_view,
        "logical_view_permutation": data["train"]["view_permutation"],
        "start_availability_filter": {
            name: {"kept": int(value["start_filter_kept"]),
                   "removed": int(value["start_filter_removed"])}
            for name, value in data.items()
        },
        "split_summary": split_summary,
        "first_exploration_validation": pseudo_strategy,
        "selected_first_strategy": selected_strategy,
        "baselines": baselines,
        "stop_continue_predictor": {
            "feature_names": feature_names,
            "feature_dim": int(train_x.shape[1]),
            "label_definition": f"CONTINUE iff max remaining target-margin gain > {EPSILON_GAIN}",
            "train_oof": oof,
            "folds": folds,
            "splits": predictor_splits,
            "selected_model": selected_model,
            "selected_threshold": selected_threshold,
        },
        "gates": gates,
        "adaptive_reports": adaptive,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=Path("data/rl_candidate_rank_v6_new50_5/cache"))
    parser.add_argument("--output", type=Path, default=Path("work_dir/sequential_start_sweep_v1"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260920)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    raw_root = args.raw_root if args.raw_root.is_absolute() else root / args.raw_root
    output = args.output if args.output.is_absolute() else root / args.output
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(json.dumps({"SEQUENTIAL_START_SWEEP_START": True, "starts": [1, 2, 3],
                      "raw_root": str(raw_root), "output": str(output), "device": str(device),
                      "independent_training_per_start": True,
                      "future_inputs_to_policy": False}), flush=True)
    base_data = {
        "train": prepare_variable_data(raw_root, "dqn_train", TRAIN_CLASSES),
        "pseudo": prepare_variable_data(raw_root, "val", PSEUDO_CLASSES),
        "seen": prepare_variable_data(raw_root, "seen_test", SEEN_CLASSES),
        "unseen": prepare_variable_data(raw_root, "test", UNSEEN_CLASSES),
    }
    all_results = {}
    for start in (1, 2, 3):
        start_output = output / f"start_{start}"
        start_output.mkdir(parents=True, exist_ok=True)
        print(json.dumps({"START_VIEW_RUN": start}), flush=True)
        all_results[str(start)] = run_one_start(base_data, start, device, args.seed + start * 100, start_output)
        (start_output / "summary.json").write_text(
            json.dumps(json_safe(all_results[str(start)]), indent=2), encoding="utf-8"
        )
        print(json.dumps({"START_VIEW_COMPLETE": start,
                          "gate_a": all_results[str(start)]["gates"]["gate_a"],
                          "gate_b": all_results[str(start)]["gates"].get("gate_b")}, indent=2), flush=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    summary = {
        "version": "sequential_start_sweep_v1",
        "starts_tested": [1, 2, 3],
        "protocol": "each start independently trained; active module 40/10/5; frozen 50-seen VPOCLIP recognizer",
        "results": all_results,
    }
    (output / "summary.json").write_text(json.dumps(json_safe(summary), indent=2), encoding="utf-8")
    print(json.dumps({"SEQUENTIAL_START_SWEEP_COMPLETE": True, "output": str(output),
                      "starts": [1, 2, 3]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
