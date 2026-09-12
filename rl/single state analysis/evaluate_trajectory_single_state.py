"""Evaluate standalone-second-view trajectory policies on true unseen classes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
import sys

sys.path.insert(0, str(PROJECT_ROOT))

from rl import multistep_angle_object_trajectory_ablation as ablation  # noqa: E402
from rl import multistep_angle_object_trajectory_policy_v1 as base  # noqa: E402
from rl import multistep_angle_object_trajectory_policy_v4 as v4  # noqa: E402
from rl import multistep_angle_object_trajectory_policy_v5_no_category_id as v5  # noqa: E402
from rl import multistep_angle_object_trajectory_policy_v6_no_object as v6  # noqa: E402
from train_trajectory_single_state import (  # noqa: E402
    EVAL_SEEDS,
    NEW_TRUE_UNSEEN,
    OLD_TRUE_UNSEEN,
    configure_variant,
    evaluate_one_step,
    load_raw,
)


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def ci95(values: Sequence[float]) -> list[float]:
    values = [float(x) for x in values]
    if len(values) < 2:
        return [float(np.mean(values)), float(np.mean(values))]
    mean = float(np.mean(values))
    standard_error = float(np.std(values, ddof=1) / np.sqrt(len(values)))
    critical = 2.045229642132703 if len(values) >= 30 else 4.302652729911275
    return [mean - critical * standard_error, mean + critical * standard_error]


def evaluate_variant(
    variant: str,
    split: str,
    cache_root: Path,
    track_root: Path,
    policy_root: Path,
    output_root: Path,
    eval_seeds: Sequence[int],
) -> dict[str, Any]:
    model_cls, attach_tracks, state_builder = configure_variant(variant)
    device = torch.device("cuda:0")
    raw = load_raw(cache_root, track_root, "test", device, attach_tracks)
    classes = OLD_TRUE_UNSEEN if split == "old" else NEW_TRUE_UNSEEN
    bank = base.class_bank_tensor(classes, device)
    eligible = torch.isin(raw.labels, bank) & (raw.valid.sum(-1) >= 2)
    episodes = eligible.nonzero().flatten()
    expected = 414 if split == "old" else 271
    if split == "new":
        # The new raw test cache contains all classes; only the formal five
        # unseen classes enter the final evaluation.
        expected = int(len(episodes))
    if split == "old" and len(episodes) != expected:
        raise RuntimeError(f"old single-state test expected {expected}, got {len(episodes)}")

    by_train_seed: dict[str, Any] = {}
    for train_seed in (20260909, 20260910, 20260911):
        checkpoint = policy_root / f"seed_{train_seed}" / "best.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        model = model_cls().to(device)
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(saved["online"], strict=True)
        model.eval()
        rows: dict[str, Any] = {}
        for start in range(base.NUM_VIEWS):
            rows[f"fixed{start}"] = {}
            for eval_seed in eval_seeds:
                report = evaluate_one_step(
                    model, state_builder, raw, classes, f"fixed{start}", int(eval_seed + start * 1000)
                )
                rows[f"fixed{start}"][str(eval_seed)] = report["metrics"]
            # Fixed-start policy is deterministic; keep the first report and
            # average only the random baseline over the 30 seeds.
            fixed_rows = list(rows[f"fixed{start}"].values())
            first = fixed_rows[0]
            aggregate = {"episodes": int(episodes.numel())}
            metric_names = (
                "single_start", "policy_second_single", "policy_fused",
                "random_second_single", "random_fused", "oracle_second_single", "oracle_fused",
                "policy_second_minus_random_second", "policy_fused_minus_random_fused",
            )
            for name in metric_names:
                if name.startswith("random_") or name.endswith("minus_random_second") or name.endswith("minus_random_fused"):
                    values = [float(row[name]["top1"]) for row in fixed_rows]
                    aggregate[name] = {"top1": float(np.mean(values)), "std": float(np.std(values))}
                else:
                    aggregate[name] = first[name]
            rows[f"fixed{start}"]["aggregate"] = aggregate
        rows["random_start"] = {}
        for eval_seed in eval_seeds:
            report = evaluate_one_step(
                model, state_builder, raw, classes, "random_start", int(eval_seed)
            )
            rows["random_start"][str(eval_seed)] = report["metrics"]
        by_train_seed[str(train_seed)] = rows
        del model
        torch.cuda.empty_cache()

    fixed_aggregate: dict[str, Any] = {}
    for start in range(base.NUM_VIEWS):
        entries = [by_train_seed[str(seed)][f"fixed{start}"]["aggregate"] for seed in (20260909, 20260910, 20260911)]
        item: dict[str, Any] = {"episodes": entries[0]["episodes"]}
        for name in (
            "single_start", "policy_second_single", "policy_fused",
            "random_second_single", "random_fused", "oracle_second_single", "oracle_fused",
            "policy_second_minus_random_second", "policy_fused_minus_random_fused",
        ):
            values = [float(entry[name]["top1"]) for entry in entries]
            item[name] = {"mean": float(np.mean(values)), "std": float(np.std(values)), "ci95": ci95(values)}
        fixed_aggregate[f"fixed{start}"] = item

    random_aggregate: dict[str, Any] = {"episodes": int(episodes.numel())}
    for name in (
        "single_start", "policy_second_single", "policy_fused",
        "random_second_single", "random_fused", "oracle_second_single", "oracle_fused",
        "policy_second_minus_random_second", "policy_fused_minus_random_fused",
    ):
        values = []
        for eval_seed in eval_seeds:
            rows = [by_train_seed[str(train_seed)]["random_start"][str(eval_seed)] for train_seed in (20260909, 20260910, 20260911)]
            values.append(float(np.mean([row[name]["top1"] for row in rows])))
        random_aggregate[name] = {"mean": float(np.mean(values)), "std": float(np.std(values)), "ci95": ci95(values)}

    result = {
        "experiment": "single_state_trajectory",
        "split": split,
        "variant": variant,
        "class_bank": classes,
        "episodes": int(episodes.numel()),
        "train_seeds": [20260909, 20260910, 20260911],
        "eval_seeds": [int(x) for x in eval_seeds],
        "target": "standalone candidate-view Top-1 correctness",
        "fixed_start_aggregate": fixed_aggregate,
        "random_start_aggregate": random_aggregate,
        "by_train_seed": by_train_seed,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    dump(output_root / "summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("v1", "v4", "v5", "v6", "object_only", "geometry_only"), required=True)
    parser.add_argument("--split", choices=("old", "new"), required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--track-root", type=Path, required=True)
    parser.add_argument("--policy-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=EVAL_SEEDS)
    args = parser.parse_args()
    result = evaluate_variant(
        args.variant, args.split, args.cache_root, args.track_root,
        args.policy_root, args.output_root, args.eval_seeds,
    )
    print("SINGLE_STATE_TRAJECTORY_EVALUATION_COMPLETE", args.split, args.variant, flush=True)
    print(json.dumps({"fixed": result["fixed_start_aggregate"], "random_start": result["random_start_aggregate"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
