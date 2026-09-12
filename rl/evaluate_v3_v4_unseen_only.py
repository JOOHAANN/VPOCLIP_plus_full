"""Evaluate completed v3/v4 checkpoints on the strict five-way unseen test only."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import torch

from . import multistep_angle_object_trajectory_policy_v3 as v3
from . import multistep_angle_object_trajectory_policy_v4 as v4
from .multistep_angle_object_trajectory_policy_v1 import (
    attach_object_tracks as base_attach_object_tracks,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEEDS = (20260909, 20260910)
PROTOCOLS = ("fixed0", "fixed1", "fixed2", "fixed3", "random_start")


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def evaluate_variant(
    module: Any,
    name: str,
    seeds: list[int],
    output_root: Path,
    unseen_classes: list[int] | None,
) -> None:
    v1 = module.v1
    v1.CACHE_ROOT = module.CACHE_ROOT
    v1.TRACK_ROOT = module.TRACK_ROOT
    v1.attach_object_tracks = (
        module.attach_object_tracks_v3
        if name == "v3"
        else base_attach_object_tracks
    )
    v1.build_trajectory_state = (
        module.build_trajectory_state_v3
        if name == "v3"
        else module.build_trajectory_state_v4
    )
    v1.AngleObjectTrajectoryPolicy = (
        module.ObjectSizeTrajectoryPolicy
        if name == "v3"
        else module.BodyOrientationTrajectoryPolicy
    )

    device = torch.device("cuda:0")
    raw: dict[str, Any] = {}
    for split in ("test",):
        raw[split] = v1.GPUCache(module.CACHE_ROOT / split, device)
        v1.attach_view_valid(raw[split], module.CACHE_ROOT, split)
        if name == "v3":
            module.attach_object_tracks_v3(raw[split], module.TRACK_ROOT, split)
        else:
            base_attach_object_tracks(raw[split], module.TRACK_ROOT, split)

    classes = sorted(set(unseen_classes or v1.TRUE_UNSEEN_CLASSES))
    result: dict[str, Any] = {
        "variant": name,
        "split": "test",
        "protocol": "true_unseen_5way_only",
        "class_bank": [int(x) for x in classes],
        "seeds": {},
    }
    for seed in seeds:
        checkpoint = module.OUTPUT_ROOT / f"seed_{seed}" / "best.pt"
        if not checkpoint.exists():
            result["seeds"][str(seed)] = {"status": "not_ready", "checkpoint": str(checkpoint)}
            continue
        model = v1.AngleObjectTrajectoryPolicy().to(device)
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(saved["online"], strict=True)
        model.eval()
        seed_result: dict[str, Any] = {"status": "ok", "checkpoint": str(checkpoint)}
        with torch.inference_mode():
            for protocol in PROTOCOLS:
                report = v1.evaluate_protocol(
                    model, raw["test"], classes, protocol, 20260910 + len(protocol)
                )
                seed_result[protocol] = report["metrics"]
        result["seeds"][str(seed)] = seed_result
        del model, saved
        torch.cuda.empty_cache()
    dump(output_root / name / "unseen_only.json", result)
    print("V3_V4_UNSEEN_ONLY", json.dumps(result, ensure_ascii=False), flush=True)
    del raw
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--variants", choices=("v3", "v4"), nargs="+", default=["v3", "v4"])
    parser.add_argument("--unseen-classes", type=int, nargs="+", default=None)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "work_dir" / "multistep_v3_v4_unseen_only_interim",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.set_float32_matmul_precision("high")
    modules = {"v3": v3, "v4": v4}
    for name in args.variants:
        evaluate_variant(modules[name], name, args.seeds, args.output_root, args.unseen_classes)


if __name__ == "__main__":
    main()
