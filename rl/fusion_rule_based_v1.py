"""Experiment B: fixed rule-based weighted logit fusion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from . import fusion_lightweight_gating_v1 as gate_experiment
from . import fusion_weighted_common_v1 as common


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gate-root", type=Path, required=True)
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=common.EVAL_SEEDS)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda:0")
    gate_summary = json.loads((args.gate_root / "summary.json").read_text()) if (args.gate_root / "summary.json").exists() else None
    if gate_summary is None:
        raise FileNotFoundError(args.gate_root / "summary.json")
    checkpoint = Path(gate_summary["selected_checkpoint"])
    gate = gate_experiment.load_gate(checkpoint, device)
    metadata = {
        "experiment": "fusion_rule_based_v1",
        "cache_root": str(args.cache_root),
        "gate_checkpoint_for_common_comparison": str(checkpoint),
        "eval_seeds": args.eval_seeds,
        "methods": ["equal_mean", "lightweight_gating", "rule_cls", "rule_geo", "rule_combined"],
        "rule_weights": {
            "rule_cls": {"entropy": 0.5, "margin": 0.5},
            "rule_geo": {"visibility": 0.5, "orientation": 0.5},
            "rule_combined": {"entropy": 0.30, "margin": 0.30, "visibility": 0.20, "orientation": 0.20},
        },
        "orientation_rule_degrees": {"front": 0.8, "30_to_90": 1.0, "150": 0.4, "back": 0.2},
        "orientation_source": "sample-specific view_geometry azimuth; angle=acos(cos_azimuth)",
        "visibility_source": "13-frame cached pose joint-presence ratio",
        "no_true_unseen_tuning": True,
        "recognizer_frozen": True,
        "anchor_preserved": True,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    common.dump(args.output_root / "config_resolved.json", metadata)
    results = common.evaluate_all(args.cache_root, args.output_root, gate, args.eval_seeds)
    common.dump(args.output_root / "summary.json", {**metadata, "results": results})
    print("RULE_BASED_EXPERIMENT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
