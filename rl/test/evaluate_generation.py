"""Evaluate every seed of an isolated iterative RL generation.

This is deliberately a pure policy evaluation: the policy always emits the
argmax candidate action.  Random/fixed/oracle are reported only as baselines;
there is no gate or fallback in this script.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from ..causal_rank_v6 import evaluate
from ..feature_ablation_diagnostic import FeatureRanker
from ..train_rl_fast import GPUCache


ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("generation", default="generation_02")
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / "logs/rl_candidate_rank_v6_new50_5/config.yaml").read_text())
    device = torch.device(cfg["runtime"]["device"])
    root = Path(cfg["cache"]["root"])
    out = ROOT / "work_dir/active_view_iterative_test" / args.generation
    cache = GPUCache(root / "test", device)
    present = torch.from_numpy(np.load(root / "test/view_valid.npy")).to(device)
    classes = cfg["evaluation"]["unseen_class_ids"]
    baselines = {"fixed0": 0.5314009785652161, "random": 0.5386473536491394}
    protocols = ("fixed0",) if args.generation in ("generation_15", "generation_16", "generation_17", "generation_18") else ("fixed0", "random")
    result = {"generation": args.generation, "primary_protocol": "fixed0",
              "baseline_random": baselines, "variants": {}}
    for variant_dir in sorted(p for p in out.iterdir() if p.is_dir() and p.name.startswith("g")):
        rows = []
        for seed_dir in sorted(variant_dir.glob("seed_*")):
            checkpoint = seed_dir / "best.pt"
            if not checkpoint.exists():
                continue
            if args.generation == "generation_03":
                from .generation_03 import make_model
                net = make_model(variant_dir.name).to(device)
            elif args.generation in ("generation_04", "generation_05"):
                from .generation_04 import LowCapacityPolicy
                net = LowCapacityPolicy(variant_dir.name).to(device)
            elif args.generation in ("generation_06", "generation_07", "generation_08", "generation_09", "generation_10", "generation_11", "generation_12", "generation_13", "generation_14", "generation_15", "generation_16", "generation_17", "generation_18"):
                if args.generation == "generation_09":
                    from .generation_09 import SetPolicy
                    net = SetPolicy(variant_dir.name).to(device)
                elif args.generation in ("generation_10", "generation_11"):
                    from .generation_10 import TransitionPolicy
                    net = TransitionPolicy(variant_dir.name).to(device)
                elif args.generation in ("generation_12", "generation_13"):
                    from .generation_12 import SkeletonTransitionPolicy
                    net = SkeletonTransitionPolicy(variant_dir.name).to(device)
                elif args.generation == "generation_14":
                    from .generation_14 import ExpectedGainPolicy
                    net = ExpectedGainPolicy(variant_dir.name).to(device)
                elif args.generation == "generation_15":
                    from .generation_15 import DirectionalPolicy
                    net = DirectionalPolicy(variant_dir.name).to(device)
                elif args.generation == "generation_16":
                    from .generation_16 import CanonicalSkeletonPolicy
                    net = CanonicalSkeletonPolicy(variant_dir.name).to(device)
                elif args.generation == "generation_17":
                    from .generation_17 import TemporalSkeletonPolicy
                    net = TemporalSkeletonPolicy(variant_dir.name).to(device)
                elif args.generation == "generation_18":
                    from .generation_18 import FactorizedSkeletonPolicy
                    net = FactorizedSkeletonPolicy(variant_dir.name).to(device)
                else:
                    from .generation_06 import SemanticPolicy
                    net = SemanticPolicy(variant_dir.name).to(device)
            else:
                net = FeatureRanker(True, False).to(device)
            saved = torch.load(checkpoint, map_location=device, weights_only=False)
            net.load_state_dict(saved["online"])
            net.eval()
            row = {"seed": int(seed_dir.name.split("_")[-1]), "checkpoint": str(checkpoint)}
            for protocol in protocols:
                report = evaluate(net, cache, present, classes, protocol)
                row[protocol] = {
                    "policy": report["metrics"]["policy"]["top1"],
                    "random": report["metrics"]["random"]["top1"],
                    "random_exact": report["metrics"]["random_exact"]["top1"],
                    "oracle": report["metrics"]["oracle"]["top1"],
                    "delta_vs_fixed_baseline": report["metrics"]["policy"]["top1"] - baselines["fixed0"],
                    "delta_vs_random_baseline": report["metrics"]["policy"]["top1"] - baselines["random"],
                }
            rows.append(row)
            del net
            torch.cuda.empty_cache()
        result["variants"][variant_dir.name] = rows
    path = out / "per_seed_test.json"
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"WROTE {path}")


if __name__ == "__main__":
    main()
