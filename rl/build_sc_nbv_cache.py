"""Build causal SC-NBV context/utility caches from the frozen RL cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from .sc_nbv import build_split_cache


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="logs/rl_candidate_rank_v6_new50_5/config.yaml")
    parser.add_argument("--raw-root", default=None)
    parser.add_argument("--output", default="data/sc_nbv_current_v1/cache")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--semantic-temperature", type=float, default=0.07)
    parser.add_argument("--tie-threshold", type=float, default=0.02)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    # New SC-NBV handoff configs use ``raw_cache``; accepting the legacy
    # ``cache.root`` form keeps this builder usable with old RL configs too.
    if args.raw_root:
        raw_root = Path(args.raw_root)
    elif "raw_cache" in cfg:
        raw_root = Path(cfg["raw_cache"])
    else:
        raw_root = Path(cfg["cache"]["root"])
    if not raw_root.is_absolute():
        raw_root = (cfg_path.parent / raw_root).resolve()
    output = Path(args.output)
    if not output.is_absolute():
        output = root / output
    split_cfg = cfg.get("split", {})
    evaluation_cfg = cfg["evaluation"]
    seen = list(split_cfg.get("seen_classes", evaluation_cfg.get("seen_class_ids", [])))
    train_classes = list(split_cfg.get("train_classes", cfg.get("v6", {}).get("policy_train_classes", [])))
    pseudo_classes = list(split_cfg.get("pseudo_unseen_classes", cfg.get("v6", {}).get("policy_holdout_classes", [])))
    unseen_classes = list(split_cfg.get("unseen_classes", evaluation_cfg.get("unseen_class_ids", [])))
    if not train_classes or not pseudo_classes or not seen or not unseen_classes:
        raise ValueError("config must define train/pseudo-unseen/seen/unseen class lists")
    for left_name, left, right_name, right in (
        ("train", train_classes, "pseudo_unseen", pseudo_classes),
        ("seen", seen, "unseen", unseen_classes),
        ("train", train_classes, "unseen", unseen_classes),
        ("pseudo_unseen", pseudo_classes, "unseen", unseen_classes),
    ):
        overlap = sorted(set(left) & set(right))
        if overlap:
            raise ValueError(f"class split leakage: {left_name}/{right_name}: {overlap}")
    if not (set(train_classes) | set(pseudo_classes)).issubset(set(seen)):
        raise ValueError("policy train and pseudo-unseen classes must belong to seen classes")
    splits = {
        "train": ("dqn_train", train_classes, train_classes),
        "pseudo_unseen": ("val", pseudo_classes, pseudo_classes),
        "seen_test": ("seen_test", seen, seen),
        "unseen": ("test", unseen_classes, unseen_classes),
    }
    summary = {
        "format": "sc_nbv_cache_bundle_v1",
        "raw_root": str(raw_root),
        "output_root": str(output),
        "recognizer_checkpoint": cfg.get("recognizer", {}).get("checkpoint"),
        "splits": {},
    }
    train_task_bank_size = cfg.get("data", {}).get("train_task_bank_size")
    train_task_seed = int(cfg.get("data", {}).get("train_task_seed", 20260909))
    top_prototype_count = int(cfg.get("data", {}).get("top_prototype_count", 2))
    if top_prototype_count < 1 or top_prototype_count > int(args.top_k):
        raise ValueError("data.top_prototype_count must be between 1 and --top-k")
    semantic_bank_cfg = cfg.get("data", {}).get("semantic_class_bank")
    if semantic_bank_cfg is None:
        semantic_bank = None
    elif isinstance(semantic_bank_cfg, str) and semantic_bank_cfg.lower() == "all":
        semantic_bank = list(range(55))
    else:
        semantic_bank = [int(value) for value in semantic_bank_cfg]
    if semantic_bank is not None and not semantic_bank:
        raise ValueError("data.semantic_class_bank must not be empty")
    utility_bank_cfg = cfg.get("data", {}).get("utility_class_bank")
    angle_quality_csv = cfg.get("data", {}).get("angle_quality_csv")
    if angle_quality_csv is not None:
        angle_quality_csv = Path(angle_quality_csv)
        if not angle_quality_csv.is_absolute():
            angle_quality_csv = (cfg_path.parent / angle_quality_csv).resolve()

    def resolve_utility_bank(spec, split_name, allowed_labels):
        """Resolve an optional target-only class bank without changing state inputs."""
        if spec is None:
            return None
        if isinstance(spec, dict):
            spec = spec.get(split_name, spec.get("default"))
            if spec is None:
                return None
        if isinstance(spec, str):
            key = spec.lower()
            if key == "all":
                return list(range(55))
            if key in {"allowed", split_name}:
                return list(allowed_labels)
            if key == "train":
                return list(train_classes)
            if key in {"pseudo", "pseudo_unseen"}:
                return list(pseudo_classes)
            if key in {"seen", "seen_test"}:
                return list(seen)
            if key == "unseen":
                return list(unseen_classes)
            raise ValueError(f"unknown data.utility_class_bank value: {spec}")
        return [int(value) for value in spec]
    for name, (raw_split, bank, allowed) in splits.items():
        print("SC_NBV_CACHE_START", name, flush=True)
        utility_bank = resolve_utility_bank(utility_bank_cfg, name, allowed)
        raw_split_path = raw_root / raw_split
        # A raw test cache can contain both seen and unseen labels.  Reusing
        # it for the filtered seen-test context avoids duplicating all feature
        # arrays into a second directory while preserving the label filter
        # passed to build_split_cache.
        if not raw_split_path.exists() and raw_split == "seen_test":
            raw_split_path = raw_root / "test"
        metadata = build_split_cache(
            raw_split_path,
            output / name,
            name,
            semantic_bank if semantic_bank is not None else bank,
            allowed_labels=allowed,
            top_k=args.top_k,
            semantic_temperature=args.semantic_temperature,
            tie_threshold=args.tie_threshold,
            task_bank_size=(int(train_task_bank_size) if name == "train" and train_task_bank_size else None),
            task_seed=train_task_seed,
            utility_class_bank=utility_bank,
            top_prototype_count=top_prototype_count,
            angle_quality_csv=angle_quality_csv,
        )
        summary["splits"][name] = metadata
        print("SC_NBV_CACHE_COMPLETE", json.dumps(metadata), flush=True)
    (output / "bundle_metadata.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
