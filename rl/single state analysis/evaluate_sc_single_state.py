"""Evaluate single-state SC-NBV v9--v24 checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
import sys

sys.path.insert(0, str(PROJECT_ROOT))

from rl import sc_nbv  # noqa: E402
from train_sc_single_state import (  # noqa: E402
    NEW_CONFIG,
    NEW_RAW_CACHE,
    NEW_SC_CACHE,
    old_config,
)


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def ci95(values: Sequence[float]) -> list[float]:
    values = [float(x) for x in values]
    mean = float(np.mean(values))
    if len(values) < 2:
        return [mean, mean]
    critical = 2.045229642132703 if len(values) >= 30 else 4.302652729911275
    half = critical * float(np.std(values, ddof=1) / np.sqrt(len(values)))
    return [mean - half, mean + half]


def manifest(version: int, split: str) -> tuple[dict[str, Any], str, Path, Path]:
    source = old_config(version)
    cfg = yaml.safe_load(source.read_text(encoding="utf-8"))
    source_summary = Path(cfg["output_root"]) / "summary.json"
    selected = json.loads(source_summary.read_text(encoding="utf-8"))["selected_variant"]
    variant_cfg = cfg["train"]["variants"][selected]
    if split == "new":
        new_cfg = yaml.safe_load(NEW_CONFIG.read_text(encoding="utf-8"))
        class_split = new_cfg["split"]
        sc_root, raw_root = NEW_SC_CACHE, NEW_RAW_CACHE
    else:
        class_split = cfg["split"]
        sc_root, raw_root = Path(cfg["sc_cache"]), Path(cfg["raw_cache"])
    return {
        "version": version,
        "variant": selected,
        "variant_config": variant_cfg,
        "class_split": class_split,
        "sc_root": sc_root,
        "raw_root": raw_root,
        "source_config": source,
    }, selected, sc_root, raw_root


@torch.inference_mode()
def evaluate_model(
    model: torch.nn.Module,
    context: sc_nbv.SCNBVData,
    raw: sc_nbv.RawViewData,
    classes: list[int],
    protocol: str,
    seed: int,
) -> dict[str, Any]:
    bank = torch.as_tensor(sorted(classes), device=raw.device, dtype=torch.long)
    eligible = torch.isin(raw.labels, bank) & (raw.valid.sum(-1) >= 2)
    episodes = eligible.nonzero().flatten()
    rng = np.random.default_rng(seed)
    starts = []
    for row in raw.valid[episodes].cpu().numpy():
        choices = np.flatnonzero(row)
        starts.append(int(0 if protocol == "fixed0" and row[0] else (rng.choice(choices) if protocol == "random_start" else choices[0])))
    starts = torch.as_tensor(starts, device=raw.device, dtype=torch.long)
    rows = context.rows_for(episodes, starts)
    batch = context.batch(rows)
    valid = raw.valid[episodes] & raw.reachable[episodes, starts]
    valid[torch.arange(len(episodes), device=raw.device), starts] = False
    keep = valid.any(-1)
    episodes, starts, valid = episodes[keep], starts[keep], valid[keep]
    batch = {key: value[keep] for key, value in batch.items()}
    q = model(sc_nbv.causal_batch(batch))
    policy = q.masked_fill(~valid, -torch.inf).argmax(-1)
    random_action = torch.empty_like(policy)
    for row, choices in enumerate(valid.cpu().numpy()):
        random_action[row] = int(rng.choice(np.flatnonzero(choices)))
    local = torch.searchsorted(bank, raw.labels[episodes])
    candidate_scores = raw.logits[episodes].index_select(-1, bank)
    correctness = candidate_scores.argmax(-1).eq(local[:, None]) & valid
    p_second = correctness.gather(1, policy[:, None]).squeeze(1)
    r_second = correctness.gather(1, random_action[:, None]).squeeze(1)
    rows = torch.arange(len(episodes), device=raw.device)
    s_start = candidate_scores[rows, starts].argmax(-1).eq(local)
    p_fused = (0.5 * (raw.logits[episodes, starts] + raw.logits[episodes, policy])).index_select(-1, bank).argmax(-1).eq(local)
    r_fused = (0.5 * (raw.logits[episodes, starts] + raw.logits[episodes, random_action])).index_select(-1, bank).argmax(-1).eq(local)
    oracle_action = correctness.float().argmax(-1)
    o_second = correctness.gather(1, oracle_action[:, None]).squeeze(1)
    o_fused = (0.5 * (raw.logits[episodes, starts] + raw.logits[episodes, oracle_action])).index_select(-1, bank).argmax(-1).eq(local)

    def top1(x: torch.Tensor) -> dict[str, float]:
        return {"top1": float(x.float().mean())}

    pm, rm, pf, rf = top1(p_second), top1(r_second), top1(p_fused), top1(r_fused)
    return {
        "protocol": protocol,
        "episodes": int(len(episodes)),
        "metrics": {
            "single_start": top1(s_start),
            "policy_second_single": pm,
            "policy_fused": pf,
            "random_second_single": rm,
            "random_fused": rf,
            "oracle_second_single": top1(o_second),
            "oracle_fused": top1(o_fused),
            "policy_second_minus_random_second": {"top1": pm["top1"] - rm["top1"]},
            "policy_fused_minus_random_fused": {"top1": pf["top1"] - rf["top1"]},
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=int, choices=list(range(9, 25)), required=True)
    parser.add_argument("--split", choices=("old", "new"), required=True)
    parser.add_argument("--policy-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=list(range(20260920, 20260950)))
    args = parser.parse_args()
    device = torch.device("cuda:0")
    info, variant, sc_root, raw_root = manifest(args.version, args.split)
    context = sc_nbv.SCNBVData(sc_root / "unseen", device)
    raw = sc_nbv.RawViewData(raw_root / "test", device)
    classes = info["class_split"]["unseen_classes"]
    by_seed = {}
    for train_seed in (20260909, 20260910, 20260911):
        checkpoint = args.policy_root / variant / f"seed_{train_seed}" / "best.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        model = sc_nbv.make_selector_model(info["variant_config"]).to(device)
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(saved["model"], strict=True)
        model.eval()
        rows = {"fixed0": {}, "random_start": {}}
        for protocol in rows:
            for eval_seed in args.eval_seeds:
                rows[protocol][str(eval_seed)] = evaluate_model(
                    model, context, raw, list(classes), protocol, int(eval_seed)
                )["metrics"]
        by_seed[str(train_seed)] = rows
        del model
        torch.cuda.empty_cache()

    aggregates = {}
    metric_names = (
        "single_start", "policy_second_single", "policy_fused",
        "random_second_single", "random_fused", "oracle_second_single", "oracle_fused",
        "policy_second_minus_random_second", "policy_fused_minus_random_fused",
    )
    for protocol in ("fixed0", "random_start"):
        aggregates[protocol] = {}
        for name in metric_names:
            values = []
            for eval_seed in args.eval_seeds:
                values.append(float(np.mean([
                    by_seed[str(train_seed)][protocol][str(eval_seed)][name]["top1"]
                    for train_seed in (20260909, 20260910, 20260911)
                ])))
            aggregates[protocol][name] = {"mean": float(np.mean(values)), "std": float(np.std(values)), "ci95": ci95(values)}
    result = {
        "experiment": "single_state_sc_nbv",
        "version": args.version,
        "split": args.split,
        "variant": variant,
        "class_bank": list(classes),
        "episodes": int((torch.isin(raw.labels, torch.as_tensor(classes, device=device)) & (raw.valid.sum(-1) >= 2)).sum()),
        "train_seeds": [20260909, 20260910, 20260911],
        "eval_seeds": [int(x) for x in args.eval_seeds],
        "target": "standalone candidate-view Top-1 correctness",
        "aggregates": aggregates,
        "by_train_seed": by_seed,
    }
    dump(args.output_root / "summary.json", result)
    print("SINGLE_STATE_SC_EVALUATION_COMPLETE", args.split, args.version, variant, flush=True)
    print(json.dumps(aggregates, indent=2), flush=True)


if __name__ == "__main__":
    main()
