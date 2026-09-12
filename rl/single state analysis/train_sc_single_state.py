"""Train canonical SC-NBV v9--v24 architectures with a single-view target.

The existing SC-NBV trainer is reused only as an optimizer/model harness. Its
source and all old outputs remain unchanged. This wrapper replaces the
privileged offline utility with standalone candidate-view correctness and
replaces validation with the same standalone-second-view metric.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
import sys

sys.path.insert(0, str(PROJECT_ROOT))

from rl import sc_nbv  # noqa: E402
from rl import train_sc_nbv as trainer  # noqa: E402


OLD_CONFIG_ROOT = PROJECT_ROOT / "rl" / "handoff_configs"
NEW_CONFIG = OLD_CONFIG_ROOT / "sc_nbv_new50_5_group1_stage_b_sensor.yaml"
NEW_SC_CACHE = PROJECT_ROOT / "data" / "sc_nbv_new50_5_group1_stage_b_sensor" / "cache"
NEW_RAW_CACHE = PROJECT_ROOT / "data" / "rl_raw_new50_5_group1_stage_b" / "cache"


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def old_config(version: int) -> Path:
    matches = sorted(OLD_CONFIG_ROOT.glob(f"sc_nbv_current_v{version}*.yaml"))
    if not matches:
        raise FileNotFoundError(f"no config for v{version}")
    return matches[0]


_ORIGINAL_UTILITY = trainer._utility_from_task_mask


def standalone_utility(
    current_logits: torch.Tensor,
    future_logits: torch.Tensor,
    labels: torch.Tensor,
    valid: torch.Tensor,
    class_mask: torch.Tensor,
    mode: str,
    temperature: float = 2.0,
) -> torch.Tensor:
    """Return binary standalone candidate correctness for the new target."""

    if mode != "single_correctness":
        return _ORIGINAL_UTILITY(
            current_logits, future_logits, labels, valid, class_mask, mode,
            temperature=temperature,
        )
    # static_utility_mode uses one fixed class pool for the batch. The target
    # intentionally ignores current_logits and never computes a fusion.
    bank = class_mask[0].nonzero().flatten()
    scores = future_logits.index_select(-1, bank)
    local = torch.searchsorted(bank, labels)
    correct = scores.argmax(-1).eq(local[:, None]).float()
    return correct.masked_fill(~valid, 0.0)


@torch.inference_mode()
def evaluate_single_state(
    model: torch.nn.Module,
    contexts: sc_nbv.SCNBVData,
    raw: sc_nbv.RawViewData,
    class_bank: list[int],
    protocol: str = "fixed0",
    seed: int = 20260909,
    batch_size: int = 4096,
) -> dict[str, Any]:
    """Evaluate standalone second view and first+second fusion."""

    device = raw.device
    bank = torch.as_tensor(sorted(class_bank), device=device, dtype=torch.long)
    eligible = torch.isin(raw.labels, bank) & (raw.valid.sum(-1) >= 2)
    episode_ids = eligible.nonzero().flatten()
    if not len(episode_ids):
        raise ValueError("no episodes in requested class bank")
    rng = __import__("numpy").random.default_rng(seed)
    starts = []
    for row in raw.valid[episode_ids].cpu().numpy():
        choices = __import__("numpy").flatnonzero(row)
        if protocol == "fixed0" and row[0]:
            starts.append(0)
        elif protocol == "random_start":
            starts.append(int(rng.choice(choices)))
        else:
            starts.append(int(choices[0]))
    starts = torch.as_tensor(starts, device=device, dtype=torch.long)
    rows = contexts.rows_for(episode_ids, starts)
    context_batch = contexts.batch(rows)
    valid = raw.valid[episode_ids] & raw.reachable[episode_ids, starts]
    valid[torch.arange(len(episode_ids), device=device), starts] = False
    keep = valid.any(-1)
    episode_ids, starts, valid = episode_ids[keep], starts[keep], valid[keep]
    context_batch = {key: value[keep] for key, value in context_batch.items()}
    n = len(episode_ids)
    q_parts = []
    model.eval()
    for begin in range(0, n, batch_size):
        batch = {key: value[begin:begin + batch_size] for key, value in context_batch.items()}
        q_parts.append(model(sc_nbv.causal_batch(batch)))
    policy_q = torch.cat(q_parts, dim=0)
    policy_action = policy_q.masked_fill(~valid, -torch.inf).argmax(-1)
    random_action = torch.empty_like(policy_action)
    np = __import__("numpy")
    for row, choices in enumerate(valid.cpu().numpy()):
        random_action[row] = int(rng.choice(np.flatnonzero(choices)))

    local = torch.searchsorted(bank, raw.labels[episode_ids])
    candidate_scores = raw.logits[episode_ids].index_select(-1, bank)
    candidate_correct = candidate_scores.argmax(-1).eq(local[:, None]) & valid
    policy_second = candidate_correct.gather(1, policy_action[:, None]).squeeze(1)
    random_second = candidate_correct.gather(1, random_action[:, None]).squeeze(1)
    rows = torch.arange(len(episode_ids), device=device)
    start_correct = candidate_scores[rows, starts].argmax(-1).eq(local)
    fused_policy_scores = 0.5 * (raw.logits[episode_ids, starts] + raw.logits[episode_ids, policy_action])
    fused_random_scores = 0.5 * (raw.logits[episode_ids, starts] + raw.logits[episode_ids, random_action])
    fused_policy = fused_policy_scores.index_select(-1, bank).argmax(-1).eq(local)
    fused_random = fused_random_scores.index_select(-1, bank).argmax(-1).eq(local)
    oracle_action = candidate_correct.float().argmax(-1)
    oracle_single = candidate_correct.gather(1, oracle_action[:, None]).squeeze(1)
    oracle_fused_scores = 0.5 * (raw.logits[episode_ids, starts] + raw.logits[episode_ids, oracle_action])
    oracle_fused = oracle_fused_scores.index_select(-1, bank).argmax(-1).eq(local)

    def top1(value: torch.Tensor) -> dict[str, float]:
        return {"top1": float(value.float().mean())}

    policy_value, random_value = top1(policy_second), top1(random_second)
    policy_fused_value, random_fused_value = top1(fused_policy), top1(fused_random)
    metrics = {
        "single_start": top1(start_correct),
        "policy_second_single": policy_value,
        "policy_fused": policy_fused_value,
        "random_second_single": random_value,
        "random_fused": random_fused_value,
        "oracle_second_single": top1(oracle_single),
        "oracle_fused": top1(oracle_fused),
        "policy_second_minus_random_second": {"top1": policy_value["top1"] - random_value["top1"]},
        "policy_fused_minus_random_fused": {"top1": policy_fused_value["top1"] - random_fused_value["top1"]},
    }
    # Compatibility fields expected by the preserved train_sc_nbv train loop.
    metrics["sc_nbv"] = {
        "top1": policy_value["top1"],
        "mean_margin_gain": 0.0,
        "mean_oracle_regret": float((candidate_correct.float().sum(-1) - policy_second.float()).mean()),
        "oracle_hit_top1": float(policy_second.float().mean()),
    }
    metrics["random_expected"] = {"top1": random_value["top1"], "mean_margin_gain": 0.0}
    return {
        "protocol": protocol,
        "episodes": int(n),
        "class_bank": [int(x) for x in class_bank],
        "candidate_count_mean": float(valid.sum(-1).float().mean()),
        "metrics": metrics,
    }


def prepare_config(version: int, split: str, output_root: Path) -> tuple[dict[str, Any], Path, str, dict[str, Any]]:
    source_path = old_config(version)
    cfg = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    source_summary = Path(cfg["output_root"]) / "summary.json"
    if not source_summary.exists():
        raise FileNotFoundError(source_summary)
    selected = json.loads(source_summary.read_text(encoding="utf-8"))["selected_variant"]
    variant_config = copy.deepcopy(cfg["train"]["variants"][selected])
    if split == "new":
        new_cfg = yaml.safe_load(NEW_CONFIG.read_text(encoding="utf-8"))
        cfg["split"] = copy.deepcopy(new_cfg["split"])
        cfg["raw_cache"] = str(NEW_RAW_CACHE)
        cfg["sc_cache"] = str(NEW_SC_CACHE)
    cfg["output_root"] = str(output_root)
    cfg["_config_path"] = str(source_path)
    cfg.setdefault("train", {})["validation_metric"] = "top1_gain"
    cfg["train"]["validation_protocol"] = "fixed0"
    original_start_views = variant_config.get("training_start_views")
    variant_config["training_start_views"] = None
    variant_config["static_utility_mode"] = "single_correctness"
    variant_config["dynamic_task_banks"] = False
    variant_config["resample_task_banks"] = False
    variant_config["auxiliary_transition"] = False
    variant_config["oracle_ce_weight"] = 0.0
    variant_config["single_state_original_training_start_views"] = original_start_views
    return cfg, source_path, str(selected), variant_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=int, choices=list(range(9, 25)), required=True)
    parser.add_argument("--split", choices=("old", "new"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--batches-per-epoch", type=int, default=48)
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260909, 20260910, 20260911])
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("single-state SC-NBV training requires CUDA")
    device = torch.device("cuda:0")
    cfg, source_path, variant, variant_config = prepare_config(args.version, args.split, args.output_root)
    cfg["train"]["epochs"] = args.epochs
    cfg["train"]["batch_size"] = args.batch_size
    cfg["train"]["batches_per_epoch"] = args.batches_per_epoch
    cfg["runtime"]["seeds"] = [int(x) for x in args.seeds]
    cfg["single_state_target"] = "standalone candidate-view Top-1 correctness"
    cfg["single_state_source_config"] = str(source_path)
    cfg["single_state_variant"] = variant
    dump(args.output_root / "config_resolved.json", cfg)

    # Process-local monkey patches; source code and old runs are untouched.
    trainer._utility_from_task_mask = standalone_utility
    trainer.evaluate_policy = evaluate_single_state

    cache_root = Path(cfg["sc_cache"])
    raw_root = Path(cfg["raw_cache"])
    train = sc_nbv.SCNBVData(cache_root / "train", device)
    pseudo = sc_nbv.SCNBVData(cache_root / "pseudo_unseen", device)
    raw_pseudo = sc_nbv.RawViewData(raw_root / "val", device)
    output_variant = args.output_root / variant
    output_variant.mkdir(parents=True, exist_ok=True)
    records = []
    for seed in args.seeds:
        print("SINGLE_STATE_SC_TRAIN_START", args.split, args.version, variant, seed, flush=True)
        records.append(
            trainer.train_seed(
                variant, variant_config, train, pseudo, raw_pseudo, cfg,
                int(seed), output_variant, device,
            )
        )
    dump(
        output_variant / "selection.json",
        {"variant": variant, "target": "standalone candidate-view Top-1 correctness", "seeds": records, "test_was_not_used_for_selection": True},
    )
    dump(
        args.output_root / "complete.json",
        {"experiment": "single_state_sc_nbv", "version": args.version, "split": args.split, "variant": variant, "source_config": str(source_path), "target": "standalone candidate-view Top-1 correctness", "runs": records},
    )
    print("SINGLE_STATE_SC_TRAIN_COMPLETE", args.split, args.version, variant, flush=True)


if __name__ == "__main__":
    main()
