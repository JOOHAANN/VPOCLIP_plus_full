"""Evaluate the canonical SC-NBV v9--v24 models on the historical split.

Each generation is represented by the variant selected by its own validation
``summary.json``.  The recognizer/cache remains frozen; this script only
replays action selection and computes the same two-view/three-view fusion
metrics used by the trajectory-policy comparison.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from .sc_nbv import SCNBVData, RawViewData, causal_batch, make_selector_model


ROOT = Path(__file__).resolve().parents[1]
CLASS_BANK = [0, 2, 26, 34, 50]
TRAIN_SEEDS = [20260909, 20260910, 20260911]
EVAL_SEEDS = list(range(20260920, 20260950))
NUM_VIEWS = 4


def _resolved(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


def _load_manifest(version: int) -> dict[str, Any]:
    config_path = next(
        (ROOT / "rl" / "handoff_configs").glob(
            f"sc_nbv_current_v{version}*.yaml"
        )
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_root = _resolved(config["output_root"])
    summary = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
    variant = str(summary["selected_variant"])
    variant_config = config["train"]["variants"][variant]
    return {
        "version": version,
        "config_path": str(config_path),
        "config": config,
        "cache_root": _resolved(config["sc_cache"]),
        "raw_root": _resolved(config["raw_cache"]),
        "output_root": output_root,
        "variant": variant,
        "variant_config": variant_config,
    }


def _eligible(raw: RawViewData, bank: torch.Tensor) -> torch.Tensor:
    return torch.isin(raw.labels, bank) & (raw.valid.sum(-1) >= 3)


def _valid_candidates(
    raw: RawViewData,
    episodes: torch.Tensor,
    current: torch.Tensor,
    history: torch.Tensor,
) -> torch.Tensor:
    valid = raw.valid[episodes] & raw.reachable[episodes, current]
    valid.scatter_(1, current[:, None], False)
    if history.shape[1] > 1:
        valid.scatter_(1, history, False)
    return valid


@torch.inference_mode()
def _policy_path(
    model: torch.nn.Module,
    context: SCNBVData,
    raw: RawViewData,
    episodes: torch.Tensor,
    starts: torch.Tensor,
    moves: int,
) -> torch.Tensor:
    history = starts[:, None]
    for _ in range(moves):
        current = history[:, -1]
        rows = context.rows_for(episodes, current)
        q = model(causal_batch(context.batch(rows)))
        valid = _valid_candidates(raw, episodes, current, history)
        action = q.masked_fill(~valid, -torch.inf).argmax(-1)
        history = torch.cat((history, action[:, None]), dim=1)
    return history


def _starts_random(raw: RawViewData, episodes: torch.Tensor, seed: int) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    choices = raw.valid[episodes].cpu().numpy()
    starts = [int(rng.choice(np.flatnonzero(row))) for row in choices]
    return torch.as_tensor(starts, device=raw.device, dtype=torch.long)


def _random_path(
    raw: RawViewData,
    episodes: torch.Tensor,
    starts: torch.Tensor,
    moves: int,
    seed: int,
) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    valid = raw.valid[episodes].cpu().numpy()
    paths: list[list[int]] = []
    for index, start in enumerate(starts.cpu().tolist()):
        remaining = [
            view for view in np.flatnonzero(valid[index]).tolist() if view != int(start)
        ]
        rng.shuffle(remaining)
        if len(remaining) < moves:
            raise RuntimeError("old split does not have enough valid views for moves")
        paths.append([int(start), *remaining[:moves]])
    return torch.as_tensor(paths, device=raw.device, dtype=torch.long)


def _margin(scores: torch.Tensor, local: torch.Tensor) -> torch.Tensor:
    target = scores.gather(-1, local[:, None]).squeeze(-1)
    negative = scores.clone()
    negative.scatter_(1, local[:, None], -torch.inf)
    return target - negative.max(-1).values


def _path_metrics(
    raw: RawViewData,
    episodes: torch.Tensor,
    paths: torch.Tensor,
    bank: torch.Tensor,
) -> dict[str, float]:
    scores = raw.logits[episodes[:, None], paths].mean(1).index_select(-1, bank)
    local = torch.searchsorted(bank, raw.labels[episodes])
    correct = scores.argmax(-1).eq(local)
    probabilities = torch.softmax(scores, -1)
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(-1)
    entropy /= math.log(float(len(bank)))
    cost = sum(
        raw.cost[episodes, paths[:, index], paths[:, index + 1]]
        for index in range(paths.shape[1] - 1)
    )
    return {
        "top1": float(correct.float().mean()),
        "mean_entropy": float(entropy.mean()),
        "mean_movement_cost": float(cost.nan_to_num(1.0).mean()),
        "episodes": float(len(episodes)),
    }


def _single_metrics(
    raw: RawViewData,
    episodes: torch.Tensor,
    starts: torch.Tensor,
    bank: torch.Tensor,
) -> dict[str, float]:
    scores = raw.logits[episodes, starts].index_select(-1, bank)
    local = torch.searchsorted(bank, raw.labels[episodes])
    return {
        "top1": float(scores.argmax(-1).eq(local).float().mean()),
        "episodes": float(len(episodes)),
    }


def _oracle_path(
    raw: RawViewData,
    episodes: torch.Tensor,
    starts: torch.Tensor,
    bank: torch.Tensor,
    moves: int,
) -> torch.Tensor:
    result: list[list[int]] = []
    labels = raw.labels[episodes]
    for row, (episode, start) in enumerate(
        zip(episodes.cpu().tolist(), starts.cpu().tolist())
    ):
        valid = raw.valid[episode].nonzero().flatten().cpu().tolist()
        tails = [view for view in valid if view != int(start)]
        candidates = []
        for tail in __import__("itertools").permutations(tails, moves):
            candidates.append([int(start), *tail])
        paths = torch.as_tensor(candidates, device=raw.device, dtype=torch.long)
        ep = torch.full((len(candidates),), int(episode), device=raw.device, dtype=torch.long)
        scores = raw.logits[ep[:, None], paths].mean(1).index_select(-1, bank)
        local = torch.searchsorted(bank, labels[row : row + 1]).expand(len(candidates))
        margins = _margin(scores, local)
        result.append(candidates[int(margins.argmax())])
    return torch.as_tensor(result, device=raw.device, dtype=torch.long)


def _evaluate_model(
    manifest: dict[str, Any],
    raw: RawViewData,
    bank: torch.Tensor,
    eval_seeds: list[int],
) -> dict[str, Any]:
    context = SCNBVData(manifest["cache_root"] / "unseen", raw.device)
    eligible = _eligible(raw, bank)
    episodes = eligible.nonzero().flatten()
    if len(episodes) != 414:
        raise RuntimeError(f"v{manifest['version']} expected 414 old-split episodes, got {len(episodes)}")
    fixed: dict[str, Any] = {}
    random_start: dict[str, Any] = {}
    for train_seed in TRAIN_SEEDS:
        checkpoint = manifest["output_root"] / manifest["variant"] / f"seed_{train_seed}" / "best.pt"
        model = make_selector_model(manifest["variant_config"]).to(raw.device)
        payload = torch.load(checkpoint, map_location=raw.device, weights_only=False)
        model.load_state_dict(payload["model"], strict=True)
        model.eval()
        fixed[str(train_seed)] = {}
        for start in range(NUM_VIEWS):
            fixed_episodes = (eligible & raw.valid[:, start]).nonzero().flatten()
            fixed_starts = torch.full_like(fixed_episodes, start)
            fixed[str(train_seed)][f"fixed{start}"] = {}
            for moves in (1, 2):
                policy = _policy_path(model, context, raw, fixed_episodes, fixed_starts, moves)
                oracle = _oracle_path(raw, fixed_episodes, fixed_starts, bank, moves)
                random_rows = [
                    _path_metrics(
                        raw,
                        fixed_episodes,
                        _random_path(raw, fixed_episodes, fixed_starts, moves, seed + 100 * start + 1000 * moves),
                        bank,
                    )
                    for seed in eval_seeds
                ]
                fixed[str(train_seed)][f"fixed{start}"][f"moves_{moves}"] = {
                    "single": _single_metrics(raw, fixed_episodes, fixed_starts, bank),
                    "policy": _path_metrics(raw, fixed_episodes, policy, bank),
                    "random_mean": {key: float(np.mean([row[key] for row in random_rows])) for key in random_rows[0]},
                    "random_std": {key: float(np.std([row[key] for row in random_rows])) for key in random_rows[0]},
                    "oracle": _path_metrics(raw, fixed_episodes, oracle, bank),
                    "episodes": int(len(fixed_episodes)),
                }
        random_start[str(train_seed)] = {}
        for seed in eval_seeds:
            starts = _starts_random(raw, episodes, seed)
            random_start[str(train_seed)][str(seed)] = {}
            for moves in (1, 2):
                policy = _policy_path(model, context, raw, episodes, starts, moves)
                random = _random_path(raw, episodes, starts, moves, seed + 1000 * moves)
                oracle = _oracle_path(raw, episodes, starts, bank, moves)
                random_start[str(train_seed)][str(seed)][f"moves_{moves}"] = {
                    "single": _single_metrics(raw, episodes, starts, bank),
                    "policy": _path_metrics(raw, episodes, policy, bank),
                    "random": _path_metrics(raw, episodes, random, bank),
                    "oracle": _path_metrics(raw, episodes, oracle, bank),
                    "episodes": int(len(episodes)),
                }
        del model
        torch.cuda.empty_cache()

    fixed_aggregates: dict[str, Any] = {}
    for start in range(NUM_VIEWS):
        for moves in (1, 2):
            rows = [fixed[str(seed)][f"fixed{start}"][f"moves_{moves}"] for seed in TRAIN_SEEDS]
            fixed_aggregates[f"fixed{start}_moves_{moves}"] = {
                "episodes": rows[0]["episodes"],
                "single_top1": rows[0]["single"]["top1"],
                "policy_top1_mean": float(np.mean([row["policy"]["top1"] for row in rows])),
                "policy_top1_std_over_train_seeds": float(np.std([row["policy"]["top1"] for row in rows])),
                "random_top1_mean": float(np.mean([row["random_mean"]["top1"] for row in rows])),
                "random_top1_std_mean": float(np.mean([row["random_std"]["top1"] for row in rows])),
                "oracle_top1_mean": float(np.mean([row["oracle"]["top1"] for row in rows])),
                "policy_minus_random": float(
                    np.mean([row["policy"]["top1"] for row in rows])
                    - np.mean([row["random_mean"]["top1"] for row in rows])
                ),
            }
    random_aggregates: dict[str, Any] = {}
    for moves in (1, 2):
        policy_rows = []
        random_rows = []
        oracle_rows = []
        single_rows = []
        for seed in eval_seeds:
            key = str(seed)
            policy_rows.append(float(np.mean([
                random_start[str(train_seed)][key][f"moves_{moves}"]["policy"]["top1"]
                for train_seed in TRAIN_SEEDS
            ])))
            random_rows.append(random_start[str(TRAIN_SEEDS[0])][key][f"moves_{moves}"]["random"]["top1"])
            oracle_rows.append(float(np.mean([
                random_start[str(train_seed)][key][f"moves_{moves}"]["oracle"]["top1"]
                for train_seed in TRAIN_SEEDS
            ])))
            single_rows.append(random_start[str(TRAIN_SEEDS[0])][key][f"moves_{moves}"]["single"]["top1"])
        random_aggregates[f"random_start_moves_{moves}"] = {
            "episodes": int(len(episodes)),
            "single_top1_mean": float(np.mean(single_rows)),
            "policy_top1_mean_over_train_seeds": float(np.mean(policy_rows)),
            "policy_top1_std_over_eval_seeds": float(np.std(policy_rows)),
            "random_top1_mean_over_eval_seeds": float(np.mean(random_rows)),
            "random_top1_std_over_eval_seeds": float(np.std(random_rows)),
            "oracle_top1_mean_over_train_seeds": float(np.mean(oracle_rows)),
            "policy_minus_random": float(np.mean(policy_rows) - np.mean(random_rows)),
        }
    return {
        "version": manifest["version"],
        "variant": manifest["variant"],
        "config_path": manifest["config_path"],
        "cache_root": str(manifest["cache_root"]),
        "raw_root": str(manifest["raw_root"]),
        "policy_root": str(manifest["output_root"]),
        "episodes": int(len(episodes)),
        "class_bank": CLASS_BANK,
        "train_seeds": TRAIN_SEEDS,
        "random_start_eval_seeds": eval_seeds,
        "fixed_results_by_train_seed": fixed,
        "random_start_results_by_train_seed": random_start,
        "fixed_aggregates": fixed_aggregates,
        "random_start_aggregates": random_aggregates,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--versions", type=int, nargs="+", default=list(range(9, 25)))
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=EVAL_SEEDS)
    args = parser.parse_args()
    device = torch.device("cuda:0")
    raw_root = _resolved("data/rl_candidate_rank_v6_new50_5/cache") / "test"
    raw = RawViewData(raw_root, device)
    bank = torch.as_tensor(CLASS_BANK, device=device, dtype=torch.long)
    output = args.output_root
    output.mkdir(parents=True, exist_ok=True)
    all_results = {}
    for version in args.versions:
        manifest = _load_manifest(version)
        result = _evaluate_model(manifest, raw, bank, list(args.eval_seeds))
        version_root = output / f"v{version}"
        version_root.mkdir(parents=True, exist_ok=True)
        (version_root / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        all_results[f"v{version}"] = {
            "variant": result["variant"],
            "fixed_aggregates": result["fixed_aggregates"],
            "random_start_aggregates": result["random_start_aggregates"],
        }
        print("SC_NBV_OLD_SPLIT_VERSION_COMPLETE", version, result["variant"], flush=True)
    (output / "index.json").write_text(json.dumps({
        "experiment": "sc_nbv_v9_v24_old_split_30seeds",
        "class_bank": CLASS_BANK,
        "episodes": int((torch.isin(raw.labels, bank) & (raw.valid.sum(-1) >= 3)).sum()),
        "train_seeds": TRAIN_SEEDS,
        "random_start_eval_seeds": list(args.eval_seeds),
        "versions": all_results,
    }, indent=2), encoding="utf-8")
    print("SC_NBV_V9_V24_OLD_SPLIT_30SEED_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
