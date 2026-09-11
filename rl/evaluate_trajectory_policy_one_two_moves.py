"""Evaluate the trained trajectory policy after one and two moves."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .multistep_angle_object_trajectory_policy_v1 import (
    CACHE_ROOT,
    TRACK_ROOT,
    TRUE_UNSEEN_CLASSES,
    AngleObjectTrajectoryPolicy,
    GPUCache,
    attach_object_tracks,
    attach_view_valid,
    build_trajectory_state,
    class_bank_tensor,
)


ROOT = CACHE_ROOT.parents[2]
POLICY_ROOT = ROOT / "work_dir" / "multistep_angle_object_trajectory_policy_v1"
OUTPUT_ROOT = POLICY_ROOT / "one_two_move_test"


@torch.inference_mode()
def policy_path(model: torch.nn.Module, raw: GPUCache, episodes: torch.Tensor, starts: torch.Tensor, moves: int) -> torch.Tensor:
    history = starts[:, None]
    actions = []
    for stage in range(1, moves + 1):
        state = build_trajectory_state(raw, episodes, history, torch.full_like(episodes, stage))
        action = model(state).masked_fill(~state["mask"], -torch.inf).argmax(-1)
        actions.append(action)
        history = torch.cat((history, action[:, None]), dim=1)
    return torch.cat((starts[:, None], torch.stack(actions, dim=1)), dim=1)


def path_metrics(raw: GPUCache, episodes: torch.Tensor, paths: torch.Tensor, bank: torch.Tensor) -> dict[str, float]:
    scores = raw.logits[episodes[:, None], paths].mean(1).index_select(-1, bank)
    local = torch.searchsorted(bank, raw.labels[episodes])
    correct = scores.argmax(-1).eq(local)
    probs = F.softmax(scores, -1)
    entropy = -(probs * probs.clamp_min(1e-12).log()).sum(-1) / math.log(float(len(bank)))
    cost = sum(raw.cost[episodes, paths[:, i], paths[:, i + 1]] for i in range(paths.shape[1] - 1))
    return {
        "top1": float(correct.float().mean()),
        "mean_entropy": float(entropy.mean()),
        "mean_movement_cost": float(cost.nan_to_num(1.0).mean()),
        "episodes": float(len(episodes)),
    }


def oracle_path(raw: GPUCache, episodes: torch.Tensor, starts: torch.Tensor, bank: torch.Tensor, moves: int) -> torch.Tensor:
    paths = []
    for episode, start in zip(episodes.cpu().tolist(), starts.cpu().tolist()):
        candidates = [
            [int(start), *tail]
            for tail in itertools.permutations([v for v in range(4) if v != int(start)], moves)
        ]
        candidate_paths = torch.as_tensor(candidates, device=raw.device, dtype=torch.long)
        ep = torch.full((len(candidates),), int(episode), device=raw.device, dtype=torch.long)
        scores = raw.logits[ep[:, None], candidate_paths].mean(1).index_select(-1, bank)
        local = torch.searchsorted(bank, raw.labels[ep])
        target = scores.gather(-1, local[:, None]).squeeze(-1)
        rival = scores.clone()
        rival.scatter_(1, local[:, None], -torch.inf)
        paths.append(candidates[int((target - rival.amax(-1)).argmax())])
    return torch.as_tensor(paths, device=raw.device, dtype=torch.long)


def random_path(starts: torch.Tensor, moves: int, seed: int, device: torch.device) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    result = []
    for start in starts.cpu().tolist():
        remaining = [v for v in range(4) if v != int(start)]
        rng.shuffle(remaining)
        result.append([int(start), *remaining[:moves]])
    return torch.as_tensor(result, device=device, dtype=torch.long)


def single_metrics(raw: GPUCache, episodes: torch.Tensor, starts: torch.Tensor, bank: torch.Tensor) -> dict[str, float]:
    scores = raw.logits[episodes, starts].index_select(-1, bank)
    local = torch.searchsorted(bank, raw.labels[episodes])
    return {"top1": float(scores.argmax(-1).eq(local).float().mean()), "episodes": float(len(episodes))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--track-root", type=Path, default=TRACK_ROOT)
    parser.add_argument("--policy-root", type=Path, default=POLICY_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260909, 20260910, 20260911])
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda:0")
    raw = GPUCache(args.cache_root / "test", device)
    attach_view_valid(raw, args.cache_root, "test")
    attach_object_tracks(raw, args.track_root, "test")
    bank = class_bank_tensor(TRUE_UNSEEN_CLASSES, device)
    eligible = torch.isin(raw.labels, bank) & (raw.valid.sum(-1) >= 3)
    episodes = eligible.nonzero().flatten()
    args.output_root.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    for seed in args.seeds:
        checkpoint = args.policy_root / f"seed_{seed}" / "best.pt"
        model = AngleObjectTrajectoryPolicy().to(device)
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(payload["online"], strict=True)
        model.eval()
        results[str(seed)] = {}
        for start in range(4):
            starts = torch.full_like(episodes, start)
            results[str(seed)][f"fixed{start}"] = {}
            for moves in (1, 2):
                policy = policy_path(model, raw, episodes, starts, moves)
                random = random_path(starts, moves, 20260910 + 10 * moves + start, device)
                oracle = oracle_path(raw, episodes, starts, bank, moves)
                result = {
                    "single": single_metrics(raw, episodes, starts, bank),
                    "random": path_metrics(raw, episodes, random, bank),
                    "policy": path_metrics(raw, episodes, policy, bank),
                    "oracle": path_metrics(raw, episodes, oracle, bank),
                    "moves": moves,
                    "fusion_views": moves + 1,
                }
                results[str(seed)][f"fixed{start}"][f"moves_{moves}"] = result
                print("ONE_TWO_MOVE", seed, start, moves, json.dumps(result), flush=True)
        del model
        torch.cuda.empty_cache()
    aggregates = {}
    for start in range(4):
        for moves in (1, 2):
            rows = [results[str(seed)][f"fixed{start}"][f"moves_{moves}"] for seed in args.seeds]
            aggregates[f"fixed{start}_moves_{moves}"] = {
                "single": float(np.mean([r["single"]["top1"] for r in rows])),
                "random": float(np.mean([r["random"]["top1"] for r in rows])),
                "policy_mean": float(np.mean([r["policy"]["top1"] for r in rows])),
                "policy_std": float(np.std([r["policy"]["top1"] for r in rows])),
                "oracle": float(np.mean([r["oracle"]["top1"] for r in rows])),
                "policy_cost_mean": float(np.mean([r["policy"]["mean_movement_cost"] for r in rows])),
                "random_cost_mean": float(np.mean([r["random"]["mean_movement_cost"] for r in rows])),
            }
    summary = {
        "experiment": "trajectory_policy_one_two_moves_true_unseen",
        "class_bank": [int(x) for x in TRUE_UNSEEN_CLASSES],
        "episodes": int(len(episodes)),
        "results_by_seed": results,
        "aggregates": aggregates,
    }
    (args.output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("ONE_TWO_MOVE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
