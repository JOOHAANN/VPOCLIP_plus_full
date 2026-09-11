"""Evaluate three moves: start view plus all remaining three views.

The experiment intentionally reuses the trained trajectory policy checkpoints
and does not retrain them.  With four valid views, every legal three-move path
observes the same set of views, so final four-view HAR accuracy should be
order-invariant; the meaningful policy effect is path/movement cost.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

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
OUTPUT_ROOT = POLICY_ROOT / "four_view_three_move_test"


@torch.inference_mode()
def policy_paths(model: torch.nn.Module, raw: GPUCache, episodes: torch.Tensor, starts: torch.Tensor) -> torch.Tensor:
    history = starts[:, None]
    actions = []
    for stage in (1, 2, 3):
        state = build_trajectory_state(raw, episodes, history, torch.full_like(episodes, stage))
        action = model(state).masked_fill(~state["mask"], -torch.inf).argmax(-1)
        actions.append(action)
        history = torch.cat((history, action[:, None]), dim=1)
    return torch.cat((starts[:, None], torch.stack(actions, dim=1)), dim=1)


def metrics(raw: GPUCache, episodes: torch.Tensor, paths: torch.Tensor, bank: torch.Tensor) -> dict[str, float]:
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


def single_metrics(raw: GPUCache, episodes: torch.Tensor, starts: torch.Tensor, bank: torch.Tensor) -> dict[str, float]:
    scores = raw.logits[episodes, starts].index_select(-1, bank)
    local = torch.searchsorted(bank, raw.labels[episodes])
    return {"top1": float(scores.argmax(-1).eq(local).float().mean()), "mean_movement_cost": 0.0, "episodes": float(len(episodes))}


def random_paths(episodes: torch.Tensor, starts: torch.Tensor, seed: int) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    values = []
    for start in starts.detach().cpu().tolist():
        remaining = [view for view in range(4) if view != int(start)]
        rng.shuffle(remaining)
        values.append([int(start), *remaining])
    return torch.as_tensor(values, device=episodes.device, dtype=torch.long)


def four_view_metrics(raw: GPUCache, episodes: torch.Tensor, bank: torch.Tensor) -> dict[str, float]:
    weights = raw.valid[episodes].float().unsqueeze(-1)
    scores = ((raw.logits[episodes] * weights).sum(1) / weights.sum(1).clamp_min(1.0)).index_select(-1, bank)
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
    eligible = torch.isin(raw.labels, bank) & (raw.valid.sum(-1) == 4)
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
            start_tensor = torch.full_like(episodes, start)
            paths = policy_paths(model, raw, episodes, start_tensor)
            random = random_paths(episodes, start_tensor, 20260910 + start)
            result = {
                "single": single_metrics(raw, episodes, start_tensor, bank),
                "random_four_view": metrics(raw, episodes, random, bank),
                "policy_four_view": metrics(raw, episodes, paths, bank),
                "all_valid_four_view": four_view_metrics(raw, episodes, bank),
                "policy_path_matches_all_views": bool(torch.equal(torch.sort(paths, dim=1).values, torch.arange(4, device=device)[None].expand(len(paths), -1))),
            }
            results[str(seed)][f"fixed{start}"] = result
            print("FOUR_VIEW_THREE_MOVE", seed, start, json.dumps(result), flush=True)
        del model
        torch.cuda.empty_cache()
    summary = {
        "experiment": "four_view_fusion_after_three_moves",
        "class_bank": [int(x) for x in TRUE_UNSEEN_CLASSES],
        "episodes": int(len(episodes)),
        "interpretation": "all legal paths visit the same four views; policy can change order and movement cost, not final mean-logit evidence",
        "results_by_seed": results,
    }
    (args.output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("FOUR_VIEW_THREE_MOVE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
