"""Evaluate v4/v5/v6 trajectory policies on the historical 414-sample split.

This is deliberately separate from the current 50/5 evaluator.  The old
protocol holds out class IDs [0, 2, 26, 34, 50], uses the historical cache,
and repeats random paths over ten evaluation seeds so its output can be
compared with the tables in Trajectory_Conditioned_MultiStep_Active_View_Policy.md.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from . import multistep_angle_object_trajectory_policy_v1 as base
from . import multistep_angle_object_trajectory_policy_v3 as v3
from . import multistep_angle_object_trajectory_policy_v4 as v4
from . import multistep_angle_object_trajectory_policy_v5_no_category_id as v5
from . import multistep_angle_object_trajectory_policy_v6_no_object as v6
from . import multistep_angle_object_trajectory_ablation as ablation


OLD_UNSEEN = [0, 2, 26, 34, 50]
TRAIN_SEEDS = [20260909, 20260910, 20260911]
EVAL_SEEDS = list(range(20260920, 20260930))


def configure_variant(name: str) -> tuple[type[torch.nn.Module], Any]:
    if name == "v1":
        # In a fresh evaluator process these are the unmodified v1 symbols.
        return base.AngleObjectTrajectoryPolicy, base.attach_object_tracks
    if name == "v3":
        base.build_trajectory_state = v3.build_trajectory_state_v3
        base.AngleObjectTrajectoryPolicy = v3.ObjectSizeTrajectoryPolicy
        base.attach_object_tracks = v3.attach_object_tracks_v3
        return v3.ObjectSizeTrajectoryPolicy, v3.attach_object_tracks_v3
    if name == "v4":
        base.build_trajectory_state = v4.build_trajectory_state_v4
        base.AngleObjectTrajectoryPolicy = v4.BodyOrientationTrajectoryPolicy
        base.attach_object_tracks = base.attach_object_tracks
        return v4.BodyOrientationTrajectoryPolicy, base.attach_object_tracks
    if name == "v5":
        base.build_trajectory_state = v5.build_trajectory_state_v5
        base.AngleObjectTrajectoryPolicy = v5.CategoryFreeObjectTrajectoryPolicy
        base.attach_object_tracks = v5.v1.attach_object_tracks
        return v5.CategoryFreeObjectTrajectoryPolicy, v5.v1.attach_object_tracks
    if name == "v6":
        base.build_trajectory_state = v6.build_trajectory_state_v5
        base.AngleObjectTrajectoryPolicy = v6.PersonGeometryTrajectoryPolicy
        base.attach_object_tracks = v6.attach_no_object_tracks
        return v6.PersonGeometryTrajectoryPolicy, v6.attach_no_object_tracks
    if name == "object_only":
        base.build_trajectory_state = ablation.build_object_only_state
        base.AngleObjectTrajectoryPolicy = ablation.ObjectOnlyTrajectoryPolicy
        base.attach_object_tracks = base.attach_object_tracks
        return ablation.ObjectOnlyTrajectoryPolicy, base.attach_object_tracks
    if name == "geometry_only":
        base.build_trajectory_state = ablation.build_geometry_only_state
        base.AngleObjectTrajectoryPolicy = ablation.GeometryOnlyPolicy
        base.attach_object_tracks = ablation.attach_no_object_tracks
        return ablation.GeometryOnlyPolicy, ablation.attach_no_object_tracks
    raise ValueError(name)


@torch.inference_mode()
def policy_path(
    model: torch.nn.Module,
    raw: base.GPUCache,
    episodes: torch.Tensor,
    starts: torch.Tensor,
    moves: int,
) -> torch.Tensor:
    history = starts[:, None]
    actions = []
    for stage in range(1, moves + 1):
        state = base.build_trajectory_state(
            raw, episodes, history, torch.full_like(episodes, stage)
        )
        action = model(state).masked_fill(~state["mask"], -torch.inf).argmax(-1)
        actions.append(action)
        history = torch.cat((history, action[:, None]), dim=1)
    return torch.cat((starts[:, None], torch.stack(actions, dim=1)), dim=1)


def path_metrics(
    raw: base.GPUCache,
    episodes: torch.Tensor,
    paths: torch.Tensor,
    bank: torch.Tensor,
) -> dict[str, float]:
    scores = raw.logits[episodes[:, None], paths].mean(1).index_select(-1, bank)
    local = torch.searchsorted(bank, raw.labels[episodes])
    correct = scores.argmax(-1).eq(local)
    probs = F.softmax(scores, -1)
    entropy = -(probs * probs.clamp_min(1e-12).log()).sum(-1) / np.log(len(bank))
    cost = sum(
        raw.cost[episodes, paths[:, i], paths[:, i + 1]]
        for i in range(paths.shape[1] - 1)
    )
    return {
        "top1": float(correct.float().mean()),
        "mean_entropy": float(entropy.mean()),
        "mean_movement_cost": float(cost.nan_to_num(1.0).mean()),
        "episodes": float(len(episodes)),
    }


def single_metrics(
    raw: base.GPUCache,
    episodes: torch.Tensor,
    starts: torch.Tensor,
    bank: torch.Tensor,
) -> dict[str, float]:
    scores = raw.logits[episodes, starts].index_select(-1, bank)
    local = torch.searchsorted(bank, raw.labels[episodes])
    return {"top1": float(scores.argmax(-1).eq(local).float().mean()), "episodes": float(len(episodes))}


def random_path(starts: torch.Tensor, moves: int, seed: int, device: torch.device) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    paths = []
    for start in starts.cpu().tolist():
        remaining = [v for v in range(base.NUM_VIEWS) if v != int(start)]
        rng.shuffle(remaining)
        paths.append([int(start), *remaining[:moves]])
    return torch.as_tensor(paths, device=device, dtype=torch.long)


def oracle_path(
    raw: base.GPUCache,
    episodes: torch.Tensor,
    starts: torch.Tensor,
    bank: torch.Tensor,
    moves: int,
) -> torch.Tensor:
    result = []
    for episode, start in zip(episodes.cpu().tolist(), starts.cpu().tolist()):
        candidates = [
            [int(start), *tail]
            for tail in itertools.permutations(
                [v for v in range(base.NUM_VIEWS) if v != int(start)], moves
            )
        ]
        paths = torch.as_tensor(candidates, device=raw.device, dtype=torch.long)
        ep = torch.full((len(candidates),), int(episode), device=raw.device, dtype=torch.long)
        scores = raw.logits[ep[:, None], paths].mean(1).index_select(-1, bank)
        local = torch.searchsorted(bank, raw.labels[ep])
        target = scores.gather(-1, local[:, None]).squeeze(-1)
        rival = scores.clone()
        rival.scatter_(1, local[:, None], -torch.inf)
        result.append(candidates[int((target - rival.amax(-1)).argmax())])
    return torch.as_tensor(result, device=raw.device, dtype=torch.long)


def evaluate_variant(
    name: str,
    cache_root: Path,
    track_root: Path,
    policy_root: Path,
    output_root: Path,
    eval_seeds: list[int],
) -> dict[str, Any]:
    model_cls, attach_tracks = configure_variant(name)
    device = torch.device("cuda:0")
    raw = base.GPUCache(cache_root / "test", device)
    base.attach_view_valid(raw, cache_root, "test")
    attach_tracks(raw, track_root, "test")
    bank = base.class_bank_tensor(OLD_UNSEEN, device)
    eligible = torch.isin(raw.labels, bank) & (raw.valid.sum(-1) >= 3)
    episodes = eligible.nonzero().flatten()
    if len(episodes) != 414:
        raise RuntimeError(f"old protocol expected 414 test episodes, got {len(episodes)}")

    results: dict[str, Any] = {}
    for seed in TRAIN_SEEDS:
        checkpoint = policy_root / f"seed_{seed}" / "best.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        model = model_cls().to(device)
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(payload["online"], strict=True)
        model.eval()
        results[str(seed)] = {}
        for start in range(base.NUM_VIEWS):
            starts = torch.full_like(episodes, start)
            key = f"fixed{start}"
            results[str(seed)][key] = {}
            for moves in (1, 2):
                policy = policy_path(model, raw, episodes, starts, moves)
                oracle = oracle_path(raw, episodes, starts, bank, moves)
                random_values = []
                for eval_seed in eval_seeds:
                    random = random_path(
                        starts, moves, eval_seed + 100 * start + 1000 * moves, device
                    )
                    random_values.append(path_metrics(raw, episodes, random, bank))
                policy_metrics = path_metrics(raw, episodes, policy, bank)
                oracle_metrics = path_metrics(raw, episodes, oracle, bank)
                results[str(seed)][key][f"moves_{moves}"] = {
                    "single": single_metrics(raw, episodes, starts, bank),
                    "policy": policy_metrics,
                    "oracle": oracle_metrics,
                    "random_mean": {
                        metric: float(np.mean([row[metric] for row in random_values]))
                        for metric in random_values[0]
                    },
                    "random_std": {
                        metric: float(np.std([row[metric] for row in random_values]))
                        for metric in random_values[0]
                    },
                    "moves": moves,
                    "fusion_views": moves + 1,
                }
        del model
        torch.cuda.empty_cache()

    aggregates: dict[str, Any] = {}
    for start in range(base.NUM_VIEWS):
        for moves in (1, 2):
            rows = [results[str(seed)][f"fixed{start}"][f"moves_{moves}"] for seed in TRAIN_SEEDS]
            aggregates[f"fixed{start}_moves_{moves}"] = {
                "single": rows[0]["single"]["top1"],
                "random_top1_mean": float(np.mean([r["random_mean"]["top1"] for r in rows])),
                "random_top1_std_mean": float(np.mean([r["random_std"]["top1"] for r in rows])),
                "policy_top1_mean": float(np.mean([r["policy"]["top1"] for r in rows])),
                "policy_top1_std": float(np.std([r["policy"]["top1"] for r in rows])),
                "oracle_top1": rows[0]["oracle"]["top1"],
                "policy_minus_random": float(
                    np.mean([r["policy"]["top1"] for r in rows])
                    - np.mean([r["random_mean"]["top1"] for r in rows])
                ),
            }
    output_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "experiment": "trajectory_policy_one_two_moves_old_split",
        "variant": name,
        "class_bank": OLD_UNSEEN,
        "episodes": int(len(episodes)),
        "cache_root": str(cache_root),
        "track_root": str(track_root),
        "train_seeds": TRAIN_SEEDS,
        "random_eval_seeds": eval_seeds,
        "results_by_seed": results,
        "aggregates": aggregates,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        choices=("v1", "v4", "v5", "v6", "object_only", "geometry_only"),
        required=True,
    )
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--track-root", type=Path, required=True)
    parser.add_argument("--policy-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--eval-seeds", type=int, nargs="+", default=list(range(20260920, 20260950))
    )
    args = parser.parse_args()
    summary = evaluate_variant(
        args.variant,
        args.cache_root,
        args.track_root,
        args.policy_root,
        args.output_root,
        args.eval_seeds,
    )
    print("OLD_SPLIT_VARIANT_EVALUATION_COMPLETE", args.variant, flush=True)
    print(json.dumps(summary["aggregates"], indent=2), flush=True)


if __name__ == "__main__":
    main()
