"""Evaluate feature-level-fusion trajectory policies on the new 50/5 split.

The protocol is intentionally the same as the preserved 2-D anchor:

* true unseen classes are [14,15,25,39,46] (5-way only);
* fixed starts view0/view1/view2/view3;
* random start independently sampled for each of 30 evaluation seeds;
* one move (two fused views) and two moves (three fused views);
* three policy training seeds and the same random path baselines.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .. import multistep_angle_object_trajectory_policy_v1 as base
from .. import multistep_angle_object_trajectory_policy_v4 as v4
from .. import multistep_angle_object_trajectory_policy_v5_no_category_id as v5
from .. import multistep_angle_object_trajectory_policy_v6_no_object as v6
from .. import multistep_angle_object_trajectory_ablation as ablation
from . import feature_fusion as ff


TRAIN_SEEDS = [20260909, 20260910, 20260911]
EVAL_SEEDS = list(range(20260920, 20260950))
TRUE_UNSEEN = [14, 15, 25, 39, 46]


def configure_variant(name: str) -> type[torch.nn.Module]:
    if name == "v4":
        base.build_trajectory_state = v4.build_trajectory_state_v4
        base.AngleObjectTrajectoryPolicy = v4.BodyOrientationTrajectoryPolicy
        base.attach_object_tracks = base.attach_object_tracks
        return v4.BodyOrientationTrajectoryPolicy
    if name == "v5":
        base.build_trajectory_state = v5.build_trajectory_state_v5
        base.AngleObjectTrajectoryPolicy = v5.CategoryFreeObjectTrajectoryPolicy
        base.attach_object_tracks = base.attach_object_tracks
        return v5.CategoryFreeObjectTrajectoryPolicy
    if name == "v6":
        base.build_trajectory_state = v6.build_trajectory_state_v5
        base.AngleObjectTrajectoryPolicy = v6.PersonGeometryTrajectoryPolicy
        base.attach_object_tracks = v6.attach_no_object_tracks
        return v6.PersonGeometryTrajectoryPolicy
    if name == "object_only":
        base.build_trajectory_state = ablation.build_object_only_state
        base.AngleObjectTrajectoryPolicy = ablation.ObjectOnlyTrajectoryPolicy
        base.attach_object_tracks = base.attach_object_tracks
        return ablation.ObjectOnlyTrajectoryPolicy
    if name == "geometry_only":
        base.build_trajectory_state = ablation.build_geometry_only_state
        base.AngleObjectTrajectoryPolicy = ablation.GeometryOnlyPolicy
        base.attach_object_tracks = ablation.attach_no_object_tracks
        return ablation.GeometryOnlyPolicy
    raise ValueError(name)


@torch.no_grad()
def policy_path(
    model: torch.nn.Module,
    raw: Any,
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


def random_path(starts: torch.Tensor, moves: int, seed: int, device: torch.device) -> torch.Tensor:
    # Keep the anchor's exact random-path protocol: shuffle the remaining
    # physical view indices, without adding a new reachability rule.
    rng = np.random.default_rng(seed)
    paths = []
    for start in starts.cpu().tolist():
        remaining = [v for v in range(base.NUM_VIEWS) if v != int(start)]
        rng.shuffle(remaining)
        paths.append([int(start), *remaining[:moves]])
    return torch.as_tensor(paths, device=device, dtype=torch.long)


def oracle_path(
    raw: Any,
    fusion: ff.FeatureFusion512,
    scale: float,
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
        scores = ff.score_features(raw, fusion, ep, paths, scale).index_select(-1, bank)
        local = torch.searchsorted(bank, raw.labels[ep])
        target = scores.gather(-1, local[:, None]).squeeze(-1)
        rival = scores.clone()
        rival.scatter_(1, local[:, None], -torch.inf)
        result.append(candidates[int((target - rival.amax(-1)).argmax())])
    return torch.as_tensor(result, device=raw.device, dtype=torch.long)


@torch.no_grad()
def evaluate_variant(
    name: str,
    cache_root: Path,
    track_root: Path,
    fusion_root: Path,
    policy_root: Path,
    output_root: Path,
    eval_seeds: list[int],
    expected_episodes: int,
) -> dict[str, Any]:
    model_cls = configure_variant(name)
    device = torch.device("cuda:0")
    raw = base.GPUCache(cache_root / "test", device)
    base.attach_view_valid(raw, cache_root, "test")
    base.attach_object_tracks(raw, track_root, "test")
    bank = base.class_bank_tensor(TRUE_UNSEEN, device)
    eligible = torch.isin(raw.labels, bank) & (raw.valid.sum(-1) >= 3)
    episodes = eligible.nonzero().flatten()
    if len(episodes) != expected_episodes:
        raise RuntimeError(f"expected {expected_episodes} true-unseen episodes, got {len(episodes)}")

    fixed_results: dict[str, Any] = {}
    random_results: dict[str, Any] = {}
    for train_seed in TRAIN_SEEDS:
        fusion_path = fusion_root / f"seed_{train_seed}" / "best.pt"
        policy_path_file = policy_root / f"seed_{train_seed}" / "best.pt"
        fusion = ff.FeatureFusion512().to(device)
        fusion_payload = torch.load(fusion_path, map_location=device, weights_only=False)
        fusion.load_state_dict(fusion_payload["model"], strict=True)
        fusion.eval()
        scale = float(fusion_payload.get("logit_scale", ff.infer_logit_scale(raw)))
        model = model_cls().to(device)
        payload = torch.load(policy_path_file, map_location=device, weights_only=False)
        model.load_state_dict(payload["online"], strict=True)
        model.eval()
        fixed_results[str(train_seed)] = {}
        for start in range(base.NUM_VIEWS):
            fixed_episodes = (eligible & raw.valid[:, start]).nonzero().flatten()
            fixed_starts = torch.full_like(fixed_episodes, start)
            fixed_results[str(train_seed)][f"fixed{start}"] = {}
            for moves in (1, 2):
                policy = policy_path(model, raw, fixed_episodes, fixed_starts, moves)
                oracle = oracle_path(raw, fusion, scale, fixed_episodes, fixed_starts, bank, moves)
                random_values = []
                for eval_seed in eval_seeds:
                    random_values.append(
                        ff.path_metrics(
                            raw, fusion, fixed_episodes,
                            random_path(fixed_starts, moves, eval_seed + 100 * start + 1000 * moves, device),
                            bank, scale,
                        )
                    )
                fixed_results[str(train_seed)][f"fixed{start}"][f"moves_{moves}"] = {
                    "single": ff.single_metrics(raw, fusion, fixed_episodes, fixed_starts, bank, scale),
                    "policy": ff.path_metrics(raw, fusion, fixed_episodes, policy, bank, scale),
                    "oracle": ff.path_metrics(raw, fusion, fixed_episodes, oracle, bank, scale),
                    "random_mean": {metric: float(np.mean([row[metric] for row in random_values])) for metric in random_values[0]},
                    "random_std": {metric: float(np.std([row[metric] for row in random_values])) for metric in random_values[0]},
                    "episodes": int(len(fixed_episodes)),
                    "moves": moves,
                    "fusion_views": moves + 1,
                }
        random_results[str(train_seed)] = {}
        for eval_seed in eval_seeds:
            starts = base.choose_random_starts(raw, episodes, eval_seed)
            random_results[str(train_seed)][str(eval_seed)] = {}
            for moves in (1, 2):
                policy = policy_path(model, raw, episodes, starts, moves)
                random = random_path(starts, moves, eval_seed + 1000 * moves, device)
                oracle = oracle_path(raw, fusion, scale, episodes, starts, bank, moves)
                random_results[str(train_seed)][str(eval_seed)][f"moves_{moves}"] = {
                    "single": ff.single_metrics(raw, fusion, episodes, starts, bank, scale),
                    "policy": ff.path_metrics(raw, fusion, episodes, policy, bank, scale),
                    "random": ff.path_metrics(raw, fusion, episodes, random, bank, scale),
                    "oracle": ff.path_metrics(raw, fusion, episodes, oracle, bank, scale),
                    "episodes": int(len(episodes)),
                    "moves": moves,
                    "fusion_views": moves + 1,
                }
        del fusion, model
        torch.cuda.empty_cache()

    fixed_aggregates: dict[str, Any] = {}
    for start in range(base.NUM_VIEWS):
        for moves in (1, 2):
            rows = [fixed_results[str(seed)][f"fixed{start}"][f"moves_{moves}"] for seed in TRAIN_SEEDS]
            fixed_aggregates[f"fixed{start}_moves_{moves}"] = {
                "episodes": rows[0]["episodes"],
                "single_top1": rows[0]["single"]["top1"],
                "policy_top1_mean": float(np.mean([r["policy"]["top1"] for r in rows])),
                "policy_top1_std_over_train_seeds": float(np.std([r["policy"]["top1"] for r in rows])),
                "random_top1_mean": float(np.mean([r["random_mean"]["top1"] for r in rows])),
                "random_top1_std_mean": float(np.mean([r["random_std"]["top1"] for r in rows])),
                "oracle_top1_mean": float(np.mean([r["oracle"]["top1"] for r in rows])),
                "policy_minus_random": float(np.mean([r["policy"]["top1"] for r in rows]) - np.mean([r["random_mean"]["top1"] for r in rows])),
            }
    random_aggregates: dict[str, Any] = {}
    for moves in (1, 2):
        policy_rows, random_rows, oracle_rows = [], [], []
        for eval_seed in eval_seeds:
            policy_rows.append(float(np.mean([random_results[str(seed)][str(eval_seed)][f"moves_{moves}"]["policy"]["top1"] for seed in TRAIN_SEEDS])))
            random_rows.append(random_results[str(TRAIN_SEEDS[0])][str(eval_seed)][f"moves_{moves}"]["random"]["top1"])
            oracle_rows.append(float(np.mean([random_results[str(seed)][str(eval_seed)][f"moves_{moves}"]["oracle"]["top1"] for seed in TRAIN_SEEDS])))
        random_aggregates[f"random_start_moves_{moves}"] = {
            "episodes": int(len(episodes)),
            "single_top1_mean": float(np.mean([random_results[str(TRAIN_SEEDS[0])][str(s)][f"moves_{moves}"]["single"]["top1"] for s in eval_seeds])),
            "policy_top1_mean_over_train_seeds": float(np.mean(policy_rows)),
            "policy_top1_std_over_eval_seeds": float(np.std(policy_rows)),
            "random_top1_mean_over_eval_seeds": float(np.mean(random_rows)),
            "random_top1_std_over_eval_seeds": float(np.std(random_rows)),
            "oracle_top1_mean_over_train_seeds": float(np.mean(oracle_rows)),
            "policy_minus_random": float(np.mean(policy_rows) - np.mean(random_rows)),
        }
    output_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "experiment": "trajectory_feature_level_fusion_new_50_5_unseen_30_seed",
        "variant": name,
        "class_bank": TRUE_UNSEEN,
        "episodes": int(len(episodes)),
        "cache_root": str(cache_root),
        "track_root": str(track_root),
        "fusion_root": str(fusion_root),
        "policy_root": str(policy_root),
        "train_seeds": TRAIN_SEEDS,
        "random_start_eval_seeds": eval_seeds,
        "fusion": "FeatureFusion512 then text-prototype cosine similarity",
        "anchor": "/home/youhan/ws/VPOCLIP_plus_full/work_dir/trajectory_g1_pseudo_selected_stage_b_2d",
        "fixed_results_by_train_seed": fixed_results,
        "random_start_results_by_train_seed": random_results,
        "fixed_aggregates": fixed_aggregates,
        "random_start_aggregates": random_aggregates,
    }
    (output_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("v4", "v5", "v6", "object_only", "geometry_only"), required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--track-root", type=Path, required=True)
    parser.add_argument("--fusion-root", type=Path, required=True)
    parser.add_argument("--policy-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-episodes", type=int, default=271)
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=EVAL_SEEDS)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    summary = evaluate_variant(args.variant, args.cache_root, args.track_root, args.fusion_root, args.policy_root, args.output_root, args.eval_seeds, args.expected_episodes)
    print("FEATURE_FUSION_EVALUATION_COMPLETE", args.variant, flush=True)
    print(json.dumps({"fixed": summary["fixed_aggregates"], "random_start": summary["random_start_aggregates"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()

