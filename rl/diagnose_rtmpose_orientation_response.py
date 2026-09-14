"""Test orientation response of the original RTMPose-17 trajectory variants.

This diagnostic uses the original 2-D cache, whose policy state is built from
13-frame RTMPose-17 sequences.  It changes only the explicit torso-axis
orientation channels of v4.  v5/v6 have no such channels, so their state is
unchanged by an orientation-only intervention.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import multistep_angle_object_trajectory_policy_v1 as base
from . import multistep_angle_object_trajectory_policy_v4 as v4
from . import multistep_angle_object_trajectory_policy_v5_no_category_id as v5
from . import multistep_angle_object_trajectory_policy_v6_no_object as v6


TRAIN_SEEDS = [20260909, 20260910, 20260911]
EVAL_SEEDS = list(range(20260920, 20260950))
TRUE_UNSEEN = [14, 15, 25, 39, 46]
SHIFTS_DEG = [-90, -60, -30, 0, 30, 60, 90]


def rotate_sincos(value: torch.Tensor, delta_rad: float) -> torch.Tensor:
    sine = value[..., 0]
    cosine = value[..., 1]
    c = math.cos(delta_rad)
    s = math.sin(delta_rad)
    return torch.stack((sine * c + cosine * s, cosine * c - sine * s), -1)


def configure_variant(variant: str, cache_root: Path, track_root: Path) -> Any:
    base.CACHE_ROOT = cache_root
    base.TRACK_ROOT = track_root
    if variant == "v4":
        base.build_trajectory_state = v4.build_trajectory_state_v4
        base.AngleObjectTrajectoryPolicy = v4.BodyOrientationTrajectoryPolicy
        base.attach_object_tracks = base.attach_object_tracks
    elif variant == "v5":
        base.build_trajectory_state = v5.build_trajectory_state_v5
        base.AngleObjectTrajectoryPolicy = v5.CategoryFreeObjectTrajectoryPolicy
        base.attach_object_tracks = base.attach_object_tracks
    elif variant == "v6":
        base.build_trajectory_state = v6.build_trajectory_state_v5
        base.AngleObjectTrajectoryPolicy = v6.PersonGeometryTrajectoryPolicy
        base.attach_object_tracks = v6.attach_no_object_tracks
    else:
        raise ValueError(variant)
    return base


def load_raw(module: Any, cache_root: Path, track_root: Path, device: torch.device) -> Any:
    raw = module.GPUCache(cache_root / "test", device)
    module.attach_view_valid(raw, cache_root, "test")
    if module.build_trajectory_state is not v6.build_trajectory_state_v5:
        module.attach_object_tracks(raw, track_root, "test")
    return raw


def build_state(module: Any, raw: Any, episodes: torch.Tensor, starts: torch.Tensor) -> dict[str, torch.Tensor]:
    history = starts[:, None]
    count = torch.ones_like(starts)
    return module.build_trajectory_state(raw, episodes, history, count)


def counterfactual_state(
    state: dict[str, torch.Tensor], variant: str, shift_deg: float
) -> dict[str, torch.Tensor]:
    out = {key: value.clone() for key, value in state.items()}
    if variant == "v4" and shift_deg != 0:
        out["person_trajectory"][..., -2:] = rotate_sincos(
            out["person_trajectory"][..., -2:], math.radians(shift_deg)
        )
    return out


def ci95(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return 1.96 * float(np.std(values, ddof=1)) / math.sqrt(len(values))


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda:0")
    module = configure_variant(args.variant, args.cache_root, args.track_root)
    raw = load_raw(module, args.cache_root, args.track_root, device)
    bank = torch.as_tensor(TRUE_UNSEEN, device=device, dtype=torch.long)
    eligible = torch.isin(raw.labels, bank) & (raw.valid.sum(-1) >= 3)
    episodes = eligible.nonzero().flatten()

    result: dict[str, Any] = {
        "experiment": "original_rtmpose17_torso_orientation_counterfactual",
        "variant": args.variant,
        "class_bank": TRUE_UNSEEN,
        "episodes": int(len(episodes)),
        "train_seeds": TRAIN_SEEDS,
        "eval_seeds": EVAL_SEEDS,
        "shifts_deg": SHIFTS_DEG,
        "intervention": (
            "rotate only v4 RTMPose torso-axis sin/cos channels; "
            "keep person center/velocity, objects, geometry and masks fixed"
        ),
        "state_orientation_source": {
            "v4": "2-D shoulder-midpoint minus hip-midpoint torso axis",
            "v5": "no explicit orientation channels",
            "v6": "no explicit orientation channels",
        }[args.variant],
        "per_train_seed": {},
    }

    for train_seed in TRAIN_SEEDS:
        checkpoint = args.policy_root / f"seed_{train_seed}" / "best.pt"
        model = module.AngleObjectTrajectoryPolicy().to(device)
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(payload["online"], strict=True)
        model.eval()
        by_shift: dict[str, list[dict[str, float]]] = {str(x): [] for x in SHIFTS_DEG}
        with torch.inference_mode():
            for eval_seed in EVAL_SEEDS:
                starts = module.choose_random_starts(raw, episodes, eval_seed)
                state = build_state(module, raw, episodes, starts)
                baseline_scores = model(state).masked_fill(~state["mask"], -torch.inf)
                baseline_actions = baseline_scores.argmax(-1)
                for shift in SHIFTS_DEG:
                    cf = counterfactual_state(state, args.variant, shift)
                    scores = model(cf).masked_fill(~cf["mask"], -torch.inf)
                    actions = scores.argmax(-1)
                    state_diff = max(
                        float((cf[key] - state[key]).abs().max().detach())
                        for key in state
                        if torch.is_floating_point(state[key])
                    )
                    by_shift[str(shift)].append(
                        {
                            "action_switch_rate": float(
                                actions.ne(baseline_actions).float().mean()
                            ),
                            "max_state_abs_diff": state_diff,
                        }
                    )
        result["per_train_seed"][str(train_seed)] = {
            "checkpoint": str(checkpoint),
            "by_shift": by_shift,
        }
        del model, payload
        torch.cuda.empty_cache()

    aggregates: dict[str, Any] = {}
    for shift in SHIFTS_DEG:
        rows = []
        for index, _ in enumerate(EVAL_SEEDS):
            per_checkpoint = [
                result["per_train_seed"][str(seed)]["by_shift"][str(shift)][index]
                for seed in TRAIN_SEEDS
            ]
            rows.append(
                {
                    "action_switch_rate": float(
                        np.mean([row["action_switch_rate"] for row in per_checkpoint])
                    ),
                    "max_state_abs_diff": float(
                        np.max([row["max_state_abs_diff"] for row in per_checkpoint])
                    ),
                }
            )
        switch = [row["action_switch_rate"] for row in rows]
        state_diff = [row["max_state_abs_diff"] for row in rows]
        aggregates[str(shift)] = {
            "action_switch_rate": {
                "mean": float(np.mean(switch)),
                "std": float(np.std(switch, ddof=1)),
                "ci95": ci95(switch),
            },
            "max_state_abs_diff": {
                "max": float(np.max(state_diff)),
                "mean": float(np.mean(state_diff)),
            },
        }
    result["aggregates"] = aggregates
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("v4", "v5", "v6"), required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--track-root", type=Path, required=True)
    parser.add_argument("--policy-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result["aggregates"], indent=2), flush=True)


if __name__ == "__main__":
    main()
