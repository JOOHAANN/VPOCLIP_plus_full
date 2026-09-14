"""Counterfactual test of body-orientation conditioning.

For each test episode and random valid starting view, the policy state is
constructed normally.  We then rotate every body-relative candidate bearing
by a common angle while leaving the person trajectory, object trajectory,
reachability, logits, and checkpoint unchanged.  A change in the selected
action is therefore attributable to the angle input of the selector.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import train_ranked_variant as adapter
from .. import multistep_angle_object_trajectory_policy_v1 as base


TRAIN_SEEDS = [20260909, 20260910, 20260911]
DEFAULT_EVAL_SEEDS = list(range(20260920, 20260950))
TRUE_UNSEEN = [14, 15, 25, 39, 46]
SHIFTS_DEG = [-90, -60, -30, 0, 30, 60, 90]


def _rotate_sincos(value: torch.Tensor, delta_rad: float) -> torch.Tensor:
    sine = value[..., 0]
    cosine = value[..., 1]
    c = math.cos(delta_rad)
    s = math.sin(delta_rad)
    return torch.stack((sine * c + cosine * s, cosine * c - sine * s), -1)


def _rotate_state(
    state: dict[str, torch.Tensor], shift_deg: float, variant: str
) -> dict[str, torch.Tensor]:
    if shift_deg == 0:
        return state
    delta = math.radians(float(shift_deg))
    out = dict(state)
    out["current_relative_angle"] = _rotate_sincos(
        state["current_relative_angle"], delta
    )
    candidate = state["candidate"].clone()
    candidate[..., :2] = _rotate_sincos(candidate[..., :2], delta)
    candidate[..., 4:6] = _rotate_sincos(candidate[..., 4:6], delta)
    out["candidate"] = candidate

    # v4 has an explicit per-view body-yaw branch in addition to the angle
    # geometry.  Rotate that branch as well for a consistent counterfactual.
    if variant == "v4" and "person_trajectory" in out:
        person = out["person_trajectory"].clone()
        person[..., -2:] = _rotate_sincos(person[..., -2:], delta)
        out["person_trajectory"] = person
    return out


def _configure(args: argparse.Namespace) -> tuple[Any, Path | None, dict[str, Any]]:
    return adapter.configure_variant(args, args.output_root)


def _load_raw(
    module: Any,
    cache_root: Path,
    track_root: Path | None,
    device: torch.device,
) -> Any:
    raw = module.GPUCache(cache_root / "test", device)
    module.attach_view_valid(raw, cache_root, "test")
    if track_root is not None:
        module.attach_object_tracks(raw, track_root, "test")
    return raw


def _build_state(
    variant: str,
    module: Any,
    model: torch.nn.Module,
    raw: Any,
    episodes: torch.Tensor,
    starts: torch.Tensor,
    shift_deg: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    history = starts[:, None]
    count = torch.ones_like(starts)
    if variant == "v2_legacy":
        state = module.build_angle_state(raw, episodes, history, count)
    else:
        state = module.build_trajectory_state(raw, episodes, history, count)
    baseline_state = state
    state = _rotate_state(state, shift_deg, variant)
    scores = model(state).masked_fill(~state["mask"], -torch.inf)
    actions = scores.argmax(-1)
    values = torch.topk(scores, k=2, dim=-1).values
    margins = values[:, 0] - values[:, 1]
    return actions, margins, baseline_state["candidate"]


def _metric_row(
    actions: torch.Tensor,
    baseline_actions: torch.Tensor,
    candidate: torch.Tensor,
    rank_to_original: torch.Tensor,
) -> dict[str, Any]:
    changed = actions.ne(baseline_actions)
    selected = candidate[torch.arange(len(actions), device=actions.device), actions]
    selected_angle = torch.atan2(selected[:, 0], selected[:, 1])
    circular_mean = torch.atan2(selected_angle.sin().mean(), selected_angle.cos().mean())
    selected_original = rank_to_original.gather(1, actions[:, None]).squeeze(1)
    baseline_original = rank_to_original.gather(1, baseline_actions[:, None]).squeeze(1)
    rank_hist = torch.bincount(actions, minlength=4).float() / max(len(actions), 1)
    return {
        "action_switch_rate": float(changed.float().mean()),
        "physical_view_switch_rate": float(
            selected_original.ne(baseline_original).float().mean()
        ),
        "selected_angle_circular_mean_deg": float(torch.rad2deg(circular_mean)),
        "rank_histogram": rank_hist.cpu().tolist(),
    }


def _ci95(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return 1.96 * float(np.std(values, ddof=1)) / math.sqrt(len(values))


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in (
        "action_switch_rate",
        "physical_view_switch_rate",
        "selected_angle_circular_mean_deg",
    ):
        values = [float(row[key]) for row in rows]
        result[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "ci95": _ci95(values),
        }
    hist = np.asarray([row["rank_histogram"] for row in rows], dtype=np.float64)
    result["rank_histogram_mean"] = hist.mean(0).tolist()
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    if len(args.eval_seeds) != 30:
        raise ValueError("exactly 30 evaluation seeds are required")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required")
    module, effective_track_root, provenance = _configure(args)
    raw = _load_raw(module, args.cache_root, effective_track_root, device)
    bank = base.class_bank_tensor(TRUE_UNSEEN, device)
    eligible = torch.isin(raw.labels, bank) & (raw.valid.sum(-1) >= 3)
    episodes = eligible.nonzero().flatten()
    rank_to_original = torch.from_numpy(
        np.asarray(
            np.load(args.cache_root / "test" / "rank_to_original.npy"), dtype=np.int64
        )
    ).to(device=device, dtype=torch.long)

    result: dict[str, Any] = {
        "experiment": "frame0_body_orientation_counterfactual",
        "variant": args.variant,
        "class_bank": TRUE_UNSEEN,
        "episodes": int(len(episodes)),
        "train_seeds": TRAIN_SEEDS,
        "eval_seeds": args.eval_seeds,
        "shifts_deg": SHIFTS_DEG,
        "counterfactual": "common rotation of all body-relative candidate bearings; trajectories, objects, reachability, logits and checkpoints unchanged",
        "provenance": provenance,
        "per_train_seed": {},
    }

    for train_seed in TRAIN_SEEDS:
        checkpoint = args.policy_root / f"seed_{train_seed}" / "best.pt"
        if args.variant == "v2_legacy":
            model = module.AngleObjectPolicy().to(device)
        else:
            model = module.AngleObjectTrajectoryPolicy().to(device)
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(payload["online"], strict=True)
        model.eval()

        by_shift: dict[str, list[dict[str, Any]]] = {str(x): [] for x in SHIFTS_DEG}
        for eval_seed in args.eval_seeds:
            starts = base.choose_random_starts(raw, episodes, eval_seed)
            baseline_actions, _, candidate = _build_state(
                args.variant, module, model, raw, episodes, starts, 0
            )
            for shift in SHIFTS_DEG:
                actions, margins, candidate_at_zero = _build_state(
                    args.variant, module, model, raw, episodes, starts, shift
                )
                selected_candidate = candidate_at_zero
                if shift != 0:
                    selected_candidate = candidate_at_zero.clone()
                    delta = math.radians(float(shift))
                    selected_candidate[..., :2] = _rotate_sincos(
                        selected_candidate[..., :2], delta
                    )
                    selected_candidate[..., 4:6] = _rotate_sincos(
                        selected_candidate[..., 4:6], delta
                    )
                row = _metric_row(
                    actions,
                    baseline_actions,
                    selected_candidate,
                    rank_to_original.index_select(0, episodes),
                )
                row["q_margin_mean"] = float(margins.mean())
                by_shift[str(shift)].append(row)

        result["per_train_seed"][str(train_seed)] = {
            "checkpoint": str(checkpoint),
            "by_shift": by_shift,
        }
        del model
        torch.cuda.empty_cache()

    # Average the three trained policies within each evaluation seed before
    # computing the seed-level confidence interval.
    aggregates: dict[str, Any] = {}
    for shift in SHIFTS_DEG:
        seed_rows: list[dict[str, Any]] = []
        for index, eval_seed in enumerate(args.eval_seeds):
            rows = [
                result["per_train_seed"][str(train_seed)]["by_shift"][str(shift)][
                    index
                ]
                for train_seed in TRAIN_SEEDS
            ]
            seed_rows.append(
                {
                    "action_switch_rate": float(
                        np.mean([row["action_switch_rate"] for row in rows])
                    ),
                    "physical_view_switch_rate": float(
                        np.mean([row["physical_view_switch_rate"] for row in rows])
                    ),
                    "selected_angle_circular_mean_deg": float(
                        np.mean(
                            [
                                row["selected_angle_circular_mean_deg"]
                                for row in rows
                            ]
                        )
                    ),
                    "rank_histogram": np.mean(
                        [row["rank_histogram"] for row in rows], axis=0
                    ).tolist(),
                }
            )
        aggregates[str(shift)] = _aggregate(seed_rows)
    result["aggregates"] = aggregates

    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--track-root", type=Path, required=True)
    parser.add_argument("--track-root-v3", type=Path, required=True)
    parser.add_argument("--policy-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--eval-seeds", type=int, nargs="+", default=DEFAULT_EVAL_SEEDS
    )
    args = parser.parse_args()
    result = run(args)
    print(
        "FRAME0_ORIENTATION_COUNTERFACTUAL_COMPLETE",
        args.variant,
        flush=True,
    )
    print(json.dumps(result["aggregates"], indent=2), flush=True)


if __name__ == "__main__":
    main()
