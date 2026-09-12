"""Diagnose margin and entropy for standalone second-view selection.

This replays the completed single-state trajectory experiments with exactly
the same legal-action masks, checkpoints, class banks, and evaluation seeds
as ``evaluate_trajectory_single_state.py``.  It adds confidence diagnostics
for the policy-selected and random second views, both standalone and after
two-view logit fusion.

Definitions (computed after restricting logits to the five-way test bank):

* decision margin: largest logit minus second-largest logit;
* normalized entropy: H(softmax(logits)) / log(5);
* GT margin: ground-truth-class logit minus the strongest rival logit.

The first two are prediction-confidence diagnostics.  GT margin is included
because it preserves the direction of correctness even when the prediction
is wrong.  No labels or future logits are passed to the policy; labels are
used only here for offline evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from train_trajectory_single_state import (  # noqa: E402
    EVAL_SEEDS,
    NEW_TRUE_UNSEEN,
    OLD_TRUE_UNSEEN,
    configure_variant,
    load_raw,
)
from rl import multistep_angle_object_trajectory_policy_v1 as base  # noqa: E402


TRAIN_SEEDS = (20260909, 20260910, 20260911)
VARIANTS = ("v1", "v4", "v5", "v6", "object_only", "geometry_only")


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def ci95(values: Sequence[float]) -> list[float]:
    values = np.asarray([float(x) for x in values], dtype=np.float64)
    if values.size < 2:
        return [float(values.mean()), float(values.mean())]
    critical = 2.045229642132703 if values.size >= 30 else 4.302652729911275
    half = critical * float(values.std(ddof=1)) / math.sqrt(values.size)
    mean = float(values.mean())
    return [mean - half, mean + half]


def aggregate(values: Sequence[float]) -> dict[str, float | list[float]]:
    values = [float(x) for x in values]
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "ci95": ci95(values),
    }


def confidence(scores: torch.Tensor, labels: torch.Tensor, bank: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return per-episode standalone confidence statistics."""

    ordered = scores.sort(dim=-1, descending=True).values
    decision_margin = ordered[:, 0] - ordered[:, 1]
    probabilities = torch.softmax(scores, dim=-1)
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(-1)
    entropy = entropy / math.log(float(bank.numel()))

    local = torch.searchsorted(bank, labels)
    target = scores.gather(-1, local[:, None]).squeeze(-1)
    rival = scores.clone()
    rival.scatter_(1, local[:, None], -torch.inf)
    gt_margin = target - rival.max(-1).values
    correct = scores.argmax(-1).eq(local)
    return {
        "top1": correct.float(),
        "decision_margin": decision_margin,
        "entropy": entropy,
        "gt_margin": gt_margin,
    }


def metric_means(
    raw: Any,
    episodes: torch.Tensor,
    first: torch.Tensor,
    second: torch.Tensor,
    bank: torch.Tensor,
) -> dict[str, float]:
    labels = raw.labels[episodes]
    standalone = raw.logits[episodes, second].index_select(-1, bank)
    fused = (raw.logits[episodes, first] + raw.logits[episodes, second]).mul(0.5)
    fused = fused.index_select(-1, bank)
    single_stats = confidence(standalone, labels, bank)
    fused_stats = confidence(fused, labels, bank)
    return {
        "top1": float(single_stats["top1"].mean()),
        "decision_margin": float(single_stats["decision_margin"].mean()),
        "entropy": float(single_stats["entropy"].mean()),
        "gt_margin": float(single_stats["gt_margin"].mean()),
        "fused_top1": float(fused_stats["top1"].mean()),
        "fused_decision_margin": float(fused_stats["decision_margin"].mean()),
        "fused_entropy": float(fused_stats["entropy"].mean()),
        "fused_gt_margin": float(fused_stats["gt_margin"].mean()),
    }


@torch.inference_mode()
def one_protocol(
    model: torch.nn.Module,
    state_builder: Any,
    raw: Any,
    classes: Sequence[int],
    protocol: str,
    seed: int,
) -> dict[str, Any]:
    bank = base.class_bank_tensor(classes, raw.device)
    eligible = torch.isin(raw.labels, bank) & (raw.valid.sum(-1) >= 2)
    episodes = eligible.nonzero().flatten()
    if protocol.startswith("fixed"):
        start = int(protocol.removeprefix("fixed"))
        episodes = (eligible & raw.valid[:, start]).nonzero().flatten()
        starts = torch.full_like(episodes, start)
    elif protocol == "random_start":
        starts = base.choose_random_starts(raw, episodes, seed)
    else:
        raise ValueError(protocol)

    count = torch.ones(len(episodes), device=raw.device, dtype=torch.long)
    state = state_builder(raw, episodes, starts[:, None], count)
    choiceable = state["mask"].any(-1)
    episodes = episodes[choiceable]
    starts = starts[choiceable]
    state = {key: value[choiceable] for key, value in state.items()}
    valid = state["mask"]
    policy_second = model(state).masked_fill(~valid, -torch.inf).argmax(-1)

    rng = np.random.default_rng(seed + 100_003)
    random_second = torch.empty_like(policy_second)
    for row, options in enumerate(valid.detach().cpu().numpy()):
        random_second[row] = int(rng.choice(np.flatnonzero(options)))

    policy = metric_means(raw, episodes, starts, policy_second, bank)
    random = metric_means(raw, episodes, starts, random_second, bank)
    return {
        "episodes": int(len(episodes)),
        "policy": policy,
        "random": random,
        "delta_policy_minus_random": {
            key: policy[key] - random[key] for key in policy
        },
    }


def evaluate_variant(
    variant: str,
    split: str,
    cache_root: Path,
    track_root: Path,
    policy_root: Path,
    eval_seeds: Sequence[int],
    device: torch.device,
) -> dict[str, Any]:
    model_cls, attach_tracks, state_builder = configure_variant(variant)
    raw = load_raw(cache_root, track_root, "test", device, attach_tracks)
    classes = OLD_TRUE_UNSEEN if split == "old" else NEW_TRUE_UNSEEN
    by_train_seed: dict[str, Any] = {}

    for train_seed in TRAIN_SEEDS:
        checkpoint = policy_root / variant / f"seed_{train_seed}" / "best.pt"
        model = model_cls().to(device)
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(saved["online"], strict=True)
        model.eval()
        per_protocol: dict[str, Any] = {}
        for protocol in ("fixed0", "random_start"):
            reports = [
                one_protocol(model, state_builder, raw, classes, protocol, int(seed))
                for seed in eval_seeds
            ]
            # The policy is deterministic for fixed0; random-start changes the
            # sampled initial/second view with the evaluation seed.
            per_protocol[protocol] = reports
        by_train_seed[str(train_seed)] = per_protocol
        del model
        torch.cuda.empty_cache()

    result: dict[str, Any] = {
        "variant": variant,
        "split": split,
        "class_bank": [int(x) for x in classes],
        "episodes": int(torch.isin(raw.labels, base.class_bank_tensor(classes, device)).logical_and(raw.valid.sum(-1) >= 2).sum()),
        "train_seeds": list(TRAIN_SEEDS),
        "eval_seeds": [int(x) for x in eval_seeds],
        "definitions": {
            "decision_margin": "top-1 class logit minus top-2 class logit in the five-way bank",
            "entropy": "normalized categorical entropy H(softmax(logits))/log(5)",
            "gt_margin": "ground-truth class logit minus strongest rival logit",
            "delta": "policy minus random; for entropy, negative is better",
        },
        "protocols": {},
    }

    for protocol in ("fixed0", "random_start"):
        # First average each train checkpoint over evaluation seeds, then
        # aggregate checkpoints. This mirrors the existing report's handling
        # of train seeds and evaluation seeds.
        train_level: list[dict[str, Any]] = []
        for train_seed in TRAIN_SEEDS:
            reports = by_train_seed[str(train_seed)][protocol]
            names = tuple(reports[0]["policy"].keys())
            train_level.append({
                "policy": {name: float(np.mean([r["policy"][name] for r in reports])) for name in names},
                "random": {name: float(np.mean([r["random"][name] for r in reports])) for name in names},
                "delta_policy_minus_random": {
                    name: float(np.mean([r["delta_policy_minus_random"][name] for r in reports]))
                    for name in names
                },
            })
        names = tuple(train_level[0]["policy"].keys())
        summary: dict[str, Any] = {"policy": {}, "random": {}, "delta_policy_minus_random": {}}
        for group in ("policy", "random", "delta_policy_minus_random"):
            for name in names:
                summary[group][name] = aggregate([row[group][name] for row in train_level])
        result["protocols"][protocol] = {
            "summary": summary,
            "train_seed_level": train_level,
        }
    return result


def markdown(results: list[dict[str, Any]]) -> str:
    lines = [
        "# Single-state margin and entropy diagnostics",
        "",
        "All values are computed on the true five-way unseen bank. `delta` is policy minus random; for entropy, a negative delta means lower uncertainty.",
        "",
        "## Standalone selected second view",
        "",
        "| split | variant | protocol | Top-1 policy/random | margin policy/random | Δmargin | H policy/random | ΔH | GT-margin policy/random |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in results:
        for protocol in ("fixed0", "random_start"):
            summary = item["protocols"][protocol]["summary"]
            p = summary["policy"]
            r = summary["random"]
            d = summary["delta_policy_minus_random"]
            lines.append(
                f"| {item['split']} | {item['variant']} | {protocol} | "
                f"{p['top1']['mean']*100:.2f}/{r['top1']['mean']*100:.2f}% | "
                f"{p['decision_margin']['mean']:.4f}/{r['decision_margin']['mean']:.4f} | "
                f"{d['decision_margin']['mean']:+.4f} | "
                f"{p['entropy']['mean']:.4f}/{r['entropy']['mean']:.4f} | "
                f"{d['entropy']['mean']:+.4f} | "
                f"{p['gt_margin']['mean']:.4f}/{r['gt_margin']['mean']:.4f} |"
            )
    lines += [
        "",
        "## Two-view fused diagnostics",
        "",
        "| split | variant | protocol | Top-1 policy/random | margin policy/random | Δmargin | H policy/random | ΔH | GT-margin policy/random |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in results:
        for protocol in ("fixed0", "random_start"):
            summary = item["protocols"][protocol]["summary"]
            p = summary["policy"]
            r = summary["random"]
            d = summary["delta_policy_minus_random"]
            lines.append(
                f"| {item['split']} | {item['variant']} | {protocol} | "
                f"{p['fused_top1']['mean']*100:.2f}/{r['fused_top1']['mean']*100:.2f}% | "
                f"{p['fused_decision_margin']['mean']:.4f}/{r['fused_decision_margin']['mean']:.4f} | "
                f"{d['fused_decision_margin']['mean']:+.4f} | "
                f"{p['fused_entropy']['mean']:.4f}/{r['fused_entropy']['mean']:.4f} | "
                f"{d['fused_entropy']['mean']:+.4f} | "
                f"{p['fused_gt_margin']['mean']:.4f}/{r['fused_gt_margin']['mean']:.4f} |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-cache-root", type=Path, required=True)
    parser.add_argument("--old-track-root", type=Path, required=True)
    parser.add_argument("--old-policy-root", type=Path, required=True)
    parser.add_argument("--new-cache-root", type=Path, required=True)
    parser.add_argument("--new-track-root", type=Path, required=True)
    parser.add_argument("--new-policy-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("confidence diagnostics require CUDA")
    # Match evaluate_trajectory_single_state.py, whose process uses PyTorch's
    # default/highest matmul precision.  ``high`` can change a few near-tied
    # candidate actions and would make the confidence replay disagree with
    # the published Top-1 table.
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda:0")
    results: list[dict[str, Any]] = []
    for split, cache_root, track_root, policy_root in (
        ("old", args.old_cache_root, args.old_track_root, args.old_policy_root),
        ("new", args.new_cache_root, args.new_track_root, args.new_policy_root),
    ):
        for variant in args.variants:
            print("CONFIDENCE_DIAGNOSTIC_START", split, variant, flush=True)
            result = evaluate_variant(
                variant, split, cache_root, track_root, policy_root, EVAL_SEEDS, device
            )
            results.append(result)
            print("CONFIDENCE_DIAGNOSTIC_COMPLETE", split, variant, flush=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    dump(args.output_root / "confidence_metrics.json", {"results": results})
    (args.output_root / "confidence_metrics.md").write_text(markdown(results), encoding="utf-8")
    print("CONFIDENCE_DIAGNOSTIC_ALL_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
