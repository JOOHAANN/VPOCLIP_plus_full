"""Exact path-selection comparison on the saved weighted-fusion experiment.

The policy and random predictions are read verbatim from
weighted_fusion_policy_variant_comparison_v1.  Only the missing
Preferred-Orthogonal paths are evaluated from the same frozen cache.  This
avoids rerunning a policy on CUDA and guarantees that the comparison uses the
exact saved weighted-fusion data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from . import fusion_weighted_common_v1 as common
from . import policy_orthogonal_random_comparison_v1 as helpers
from . import weighted_fusion_policy_variants_v1 as variants


ROOT = variants.ROOT
DEVICE = variants.DEVICE
SOURCE_ROOT = ROOT / "work_dir/weighted_fusion_policy_variant_comparison_v1"
OUTPUT_ROOT = ROOT / "work_dir/policy_orthogonal_random_comparison_v2_exact_weighted"
EVAL_SEEDS = variants.EVAL_SEEDS
METHODS = variants.ALL_METHODS
NEW_VARIANTS = variants.TRAJECTORY_VARIANTS
NEW_TRUE_UNSEEN = variants.NEW_TRUE_UNSEEN
NEW_SEEN = variants.NEW_SEEN
OLD_TRUE_UNSEEN = variants.OLD_TRUE_UNSEEN
OLD_SEEN = variants.OLD_SEEN


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def local_correct_rows(
    detail_path: Path,
    raw: Any,
    classes: Sequence[int],
    method: str,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Read exact saved predictions and convert them to correctness rows."""

    payload = np.load(detail_path, allow_pickle=True)
    ids = payload["episode_indices"].astype(np.int64)
    labels = raw.labels[torch.as_tensor(ids, device=raw.device)].detach().cpu().numpy().astype(np.int64)
    local = np.searchsorted(np.asarray(sorted(classes), dtype=np.int64), labels)
    predictions = payload[f"prediction_{method}"].astype(np.int64)
    correct = (predictions == local[None, :]).astype(np.float32)
    return [(ids, row) for row in correct]


def evaluate_orthogonal(
    raw: Any,
    classes: Sequence[int],
    gate: torch.nn.Module,
    protocol: str,
    moves: int,
    output_root: Path,
) -> tuple[dict[str, Any], dict[str, list[tuple[np.ndarray, np.ndarray]]]]:
    """Evaluate only the orthogonal path using the shared weighted fusion."""

    base_episodes = common.eligible_episodes(raw, classes, minimum_views=moves + 1)
    rows = {method: [] for method in METHODS}
    records: list[dict[str, Any]] = []
    bank = common.class_bank(classes, raw.device)

    for eval_seed in EVAL_SEEDS:
        if protocol.startswith("fixed"):
            start = int(protocol.removeprefix("fixed"))
            episodes = base_episodes[raw.valid[base_episodes, start]]
            starts = torch.full_like(episodes, start)
        elif protocol == "random_start":
            episodes = base_episodes
            starts = common.random_starts(raw, episodes, eval_seed)
        else:
            raise ValueError(protocol)

        path_episodes, paths = common.path_candidates(
            raw,
            episodes,
            starts,
            f"Preferred-Orthogonal-{moves + 1}",
            eval_seed + 173,
        )
        if not len(path_episodes):
            continue
        ids = path_episodes.detach().cpu().numpy()
        record = {"ids": ids, "paths": paths.detach().cpu().numpy(), "predictions": {}, "weights": {}}
        for method in METHODS:
            metrics = common.path_metrics(raw, path_episodes, paths, bank, method, gate)
            rows[method].append((ids, metrics["correct"].detach().cpu().numpy().astype(np.float32)))
            record["predictions"][method] = metrics["prediction"].detach().cpu().numpy()
            record["weights"][method] = metrics["weights"].detach().cpu().numpy().astype(np.float32)
        records.append(record)

    summaries, common_ids = variants.summarize_rows(rows, 810000 + moves * 100)
    detail_path = output_root / "details" / f"{protocol}_moves_{moves}_preferred_orthogonal.npz"
    helpers.save_details(detail_path, records, common_ids)
    summaries = dict(summaries)
    return {
        "methods": summaries,
        "episodes": int(len(common_ids)),
        "details": str(detail_path),
    }, rows


def source_json(group: str, variant: str, split: str, protocol: str, moves: int) -> Path:
    if group == "v7_old_split_414":
        return SOURCE_ROOT / group / split / protocol / f"moves_{moves}.json"
    return SOURCE_ROOT / group / variant / split / protocol / f"moves_{moves}.json"


def evaluate_setting(
    group: str,
    variant: str,
    raw: Any,
    classes: Sequence[int],
    gate: torch.nn.Module,
    split: str,
    protocol: str,
    moves: int,
) -> dict[str, Any]:
    source_path = source_json(group, variant, split, protocol, moves)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source_policy_details = Path(source["path_types"]["policy"]["details"])
    source_random_details = Path(source["path_types"]["random"]["details"])
    out_root = OUTPUT_ROOT / group / variant / split / protocol
    orth_summary, orth_rows = evaluate_orthogonal(raw, classes, gate, protocol, moves, out_root)

    result = {
        "protocol": protocol,
        "moves": moves,
        "fusion_views": moves + 1,
        "eval_seeds": list(EVAL_SEEDS),
        "source_weighted_fusion_json": str(source_path),
        "path_types": {
            "policy": source["path_types"]["policy"],
            "preferred_orthogonal": orth_summary,
            "random": source["path_types"]["random"],
        },
        "paired_vs_random": {},
    }
    for path_type, detail_path in (
        ("policy", source_policy_details),
        ("preferred_orthogonal", None),
        ("random", source_random_details),
    ):
        result["paired_vs_random"][path_type] = {}
        for method_index, method in enumerate(METHODS):
            if path_type == "preferred_orthogonal":
                left_rows = orth_rows[method]
            else:
                left_rows = local_correct_rows(detail_path, raw, classes, method)
            random_rows = local_correct_rows(source_random_details, raw, classes, method)
            result["paired_vs_random"][path_type][method] = helpers.paired_delta(
                left_rows,
                random_rows,
                820000 + method_index * 101 + moves * 1000,
            )
    return result


def run_new(gate: torch.nn.Module) -> None:
    raws = {
        "seen": variants.load_split_cache(variants.NEW_CACHE, "seen_test"),
        "true_unseen_5way": variants.load_split_cache(variants.NEW_CACHE, "test"),
    }
    for variant in NEW_VARIANTS:
        variant_out = OUTPUT_ROOT / "new_split_271" / variant
        dump(
            variant_out / "config_resolved.json",
            {
                "variant": variant,
                "source": str(SOURCE_ROOT / "new_split_271" / variant),
                "comparison": ["policy", "preferred_orthogonal", "random"],
                "fusion_methods": list(METHODS),
                "eval_seeds": EVAL_SEEDS,
                "exact_saved_policy_and_random_reused": True,
            },
        )
        for split, raw in raws.items():
            classes = NEW_SEEN if split == "seen" else NEW_TRUE_UNSEEN
            for protocol in ("fixed0", "fixed1", "fixed2", "fixed3", "random_start"):
                for moves in (1, 2):
                    result = evaluate_setting(
                        "new_split_271", variant, raw, classes, gate,
                        split, protocol, moves,
                    )
                    dump(variant_out / split / protocol / f"moves_{moves}.json", result)
    del raws
    torch.cuda.empty_cache()


def run_v7(gate: torch.nn.Module) -> None:
    raw_seen = variants.load_split_cache(variants.V7_CACHE, "seen_test")
    raw_test = variants.load_split_cache(variants.V7_CACHE, "test")
    variant_out = OUTPUT_ROOT / "v7_old_split_414" / "v7"
    dump(
        variant_out / "config_resolved.json",
        {
            "variant": "v7",
            "source": str(SOURCE_ROOT / "v7_old_split_414"),
            "comparison": ["policy", "preferred_orthogonal", "random"],
            "fusion_methods": list(METHODS),
            "eval_seeds": EVAL_SEEDS,
            "exact_saved_policy_and_random_reused": True,
        },
    )
    for split, raw in (("seen", raw_seen), ("true_unseen_5way", raw_test)):
        classes = OLD_SEEN if split == "seen" else OLD_TRUE_UNSEEN
        result = evaluate_setting(
            "v7_old_split_414", "v7", raw, classes, gate,
            split, "random_start", 1,
        )
        dump(variant_out / split / "random_start" / "moves_1.json", result)
        for start in range(4):
            result = evaluate_setting(
                "v7_old_split_414", "v7", raw, classes, gate,
                split, f"fixed{start}", 1,
            )
            dump(variant_out / split / f"fixed{start}" / "moves_1.json", result)
    del raw_seen, raw_test
    torch.cuda.empty_cache()


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    new_gate = variants.load_gate(variants.NEW_GATE_CHECKPOINT)
    old_gate_summary = SOURCE_ROOT / "gates/v7_old_split/summary.json"
    old_gate = variants.load_gate(Path(json.loads(old_gate_summary.read_text())["selected_checkpoint"]))
    run_new(new_gate)
    run_v7(old_gate)
    dump(
        OUTPUT_ROOT / "summary.json",
        {
            "experiment": "exact_saved_weighted_fusion_policy_vs_preferred_orthogonal_vs_random",
            "source_weighted_fusion_root": str(SOURCE_ROOT),
            "path_types": ["policy", "preferred_orthogonal", "random"],
            "trajectory_variants": list(NEW_VARIANTS),
            "eval_seeds": EVAL_SEEDS,
            "fusion_methods": list(METHODS),
            "exact_saved_policy_and_random_reused": True,
            "no_historical_outputs_modified": True,
        },
    )
    print("EXACT_WEIGHTED_POLICY_ORTHOGONAL_RANDOM_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
