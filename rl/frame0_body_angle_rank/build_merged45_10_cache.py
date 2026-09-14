"""Build a strict 45/10 frame-0 angle-rank cache with the 45/10 VPOCLIP.

The geometric/state arrays are linked to the already validated frame-0
anatomical-yaw cache.  Only the frozen VPOCLIP ``z``/logit arrays and text
prototypes are regenerated from the strict merged45_10 checkpoint.  This
keeps the experiment small and, importantly, does not modify any historical
cache or checkpoint.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch

from ..vpoclip_adapter import VPOCLIPAdapter


ROOT = Path("/home/youhan/ws/VPOCLIP_plus_full")
SOURCE_CACHE = ROOT / "data/frame0_body_angle_rank_g1_pseudo_selected_stage_b/cache"
OUTPUT_CACHE = ROOT / "data/frame0_body_angle_rank_merged45_10/cache"
ZSL_ROOT = ROOT / "data/merged45_10_groupAB_zsl"
VPO_CONFIG = ROOT / "logs/merged45_10_groupAB_20260913/vpoclip_selected_stage_b.yaml"
VPO_CHECKPOINT = ROOT / "work_dir/merged45_10_groupAB/selected_stage_b/last_model.pth"

UNSEEN = [1, 7, 14, 15, 18, 25, 39, 46, 52, 54]
PSEUDO = [3, 8, 13, 20, 27, 28, 30, 35, 41, 50]
SPLITS = ("dqn_train", "val", "test")


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    destination.symlink_to(source.resolve())


def sample_lookup(prefixes: list[str]) -> dict[str, tuple[str, int]]:
    lookup: dict[str, tuple[str, int]] = {}
    for prefix in prefixes:
        names = np.load(ZSL_ROOT / f"{prefix}_sample_names.npy", allow_pickle=False).astype(str)
        for index, name in enumerate(names):
            if name in lookup:
                raise RuntimeError(f"duplicate sample name across 45/10 banks: {name}")
            lookup[name] = (prefix, index)
    return lookup


def open_inputs(prefix: str) -> dict[str, np.ndarray]:
    return {
        "video": np.load(ZSL_ROOT / f"{prefix}_video.npy", mmap_mode="r", allow_pickle=False),
        "pose": np.load(ZSL_ROOT / f"{prefix}_pose_coco17.npy", mmap_mode="r", allow_pickle=False),
        "object": np.load(ZSL_ROOT / f"{prefix}_object.npy", mmap_mode="r", allow_pickle=False),
        "joint_xy": np.load(ZSL_ROOT / f"{prefix}_joint_xy_coco17.npy", mmap_mode="r", allow_pickle=False),
    }


def encode_split(
    adapter: VPOCLIPAdapter,
    split: str,
    lookup: dict[str, tuple[str, int]],
    inputs: dict[str, dict[str, np.ndarray]],
    output_dir: Path,
    batch_episodes: int = 64,
) -> dict[str, object]:
    metadata = json.loads((SOURCE_CACHE / split / "metadata.json").read_text(encoding="utf-8"))
    episodes = metadata["episodes"]
    episode_view_names: list[list[str]] = []
    for episode in episodes:
        names = [str(view["sample_name"]) for view in episode.get("views", [])]
        if not 1 <= len(names) <= 4:
            raise RuntimeError(f"invalid candidate-view count for {episode.get('base_sample')}: {len(names)}")
        episode_view_names.append(names)
    sample_names = [name for names in episode_view_names for name in names]
    missing = sorted(set(sample_names) - set(lookup))
    if missing:
        raise RuntimeError(f"{split}: {len(missing)} sample names are absent from 45/10 arrays; first={missing[:5]}")

    n = len(episodes)
    z_path = output_dir / "z.npy"
    logits_path = output_dir / "logits.npy"
    z = np.lib.format.open_memmap(z_path, mode="w+", dtype=np.float32, shape=(n, 4, 512))
    logits = np.lib.format.open_memmap(logits_path, mode="w+", dtype=np.float32, shape=(n, 4, 55))
    parity_done = False
    max_logit_error = 0.0
    for start in range(0, n, batch_episodes):
        stop = min(n, start + batch_episodes)
        flat_names = [
            name
            for names in episode_view_names[start:stop]
            for name in (names + [names[0]] * (4 - len(names)))
        ]
        rows_by_prefix: dict[str, list[int]] = {prefix: [] for prefix in inputs}
        for name in flat_names:
            prefix, index = lookup[name]
            rows_by_prefix[prefix].append(index)
        # The source arrays are split by prefix, so preserve episode/view order
        # while gathering them into one compact tensor batch.
        gathered: dict[str, list[np.ndarray]] = {key: [] for key in ("video", "pose", "object", "joint_xy")}
        for name in flat_names:
            prefix, index = lookup[name]
            for key in gathered:
                gathered[key].append(np.asarray(inputs[prefix][key][index], dtype=np.float32))
        batch = {
            key: torch.from_numpy(np.stack(values, axis=0)).to(adapter.device, non_blocking=False)
            for key, values in gathered.items()
        }
        with torch.inference_mode():
            encoded = adapter.encode_all_views({key: value.view(stop - start, 4, *value.shape[1:]) for key, value in batch.items()})
            if not parity_done:
                reference = adapter.model(
                    batch["video"], batch["pose"], batch["object"], batch["joint_xy"]
                )
                error = float((encoded["logits"].reshape(-1, 55) - reference).abs().max().item())
                if error > 1e-4:
                    raise RuntimeError(f"forward parity failed: max_logit_error={error}")
                max_logit_error = error
                parity_done = True
        z_batch = encoded["z"].detach().cpu().numpy()
        logits_batch = encoded["logits"].detach().cpu().numpy()
        for episode_index, names in enumerate(episode_view_names[start:stop]):
            valid_count = len(names)
            if valid_count < 4:
                z_batch[episode_index, valid_count:] = 0
                logits_batch[episode_index, valid_count:] = 0
        z[start:stop] = z_batch
        logits[start:stop] = logits_batch
        z.flush()
        logits.flush()
        print(f"MERGED45_10_CACHE {split} {stop}/{n}", flush=True)
    return {"episodes": n, "max_logit_error": max_logit_error}


def prepare_split(split: str) -> None:
    source = SOURCE_CACHE / split
    destination = OUTPUT_CACHE / split
    destination.mkdir(parents=True, exist_ok=True)
    # Link all state/geometry arrays and copy the metadata so the source cache
    # remains immutable and the new recognizer contract is explicit.
    linked = (
        "pose.npy", "object_map.npy", "image_quality.npy", "view_geometry.npy",
        "reachable.npy", "move_cost.npy", "labels.npy", "target_columns.npy",
        "body_yaw_deg.npy", "body_relative_angle_deg.npy", "rank_to_original.npy",
        "original_to_rank.npy", "view_valid.npy", "manifest.jsonl",
    )
    for filename in linked:
        link_or_copy(source / filename, destination / filename)
    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    metadata["recognizer"] = {
        "config": str(VPO_CONFIG),
        "checkpoint": str(VPO_CHECKPOINT),
        "protocol": "strict_zsl_45_10",
    }
    metadata["split_override"] = {
        "seen_classes": sorted(set(range(55)) - set(UNSEEN)),
        "pseudo_unseen_classes": PSEUDO,
        "unseen_classes": UNSEEN,
    }
    metadata["source_geometry_cache"] = str(SOURCE_CACHE / split)
    (destination / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    if not VPO_CONFIG.exists() or not VPO_CHECKPOINT.exists():
        raise FileNotFoundError(f"missing strict 45/10 recognizer: {VPO_CONFIG} / {VPO_CHECKPOINT}")
    if not SOURCE_CACHE.exists():
        raise FileNotFoundError(SOURCE_CACHE)
    OUTPUT_CACHE.mkdir(parents=True, exist_ok=True)
    marker = OUTPUT_CACHE / "build_summary.json"
    if marker.exists():
        print(f"MERGED45_10_CACHE_ALREADY_COMPLETE {marker}", flush=True)
        return

    unseen_lookup = sample_lookup(["trimodal_test"])
    seen_lookup = sample_lookup(["trimodal_test_seen"])
    train_lookup = sample_lookup(["trimodal_train"])
    val_lookup = sample_lookup(["trimodal_val"])
    lookup_by_split = {
        "dqn_train": train_lookup,
        "val": val_lookup,
        "test": {**unseen_lookup, **seen_lookup},
    }
    inputs = {
        "trimodal_train": open_inputs("trimodal_train"),
        "trimodal_val": open_inputs("trimodal_val"),
        "trimodal_test": open_inputs("trimodal_test"),
        "trimodal_test_seen": open_inputs("trimodal_test_seen"),
    }
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        raise RuntimeError("CUDA is required for strict 45/10 cache regeneration")
    adapter = VPOCLIPAdapter.from_config(str(VPO_CONFIG), str(VPO_CHECKPOINT), device)
    results: dict[str, object] = {}
    for split in SPLITS:
        prepare_split(split)
        results[split] = encode_split(adapter, split, lookup_by_split[split], inputs, OUTPUT_CACHE / split)
    # The evaluator expects a seen_test directory. The class filter is supplied
    # by the strict 45/10 runner, so it is exactly the same test cache.
    seen_test = OUTPUT_CACHE / "seen_test"
    if not seen_test.exists():
        seen_test.symlink_to((OUTPUT_CACHE / "test").resolve(), target_is_directory=True)
    np.save(OUTPUT_CACHE / "text_prototypes.npy", adapter.model.text_features.detach().float().cpu().numpy())
    protocol = {
        "protocol": "strict_zsl_45_10",
        "unseen_classes": UNSEEN,
        "seen_classes": sorted(set(range(55)) - set(UNSEEN)),
        "pseudo_unseen_classes": PSEUDO,
        "v_poclip_config": str(VPO_CONFIG),
        "v_poclip_checkpoint": str(VPO_CHECKPOINT),
        "geometry_source": str(SOURCE_CACHE),
        "historical_old_split_not_used": [0, 2, 26, 34, 50],
        "results": results,
    }
    (OUTPUT_CACHE / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    marker.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print("MERGED45_10_CACHE_COMPLETE", json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
