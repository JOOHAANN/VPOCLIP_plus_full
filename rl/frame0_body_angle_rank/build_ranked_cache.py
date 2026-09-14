"""Build an isolated cache whose four action slots are frame-0 angle ranks.

No old cache is modified.  The script performs a complete skeleton preflight
before writing the destination so a missing original 3-D skeleton cannot
produce a partially valid experiment.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from .frame0_body_angle import (
    LOW_CAMERAS,
    resolve_episode,
    write_csv,
    write_json,
    load_angle_csv,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_CACHE = ROOT / "data" / "rl_raw_new50_5_group1_g1_pseudo_selected_stage_b" / "cache"
DEFAULT_SOURCE_TRACKS = ROOT / "data" / "trajectory_g1_pseudo_selected_stage_b" / "object_tracks_full"
DEFAULT_SOURCE_TRACKS_V3 = ROOT / "data" / "angle_object_trajectory_v3_bbox_size"
DEFAULT_OUTPUT_CACHE = ROOT / "data" / "frame0_body_angle_rank_g1_pseudo_selected_stage_b" / "cache"
DEFAULT_OUTPUT_TRACKS = ROOT / "data" / "frame0_body_angle_rank_g1_pseudo_selected_stage_b" / "object_tracks_full"
DEFAULT_OUTPUT_TRACKS_V3 = ROOT / "data" / "frame0_body_angle_rank_g1_pseudo_selected_stage_b" / "object_tracks_v3_bbox_size"
DEFAULT_ANGLE_INDEX = ROOT.parent / "ETRI-Activity3D-RGB" / "low_view_camera_angles_split55.csv"
SPLITS = ("dqn_train", "val", "test")
VIEW_AXIS_FILES = ("z", "logits", "pose", "object_map", "image_quality", "view_geometry", "view_valid")
PAIR_AXIS_FILES = ("reachable", "move_cost")


def load_metadata(root: Path) -> dict[str, Any]:
    return json.loads((root / "metadata.json").read_text(encoding="utf-8"))


def resolve_all(
    source_cache: Path,
    angle_index: Path,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    rows = load_angle_csv(angle_index)
    all_resolved: dict[str, list[dict[str, Any]]] = {}
    csv_rows: list[dict[str, Any]] = []
    skeleton_cache: dict[Path, dict[int, tuple[np.ndarray, np.ndarray]]] = {}
    for split in SPLITS:
        split_root = source_cache / split
        metadata = load_metadata(split_root)
        episodes = metadata["episodes"]
        valid = np.load(split_root / "view_valid.npy", mmap_mode="r", allow_pickle=False)
        if valid.shape != (len(episodes), 4):
            raise ValueError(f"{split} view_valid shape {valid.shape} != {(len(episodes), 4)}")
        resolved_split: list[dict[str, Any]] = []
        for index, episode in enumerate(episodes):
            item = resolve_episode(episode, rows, valid[index], index, skeleton_cache)
            if len(item['views']) < 2:
                print(f'EXCLUDED_EPISODE split={split} index={index} valid={len(item["views"])}', flush=True)
                continue
            resolved_split.append(item)
            for view in item["views"]:
                csv_rows.append({"split": split, **view})
        all_resolved[split] = resolved_split
        print(f"FRAME0_ANGLE_PREFLIGHT split={split} episodes={len(resolved_split)}", flush=True)
    print(f"FRAME0_ANGLE_SKELETON_FILES loaded={len(skeleton_cache)}", flush=True)
    return all_resolved, csv_rows


def reorder_view_array(array: np.ndarray, resolved: list[dict[str, Any]], fill: float | bool = 0) -> np.ndarray:
    if array.ndim < 2 or array.shape[1] != 4:
        raise ValueError(f"expected view-axis array [N,4,...], got {array.shape}")
    output = np.full((len(resolved), *array.shape[1:]), fill, dtype=array.dtype)
    for episode_index, item in enumerate(resolved):
        for view in item["views"]:
            rank = int(view["angle_rank"])
            original = int(view["original_view_index"])
            output[episode_index, rank] = array[item['episode_index'], original]
    return output


def reorder_pair_array(array: np.ndarray, resolved: list[dict[str, Any]]) -> np.ndarray:
    if array.shape[1:3] != (4, 4):
        raise ValueError(f"expected pair-axis array [N,4,4,...], got {array.shape}")
    output = np.zeros((len(resolved), *array.shape[1:]), dtype=array.dtype)
    for episode_index, item in enumerate(resolved):
        mapping = item["rank_to_original"]
        for rank_i, original_i in enumerate(mapping):
            if original_i < 0:
                continue
            for rank_j, original_j in enumerate(mapping):
                if original_j < 0:
                    continue
                output[episode_index, rank_i, rank_j] = array[item['episode_index'], original_i, original_j]
    return output


def write_split(
    source_cache: Path,
    output_cache: Path,
    split: str,
    resolved: list[dict[str, Any]],
    angle_csv_name: str,
) -> None:
    source = source_cache / split
    destination = output_cache / split
    destination.mkdir(parents=True, exist_ok=False)

    for path in source.iterdir():
        if path.is_file() and path.suffix not in {".npy", ".json", ".jsonl"}:
            shutil.copy2(path, destination / path.name)

    for name in VIEW_AXIS_FILES:
        array = np.load(source / f"{name}.npy", mmap_mode="r", allow_pickle=False)
        if name == "view_geometry":
            reordered = reorder_view_array(np.asarray(array), resolved, fill=0.0).astype(np.float32, copy=False)
            angles = np.full((len(resolved), 4), np.nan, dtype=np.float32)
            for episode_index, item in enumerate(resolved):
                for view in item["views"]:
                    angles[episode_index, int(view["angle_rank"])] = float(view["angle_deg"])
            radians = np.deg2rad(np.nan_to_num(angles, nan=0.0))
            reordered[..., 0] = np.sin(radians)
            reordered[..., 1] = np.cos(radians)
            if reordered.shape[-1] > 2:
                reordered[..., 2:] = 0.0
        else:
            reordered = reorder_view_array(np.asarray(array), resolved, fill=False if name == "view_valid" else 0)
        np.save(destination / f"{name}.npy", reordered)

    for name in PAIR_AXIS_FILES:
        array = np.load(source / f"{name}.npy", mmap_mode="r", allow_pickle=False)
        reordered = reorder_pair_array(np.asarray(array), resolved)
        if name == 'reachable':
            for j in range(4):
                reordered[:, j, j] = True
        np.save(destination / f"{name}.npy", reordered)

    for name in ("labels", "target_columns", "text_prototypes"):
        source_path = source / f"{name}.npy"
        if source_path.exists():
            if name in ('labels', 'target_columns'):
                np.save(destination / source_path.name, np.load(source_path)[[i['episode_index'] for i in resolved]])
            else:
                shutil.copy2(source_path, destination / source_path.name)

    source_metadata = load_metadata(source)
    episodes = []
    for item in resolved:
        old_episode = source_metadata['episodes'][item['episode_index']]
        episode = dict(old_episode)
        new_views = []
        for view in item["views"]:
            old_view = dict(old_episode["views"][int(view["original_view_index"])])
            old_view.update(
                {
                    "angle_rank": int(view["angle_rank"]),
                    "action_index": int(view["action_index"]),
                    "original_view_index": int(view["original_view_index"]),
                    "original_camera": str(view["original_camera"]),
                    "body_relative_angle_deg": float(view["angle_deg"]),
                    "angle_source": str(view["angle_source"]),
                    "skeleton_path": str(view["skeleton_path"]),
                    "skeleton_frame_row": int(view["skeleton_frame_row"]),
                    "skeleton_indexing": str(view["skeleton_indexing"]),
                }
            )
            new_views.append(old_view)
        episode["views"] = new_views
        episode["rank_to_original"] = item["rank_to_original"]
        episode["original_to_rank"] = item["original_to_rank"]
        episode["angle_deg_by_rank"] = item["angle_deg_by_rank"]
        episodes.append(episode)

    output_metadata = dict(source_metadata)
    output_metadata["episodes"] = episodes
    output_metadata["angle_csv"] = angle_csv_name
    output_metadata["geometry_contract"] = {
        "definition": "signed bearing from frame-0 anatomical body-forward to person-to-camera direction",
        "coordinate_frame": "each video's released 3-D skeleton camera-local frame",
        "sign": "negative=body-left, positive=body-right",
        "rank": "within each episode, ascending signed angle; rank 0 is smallest and rank 3 largest",
        "action_semantics": "action k selects angle rank k, then rank_to_original maps to the source video slot",
        "camera_id_is_not_angular_order": True,
        "fallback": "none; missing skeleton is a hard error",
        "frame0_policy": "skeleton row 0 preferred; row 1 only when row 0 is absent/unusable, recorded per view",
    }
    output_metadata["human_yaw_contract"] = {
        "array": "body_yaw_deg.npy",
        "definition": "frame-0 anatomical body-forward yaw in the corresponding camera-local skeleton frame",
        "unit": "degrees",
        "encoding": "raw degrees; v4 converts to sin/cos before the policy",
        "source": "released 25-joint 3-D skeleton",
        "rtmpose_used_for_yaw": False,
    }
    output_metadata["source_cache"] = str(source_cache)
    output_metadata["frame0_angle_rank_version"] = "v1"
    (destination / "metadata.json").write_text(
        json.dumps(output_metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (destination / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for episode in episodes:
            handle.write(json.dumps(episode, ensure_ascii=False) + "\n")
    np.save(destination / "rank_to_original.npy", np.asarray([item["rank_to_original"] for item in resolved], dtype=np.int64))
    np.save(destination / "original_to_rank.npy", np.asarray([item["original_to_rank"] for item in resolved], dtype=np.int64))
    angles = np.full((len(resolved), 4), np.nan, dtype=np.float32)
    body_yaw = np.full((len(resolved), 4), np.nan, dtype=np.float32)
    for i, item in enumerate(resolved):
        for view in item["views"]:
            angles[i, int(view["angle_rank"])] = float(view["angle_deg"])
            body_yaw[i, int(view["angle_rank"])] = float(
                view["body_forward_yaw_in_camera_deg"]
            )
    np.save(destination / "body_relative_angle_deg.npy", angles)
    np.save(destination / "body_yaw_deg.npy", body_yaw)


def reorder_tracks(source_root: Path, output_root: Path, resolved_by_split: dict[str, list[dict[str, Any]]]) -> None:
    if not source_root.exists():
        raise FileNotFoundError(f"track root not found: {source_root}")
    for split in SPLITS:
        source_path = source_root / split / "object_position_track.npy"
        if not source_path.exists():
            raise FileNotFoundError(f"track file not found: {source_path}")
        array = np.load(source_path, mmap_mode="r", allow_pickle=False)
        resolved = resolved_by_split[split]
        expected = (len(resolved), 4, 13, 50)
        if tuple(array.shape[1:4]) != expected[1:]:
            raise ValueError(f"track shape {array.shape} does not start with {expected}")
        destination = output_root / split
        destination.mkdir(parents=True, exist_ok=False)
        np.save(destination / "object_position_track.npy", reorder_view_array(np.asarray(array), resolved, fill=0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-cache", type=Path, default=DEFAULT_SOURCE_CACHE)
    parser.add_argument("--source-tracks", type=Path, default=DEFAULT_SOURCE_TRACKS)
    parser.add_argument("--source-tracks-v3", type=Path, default=DEFAULT_SOURCE_TRACKS_V3)
    parser.add_argument("--angle-index", type=Path, default=DEFAULT_ANGLE_INDEX)
    parser.add_argument("--output-cache", type=Path, default=DEFAULT_OUTPUT_CACHE)
    parser.add_argument("--output-tracks", type=Path, default=DEFAULT_OUTPUT_TRACKS)
    parser.add_argument("--output-tracks-v3", type=Path, default=DEFAULT_OUTPUT_TRACKS_V3)
    args = parser.parse_args()

    for required in (args.source_cache, args.angle_index):
        if not required.exists():
            raise FileNotFoundError(required)
    if args.output_cache.exists():
        raise FileExistsError(f"refusing to overwrite isolated output: {args.output_cache}")

    resolved, csv_rows = resolve_all(args.source_cache, args.angle_index)
    print("FRAME0_ANGLE_PREFLIGHT_COMPLETE", flush=True)

    args.output_cache.mkdir(parents=True, exist_ok=False)
    sidecar = args.output_cache.parent / "frame0_body_relative_angle_rank.csv"
    write_csv(sidecar, csv_rows)
    for split in SPLITS:
        write_split(args.source_cache, args.output_cache, split, resolved[split], sidecar.name)
        print(f"FRAME0_ANGLE_CACHE_WRITTEN split={split}", flush=True)
    seen_link = args.output_cache / "seen_test"
    if not seen_link.exists():
        seen_link.symlink_to("test", target_is_directory=True)

    reorder_tracks(args.source_tracks, args.output_tracks, resolved)
    reorder_tracks(args.source_tracks_v3, args.output_tracks_v3, resolved)
    for track_root in (args.output_tracks, args.output_tracks_v3):
        link = track_root / "seen_test"
        if not link.exists():
            link.symlink_to("test", target_is_directory=True)

    summary = {
        "source_cache": str(args.source_cache),
        "output_cache": str(args.output_cache),
        "angle_index": str(args.angle_index),
        "angle_sidecar": str(sidecar),
        "splits": {split: len(resolved[split]) for split in SPLITS},
        "angle_source": "frame0_3d_anatomical_body_forward_to_camera",
        "body_yaw_source": "frame0_3d_anatomical_body_forward_in_each_camera_frame",
        "action_semantics": "rank 0..3 ascending signed angle, mapped through rank_to_original",
        "old_cache_modified": False,
        "old_camera_angle_csv_used_as_feature": False,
        "missing_skeleton_fallback": False,
    }
    write_json(args.output_cache.parent / "build_summary.json", summary)
    print("FRAME0_ANGLE_CACHE_COMPLETE", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
