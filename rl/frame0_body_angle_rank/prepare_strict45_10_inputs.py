#!/usr/bin/env python3
"""Prepare strict 45/10 manifests and the precomputed RTMPose cache.

The 45/10 VPOCLIP model has its own ten-class ZSL protocol.  This helper
keeps the RL subject split (DQN=25, val=10, test=20), includes all recordings
instead of inheriting the retired 50/5 exclusions, and ranks each candidate
by the released frame-0 3-D anatomical body-relative angle.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ..build_cache import build_episode_manifest, load_config
from .frame0_body_angle import load_angle_csv, resolve_episode


ROOT = Path(__file__).resolve().parents[2]
VPO = ROOT
WS = ROOT.parent
SOURCE = WS / "X3D_full" / "data" / "etri_rgb_allviews_cs_45_25_10_20"
RGB_ROOT = WS / "ETRI-Activity3D-RGB"
SKELETON_ROOT = WS / "ETRI-Activity3D-Skeleton"
SUBJECT_MANIFEST = WS / "ETRI_subject_split_45_25_10_20.json"
ANGLE_INDEX = RGB_ROOT / "low_view_camera_angles_split55.csv"
POSE_NPZ = WS / "CTR-GCN_17_full" / "data" / "etri_coco17" / "allviews_cs45" / "ETRI_55_CS_rtmpose_coco17_13.npz"
RUN_ROOT = VPO / "work_dir" / "frame0_body_angle_rank_strict45_10"
RAW_POSE_ROOT = RUN_ROOT / "rtmpose_cache"
CACHE_ROOT = RUN_ROOT / "cache"
CONFIG_PATH = RUN_ROOT / "build_cache.yaml"

TRUE_UNSEEN = [1, 7, 14, 15, 18, 25, 39, 46, 52, 54]
PSEUDO = [8, 27, 28, 35, 50]


def subject_groups() -> dict[str, list[str]]:
    payload = json.loads(SUBJECT_MANIFEST.read_text(encoding="utf-8"))
    groups = payload["subject_groups"]
    return {
        "dqn_train": list(groups["dqn_train"]),
        "val": list(groups["val"]),
        "test": list(groups["test"]),
    }


def write_config() -> dict[str, Any]:
    groups = subject_groups()
    config: dict[str, Any] = {
        "runtime": {"device": "cuda:0", "amp": True},
        "recognizer": {
            "config": str(VPO / "config_final_aug_entropy_single_h2.yaml"),
            "checkpoint": str(VPO / "work_dir/merged45_10_groupAB/selected_stage_b/last_model.pth"),
        },
        "data": {
            "rgb_root": str(RGB_ROOT),
            "skeleton_root": str(SKELETON_ROOT),
            "angle_csv": str(ANGLE_INDEX),
            "source_split_dir": str(SOURCE),
            "source_prefix": {"dqn_train": "dqn", "val": "val", "test": "test"},
            "subject_groups": groups,
            "window_seconds": 8.0,
            "stride_seconds": 1.0,
            "frames": 13,
            "max_windows_per_clip": 1,
            "max_episodes": None,
            "exclude_labels": [],
            "allow_variable_views": True,
            "allow_available_four_views": False,
            "allow_cross_group_views": False,
            "allow_missing_skeleton": False,
            "merge_group_actions": ["A049", "A050"],
            "minimum_views": 2,
        },
        "cache": {
            "root": str(CACHE_ROOT),
            "candidate_cameras": ["C001", "C003", "C005", "C007"],
            "protocol": "strict_zsl_45_10_frame0_body_angle_rank",
        },
        "raw": {
            "batch_size": 64,
            "decode_workers": 8,
            "x3d_input_size": 182,
            "x3d_root": str(WS / "X3D_full"),
            "x3d_config": str(WS / "VPOCLIP_plus_full/logs/merged45_10_groupAB_20260913/x3d.yaml"),
            "x3d_checkpoint": str(WS / "X3D_full/outputs/etri_allviews_cs45_merged_groupAB_7000/model_best.pth"),
            "x3d_layer": "s5",
            "ctrgcn_root": str(WS / "CTR-GCN_17_full"),
            "ctrgcn_config": str(WS / "VPOCLIP_plus_full/logs/merged45_10_groupAB_20260913/ctrgcn.yaml"),
            "ctrgcn_weights": str(WS / "CTR-GCN_17_full/work_dir/etri_coco17/allviews_cs45_merged_groupAB/runs-51-4029.pt"),
            "ctrgcn_hook_layer": "l4",
            "yolo_repo": str(WS / "yolov5_full"),
            "yolo_weights": str(WS / "yolov5_full/best.pt"),
            "rtmpose_cache_root": str(RAW_POSE_ROOT),
            "yolo_size": 640,
            "yolo_conf": 0.25,
            "yolo_iou": 0.45,
            "yolo_half": True,
            "no_yolo": False,
            "object_grid_size": 6,
            "object_value": "presence",
            "object_max_distance_weight": 10.0,
            "yolo_detect_every": 1,
        },
    }
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    # ``build_episode_manifest`` also accepts an in-memory config, but its
    # path resolver expects the same provenance field that ``load_config``
    # adds for a file-backed config.  Keep the helper usable both ways.
    config["_config_path"] = str(CONFIG_PATH)
    return config


def prepare_rtmpose() -> None:
    RAW_POSE_ROOT.mkdir(parents=True, exist_ok=True)
    required = [
        RAW_POSE_ROOT / f"{split}_x.npy"
        for split in ("dqn", "val", "test")
    ] + [
        RAW_POSE_ROOT / f"{split}_sample_names.npy"
        for split in ("dqn", "val", "test")
    ]
    if all(path.exists() for path in required):
        print("STRICT45_10_RTMPOSE_READY", flush=True)
        return
    source = np.load(POSE_NPZ, allow_pickle=True)
    for split in ("dqn", "val", "test"):
        np.save(RAW_POSE_ROOT / f"{split}_x.npy", np.asarray(source[f"x_{split}"]))
        np.save(
            RAW_POSE_ROOT / f"{split}_sample_names.npy",
            np.asarray(source[f"{split}_sample_name"]).astype(str),
        )
        np.save(RAW_POSE_ROOT / f"{split}_labels.npy", np.asarray(source[f"y_{split}"], dtype=np.int64))
        print(f"STRICT45_10_RTMPOSE_WRITTEN {split} {len(source[f'x_{split}'])}", flush=True)


def rank_manifests(config: dict[str, Any]) -> None:
    angle_rows = load_angle_csv(ANGLE_INDEX)
    skeleton_cache: dict[Path, dict[int, tuple[np.ndarray, np.ndarray]]] = {}
    summary: dict[str, Any] = {}
    for split in ("dqn_train", "val", "test"):
        split_root = CACHE_ROOT / split
        manifest = split_root / "manifest.jsonl"
        if not manifest.exists():
            build_episode_manifest(config, split)
        rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
        ranked: list[dict[str, Any]] = []
        for index, row in enumerate(rows):
            source_valid = np.ones(len(row["views"]), dtype=bool)
            resolved = resolve_episode(row, angle_rows, source_valid, index, skeleton_cache)
            originals = row["views"]
            merged_views: list[dict[str, Any]] = []
            geometry_views: list[dict[str, Any]] = []
            for item in resolved["views"]:
                original = dict(originals[int(item["original_view_index"])])
                original.update(item)
                merged_views.append(original)
                geometry_views.append(
                    {
                        "camera": str(item["camera"]),
                        "relative_bearing_deg": float(item["angle_deg"]),
                        "geometry_confidence": 1.0,
                        "geometry_source": "frame0_3d_anatomical_body_forward_to_camera",
                    }
                )
            if len(merged_views) < 2:
                continue
            updated = dict(row)
            updated["views"] = merged_views
            updated["rank_to_original"] = resolved["rank_to_original"]
            updated["original_to_rank"] = resolved["original_to_rank"]
            updated["angle_deg_by_rank"] = resolved["angle_deg_by_rank"]
            updated["frame0_geometry"] = {
                "body_yaw_deg_c001": float("nan"),
                "body_yaw_confidence": 1.0,
                "views": geometry_views,
            }
            ranked.append(updated)
        manifest.write_text(
            "".join(json.dumps(row, ensure_ascii=False, allow_nan=True) + "\n" for row in ranked),
            encoding="utf-8",
        )
        (split_root / "manifest_summary.json").write_text(
            json.dumps(
                {
                    "split": split,
                    "episodes": len(ranked),
                    "views": ["C001", "C003", "C005", "C007"],
                    "variable_views": True,
                    "minimum_views": 2,
                    "merge_group_actions": ["A049", "A050"],
                    "angle_source": "released_frame0_3d_skeleton",
                    "class_protocol": "strict_zsl_45_10",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        summary[split] = len(ranked)
        print(f"STRICT45_10_FRAME0_MANIFEST {split} {len(ranked)}", flush=True)
    (RUN_ROOT / "manifest_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    config = write_config()
    prepare_rtmpose()
    rank_manifests(config)
    print(json.dumps({"config": str(CONFIG_PATH), "cache": str(CACHE_ROOT), "summary": str(RUN_ROOT / "manifest_summary.json")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
