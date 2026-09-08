"""Create isolated RL configs after the two all-view VPO runs finish."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
WS = ROOT.parent
PYTHON = Path("/home/youhan/.conda/envs/clipgcn/bin/python")
SUBJECT_MANIFEST = WS / "ETRI_subject_split_45_25_10_20.json"
RGB_CACHE = WS / "X3D_full/data/etri_rgb_allviews_cs_45_25_10_20"
ANGLE_CSV = WS / "ETRI-Activity3D-RGB/low_view_camera_angles_split55.csv"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _last_model(index: Path) -> Path:
    payload = _read_json(index)
    value = Path(payload["last_model"])
    if not value.is_absolute():
        value = (ROOT / value).resolve()
    if not value.is_file():
        raise FileNotFoundError(value)
    return value


def _subjects() -> dict[str, list[str]]:
    payload = _read_json(SUBJECT_MANIFEST)
    groups = payload["subject_groups"]
    return {key: list(value) for key, value in groups.items()}


def _common(name: str, unseen: list[int], vpo_config: Path, vpo_checkpoint: Path,
            backbone_run: Path, output_name: str, cache_name: str, *,
            recording_manifest: Path | None = None,
            allow_cross_group_views: bool = False,
            allow_available_four_views: bool = False) -> dict[str, Any]:
    selected = _read_json(backbone_run / "selected_backbones.json")
    subjects = _subjects()
    x3d_config = backbone_run / "x3d.yaml"
    ctrgcn_config = backbone_run / "ctrgcn.yaml"
    for path in (x3d_config, ctrgcn_config):
        if not path.is_file():
            raise FileNotFoundError(path)
    seen = sorted(set(range(55)) - set(unseen))
    data: dict[str, Any] = {
        "rgb_root": str(WS / "ETRI-Activity3D-RGB"),
        "skeleton_root": str(WS / "ETRI-Activity3D-Skeleton"),
        "angle_csv": str(ANGLE_CSV),
        "source_split_dir": str(RGB_CACHE),
        "source_prefix": {"dqn_train": "dqn", "val": "val", "test": "test"},
        "exclude_labels": unseen,
        "exclude_labels_by_split": {"dqn_train": unseen, "val": unseen, "test": []},
        "allowed_labels_by_split": {"dqn_train": seen, "val": seen, "test": unseen},
        "subject_groups": subjects,
        "window_seconds": 2.0,
        "stride_seconds": 1.0,
        "frames": 13,
        "max_windows_per_clip": None,
        "recording_manifest": str(recording_manifest) if recording_manifest else None,
        # A/P samples such as A050 can place C001/C003/C005/C007 in
        # different G groups.  That is still valid for the low-view
        # candidate set, but never substitute the paired even high cameras
        # when one of the four required low cameras is missing.
        "allow_cross_group_views": bool(allow_cross_group_views),
        "allow_available_four_views": bool(allow_available_four_views),
    }
    config: dict[str, Any] = {
        "runtime": {"device": "cuda:0", "seed": 20260907, "amp": True},
        "recognizer": {
            "config": str(vpo_config), "checkpoint": str(vpo_checkpoint),
            "frozen": True, "embedding_dim": 512, "num_classes": 55,
        },
        "data": data,
        "cache": {
            "root": str(ROOT / "data" / cache_name), "num_views": 4,
            "selected_views": 2, "mmap": True,
            "candidate_cameras": ["C001", "C003", "C005", "C007"],
        },
        "state": {
            "stage": "M2", "slots": 3, "top_k": 5,
            "fusion_weights": [1.0, 1.0], "use_top5_evidence": True,
            "use_pose": True, "use_object": True, "pose_input_dim": 442,
            "object_input_dim": 1800, "image_quality_dim": 4,
            "quality_dim": 4, "view_encoding_dim": 3,
        },
        "network": {"view_encoding_dim": 3},
        "reward": {"ce_weight": 0.3, "terminal_weight": 1.0,
                   "movement_weight": 0.05, "ce_clip": 1.0,
                   "class_ids": seen},
        "agent": {
            "algorithm": "masked_dueling_q_single_step", "gamma": 0.95,
            "batch_size": 512, "replay_capacity": 100000,
            "replay_warmup": 512, "learning_rate": 1.0e-4,
            "weight_decay": 1.0e-4, "grad_clip": 10.0,
            "target_update_interval": 1000,
            "epsilon": {"start": 1.0, "end": 0.05, "decay_steps": 100000},
        },
        "fast": {
            "batch_size": 524288, "updates_per_epoch": 512,
            "replay_capacity": 100000, "selection_batch_size": 16384,
            "validation_batch_size": 16384, "amp_dtype": "bfloat16",
            "compile": True, "compile_mode": "reduce-overhead",
        },
        "raw": {
            "batch_size": 32, "decode_workers": 4, "reuse_pose_frames": True,
            "rtmpose_mode": "balanced", "x3d_input_size": 168,
            "x3d_root": str(WS / "X3D_full"), "x3d_config": str(x3d_config),
            "x3d_checkpoint": str(Path(selected["x3d"])), "x3d_layer": "s5",
            "ctrgcn_root": str(WS / "CTR-GCN_17_full"),
            "ctrgcn_config": str(ctrgcn_config),
            "ctrgcn_weights": str(Path(selected["pose"])),
            "ctrgcn_hook_layer": "l4", "yolo_repo": str(WS / "yolov5_full"),
            "yolo_weights": str(WS / "yolov5_full/best.pt"),
        },
        "train": {"epochs": 50, "validation_interval": 1,
                  "output_dir": str(ROOT / "work_dir" / output_name), "resume": None},
        "evaluation": {"splits": ["val", "test"], "class_ids": unseen,
                        "unseen_class_ids": unseen, "seen_class_ids": seen},
        "handoff": {"name": name, "vpo_config": str(vpo_config),
                    "vpo_checkpoint": str(vpo_checkpoint),
                    "backbone_record": str(backbone_run / "selected_backbones.json")},
    }
    return config


def main() -> None:
    new_run = ROOT / "logs/allviews_lowview50_5_20260907"
    old_run = ROOT / "logs/allviews_20260907"
    split = _read_json(new_run / "class_split.json")
    new_unseen = list(split["unseen_classes_zero_based"])
    old_unseen = [9, 10, 11, 17, 49]
    new_vpo = _last_model(ROOT / "work_dir/allviews_lowview50_5_final_aug_entropy_single_h2/latest_run.json")
    old_vpo = _last_model(ROOT / "work_dir/allviews_final_aug_entropy_single_h2/latest_run.json")
    base_vpo = ROOT / "config_final_aug_entropy_single_h2.yaml"
    configs = {
        "new50_5": _common(
            "new50_5", new_unseen, base_vpo, new_vpo, new_run,
            "active_view_ddqn_new50_5", "active_multiview_cache_new50_5",
            recording_manifest=new_run / "class_split.json"),
        "old50_5": _common(
            "old50_5", old_unseen, base_vpo, old_vpo, old_run,
            "active_view_ddqn_old50_5", "active_multiview_cache_old50_5",
            allow_cross_group_views=True,
            allow_available_four_views=False),
    }
    out = ROOT / "rl/handoff_configs"
    out.mkdir(parents=True, exist_ok=True)
    for name, config in configs.items():
        config["_config_path"] = str(out / f"{name}.yaml")
        (out / f"{name}.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        print(out / f"{name}.yaml", flush=True)


if __name__ == "__main__":
    main()
