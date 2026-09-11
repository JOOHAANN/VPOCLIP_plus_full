#!/usr/bin/env python3
"""Train two independent all-view 50/5 VPOCLIP variants.

The RGB tensor cache and RTMPose cache are class agnostic and are already
complete.  For each selected unseen-class group this driver retrains the two
supervised backbones, extracts aligned features, assembles a strict ZSL
dataset, then runs the established two-stage VPOCLIP recipe.  The large
feature extraction directory is intentionally shared and overwritten only
after the preceding group's VPOCLIP model and dataset have been saved in
their own directories.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml


WS = Path("/home/youhan/ws")
VPO = WS / "VPOCLIP_plus_full"
X3D = WS / "X3D_full"
CTR = WS / "CTR-GCN_17_full"
YOLO = WS / "yolov5_full"
RAW_CACHE = X3D / "data/etri_rgb_allviews_cs_45_25_10_20"
POSE_CACHE = CTR / "data/etri_coco17/allviews_cs45"
POSE_NPZ = POSE_CACHE / "ETRI_55_CS_rtmpose_coco17_13.npz"
OBJECT_DIR = VPO / "data/allviews_cs45_lowview50_5_features/object"
FEATURE_DIR = VPO / "data/new50_5_groups_shared_features"
ZSL_ROOT = VPO / "data"
SUBJECT_MANIFEST = WS / "ETRI_subject_split_45_25_10_20.json"
SELECTION = VPO / "logs/new50_5_groups_20260909/class_selection.json"
RUN_ROOT = VPO / "logs/new50_5_groups_20260909"
PYTHON = Path("/home/youhan/.conda/envs/clipgcn/bin/python")
PRETRAINED_X3D = X3D / "pretrained/x3d_s.pyth"
YOLO_WEIGHTS = YOLO / "best.pt"
X3D_TEMPLATE = X3D / "configs/x3d-s_etri_rgb_cs_45_25_10_20_zsl50_5_182.yaml"
CTR_TEMPLATE = CTR / "config/etri-coco17/ctrgcn_joint_coco17_13.yaml"
VPO_STAGE_A_TEMPLATE = VPO / "configs/vpoclip_new50_5_recipe_stage_a.yaml"
VPO_STAGE_B_TEMPLATE = VPO / "configs/vpoclip_new50_5_recipe_stage_b.yaml"

# 7000 iterations is the established X3D budget.  Batch 288 leaves a safe
# margin for the pre-existing DQN/RL process while keeping the tensor core
# workload large.
X3D_BATCH = 288
X3D_MAX_ITER = 7000
X3D_LR_STEPS = (3500, 5810)
X3D_WORKERS = 8
VPO_WORKERS = 16


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def load_selection() -> list[dict]:
    payload = json.loads(SELECTION.read_text(encoding="utf-8"))
    groups = payload["groups"]
    if len(groups) != 2:
        raise ValueError("selection manifest must contain exactly two groups")
    used: set[int] = set()
    for group in groups:
        labels = [int(x) for x in group["unseen_zero_based"]]
        if len(labels) != 5 or len(set(labels)) != 5:
            raise ValueError(f"invalid class selection: {group}")
        if used.intersection(labels):
            raise ValueError("the two groups unexpectedly share an unseen class")
        used.update(labels)
    return groups


def configure_x3d(group: dict, group_root: Path) -> Path:
    cfg = yaml.safe_load(X3D_TEMPLATE.read_text(encoding="utf-8"))
    output = X3D / "outputs" / f"etri_allviews_cs45_new50_5_{group['name']}_7000"
    cfg["OUTPUT_DIR"] = str(output)
    cfg["TRAIN"]["MAX_ITER"] = X3D_MAX_ITER
    cfg["TRAIN"]["SAVE_STEP"] = 1000
    cfg["TRAIN"]["EVAL_STEP"] = 1000
    cfg["TEST"]["CHECKPOINT"] = str(output / "model_final.pth")
    cfg["TEST"]["OUTPUT_DIR"] = str(output / "test")
    for split in ("TRAIN", "TEST"):
        cfg["DATASETS"][split]["DATA_DIR"] = str(RAW_CACHE)
        cfg["DATASETS"][split]["ANNOTATION_DIR"] = str(RAW_CACHE)
    path = group_root / "x3d.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def configure_ctr(group: dict, group_root: Path) -> tuple[Path, Path]:
    cfg = yaml.safe_load(CTR_TEMPLATE.read_text(encoding="utf-8"))
    output = CTR / "work_dir/etri_coco17" / f"allviews_cs45_new50_5_{group['name']}"
    cfg["work_dir"] = str(output)
    for split in ("train_feeder_args", "test_feeder_args"):
        cfg[split]["data_path"] = str(POSE_NPZ)
        cfg[split]["exclude_classes"] = [int(x) for x in sorted(group["unseen_zero_based"])]
    cfg.update(
        batch_size=512,
        test_batch_size=512,
        num_worker=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        cudnn_benchmark=True,
        cudnn_deterministic=False,
        save_interval=1,
    )
    path = group_root / "ctrgcn.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path, output


def configure_vpoclip(group: dict, group_root: Path, zsl_dir: Path) -> tuple[Path, Path, Path, Path]:
    """Materialise absolute configs so later evaluation is independent of scratch dirs."""
    group_name = group["name"]
    work_root = VPO / "work_dir" / f"new50_5_{group_name}"
    stage_a = work_root / "recipe_stage_a"
    stage_b = work_root / "recipe_stage_b"

    a = yaml.safe_load(VPO_STAGE_A_TEMPLATE.read_text(encoding="utf-8"))
    for split in ("train", "val"):
        a["data"][split]["data_dir"] = str(zsl_dir)
        a["data"][split]["prefix"] = f"trimodal_{split}"
    a["data"]["dataloader"]["num_workers"] = VPO_WORKERS
    a["data"]["dataloader"]["persistent_workers"] = True
    a["data"]["text"]["cache_output"] = str(stage_a / "text.pt")
    a["outputs"] = {
        "work_dir": str(stage_a),
        "auto_run_dir": False,
        "best_model": str(stage_a / "best_model.pth"),
        "last_model": str(stage_a / "last_model.pth"),
        "history": str(stage_a / "history.json"),
        "train_curve": str(stage_a / "train_curve.png"),
        "test_results": str(stage_a / "unseen_test_results.json"),
    }
    a_path = group_root / "vpoclip_stage_a.yaml"
    a_path.write_text(yaml.safe_dump(a, sort_keys=False), encoding="utf-8")

    b = yaml.safe_load(VPO_STAGE_B_TEMPLATE.read_text(encoding="utf-8"))
    for split in ("train", "val"):
        b["data"][split]["data_dir"] = str(zsl_dir)
        b["data"][split]["prefix"] = f"trimodal_{split}"
    b["data"]["dataloader"]["num_workers"] = VPO_WORKERS
    b["data"]["dataloader"]["persistent_workers"] = True
    b["data"]["text"]["cache_output"] = str(stage_b / "text.pt")
    b["train"]["init_checkpoint"] = str(stage_a / "best_model.pth")
    b["outputs"] = {
        "work_dir": str(stage_b),
        "auto_run_dir": False,
        "best_model": str(stage_b / "best_model.pth"),
        "last_model": str(stage_b / "last_model.pth"),
        "history": str(stage_b / "history.json"),
        "train_curve": str(stage_b / "train_curve.png"),
        "test_results": str(stage_b / "unseen_test_results.json"),
    }
    b_path = group_root / "vpoclip_stage_b.yaml"
    b_path.write_text(yaml.safe_dump(b, sort_keys=False), encoding="utf-8")
    return a_path, b_path, stage_a, stage_b


def run_stage(group_root: Path, name: str, cwd: Path, command: list[object], *, env_extra: dict[str, str] | None = None,
              required: list[Path] | None = None) -> None:
    marker = group_root / f"{name}.done"
    if marker.exists():
        print(f"[{now()}] SKIP {name} (marker exists)", flush=True)
        return
    log_path = group_root / f"{name}.log"
    env = os.environ.copy()
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "CUDA_VISIBLE_DEVICES": "0",
            "OMP_NUM_THREADS": "8",
            "MKL_NUM_THREADS": "8",
            "OPENBLAS_NUM_THREADS": "1",
            "PYTHONPATH": str(cwd) + os.pathsep + env.get("PYTHONPATH", ""),
        }
    )
    if env_extra:
        env.update(env_extra)
    argv = [str(x) for x in command]
    print(f"[{now()}] START {name}: {' '.join(argv)}", flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"[{now()}] COMMAND {' '.join(argv)}\n")
        log.flush()
        result = subprocess.run(argv, cwd=str(cwd), env=env, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"{name} failed with exit code {result.returncode}; see {log_path}")
    if required:
        missing = [str(path) for path in required if not path.exists() or path.stat().st_size == 0]
        if missing:
            raise RuntimeError(f"{name} completed but required outputs are missing: {missing}")
    marker.write_text(now() + "\n", encoding="utf-8")
    print(f"[{now()}] DONE {name}", flush=True)


def select_pose_checkpoint(ctr_output: Path, group_root: Path) -> Path:
    log_path = ctr_output / "log.txt"
    if not log_path.exists():
        raise FileNotFoundError(log_path)
    epochs = re.findall(r"Epoch number:\s*(\d+)", log_path.read_text(encoding="utf-8", errors="replace"))
    if not epochs:
        raise RuntimeError(f"No best epoch found in {log_path}")
    best_epoch = int(epochs[-1])
    candidates = sorted(ctr_output.glob(f"runs-{best_epoch}-*.pt"))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one CTR checkpoint for best epoch {best_epoch}, found {candidates}")
    return candidates[0]


def validate_zsl(zsl_dir: Path, unseen: list[int], group_root: Path) -> None:
    metadata = json.loads((zsl_dir / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("unseen_classes_zero_based") != sorted(unseen):
        raise RuntimeError(f"assembled metadata has wrong unseen classes: {metadata}")
    counts = metadata.get("counts", {})
    for prefix in ("trimodal_train", "trimodal_val", "trimodal_test"):
        if int(counts.get(prefix, 0)) <= 0:
            raise RuntimeError(f"assembled dataset has no samples for {prefix}")
    write_json(group_root / "assembled_metadata.json", metadata)


def run_group(group: dict) -> None:
    name = group["name"]
    unseen = [int(x) for x in group["unseen_zero_based"]]
    group_root = RUN_ROOT / name
    group_root.mkdir(parents=True, exist_ok=True)
    zsl_dir = ZSL_ROOT / f"new50_5_{name}_zsl"
    x3d_config = configure_x3d(group, group_root)
    ctr_config, ctr_output = configure_ctr(group, group_root)
    a_config, b_config, stage_a_dir, stage_b_dir = configure_vpoclip(group, group_root, zsl_dir)
    x3d_output = X3D / "outputs" / f"etri_allviews_cs45_new50_5_{name}_7000"
    x3d_best = x3d_output / "model_best.pth"
    pose_weights_file = group_root / "pose_checkpoint.txt"
    feature_x3d = FEATURE_DIR / "x3d"
    feature_pose = FEATURE_DIR / "pose"
    feature_x3d.mkdir(parents=True, exist_ok=True)
    feature_pose.mkdir(parents=True, exist_ok=True)

    write_json(
        group_root / "run_manifest.json",
        {
            "group": group,
            "unseen_classes_zero_based": unseen,
            "unseen_actions": [f"A{x + 1:03d}" for x in unseen],
            "raw_cache": str(RAW_CACHE),
            "rtmpose_cache": str(POSE_NPZ),
            "object_features_reused": str(OBJECT_DIR),
            "x3d_output": str(x3d_output),
            "ctr_output": str(ctr_output),
            "zsl_dataset": str(zsl_dir),
            "vpo_stage_a": str(stage_a_dir),
            "vpo_stage_b": str(stage_b_dir),
            "started_at": now(),
        },
    )

    run_stage(
        group_root,
        "01_x3d_train",
        X3D,
        [
            PYTHON,
            "tools/train_etri_x3d.py",
            "--config",
            x3d_config,
            "--data-dir",
            RAW_CACHE,
            "--output-dir",
            x3d_output,
            "--pretrained",
            PRETRAINED_X3D,
            "--exclude-classes",
            *sorted(unseen),
            "--workers",
            X3D_WORKERS,
            "--prefetch-factor",
            2,
            "--batch-size",
            X3D_BATCH,
            "--max-iter",
            X3D_MAX_ITER,
            "--warmup-iterations",
            400,
            "--lr-steps",
            *X3D_LR_STEPS,
            "--save-step",
            1000,
            "--eval-step",
            1000,
            "--resume",
        ],
        required=[x3d_best, x3d_output / "training_summary.json"],
    )

    run_stage(
        group_root,
        "02_ctrgcn_train",
        CTR,
        [PYTHON, "main.py", "--config", ctr_config],
        required=[ctr_output / "log.txt"],
    )
    pose_weights = select_pose_checkpoint(ctr_output, group_root)
    pose_weights_file.write_text(str(pose_weights.resolve()) + "\n", encoding="utf-8")
    write_json(
        group_root / "selected_backbones.json",
        {"x3d": str(x3d_best.resolve()), "pose": str(pose_weights.resolve()), "yolo": str(YOLO_WEIGHTS.resolve())},
    )

    for split in ("train", "val", "test"):
        run_stage(
            group_root,
            f"03_x3d_extract_{split}",
            X3D,
            [
                PYTHON,
                "tools/extract_x3d_features.py",
                "--config",
                x3d_config,
                "--checkpoint",
                x3d_best,
                "--data-dir",
                RAW_CACHE,
                "--split",
                split,
                "--output",
                feature_x3d / f"{split}_video.npy",
                "--batch-size",
                512,
                "--num-workers",
                8,
                "--tensor-resize-size",
                182,
                "--spatial-size",
                6,
                "--save-sidecars",
            ],
            required=[feature_x3d / f"{split}_video.npy", feature_x3d / f"{split}_video.labels.npy"],
        )
        run_stage(
            group_root,
            f"04_pose_extract_{split}",
            VPO,
            [
                PYTHON,
                "tools/extract_current_ctrgcn_features.py",
                "--ctrgcn-root",
                CTR,
                "--config",
                ctr_config,
                "--weights",
                pose_weights,
                "--npz",
                POSE_NPZ,
                "--split",
                split,
                "--output-dir",
                feature_pose,
                "--batch-size",
                1024,
            ],
            required=[feature_pose / f"{split}_pose_coco17.npy", feature_pose / f"{split}_labels.npy"],
        )

    run_stage(
        group_root,
        "05_assemble_zsl",
        VPO,
        [
            PYTHON,
            "tools/assemble_current_45_25_10_20.py",
            "--cache-dir",
            RAW_CACHE,
            "--x3d-dir",
            feature_x3d,
            "--pose-dir",
            feature_pose,
            "--object-dir",
            OBJECT_DIR,
            "--zsl-dir",
            zsl_dir,
            "--closed-dir",
            group_root / "closed_unused",
            "--manifest",
            SUBJECT_MANIFEST,
            "--unseen-classes",
            *sorted(unseen),
            "--zsl-only",
        ],
        required=[zsl_dir / "metadata.json", zsl_dir / "trimodal_train_labels.npy", zsl_dir / "trimodal_test_labels.npy"],
    )
    validate_zsl(zsl_dir, unseen, group_root)

    run_stage(
        group_root,
        "06_vpoclip_stage_a",
        VPO,
        [PYTHON, "train.py", "--config", a_config],
        required=[stage_a_dir / "best_model.pth", stage_a_dir / "last_model.pth"],
    )
    run_stage(
        group_root,
        "07_vpoclip_stage_b",
        VPO,
        [PYTHON, "train.py", "--config", b_config],
        required=[stage_b_dir / "best_model.pth", stage_b_dir / "last_model.pth"],
    )
    write_json(
        group_root / "complete.json",
        {
            "completed_at": now(),
            "group": group,
            "unseen_classes_zero_based": unseen,
            "zsl_dataset": str(zsl_dir),
            "stage_a_best": str((stage_a_dir / "best_model.pth").resolve()),
            "stage_a_last": str((stage_a_dir / "last_model.pth").resolve()),
            "stage_b_best": str((stage_b_dir / "best_model.pth").resolve()),
            "stage_b_last": str((stage_b_dir / "last_model.pth").resolve()),
        },
    )
    print(f"[{now()}] COMPLETE {name}", flush=True)


def main() -> None:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    groups = load_selection()
    write_json(RUN_ROOT / "pipeline_manifest.json", {"started_at": now(), "groups": groups})
    # Keep a lightweight telemetry stream in the same run directory.  It does
    # not alter training and makes post-run utilization/thermal checks easy.
    telemetry_path = RUN_ROOT / "gpu_telemetry.csv"
    telemetry = None
    try:
        telemetry = telemetry_path.open("a", encoding="utf-8")
        gpu_monitor = subprocess.Popen(
            [
                "nvidia-smi",
                "--query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.free,power.draw",
                "--format=csv",
                "-l",
                "10",
            ],
            stdout=telemetry,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except OSError as exc:
        gpu_monitor = None
        print(f"[{now()}] GPU telemetry unavailable: {exc}", flush=True)
    try:
        for group in groups:
            run_group(group)
    finally:
        if gpu_monitor is not None:
            gpu_monitor.terminate()
            try:
                gpu_monitor.wait(timeout=5)
            except subprocess.TimeoutExpired:
                gpu_monitor.kill()
        if telemetry is not None:
            telemetry.close()
    write_json(RUN_ROOT / "complete.json", {"completed_at": now(), "groups": groups})
    print(f"[{now()}] ALL GROUPS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
