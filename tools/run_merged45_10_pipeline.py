#!/usr/bin/env python3
"""Run the strict merged-group-A/group-B 45/10 VPOCLIP experiment.

The two previous five-class unseen sets are merged into one ten-class test
bank.  X3D and CTR-GCN are retrained with all ten labels excluded, then the
three aligned streams are assembled and the established pseudo-unseen
checkpoint-selection recipe is run before the final VPOCLIP training.

Every expensive operation has a ``.done`` marker and an isolated output
directory, so an interrupted tmux session can be resumed without overwriting
the earlier 50/5 experiments.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import yaml


WS = Path("/home/youhan/ws")
VPO = WS / "VPOCLIP_plus_full"
X3D = WS / "X3D_full"
CTR = WS / "CTR-GCN_17_full"
YOLO = WS / "yolov5_full"

PYTHON = Path("/home/youhan/.conda/envs/clipgcn/bin/python")
RUN_ROOT = VPO / "logs/merged45_10_groupAB_20260913"
RAW_CACHE = X3D / "data/etri_rgb_allviews_cs_45_25_10_20"
SUBJECT_MANIFEST = WS / "ETRI_subject_split_45_25_10_20.json"
POSE_NPZ = CTR / "data/etri_coco17/allviews_cs45/ETRI_55_CS_rtmpose_coco17_13.npz"
OBJECT_DIR = VPO / "data/allviews_cs45_lowview50_5_features/object"

FEATURE_ROOT = VPO / "data/merged45_10_groupAB_features"
X3D_FEATURE_DIR = FEATURE_ROOT / "x3d"
POSE_FEATURE_DIR = FEATURE_ROOT / "pose"
ZSL_DIR = VPO / "data/merged45_10_groupAB_zsl"

X3D_OUTPUT = X3D / "outputs/etri_allviews_cs45_merged_groupAB_7000"
CTR_OUTPUT = CTR / "work_dir/etri_coco17/allviews_cs45_merged_groupAB"

X3D_TEMPLATE = X3D / "configs/x3d-s_etri_rgb_cs_45_25_10_20_zsl50_5_182.yaml"
CTR_TEMPLATE = CTR / "config/etri-coco17/ctrgcn_joint_coco17_13.yaml"
VPO_BASE_A = VPO / "config_final_aug.yaml"
VPO_BASE_B = VPO / "config_final_aug_entropy_single_h2.yaml"
TEXT_XLSX = VPO / "ntu55_global_descriptions.xlsx"
PRETRAINED_X3D = X3D / "pretrained/x3d_s.pyth"
YOLO_WEIGHTS = YOLO / "best.pt"

# Labels are zero based throughout the cache and model.  These are the union
# of group1 [A040,A016,A026,A015,A047] and group2
# [A019,A053,A008,A055,A002], sorted for an auditable 45/10 protocol.
TRUE_UNSEEN = [1, 7, 14, 15, 18, 25, 39, 46, 52, 54]
PSEUDO_GROUPS = {
    "p3": [8, 27, 28, 35, 50],
    "p4": [3, 13, 20, 30, 41],
}

# 192 keeps the complete CUDA reservation below the user's 80-GiB target;
# the previous 288-batch attempt is resumable from model_001000.pth.
X3D_BATCH = 192
X3D_MAX_ITER = 7000
CTR_EPOCHS = 65
VPO_BATCH = 2048
VPO_WORKERS = 16


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def configure_x3d() -> Path:
    cfg = yaml.safe_load(X3D_TEMPLATE.read_text(encoding="utf-8"))
    cfg["OUTPUT_DIR"] = str(X3D_OUTPUT)
    cfg["TRAIN"]["MAX_ITER"] = X3D_MAX_ITER
    cfg["TRAIN"]["SAVE_STEP"] = 1000
    cfg["TRAIN"]["EVAL_STEP"] = 1000
    cfg["TEST"]["CHECKPOINT"] = str(X3D_OUTPUT / "model_final.pth")
    cfg["TEST"]["OUTPUT_DIR"] = str(X3D_OUTPUT / "test")
    for split in ("TRAIN", "TEST"):
        cfg["DATASETS"][split]["DATA_DIR"] = str(RAW_CACHE)
        cfg["DATASETS"][split]["ANNOTATION_DIR"] = str(RAW_CACHE)
    path = RUN_ROOT / "x3d.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def configure_ctr() -> Path:
    cfg = yaml.safe_load(CTR_TEMPLATE.read_text(encoding="utf-8"))
    cfg["work_dir"] = str(CTR_OUTPUT)
    for split in ("train_feeder_args", "test_feeder_args"):
        cfg[split]["data_path"] = str(POSE_NPZ)
        cfg[split]["exclude_classes"] = TRUE_UNSEEN
    cfg.update(
        batch_size=512,
        test_batch_size=512,
        num_worker=16,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        cudnn_benchmark=True,
        cudnn_deterministic=False,
        save_interval=1,
        save_epoch=0,
        eval_interval=1,
        num_epoch=CTR_EPOCHS,
    )
    path = RUN_ROOT / "ctrgcn.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def vpo_config(
    name: str,
    base: Path,
    output_dir: Path,
    *,
    pseudo: list[int] | None = None,
    init_checkpoint: Path | None = None,
    epochs: int,
    selection: str,
) -> Path:
    data = {
        "train": {
            "data_dir": str(ZSL_DIR),
            "prefix": "trimodal_train",
            "pose_suffix": "_coco17",
            "mmap": True,
        },
        "val": {
            "data_dir": str(ZSL_DIR),
            "prefix": "trimodal_val",
            "pose_suffix": "_coco17",
            "mmap": True,
        },
        "text": {
            "xlsx": str(TEXT_XLSX),
            "id_column": "ID",
            "text_column": "global_description",
            "label_offset": 1,
            "prompt_template": "{global_description}",
            "cache_output": str(output_dir / "text.pt"),
        },
        "dataloader": {
            "batch_size": VPO_BATCH,
            "num_workers": VPO_WORKERS,
            "persistent_workers": True,
            "prefetch_factor": 4,
            "train_shuffle": True,
            "val_shuffle": False,
            "class_balanced_sampling": True,
            "pin_memory": True,
            "drop_last": True,
        },
    }
    if pseudo:
        data["train"]["pseudo_unseen"] = list(pseudo)
        data["pseudo_val"] = {
            "classes": list(pseudo),
            "sources": [
                {
                    "data_dir": str(ZSL_DIR),
                    "prefix": "trimodal_train",
                    "pose_suffix": "_coco17",
                    "mmap": True,
                },
                {
                    "data_dir": str(ZSL_DIR),
                    "prefix": "trimodal_val",
                    "pose_suffix": "_coco17",
                    "mmap": True,
                },
            ],
        }

    train = {
        "epochs": int(epochs),
        "val_interval": 1,
        "model_selection": selection,
        "checkpoint_every": 0,
        "save_every": 1,
        "swa_start": None,
        "scheduler": "cosine",
        "min_lr": 1.0e-6,
        "grad_clip_norm": 1.0,
        "optimizer": {"name": "AdamW", "lr": 1.0e-4, "weight_decay": 1.0e-4},
    }
    if base == VPO_BASE_B:
        train.update(
            freeze_batchnorm=True,
            optimizer={"name": "AdamW", "lr": 2.0e-6, "weight_decay": 0.0},
            loss={
                "mode": "cross_entropy_entropy",
                "entropy_weight": 2.0,
                "entropy_temperature": 0.1,
                "entropy_scope": "all",
                "entropy_normalize": True,
                "entropy_correct_only": True,
                "entropy_start_epoch": 0,
                "entropy_warmup_epochs": 5,
                "scale_independent": True,
                "anchor_weight": 100.0,
                "anchor_scope": "visual",
            },
        )
    if init_checkpoint is not None:
        train["init_checkpoint"] = str(init_checkpoint)

    config = {
        "base_config": str(base),
        "runtime": {
            "device": "cuda:0",
            "seed": 20260913,
            "amp": True,
            "no_show": True,
            "cudnn_benchmark": True,
            "allow_tf32": True,
            "matmul_precision": "high",
        },
        "data": data,
        "train": train,
        "outputs": {
            "work_dir": str(output_dir),
            "auto_run_dir": False,
            "run_name": None,
            "best_model": str(output_dir / "best_model.pth"),
            "last_model": str(output_dir / "last_model.pth"),
            "history": str(output_dir / "history.json"),
            "train_curve": str(output_dir / "train_curve.png"),
            "test_results": str(output_dir / "test_results.json"),
        },
    }
    path = RUN_ROOT / f"{name}.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def run_stage(
    name: str,
    cwd: Path,
    command: list[object],
    *,
    required: list[Path] | None = None,
    env_extra: dict[str, str] | None = None,
) -> None:
    marker = RUN_ROOT / f"{name}.done"
    if marker.exists():
        print(f"[{now()}] SKIP {name} (marker exists)", flush=True)
        return

    log_path = RUN_ROOT / f"{name}.log"
    env = os.environ.copy()
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": "0",
            "PYTHONPATH": str(cwd) + os.pathsep + env.get("PYTHONPATH", ""),
        }
    )
    # X3D benefits from a few BLAS threads; feature/VPO stages are mostly
    # tensor copies and should not oversubscribe the 32-core host.
    if "x3d" in name.lower() and "extract" not in name.lower():
        env.update(OMP_NUM_THREADS="8", MKL_NUM_THREADS="8", OPENBLAS_NUM_THREADS="1")
    elif "ctr" in name.lower() or "pose" in name.lower():
        env.update(OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1")
    else:
        env.update(
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            NUMEXPR_NUM_THREADS="1",
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
        missing = [str(p) for p in required if not p.exists() or p.stat().st_size == 0]
        if missing:
            raise RuntimeError(f"{name} completed but required outputs are missing: {missing}")
    marker.write_text(now() + "\n", encoding="utf-8")
    print(f"[{now()}] DONE {name}", flush=True)


def select_ctr_checkpoint() -> Path:
    log_path = CTR_OUTPUT / "log.txt"
    if not log_path.exists():
        raise FileNotFoundError(log_path)
    epochs = re.findall(r"Epoch number:\s*(\d+)", log_path.read_text(encoding="utf-8", errors="replace"))
    if not epochs:
        raise RuntimeError(f"No best epoch recorded in {log_path}")
    best_epoch = int(epochs[-1])
    candidates = sorted(CTR_OUTPUT.glob(f"runs-{best_epoch}-*.pt"))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one CTR checkpoint at epoch {best_epoch}, found {candidates}")
    return candidates[0]


def validate_assembled_dataset() -> dict:
    metadata = json.loads((ZSL_DIR / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("unseen_classes_zero_based") != TRUE_UNSEEN:
        raise RuntimeError(f"Wrong merged unseen labels in metadata: {metadata}")
    if metadata.get("protocol") != "strict_zsl_45_10":
        raise RuntimeError(f"Unexpected protocol metadata: {metadata.get('protocol')}")
    counts = metadata.get("counts", {})
    for prefix in ("trimodal_train", "trimodal_val", "trimodal_test", "trimodal_test_seen"):
        if int(counts.get(prefix, 0)) <= 0:
            raise RuntimeError(f"No samples in {prefix}")
    write_json(RUN_ROOT / "assembled_metadata.json", metadata)
    return metadata


def cleanup_intermediate_checkpoints(pseudo_dirs: list[Path]) -> None:
    """Remove only generated pseudo-probe weights after final training.

    Histories, configs, selector output, final VPO weights, and all backbone
    weights remain.  Probe ``best/last`` files are redundant after epoch
    selection and otherwise consume ~2.7 GiB on this disk.
    """
    removed = []
    for directory in pseudo_dirs:
        for filename in ("best_model.pth", "last_model.pth"):
            path = directory / filename
            if path.exists():
                path.unlink()
                removed.append(str(path))
    write_json(RUN_ROOT / "intermediate_checkpoint_cleanup.json", {"removed": removed, "at": now()})


def main() -> None:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    FEATURE_ROOT.mkdir(parents=True, exist_ok=True)
    X3D_FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    POSE_FEATURE_DIR.mkdir(parents=True, exist_ok=True)

    if len(set(TRUE_UNSEEN)) != 10 or any(c < 0 or c >= 55 for c in TRUE_UNSEEN):
        raise RuntimeError("TRUE_UNSEEN must contain ten unique 0..54 labels")
    if any(set(TRUE_UNSEEN).intersection(classes) for classes in PSEUDO_GROUPS.values()):
        raise RuntimeError("Pseudo groups overlap the true ten-class unseen set")
    if not all(path.exists() for path in (RAW_CACHE, POSE_NPZ, OBJECT_DIR, PRETRAINED_X3D, TEXT_XLSX, YOLO_WEIGHTS)):
        missing = [str(p) for p in (RAW_CACHE, POSE_NPZ, OBJECT_DIR, PRETRAINED_X3D, TEXT_XLSX, YOLO_WEIGHTS) if not p.exists()]
        raise FileNotFoundError(f"Required input missing: {missing}")

    free_gib = shutil.disk_usage(WS).free / 2**30
    # New X3D+pose features plus the assembled 45/10 copies need about 46 GiB;
    # refuse to start a long run if another job has consumed the safety margin.
    print(f"[{now()}] disk free before run: {free_gib:.1f} GiB", flush=True)
    if free_gib < 50.0:
        raise RuntimeError(f"Only {free_gib:.1f} GiB free; need at least 50 GiB for isolated caches")

    x3d_cfg = configure_x3d()
    ctr_cfg = configure_ctr()
    work_root = VPO / "work_dir/merged45_10_groupAB"
    pseudo_a = {name: work_root / f"pseudo_{name}_stage_a" for name in PSEUDO_GROUPS}
    pseudo_b = {name: work_root / f"pseudo_{name}_stage_b" for name in PSEUDO_GROUPS}
    final_a = work_root / "selected_stage_a"
    final_b = work_root / "selected_stage_b"
    vpo_cfg = {
        "p3_a": vpo_config("vpoclip_p3_stage_a", VPO_BASE_A, pseudo_a["p3"], pseudo=PSEUDO_GROUPS["p3"], epochs=300, selection="pseudo_zsl"),
        "p4_a": vpo_config("vpoclip_p4_stage_a", VPO_BASE_A, pseudo_a["p4"], pseudo=PSEUDO_GROUPS["p4"], epochs=300, selection="pseudo_zsl"),
        "p3_b": vpo_config("vpoclip_p3_stage_b", VPO_BASE_B, pseudo_b["p3"], pseudo=PSEUDO_GROUPS["p3"], init_checkpoint=pseudo_a["p3"] / "best_model.pth", epochs=20, selection="pseudo_zsl"),
        "p4_b": vpo_config("vpoclip_p4_stage_b", VPO_BASE_B, pseudo_b["p4"], pseudo=PSEUDO_GROUPS["p4"], init_checkpoint=pseudo_a["p4"] / "best_model.pth", epochs=20, selection="pseudo_zsl"),
        "final_a": vpo_config("vpoclip_selected_stage_a", VPO_BASE_A, final_a, epochs=300, selection="val_accuracy"),
        "final_b": vpo_config("vpoclip_selected_stage_b", VPO_BASE_B, final_b, init_checkpoint=final_a / "last_model.pth", epochs=20, selection="val_accuracy"),
    }

    write_json(
        RUN_ROOT / "run_manifest.json",
        {
            "started_at": now(),
            "protocol": "strict_zsl_45_10",
            "true_unseen_zero_based": TRUE_UNSEEN,
            "true_unseen_actions": [f"A{x + 1:03d}" for x in TRUE_UNSEEN],
            "seen_zero_based": [x for x in range(55) if x not in TRUE_UNSEEN],
            "pseudo_groups": PSEUDO_GROUPS,
            "subject_manifest": str(SUBJECT_MANIFEST),
            "raw_allview_cache": str(RAW_CACHE),
            "x3d_output": str(X3D_OUTPUT),
            "ctr_output": str(CTR_OUTPUT),
            "x3d_feature_dir": str(X3D_FEATURE_DIR),
            "pose_feature_dir": str(POSE_FEATURE_DIR),
            "object_feature_dir_reused": str(OBJECT_DIR),
            "zsl_dataset": str(ZSL_DIR),
            "vpo_configs": {key: str(value) for key, value in vpo_cfg.items()},
            "compute": {"x3d_batch": X3D_BATCH, "x3d_max_iter": X3D_MAX_ITER, "vpo_batch": VPO_BATCH, "vpo_workers": VPO_WORKERS},
        },
    )

    run_stage(
        "01_x3d_train", X3D,
        [PYTHON, "tools/train_etri_x3d.py", "--config", x3d_cfg, "--data-dir", RAW_CACHE,
         "--output-dir", X3D_OUTPUT, "--pretrained", PRETRAINED_X3D,
         "--exclude-classes", *TRUE_UNSEEN, "--workers", 8, "--prefetch-factor", 2,
         "--batch-size", X3D_BATCH, "--max-iter", X3D_MAX_ITER,
         "--warmup-iterations", 400, "--lr-steps", 3500, 5810,
         "--save-step", 1000, "--eval-step", 1000, "--resume"],
        required=[X3D_OUTPUT / "model_best.pth", X3D_OUTPUT / "model_final.pth", X3D_OUTPUT / "training_summary.json"],
    )

    run_stage("02_ctrgcn_train", CTR, [PYTHON, "main.py", "--config", ctr_cfg], required=[CTR_OUTPUT / "log.txt"])
    pose_weights_file = RUN_ROOT / "pose_checkpoint.txt"
    if not pose_weights_file.exists():
        pose_weights = select_ctr_checkpoint()
        pose_weights_file.write_text(str(pose_weights.resolve()) + "\n", encoding="utf-8")
    else:
        pose_weights = Path(pose_weights_file.read_text(encoding="utf-8").strip())
        if not pose_weights.exists():
            pose_weights = select_ctr_checkpoint()
            pose_weights_file.write_text(str(pose_weights.resolve()) + "\n", encoding="utf-8")
    write_json(RUN_ROOT / "selected_backbones.json", {"x3d": str((X3D_OUTPUT / "model_best.pth").resolve()), "pose": str(pose_weights.resolve()), "yolo": str(YOLO_WEIGHTS.resolve())})

    for split in ("train", "val", "test"):
        run_stage(
            f"03_x3d_extract_{split}", X3D,
            [PYTHON, "tools/extract_x3d_features.py", "--config", x3d_cfg,
             "--checkpoint", X3D_OUTPUT / "model_best.pth", "--data-dir", RAW_CACHE,
             "--split", split, "--output", X3D_FEATURE_DIR / f"{split}_video.npy",
             "--batch-size", 512, "--num-workers", 8, "--tensor-resize-size", 182,
             "--spatial-size", 6, "--save-sidecars"],
            required=[X3D_FEATURE_DIR / f"{split}_video.npy", X3D_FEATURE_DIR / f"{split}_video.labels.npy"],
        )
        run_stage(
            f"04_pose_extract_{split}", VPO,
            [PYTHON, "tools/extract_current_ctrgcn_features.py", "--ctrgcn-root", CTR,
             "--config", ctr_cfg, "--weights", pose_weights, "--npz", POSE_NPZ,
             "--split", split, "--output-dir", POSE_FEATURE_DIR, "--batch-size", 1024],
            required=[POSE_FEATURE_DIR / f"{split}_pose_coco17.npy", POSE_FEATURE_DIR / f"{split}_labels.npy"],
        )

    run_stage(
        "05_assemble_zsl", VPO,
        [PYTHON, "tools/assemble_current_45_25_10_20.py", "--cache-dir", RAW_CACHE,
         "--x3d-dir", X3D_FEATURE_DIR, "--pose-dir", POSE_FEATURE_DIR,
         "--object-dir", OBJECT_DIR, "--zsl-dir", ZSL_DIR,
         "--closed-dir", RUN_ROOT / "closed_unused", "--manifest", SUBJECT_MANIFEST,
         "--unseen-classes", *TRUE_UNSEEN, "--zsl-only"],
        required=[ZSL_DIR / "metadata.json", ZSL_DIR / "trimodal_train_labels.npy", ZSL_DIR / "trimodal_test_labels.npy", ZSL_DIR / "trimodal_test_seen_labels.npy"],
    )
    metadata = validate_assembled_dataset()

    for tag, key in (("06_p3_stage_a", "p3_a"), ("07_p4_stage_a", "p4_a"), ("08_p3_stage_b", "p3_b"), ("09_p4_stage_b", "p4_b")):
        run_stage(tag, VPO, [PYTHON, "train.py", "--config", vpo_cfg[key]])
        # Explicit required checks are kept separate because each generated
        # config has absolute output paths and is easier to audit this way.
        out_dir = pseudo_a["p3"] if key == "p3_a" else pseudo_a["p4"] if key == "p4_a" else pseudo_b["p3"] if key == "p3_b" else pseudo_b["p4"]
        for output in (out_dir / "best_model.pth", out_dir / "last_model.pth", out_dir / "history.json"):
            if not output.exists() or output.stat().st_size == 0:
                raise RuntimeError(f"{tag} missing output {output}")

    run_stage(
        "10_select_pseudo", VPO,
        [PYTHON, "tools/select_merged45_10_pseudo_epochs.py",
         "--stage-a", pseudo_a["p3"] / "history.json", pseudo_a["p4"] / "history.json",
         "--stage-b", pseudo_b["p3"] / "history.json", pseudo_b["p4"] / "history.json",
         "--data-dir", ZSL_DIR, "--output", RUN_ROOT / "selection.json", "--smooth-a", 9, "--smooth-b", 3],
        required=[RUN_ROOT / "selection.json"],
    )
    selection = json.loads((RUN_ROOT / "selection.json").read_text(encoding="utf-8"))
    stage_a_epochs = int(selection["recommended_stage_a_epochs"])
    stage_b_epochs = int(selection["recommended_stage_b_epochs"])
    write_json(RUN_ROOT / "selected_epochs.json", {"stage_a": stage_a_epochs, "stage_b": stage_b_epochs, "selection": str(RUN_ROOT / "selection.json")})

    run_stage(
        "11_full_stage_a", VPO,
        [PYTHON, "train.py", "--config", vpo_cfg["final_a"], "--epochs", stage_a_epochs],
        required=[final_a / "best_model.pth", final_a / "last_model.pth", final_a / "history.json"],
    )
    run_stage(
        "12_full_stage_b", VPO,
        [PYTHON, "train.py", "--config", vpo_cfg["final_b"], "--epochs", stage_b_epochs],
        required=[final_b / "best_model.pth", final_b / "last_model.pth", final_b / "history.json"],
    )

    final_checkpoint = final_b / "last_model.pth"
    run_stage(
        "13_eval_unseen_zsl", VPO,
        [PYTHON, "test.py", "--config", vpo_cfg["final_b"], "--checkpoint", final_checkpoint,
         "--split-dir", ZSL_DIR, "--class-split-dir", ZSL_DIR, "--prefix", "trimodal_test",
         "--candidate-scope", "unseen", "--sample-scope", "all", "--batch-size", 4096,
         "--num-workers", 8, "--output", final_b / "unseen_zsl.json"],
        required=[final_b / "unseen_zsl.json"],
    )
    run_stage(
        "14_eval_seen_zsl", VPO,
        [PYTHON, "test.py", "--config", vpo_cfg["final_b"], "--checkpoint", final_checkpoint,
         "--split-dir", ZSL_DIR, "--class-split-dir", ZSL_DIR, "--prefix", "trimodal_test_seen",
         "--candidate-scope", "seen", "--sample-scope", "all", "--batch-size", 4096,
         "--num-workers", 8, "--output", final_b / "seen_zsl.json"],
        required=[final_b / "seen_zsl.json"],
    )

    cleanup_intermediate_checkpoints(list(pseudo_a.values()) + list(pseudo_b.values()))
    summary = {
        "completed_at": now(),
        "protocol": "strict_zsl_45_10",
        "true_unseen_zero_based": TRUE_UNSEEN,
        "true_unseen_actions": [f"A{x + 1:03d}" for x in TRUE_UNSEEN],
        "seen_count": 45,
        "subject_split": str(SUBJECT_MANIFEST),
        "dataset_metadata": str(ZSL_DIR / "metadata.json"),
        "dataset_counts": metadata.get("counts"),
        "x3d_checkpoint": str((X3D_OUTPUT / "model_best.pth").resolve()),
        "pose_checkpoint": str(pose_weights.resolve()),
        "vpo_final_checkpoint_selected": str(final_checkpoint.resolve()),
        "vpo_final_best_checkpoint": str((final_b / "best_model.pth").resolve()),
        "pseudo_selection": str((RUN_ROOT / "selection.json").resolve()),
        "selected_stage_a_epochs": stage_a_epochs,
        "selected_stage_b_epochs": stage_b_epochs,
        "unseen_eval": str((final_b / "unseen_zsl.json").resolve()),
        "seen_eval": str((final_b / "seen_zsl.json").resolve()),
    }
    write_json(RUN_ROOT / "complete.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print(f"[{now()}] MERGED_45_10_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
