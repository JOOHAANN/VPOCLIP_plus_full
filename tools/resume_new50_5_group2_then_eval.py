#!/usr/bin/env python3
"""Resume the interrupted second group with an 80-GiB-safe X3D batch.

Group 1 is already complete.  This driver resumes only group 2 from its
latest X3D checkpoint, completes CTR-GCN, feature extraction and both VPOCLIP
stages, then runs the requested seen/unseen ZSL evaluation before writing a
completion marker.  It intentionally uses a smaller X3D batch than the first
run so peak allocation stays below the user's 80-GiB budget.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
import os
import subprocess
from pathlib import Path


VPO = Path("/home/youhan/ws/VPOCLIP_plus_full")
RUN_ROOT = VPO / "logs/new50_5_groups_20260909"
SELECTION = RUN_ROOT / "class_selection.json"
FEATURE_DIR = VPO / "data/new50_5_groups_shared_features"
GROUP_NAME = "group2"
PYTHON = Path("/home/youhan/.conda/envs/clipgcn/bin/python")


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def load_driver():
    path = VPO / "tools/run_new50_5_groups_pipeline.py"
    spec = importlib.util.spec_from_file_location("new50_driver_resume", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_group2(driver, group: dict) -> None:
    # 192 gives a generous margin under 80 GiB, including the prefetched
    # training batch and the validation loader (2x batch).
    driver.X3D_BATCH = 192
    driver.X3D_MAX_ITER = 7000
    driver.RUN_ROOT = RUN_ROOT
    driver.SELECTION = SELECTION
    driver.FEATURE_DIR = FEATURE_DIR
    driver.run_group(group)


def run_zsl_eval(group: dict) -> None:
    """Run strict 5-way unseen and 50-way seen evaluation on CPU."""

    script = VPO / "scripts/evaluate_new50_5_group1_last.sh"
    env = os.environ.copy()
    env.update(
        {
            "RUN_ROOT_NAME": "new50_5_groups_20260909",
            "GROUP_NAME": GROUP_NAME,
            "MODEL_VARIANT": "last",
            "CUDA_VISIBLE_DEVICES": "",
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "8",
            "MKL_NUM_THREADS": "8",
            "OPENBLAS_NUM_THREADS": "1",
        }
    )
    log_path = RUN_ROOT / GROUP_NAME / "evaluation_last.log"
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"[{now()}] queued post-training ZSL evaluation\n")
        result = subprocess.run([str(script)], cwd=str(VPO), env=env, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"second-group ZSL evaluation failed with exit code {result.returncode}")


def main() -> None:
    driver = load_driver()
    groups = driver.load_selection()
    group = next(item for item in groups if item["name"] == GROUP_NAME)
    group_root = RUN_ROOT / GROUP_NAME
    group_root.mkdir(parents=True, exist_ok=True)
    write_json(
        group_root / "resume_manifest.json",
        {
            "started_at": now(),
            "group": group,
            "resume_checkpoint": str(
                (driver.X3D / "outputs/etri_allviews_cs45_new50_5_group2_7000/model_002000.pth").resolve()
            ),
            "x3d_batch_size": 192,
            "x3d_memory_budget_gib": 80,
        },
    )
    telemetry_path = RUN_ROOT / "gpu_telemetry_resume.csv"
    telemetry = telemetry_path.open("a", encoding="utf-8")
    monitor = subprocess.Popen(
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
    try:
        run_group2(driver, group)
    finally:
        monitor.terminate()
        try:
            monitor.wait(timeout=5)
        except subprocess.TimeoutExpired:
            monitor.kill()
        telemetry.close()

    # This is deliberately after all model stages; a failed evaluation leaves
    # no completion marker and therefore cannot be mistaken for a finished run.
    run_zsl_eval(group)
    write_json(
        RUN_ROOT / "group2_resume_complete.json",
        {
            "completed_at": now(),
            "group": group,
            "evaluation": str((RUN_ROOT / GROUP_NAME / "evaluation_last/summary.json").resolve()),
            "stage_b_last": str(
                (VPO / "work_dir/new50_5_group2/recipe_stage_b/last_model.pth").resolve()
            ),
        },
    )
    print(f"[{now()}] GROUP2 RESUME + EVALUATION COMPLETE", flush=True)


if __name__ == "__main__":
    main()
