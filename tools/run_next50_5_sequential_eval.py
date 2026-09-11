#!/usr/bin/env python3
"""Queue the next two all-view 50/5 runs behind the current pipeline.

The current ``new50_5_groups_20260909`` run is allowed to finish first.  Each
new group is then processed end-to-end (X3D, CTR-GCN, aligned trimodal cache,
the two-stage VPOCLIP recipe), evaluated on seen and unseen test samples, and
only then is the next group started.  This wrapper reuses the audited driver
without sharing its output directories or feature scratch space.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
import os
import subprocess
import time
from pathlib import Path


VPO = Path("/home/youhan/ws/VPOCLIP_plus_full")
CURRENT_RUN = VPO / "logs/new50_5_groups_20260909"
RUN_ROOT = VPO / "logs/new50_5_groups_sequential_20260909"
SELECTION = RUN_ROOT / "class_selection.json"
FEATURE_DIR = VPO / "data/new50_5_groups_sequential_shared_features"
PYTHON = Path("/home/youhan/.conda/envs/clipgcn/bin/python")


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def load_driver():
    path = VPO / "tools/run_new50_5_groups_pipeline.py"
    spec = importlib.util.spec_from_file_location("new50_driver", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def wait_for_current_run() -> None:
    """Wait on the authoritative completion marker, never on a stale log line."""

    complete = CURRENT_RUN / "complete.json"
    while not complete.exists():
        if subprocess.run(
            ["tmux", "has-session", "-t", "vpoclip_new50_5_groups_20260909"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode != 0:
            raise RuntimeError(
                f"current pipeline session disappeared before completion marker: {complete}"
            )
        time.sleep(60)
    payload = json.loads(complete.read_text(encoding="utf-8"))
    if len(payload.get("groups", [])) != 2:
        raise RuntimeError(f"current pipeline marker is incomplete: {complete}")
    print(f"[{now()}] CURRENT PIPELINE COMPLETE; releasing queued run", flush=True)


def run_eval(driver, group: dict) -> None:
    """Evaluate stage-B last checkpoint before allowing the next group to run."""

    group_root = RUN_ROOT / group["name"]
    marker = group_root / "08_eval_seen_unseen.done"
    if marker.exists():
        return
    zsl_dir = VPO / "data" / f"new50_5_{group['name']}_zsl"
    cfg = group_root / "vpoclip_stage_b.yaml"
    ckpt = VPO / "work_dir" / f"new50_5_{group['name']}" / "recipe_stage_b" / "last_model.pth"
    out_dir = group_root / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    common = [
        PYTHON,
        "test.py",
        "--config",
        cfg,
        "--checkpoint",
        ckpt,
        "--split-dir",
        zsl_dir,
        "--class-split-dir",
        zsl_dir,
        "--batch-size",
        256,
        "--num-workers",
        12,
    ]
    calls = {
        "unseen5_zsl": ["--prefix", "trimodal_test", "--sample-scope", "all", "--candidate-scope", "unseen"],
        "seen50_closed": ["--prefix", "trimodal_test_seen", "--sample-scope", "all", "--candidate-scope", "seen"],
        "gzsl_unseen55": ["--prefix", "trimodal_test", "--protocol", "gzsl_unseen"],
        "gzsl_seen55": ["--prefix", "trimodal_test_seen", "--protocol", "gzsl_seen"],
    }
    env = os.environ.copy()
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "CUDA_VISIBLE_DEVICES": "0",
            "OMP_NUM_THREADS": "8",
            "MKL_NUM_THREADS": "8",
            "OPENBLAS_NUM_THREADS": "1",
            "PYTHONPATH": str(VPO) + os.pathsep + env.get("PYTHONPATH", ""),
        }
    )
    log_path = group_root / "08_eval_seen_unseen.log"
    results = {}
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"[{now()}] EVALUATION checkpoint={ckpt}\n")
        for name, args in calls.items():
            output = out_dir / f"{name}.json"
            argv = [str(x) for x in common] + [str(x) for x in args] + ["--output", str(output)]
            log.write(f"[{now()}] COMMAND {' '.join(argv)}\n")
            log.flush()
            result = subprocess.run(argv, cwd=str(VPO), env=env, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f"evaluation {name} failed with exit code {result.returncode}")
            payload = json.loads(output.read_text(encoding="utf-8"))
            results[name] = payload.get("metrics", payload)
    seen = float(results["gzsl_seen55"].get("top1_acc", 0.0))
    unseen = float(results["gzsl_unseen55"].get("top1_acc", 0.0))
    results["gzsl_harmonic_mean"] = 2 * seen * unseen / (seen + unseen) if seen + unseen else 0.0
    write_json(
        group_root / "evaluation_seen_unseen.json",
        {
            "evaluated_at": now(),
            "group": group,
            "checkpoint": str(ckpt.resolve()),
            "zsl_dataset": str(zsl_dir.resolve()),
            "metrics": results,
        },
    )
    marker.write_text(now() + "\n", encoding="utf-8")
    print(f"[{now()}] EVALUATED {group['name']} seen/unseen", flush=True)


def main() -> None:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    wait_for_current_run()

    driver = load_driver()
    # Override every scratch/output root used by the imported driver.  The
    # class-agnostic raw RGB, RTMPose and YOLO caches remain shared read-only.
    driver.SELECTION = SELECTION
    driver.RUN_ROOT = RUN_ROOT
    driver.FEATURE_DIR = FEATURE_DIR
    groups = driver.load_selection()
    write_json(RUN_ROOT / "pipeline_manifest.json", {"started_at": now(), "groups": groups})

    telemetry_path = RUN_ROOT / "gpu_telemetry.csv"
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
        for group in groups:
            driver.run_group(group)
            run_eval(driver, group)
    finally:
        monitor.terminate()
        try:
            monitor.wait(timeout=5)
        except subprocess.TimeoutExpired:
            monitor.kill()
        telemetry.close()
    write_json(RUN_ROOT / "complete.json", {"completed_at": now(), "groups": groups})
    print(f"[{now()}] ALL QUEUED GROUPS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
