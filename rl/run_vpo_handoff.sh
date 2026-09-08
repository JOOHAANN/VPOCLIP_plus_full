#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/youhan/ws/VPOCLIP_plus_full"
PY="/home/youhan/.conda/envs/clipgcn/bin/python"
LOG_DIR="${ROOT}/logs/rl_handoff"
mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_DIR}/handoff.log") 2>&1
cd "${ROOT}"

wait_for_marker() {
    local marker="$1"
    echo "WAITING_FOR ${marker}"
    while [[ ! -f "${marker}" ]]; do
        sleep 60
    done
    echo "READY ${marker}"
}

check_pipeline() {
    local log="$1"
    if rg -n 'Traceback|uncaught exception|RuntimeError:|ALL CAMERA TRAINING FAILED' "${log}"; then
        echo "ERROR unresolved failure in ${log}"
        exit 1
    fi
}

wait_for_marker "${ROOT}/logs/allviews_20260907/10_final55_aug_entropy_h5_full300_score.done"
wait_for_marker "${ROOT}/logs/allviews_lowview50_5_20260907/10_final_aug_entropy_single_h2.done"
check_pipeline "${ROOT}/logs/allviews_20260907/pipeline.log"
check_pipeline "${ROOT}/logs/allviews_lowview50_5_20260907/pipeline.log"

"${PY}" -u -m rl.prepare_handoff_configs

run_one() {
    local name="$1"
    local cfg="${ROOT}/rl/handoff_configs/${name}.yaml"
    local out
    out=$("${PY}" - <<PY
import yaml
from pathlib import Path
c=yaml.safe_load(Path("${cfg}").read_text())
print(Path("${cfg}").parent.parent / c["train"]["output_dir"] if not Path(c["train"]["output_dir"]).is_absolute() else c["train"]["output_dir"])
PY
)
    mkdir -p "${out}"
    echo "=== START ${name} cache dqn_train ==="
    "${PY}" -u -m rl.build_cache --config "${cfg}" --split dqn_train
    echo "=== START ${name} cache val ==="
    "${PY}" -u -m rl.build_cache --config "${cfg}" --split val
    echo "=== START ${name} cache test ==="
    "${PY}" -u -m rl.build_cache --config "${cfg}" --split test
    echo "=== START ${name} DDQN ==="
    "${PY}" -u -m rl.train_rl_fast --config "${cfg}"
    echo "=== START ${name} evaluation best ==="
    "${PY}" -u -m rl.evaluate_rl --config "${cfg}" --checkpoint "${out}/best.pt" --split test
    cp "${out}/evaluation_test.json" "${out}/evaluation_test_best.json"
    echo "=== START ${name} evaluation last ==="
    "${PY}" -u -m rl.evaluate_rl --config "${cfg}" --checkpoint "${out}/last.pt" --split test
    cp "${out}/evaluation_test.json" "${out}/evaluation_test_last.json"
    echo "=== COMPLETE ${name} ==="
}

run_one new50_5
run_one old50_5
date -Is
echo "status=complete"
