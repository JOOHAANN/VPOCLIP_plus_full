#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/youhan/ws/VPOCLIP_plus_full"
PYTHON_BIN="/home/youhan/.conda/envs/clipgcn/bin/python"
LOG_FILE="${PROJECT_DIR}/logs/active_rl_body_relative_v2_50_5_fast.log"

cd "${PROJECT_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1
date -Is
echo "stage=train_gpu_resident_body_relative_ddqn protocol=strict_zsl_50_5"
"${PYTHON_BIN}" -u -m rl.train_rl_fast --config rl/config_rl.yaml
date -Is
echo "status=complete"
