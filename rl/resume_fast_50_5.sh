#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/youhan/ws/VPOCLIP_plus_full"
PYTHON_BIN="/home/youhan/.conda/envs/clipgcn/bin/python"
CHECKPOINT="${PROJECT_DIR}/work_dir/active_view_ddqn/last.pt"
LOG_FILE="${PROJECT_DIR}/logs/active_rl_50_5_fast.log"

cd "${PROJECT_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1
date -Is
echo "stage=resume_gpu_resident_ddqn checkpoint=${CHECKPOINT}"
"${PYTHON_BIN}" -u -m rl.train_rl_fast \
  --config rl/config_rl.yaml \
  --resume "${CHECKPOINT}"
date -Is
echo "status=complete"
