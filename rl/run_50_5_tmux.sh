#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/youhan/ws/VPOCLIP_plus_full"
PYTHON_BIN="/home/youhan/.conda/envs/clipgcn/bin/python"
LOG_FILE="${PROJECT_DIR}/logs/active_rl_50_5.log"

cd "${PROJECT_DIR}"
mkdir -p "${PROJECT_DIR}/logs"
exec > >(tee -a "${LOG_FILE}") 2>&1

date -Is
echo "protocol=strict_zsl_50_5 split=dqn_train subjects=25 input=640x480 window=13"
echo "stage=build_dqn_train"
"${PYTHON_BIN}" -u -m rl.build_cache --config rl/config_rl.yaml --split dqn_train

echo "stage=build_val"
"${PYTHON_BIN}" -u -m rl.build_cache --config rl/config_rl.yaml --split val

echo "stage=train_ddqn"
"${PYTHON_BIN}" -u -m rl.train_rl --config rl/config_rl.yaml

date -Is
echo "status=complete"
