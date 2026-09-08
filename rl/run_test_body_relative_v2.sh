#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/youhan/ws/VPOCLIP_plus_full"
PYTHON_BIN="/home/youhan/.conda/envs/clipgcn/bin/python"
CONFIG="${PROJECT_DIR}/rl/config_rl.yaml"
OUTPUT_DIR="${PROJECT_DIR}/work_dir/active_view_ddqn_body_relative_v2"
TEST_CACHE="${PROJECT_DIR}/data/active_multiview_cache/test"
LOG_FILE="${PROJECT_DIR}/logs/active_rl_body_relative_v2_50_5_test.log"

cd "${PROJECT_DIR}"
mkdir -p "${PROJECT_DIR}/logs" "${OUTPUT_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

date -Is
echo "stage=prepare_test_cache protocol=strict_zsl_50_5 subjects=20 input=640x480 window=13"

required=(z.npy logits.npy pose.npy object_map.npy image_quality.npy view_geometry.npy reachable.npy move_cost.npy labels.npy target_columns.npy text_prototypes.npy metadata.json)
cache_complete=true
for name in "${required[@]}"; do
    if [[ ! -s "${TEST_CACHE}/${name}" ]]; then
        cache_complete=false
        break
    fi
done

if [[ "${cache_complete}" == "false" ]]; then
    "${PYTHON_BIN}" -u -m rl.build_cache --config "${CONFIG}" --split test
else
    echo "test_cache=complete action=reuse"
fi

echo "stage=evaluate_best checkpoint=epoch2"
"${PYTHON_BIN}" -u -m rl.evaluate_rl \
    --config "${CONFIG}" \
    --checkpoint "${OUTPUT_DIR}/best.pt" \
    --split test
cp "${OUTPUT_DIR}/evaluation_test.json" "${OUTPUT_DIR}/evaluation_test_best.json"

echo "stage=evaluate_last checkpoint=epoch50"
"${PYTHON_BIN}" -u -m rl.evaluate_rl \
    --config "${CONFIG}" \
    --checkpoint "${OUTPUT_DIR}/last.pt" \
    --split test
cp "${OUTPUT_DIR}/evaluation_test.json" "${OUTPUT_DIR}/evaluation_test_last.json"

date -Is
echo "status=complete"
