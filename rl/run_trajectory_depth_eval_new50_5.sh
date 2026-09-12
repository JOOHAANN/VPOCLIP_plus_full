#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/youhan/ws/VPOCLIP_plus_full"
PYTHON_BIN="/home/youhan/.conda/envs/clipgcn/bin/python"
POLICY_ROOT="$PROJECT_ROOT/work_dir/trajectory_depth_new50_5"
OUTPUT_ROOT="$POLICY_ROOT/eval_30seeds"
LOG_ROOT="$PROJECT_ROOT/logs/trajectory_depth_new50_5_eval"

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"

for variant in v4_depth v5_depth v6_depth object_only_depth; do
  echo "START $variant" | tee "$LOG_ROOT/${variant}.log"
  cd "$PROJECT_ROOT"
  "$PYTHON_BIN" -u -m rl.evaluate_trajectory_depth_variants_new_split \
    --variant "$variant" \
    --policy-root "$POLICY_ROOT/$variant" \
    --output-root "$OUTPUT_ROOT/$variant" \
    --eval-seeds 20260920 20260921 20260922 20260923 20260924 \
                 20260925 20260926 20260927 20260928 20260929 \
                 20260930 20260931 20260932 20260933 20260934 \
                 20260935 20260936 20260937 20260938 20260939 \
                 20260940 20260941 20260942 20260943 20260944 \
                 20260945 20260946 20260947 20260948 20260949 \
    >> "$LOG_ROOT/${variant}.log" 2>&1
  echo "DONE $variant" | tee -a "$LOG_ROOT/${variant}.log"
done

echo "ALL_TRAJECTORY_DEPTH_EVALUATION_COMPLETE" | tee "$LOG_ROOT/pipeline.done"
