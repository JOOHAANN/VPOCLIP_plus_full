#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/youhan/ws/VPOCLIP_plus_full"
PYTHON_BIN="/home/youhan/.conda/envs/clipgcn/bin/python"
OUTPUT_ROOT="$PROJECT_ROOT/work_dir/trajectory_depth_new50_5"
LOG_ROOT="$PROJECT_ROOT/logs/trajectory_depth_new50_5"

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"

for variant in v4_depth v5_depth v6_depth object_only_depth; do
  echo "START $variant" | tee "$LOG_ROOT/${variant}.log"
  cd "$PROJECT_ROOT"
  "$PYTHON_BIN" -u -m rl.train_trajectory_depth_variant \
    --variant "$variant" \
    --output-root "$OUTPUT_ROOT/$variant" \
    --epochs 40 \
    --batch-size 16384 \
    --updates-per-epoch 64 \
    --seeds 20260909 20260910 20260911 \
    >> "$LOG_ROOT/${variant}.log" 2>&1
  echo "DONE $variant" | tee -a "$LOG_ROOT/${variant}.log"
done

echo "ALL_TRAJECTORY_DEPTH_TRAINING_COMPLETE" | tee "$LOG_ROOT/pipeline.done"
