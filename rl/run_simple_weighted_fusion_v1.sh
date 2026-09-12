#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/youhan/ws/VPOCLIP_plus_full"
PYTHON="/home/youhan/.conda/envs/clipgcn/bin/python"
CACHE="$ROOT/data/rl_raw_new50_5_group1_g1_pseudo_selected_stage_b/cache"
GATE_OUT="$ROOT/work_dir/fusion_lightweight_gating_v1"
RULE_OUT="$ROOT/work_dir/fusion_rule_based_v1"
LOG_DIR="$ROOT/logs"

cd "$ROOT"
export PYTHONUNBUFFERED=1

"$PYTHON" -m rl.fusion_lightweight_gating_v1 \
  --cache-root "$CACHE" \
  --output-root "$GATE_OUT" \
  --epochs 40 --batch-size 16384 --updates-per-epoch 64 \
  --train-seeds 20260909 20260910 20260911 \
  --eval-seeds $(seq 20260920 20260949) \
  2>&1 | tee "$LOG_DIR/fusion_lightweight_gating_v1.log"

"$PYTHON" -m rl.fusion_rule_based_v1 \
  --cache-root "$CACHE" \
  --output-root "$RULE_OUT" \
  --gate-root "$GATE_OUT" \
  --eval-seeds $(seq 20260920 20260949) \
  2>&1 | tee "$LOG_DIR/fusion_rule_based_v1.log"

echo SIMPLE_WEIGHTED_LOGIT_FUSION_EXPERIMENTS_COMPLETE

