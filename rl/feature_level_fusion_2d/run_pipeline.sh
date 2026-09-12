#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/youhan/ws/VPOCLIP_plus_full"
PYTHON="/home/youhan/.conda/envs/clipgcn/bin/python"
CACHE="$ROOT/data/rl_raw_new50_5_group1_g1_pseudo_selected_stage_b/cache"
TRACK="$ROOT/data/trajectory_latest_vpoclip_new50_5/object_tracks_full"
OUT="$ROOT/work_dir/trajectory_g1_pseudo_selected_stage_b_feature_level_fusion_2d"
LOG="$ROOT/logs/trajectory_g1_pseudo_selected_stage_b_feature_level_fusion_2d"
FUSION="$OUT/fusion"
mkdir -p "$OUT" "$LOG"

cd "$ROOT"
export PYTHONUNBUFFERED=1

if [[ ! -f "$FUSION/training_summary.json" ]]; then
  "$PYTHON" -m rl.feature_level_fusion_2d.train_feature_fusion \
    --cache-root "$CACHE" \
    --output-root "$FUSION" \
    --epochs 40 --batch-size 32768 --updates-per-epoch 64 \
    --seeds 20260909 20260910 20260911 \
    2>&1 | tee "$LOG/00_feature_fusion_train.log"
fi

for variant in v4 v5 v6 object_only geometry_only; do
  variant_out="$OUT/$variant"
  if [[ ! -f "$variant_out/training_summary.json" ]]; then
    "$PYTHON" -m rl.feature_level_fusion_2d.train_trajectory \
      --variant "$variant" \
      --cache-root "$CACHE" \
      --track-root "$TRACK" \
      --fusion-root "$FUSION" \
      --output-root "$variant_out" \
      --epochs 40 --batch-size 16384 --updates-per-epoch 64 \
      --seeds 20260909 20260910 20260911 \
      2>&1 | tee "$LOG/01_train_${variant}.log"
  fi
done

for variant in v4 v5 v6 object_only geometry_only; do
  variant_out="$OUT/$variant"
  if [[ ! -f "$variant_out/evaluation_test_30seeds/summary.json" ]]; then
    "$PYTHON" -m rl.feature_level_fusion_2d.evaluate \
      --variant "$variant" \
      --cache-root "$CACHE" \
      --track-root "$TRACK" \
      --fusion-root "$FUSION" \
      --policy-root "$variant_out" \
      --output-root "$variant_out/evaluation_test_30seeds" \
      --expected-episodes 271 \
      --eval-seeds $(seq 20260920 20260949) \
      2>&1 | tee "$LOG/02_eval_${variant}.log"
  fi
done

echo FEATURE_LEVEL_FUSION_PIPELINE_COMPLETE

