#!/usr/bin/env bash
set -Eeuo pipefail

# Strict replacement for the retired [0,2,26,34,50] protocol.
# This script never writes the historical frame0_body_angle_rank_v1_v6 output.
ROOT="/home/youhan/ws/VPOCLIP_plus_full"
PACKAGE="rl.frame0_body_angle_rank.train_ranked_variant"
PYTHON="/home/youhan/.conda/envs/clipgcn/bin/python"
CACHE="$ROOT/data/frame0_body_angle_rank_g1_pseudo_selected_stage_b/cache"
TRACK="$ROOT/data/frame0_body_angle_rank_g1_pseudo_selected_stage_b/object_tracks_full"
TRACK_V3="$ROOT/data/frame0_body_angle_rank_g1_pseudo_selected_stage_b/object_tracks_v3_bbox_size"
OUT="$ROOT/work_dir/frame0_body_angle_rank_new50_5_v1_v6"
LOG="$ROOT/logs/frame0_body_angle_rank_new50_5_v1_v6"
UNSEEN=(14 15 25 39 46)
PSEUDO=(0 2 9 10 11 17 26 34 49 50)

mkdir -p "$LOG" "$OUT"
cd "$ROOT"

if [[ ! -f "$CACHE/../build_summary.json" ]]; then
  echo "FRAME0_ANGLE_RUN_BLOCKED ranked cache is not ready: $CACHE" >&2
  exit 2
fi

variants=(v1 v2_legacy v3 v4 v5 v6 object_only geometry_only)
for variant in "${variants[@]}"; do
  if [[ -f "$OUT/$variant/complete.json" ]]; then
    echo "FRAME0_ANGLE_SKIP $variant already complete"
    continue
  fi
  echo "FRAME0_ANGLE_LAUNCH_NEW50_5 $variant"
  "$PYTHON" -m "$PACKAGE" \
    --variant "$variant" \
    --cache-root "$CACHE" \
    --track-root "$TRACK" \
    --track-root-v3 "$TRACK_V3" \
    --output-root "$OUT/$variant" \
    --epochs 40 \
    --batch-size 16384 \
    --updates-per-epoch 64 \
    --learning-rate 1e-4 \
    --seeds 20260909 20260910 20260911 \
    --memory-fraction 0.16 \
    --unseen-classes "${UNSEEN[@]}" \
    --pseudo-unseen-classes "${PSEUDO[@]}" \
    2>&1 | tee "$LOG/$variant.log"
done

echo "FRAME0_ANGLE_NEW50_5_ALL_VARIANTS_COMPLETE"
