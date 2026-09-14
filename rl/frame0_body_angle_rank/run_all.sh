#!/usr/bin/env bash
set -Eeuo pipefail

# Isolated frame-0 anatomical-angle experiment.  The shell runs one variant
# per Python process so GPU memory is released before the next model starts.
ROOT="/home/youhan/ws/VPOCLIP_plus_full"
PACKAGE="rl.frame0_body_angle_rank.train_ranked_variant"
CACHE="$ROOT/data/frame0_body_angle_rank_g1_pseudo_selected_stage_b/cache"
TRACK="$ROOT/data/frame0_body_angle_rank_g1_pseudo_selected_stage_b/object_tracks_full"
TRACK_V3="$ROOT/data/frame0_body_angle_rank_g1_pseudo_selected_stage_b/object_tracks_v3_bbox_size"
OUT="$ROOT/work_dir/frame0_body_angle_rank_v1_v6"
LOG="$ROOT/logs/frame0_body_angle_rank_v1_v6"

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
  echo "FRAME0_ANGLE_LAUNCH $variant"
  python3 -m "$PACKAGE" \
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
    2>&1 | tee "$LOG/$variant.log"
done

echo "FRAME0_ANGLE_ALL_VARIANTS_COMPLETE"
