#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/youhan/ws/VPOCLIP_plus_full"
PYTHON_BIN="/home/youhan/.conda/envs/clipgcn/bin/python"
OUTPUT_ROOT="$PROJECT_ROOT/work_dir/trajectory_first_frame_new50_5"
LOG_ROOT="$PROJECT_ROOT/logs/trajectory_first_frame_new50_5"

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"
cd "$PROJECT_ROOT"

printf '%s\n' "FIRST_FRAME_PIPELINE_START $(date -Is)" | tee "$LOG_ROOT/pipeline.log"
printf '%s\n' "cache=$PROJECT_ROOT/data/trajectory_latest_vpoclip_new50_5/cache_compat" | tee -a "$LOG_ROOT/pipeline.log"
printf '%s\n' "track=$PROJECT_ROOT/data/trajectory_latest_vpoclip_new50_5/object_tracks_full" | tee -a "$LOG_ROOT/pipeline.log"

variants=(v1 v3 v4 v5 v6 object_only geometry_only)
start_variant="${START_VARIANT:-v1}"
started=0
for variant in "${variants[@]}"; do
    if [[ "$variant" == "$start_variant" ]]; then
        started=1
    fi
    if [[ "$started" != 1 ]]; then
        continue
    fi
    printf '%s\n' "VARIANT_START $variant $(date -Is)" | tee -a "$LOG_ROOT/pipeline.log"
    "$PYTHON_BIN" -u -m rl.first_frame_analysis.train_first_frame \
        --variant "$variant" \
        --cache-root "$PROJECT_ROOT/data/trajectory_latest_vpoclip_new50_5/cache_compat" \
        --track-root "$PROJECT_ROOT/data/trajectory_latest_vpoclip_new50_5/object_tracks_full" \
        --output-root "$OUTPUT_ROOT/$variant" \
        --epochs 40 \
        --batch-size 16384 \
        --updates-per-epoch 64 \
        --learning-rate 1e-4 \
        --seeds 20260909 20260910 20260911 \
        --true-unseen 14 15 25 39 46 \
        --pseudo-unseen 0 2 9 10 11 17 26 34 49 50 \
        2>&1 | tee "$LOG_ROOT/${variant}.log"
    printf '%s\n' "VARIANT_COMPLETE $variant $(date -Is)" | tee -a "$LOG_ROOT/pipeline.log"
done

printf '%s\n' "FIRST_FRAME_PIPELINE_COMPLETE $(date -Is)" | tee -a "$LOG_ROOT/pipeline.log"
