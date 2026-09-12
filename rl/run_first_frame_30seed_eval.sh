#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/youhan/ws/VPOCLIP_plus_full"
PYTHON_BIN="/home/youhan/.conda/envs/clipgcn/bin/python"
CACHE_ROOT="$PROJECT_ROOT/data/trajectory_latest_vpoclip_new50_5/cache_compat"
TRACK_ROOT="$PROJECT_ROOT/data/trajectory_latest_vpoclip_new50_5/object_tracks_full"
V3_TRACK_ROOT="$PROJECT_ROOT/data/angle_object_trajectory_v3_bbox_size"
POLICY_ROOT="$PROJECT_ROOT/work_dir/trajectory_first_frame_new50_5"
OUTPUT_ROOT="$PROJECT_ROOT/work_dir/trajectory_first_frame_new50_5_eval30"
LOG_ROOT="$PROJECT_ROOT/logs/trajectory_first_frame_new50_5_eval30"

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"
cd "$PROJECT_ROOT"

printf '%s\n' "FIRST_FRAME_30_SEED_EVAL_START $(date -Is)" | tee "$LOG_ROOT/pipeline.log"
printf '%s\n' "eval_seeds=20260920..20260949" | tee -a "$LOG_ROOT/pipeline.log"

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
    "$PYTHON_BIN" -u -m rl.evaluate_first_frame_30seeds \
        --variant "$variant" \
        --cache-root "$CACHE_ROOT" \
        --track-root "$TRACK_ROOT" \
        --v3-track-root "$V3_TRACK_ROOT" \
        --policy-root "$POLICY_ROOT/$variant" \
        --output-root "$OUTPUT_ROOT/$variant" \
        2>&1 | tee "$LOG_ROOT/${variant}.log"
    printf '%s\n' "VARIANT_COMPLETE $variant $(date -Is)" | tee -a "$LOG_ROOT/pipeline.log"
done

printf '%s\n' "FIRST_FRAME_30_SEED_EVAL_COMPLETE $(date -Is)" | tee -a "$LOG_ROOT/pipeline.log"
