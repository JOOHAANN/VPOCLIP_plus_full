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

printf '%s\n' "FIRST_FRAME_30_SEED_EVAL_RESUME_START $(date -Is)" | tee "$LOG_ROOT/pipeline_resume.log"
variants=(v6 object_only geometry_only)
for variant in "${variants[@]}"; do
    printf '%s\n' "VARIANT_START $variant $(date -Is)" | tee -a "$LOG_ROOT/pipeline_resume.log"
    "$PYTHON_BIN" -u -m rl.evaluate_first_frame_30seeds \
        --variant "$variant" \
        --cache-root "$CACHE_ROOT" \
        --track-root "$TRACK_ROOT" \
        --v3-track-root "$V3_TRACK_ROOT" \
        --policy-root "$POLICY_ROOT/$variant" \
        --output-root "$OUTPUT_ROOT/$variant" \
        2>&1 | tee "$LOG_ROOT/${variant}.log"
    printf '%s\n' "VARIANT_COMPLETE $variant $(date -Is)" | tee -a "$LOG_ROOT/pipeline_resume.log"
done

printf '%s\n' "FIRST_FRAME_30_SEED_EVAL_RESUME_COMPLETE $(date -Is)" | tee -a "$LOG_ROOT/pipeline_resume.log"
