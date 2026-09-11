#!/usr/bin/env bash
set -uo pipefail

ROOT=/home/youhan/ws/VPOCLIP_plus_full
PY=/home/youhan/.conda/envs/clipgcn/bin/python
CACHE=$ROOT/data/rl_candidate_rank_v6_new50_5/cache
TRACK=$ROOT/data/angle_object_trajectory_v1
POLICY_OUT=$ROOT/work_dir/multistep_angle_object_trajectory_policy_v1
PIPE_LOG=$ROOT/logs/dualrobot_pipeline_v1.log
EXP1_LOG=$ROOT/logs/exp01_dual_robot_orthogonal.log
EXP2_LOG=$ROOT/logs/exp02_synthetic_occlusion_diagnostic.log

mkdir -p "$ROOT/logs"
exec >> "$PIPE_LOG" 2>&1
date -u '+PIPELINE_START %Y-%m-%dT%H:%M:%SZ'

echo 'WAITING_FOR_TRAJECTORY_POLICY'
while true; do
    if [ -f "$POLICY_OUT/summary.json" ]; then
        echo 'TRAJECTORY_POLICY_READY'
        break
    fi
    if [ -f "$ROOT/logs/multistep_angle_object_trajectory_policy_v1.log" ] && \
       rg -q 'Traceback|RuntimeError|^Error:' "$ROOT/logs/multistep_angle_object_trajectory_policy_v1.log"; then
        echo 'TRAJECTORY_POLICY_FAILED'
        exit 20
    fi
    sleep 30
done

echo 'EXP01_START'
"$PY" -u -m rl.dual_robot_orthogonal_experiment \
    --cache-root "$CACHE" \
    --output-root "$ROOT/work_dir/exp01_dual_robot_orthogonal" \
    --seed 20260910 > "$EXP1_LOG" 2>&1
exp1_status=$?
echo "EXP01_EXIT $exp1_status"
if [ "$exp1_status" -ne 0 ] || [ ! -f "$ROOT/work_dir/exp01_dual_robot_orthogonal/summary.json" ]; then
    echo 'EXP01_FAILED'
    exit 21
fi

echo 'EXP02_START'
"$PY" -u -m rl.synthetic_occlusion_diagnostic \
    --cache-root "$CACHE" \
    --output-root "$ROOT/work_dir/exp02_synthetic_occlusion" \
    --batch-size 128 \
    --seed 20260910 > "$EXP2_LOG" 2>&1
exp2_status=$?
echo "EXP02_EXIT $exp2_status"
if [ "$exp2_status" -ne 0 ] || [ ! -f "$ROOT/work_dir/exp02_synthetic_occlusion/summary.json" ]; then
    echo 'EXP02_FAILED'
    exit 22
fi

date -u '+PIPELINE_COMPLETE %Y-%m-%dT%H:%M:%SZ'
