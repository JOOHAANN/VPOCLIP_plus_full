#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/youhan/ws/VPOCLIP_plus_full
PYTHON=/home/youhan/.conda/envs/clipgcn/bin/python
ROOT="$PROJECT/rl/single state analysis"
WORK="$PROJECT/work_dir/single_state_analysis/trajectory"
LOG="$PROJECT/logs/single_state_analysis_trajectory_pipeline.log"

mkdir -p "$WORK" "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

echo "SINGLE_STATE_TRAJECTORY_PIPELINE_START $(date -Is)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}"

for split in old new; do
  if [[ "$split" == old ]]; then
    CACHE="$PROJECT/data/rl_candidate_rank_v6_new50_5/cache"
    TRACK="$PROJECT/data/angle_object_trajectory_v1"
  else
    # cache_compat repairs the self-reachability bookkeeping required by the
    # trajectory policy; the logits/pose/geometry are identical to cache/.
    CACHE="$PROJECT/data/trajectory_latest_vpoclip_new50_5/cache_compat"
    TRACK="$PROJECT/data/trajectory_latest_vpoclip_new50_5/object_tracks_full"
  fi
  for variant in v1 v4 v5 v6 object_only geometry_only; do
    OUT="$WORK/$split/$variant"
    echo "TRAJECTORY_TRAIN_START split=$split variant=$variant $(date -Is)"
    "$PYTHON" -u "$ROOT/train_trajectory_single_state.py" \
      --variant "$variant" --split "$split" \
      --cache-root "$CACHE" --track-root "$TRACK" --output-root "$OUT" \
      --epochs 40 --batch-size 16384 --updates-per-epoch 64 \
      --learning-rate 1e-4 --weight-decay 0.04 \
      --seeds 20260909 20260910 20260911
    echo "TRAJECTORY_TRAIN_COMPLETE split=$split variant=$variant $(date -Is)"
    "$PYTHON" -u "$ROOT/evaluate_trajectory_single_state.py" \
      --variant "$variant" --split "$split" \
      --cache-root "$CACHE" --track-root "$TRACK" --policy-root "$OUT" \
      --output-root "$WORK/$split/evaluation/$variant" \
      --eval-seeds 20260920 20260921 20260922 20260923 20260924 \
                   20260925 20260926 20260927 20260928 20260929 \
                   20260930 20260931 20260932 20260933 20260934 \
                   20260935 20260936 20260937 20260938 20260939 \
                   20260940 20260941 20260942 20260943 20260944 \
                   20260945 20260946 20260947 20260948 20260949
    echo "TRAJECTORY_EVALUATION_COMPLETE split=$split variant=$variant $(date -Is)"
  done
done

echo "SINGLE_STATE_TRAJECTORY_PIPELINE_COMPLETE $(date -Is)"
