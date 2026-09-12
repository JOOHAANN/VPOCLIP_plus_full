#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/youhan/ws/VPOCLIP_plus_full
PYTHON=/home/youhan/.conda/envs/clipgcn/bin/python
ROOT="$PROJECT/rl/single state analysis"
WORK="$PROJECT/work_dir/single_state_analysis/sc_nbv"
LOG="$PROJECT/logs/single_state_analysis_sc_pipeline.log"
mkdir -p "$WORK" "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

echo "SINGLE_STATE_SC_PIPELINE_START $(date -Is)"
for split in old new; do
  for version in $(seq 9 24); do
    OUT="$WORK/$split/v$version"
    echo "SC_TRAIN_START split=$split version=$version $(date -Is)"
    "$PYTHON" -u "$ROOT/train_sc_single_state.py" \
      --version "$version" --split "$split" --output-root "$OUT" \
      --epochs 50 --batch-size 8192 --batches-per-epoch 48 \
      --seeds 20260909 20260910 20260911
    echo "SC_TRAIN_COMPLETE split=$split version=$version $(date -Is)"
    "$PYTHON" -u "$ROOT/evaluate_sc_single_state.py" \
      --version "$version" --split "$split" \
      --policy-root "$OUT" --output-root "$WORK/$split/evaluation/v$version" \
      --eval-seeds 20260920 20260921 20260922 20260923 20260924 \
                   20260925 20260926 20260927 20260928 20260929 \
                   20260930 20260931 20260932 20260933 20260934 \
                   20260935 20260936 20260937 20260938 20260939 \
                   20260940 20260941 20260942 20260943 20260944 \
                   20260945 20260946 20260947 20260948 20260949
    echo "SC_EVALUATION_COMPLETE split=$split version=$version $(date -Is)"
  done
done
echo "SINGLE_STATE_SC_PIPELINE_COMPLETE $(date -Is)"
