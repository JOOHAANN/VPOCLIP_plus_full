#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/home/youhan/ws/VPOCLIP_plus_full
PY=/home/youhan/.conda/envs/clipgcn/bin/python
LOG="$ROOT/logs/vpoclip_new50_5_recipe_20260909/train.log"
mkdir -p "$(dirname "$LOG")"
cd "$ROOT"
export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
exec > >(tee -a "$LOG") 2>&1
echo "START recipe new50_5 $(date -Is)"
echo "unseen zero-based: 0 2 26 34 50 (A001 A003 A027 A035 A051)"
echo "Stage A: 300 epochs, CE, 8 prompts, augmented, select seen-val best_model"
"$PY" -u train.py --config configs/vpoclip_new50_5_recipe_stage_a.yaml
test -s work_dir/allviews_lowview50_5_recipe_stage_a/best_model.pth
echo "Stage A complete; Stage B init=Stage A best_model.pth"
"$PY" -u train.py --config configs/vpoclip_new50_5_recipe_stage_b.yaml
test -s work_dir/allviews_lowview50_5_recipe_stage_b/last_model.pth
echo "COMPLETE recipe new50_5 $(date -Is)"
