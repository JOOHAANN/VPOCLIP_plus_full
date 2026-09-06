#!/bin/bash
# Multi-seed sweep for the 50/5 COCO-17 fusion heads.
#
# The three heads differ by up to 6 points on the 145-clip ZSL set, but no pair
# is significant and last-vs-best inside a single run already swings 5.5 points.
# This repeats each head over several seeds so the architecture effect can be
# told apart from run-to-run variance.
#
# Usage:  GPU=1 ./tools/run_seed_sweep.sh objconcat:101 objconcat:202 legacy:101
set -e
cd /workspace/VPOCLIP_plus

PY=/root/miniconda3/envs/clipgcn/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${GPU:-1}

for job in "$@"; do
  variant="${job%%:*}"
  seed="${job##*:}"
  tag="seed${seed}"
  out="work_dir/cs_50_5_coco17_${variant}/${tag}"
  if [ -f "${out}/best_model.pth" ]; then
    echo "[skip] ${variant} ${tag} already done"
    continue
  fi
  echo "==== ${variant} seed=${seed} on GPU ${CUDA_VISIBLE_DEVICES} ($(date +%H:%M)) ===="
  $PY train.py --config "config_cs_50_5_coco17_${variant}.yaml" \
    --seed "${seed}" --run-name "${tag}" \
    > "work_dir/seedsweep_${variant}_${tag}.log" 2>&1
  echo "[done] ${variant} ${tag} ($(date +%H:%M))"
done
echo "SWEEP_COMPLETE"
