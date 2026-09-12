#!/usr/bin/env bash
# Group1 (A015/A016/A026/A040/A047) pseudo-unseen VPOCLIP + skeleton unknown gate.
#
# Backbones/caches are reused read-only.  The two pseudo probes are run one at
# a time on the single GPU, then a full 50-class model is retrained at the
# epoch numbers selected from their pseudo curves.  Periodic snapshots are
# disabled in every config, so each run keeps only best/last/history.
set -euo pipefail

ROOT=/home/youhan/ws/VPOCLIP_plus_full
PY=/home/youhan/.conda/envs/clipgcn/bin/python
LOGDIR="$ROOT/logs/g1_pseudo_unknown_20260912"
mkdir -p "$LOGDIR"
cd "$ROOT"

# Worker processes only decode/mmap cached features; limiting their BLAS
# threads prevents a 16-worker loader from oversubscribing the 32-core host.
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

run_train() {
  local tag="$1"; shift
  local cfg="$1"; shift
  local done="$LOGDIR/${tag}.done"
  local log="$LOGDIR/${tag}.log"
  if [[ -f "$done" ]]; then
    echo "[skip] $tag already complete" | tee -a "$LOGDIR/pipeline.log"
    return
  fi
  echo "[$(date -Is)] START $tag: $PY train.py --config $cfg $*" | tee -a "$LOGDIR/pipeline.log"
  "$PY" train.py --config "$cfg" "$@" >"$log" 2>&1
  date -Is >"$done"
  echo "[$(date -Is)] DONE $tag" | tee -a "$LOGDIR/pipeline.log"
}

echo "[$(date -Is)] group1 pseudo protocol start" | tee "$LOGDIR/pipeline.log"
echo "true unseen held out from selection: 14,15,25,39,46" | tee -a "$LOGDIR/pipeline.log"
echo "pseudo groups: p3=[8,27,28,35,50], p4=[3,13,30,41,52]" | tee -a "$LOGDIR/pipeline.log"

# Stage A: augmented CE + eight-prompt text prototype ensemble.
run_train 01_p3_stage_a "$ROOT/config_g1_pseudo_p3_stage_a.yaml"
run_train 02_p4_stage_a "$ROOT/config_g1_pseudo_p4_stage_a.yaml"

# Stage B probes start from their own pseudo-selected Stage A checkpoints.
run_train 03_p3_stage_b "$ROOT/config_g1_pseudo_p3_stage_b.yaml"
run_train 04_p4_stage_b "$ROOT/config_g1_pseudo_p4_stage_b.yaml"

"$PY" tools/select_group1_pseudo_epochs.py \
  --stage-a \
    "$ROOT/work_dir/g1_pseudo_p3_stage_a/history.json" \
    "$ROOT/work_dir/g1_pseudo_p4_stage_a/history.json" \
  --stage-b \
    "$ROOT/work_dir/g1_pseudo_p3_stage_b/history.json" \
    "$ROOT/work_dir/g1_pseudo_p4_stage_b/history.json" \
  --data-dir "$ROOT/data/new50_5_group1_zsl" \
  --output "$LOGDIR/selection.json" \
  --smooth-a 9 --smooth-b 3 | tee "$LOGDIR/selection.log"

STAGE_A_EPOCH=$($PY -c 'import json,sys; print(json.load(open(sys.argv[1]))["recommended_stage_a_epochs"])' "$LOGDIR/selection.json")
STAGE_B_EPOCH=$($PY -c 'import json,sys; print(json.load(open(sys.argv[1]))["recommended_stage_b_epochs"])' "$LOGDIR/selection.json")
echo "selected stopping points: stage_a=${STAGE_A_EPOCH} stage_b=${STAGE_B_EPOCH}" | tee -a "$LOGDIR/pipeline.log"

# Final Stage A trains on all 50 seen classes (no pseudo hold-out).  Its exact
# duration is transferred from the held-out pseudo probes.
run_train 05_full_stage_a "$ROOT/config_g1_full_stage_a.yaml" --epochs "$STAGE_A_EPOCH"

# Final Stage B uses the same conservative entropy/anchor recipe as the best
# historical checkpoint, but its duration comes from pseudo Stage-B curves.
run_train 06_full_stage_b "$ROOT/config_g1_full_stage_b.yaml" --epochs "$STAGE_B_EPOCH"

FINAL_CKPT="$ROOT/work_dir/g1_pseudo_selected_stage_b/last_model.pth"
if [[ ! -f "$LOGDIR/07_vpo_eval.done" ]]; then
  echo "[$(date -Is)] evaluating final VPO checkpoint (report only; not used for selection)" | tee -a "$LOGDIR/pipeline.log"
  "$PY" test.py --config "$ROOT/config_g1_full_stage_b.yaml" \
    --checkpoint "$FINAL_CKPT" \
    --split-dir "$ROOT/data/new50_5_group1_zsl" \
    --class-split-dir "$ROOT/data/new50_5_group1_zsl" \
    --prefix trimodal_test --candidate-scope unseen --sample-scope all \
    --batch-size 2048 --num-workers 8 \
    --output "$ROOT/work_dir/g1_pseudo_selected_stage_b/unseen_test.json" >"$LOGDIR/07_vpo_eval_unseen.log" 2>&1
  "$PY" test.py --config "$ROOT/config_g1_full_stage_b.yaml" \
    --checkpoint "$FINAL_CKPT" \
    --split-dir "$ROOT/data/new50_5_group1_zsl" \
    --class-split-dir "$ROOT/data/new50_5_group1_zsl" \
    --prefix trimodal_test_seen --candidate-scope seen --sample-scope all \
    --batch-size 2048 --num-workers 8 \
    --output "$ROOT/work_dir/g1_pseudo_selected_stage_b/seen_test.json" >"$LOGDIR/07_vpo_eval_seen.log" 2>&1
  date -Is >"$LOGDIR/07_vpo_eval.done"
fi

# Gate calibration/training is deliberately last: the VPO checkpoint is now
# fixed, and the true unseen labels remain absent from all gate fitting steps.
if [[ ! -f "$LOGDIR/08_unknown_gate.done" ]]; then
  echo "[$(date -Is)] training skeleton unknown gate" | tee -a "$LOGDIR/pipeline.log"
  "$PY" tools/train_skeleton_unknown_gate.py \
    --config "$ROOT/config_g1_full_stage_b.yaml" \
    --checkpoint "$FINAL_CKPT" \
    --data-dir "$ROOT/data/new50_5_group1_zsl" \
    --output-dir "$ROOT/work_dir/g1_skeleton_unknown_gate" \
    --batch-size 2048 --workers 8 --gate-batch-size 8192 --epochs 100 \
    >"$LOGDIR/08_unknown_gate.log" 2>&1
  date -Is >"$LOGDIR/08_unknown_gate.done"
fi

echo "[$(date -Is)] GROUP1_PSEUDO_UNKNOWN_COMPLETE" | tee -a "$LOGDIR/pipeline.log"
