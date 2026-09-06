#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/workspace
CLIPGCN_ROOT=${CLIPGCN_ROOT:-"$ROOT/CLIPGCN"}
X3D_ROOT=${X3D_ROOT:-"$ROOT/X3D"}
CTRGCN_ROOT=${CTRGCN_ROOT:-"$ROOT/CTR-GCN"}
PYTHON=${PYTHON:-/root/miniconda3/envs/clipgcn/bin/python}

XLSX=${XLSX:-"$CLIPGCN_ROOT/Action details collection.xlsx"}
REFERENCE_METADATA=${REFERENCE_METADATA:-"$X3D_ROOT/data/clipgcn_tensor_cs_70_10_20/metadata.json"}
FULL_X3D_DIR=${FULL_X3D_DIR:-"$X3D_ROOT/data/clipgcn_tensor_cs_70_10_20"}
SHORT_WINDOW_DIR=${SHORT_WINDOW_DIR:-"$CLIPGCN_ROOT/data/windowed_2s1s_13f"}
SPECIAL_WINDOW_DIR=${SPECIAL_WINDOW_DIR:-"$CLIPGCN_ROOT/data/windowed_start_anchor_2s1s_13f"}
X3D_OUT_DIR=${X3D_OUT_DIR:-"$X3D_ROOT/data/clipgcn_action_detail_raw_13f"}
CTRGCN_OUT=${CTRGCN_OUT:-"$CTRGCN_ROOT/data/etri/ETRI_P1_P230_action_detail_raw_13f.npz"}

RESIZE_DEVICE=${RESIZE_DEVICE:-cuda:1}
DECODE_WORKERS=${DECODE_WORKERS:-8}
GPU_BATCH_SIZE=${GPU_BATCH_SIZE:-32}
VIDEO_TIMEOUT=${VIDEO_TIMEOUT:-120}

log_step() {
  printf '\n[%s] %s\n' "$(date '+%F %T')" "$*"
}

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Required file is missing: $1" >&2
    exit 1
  fi
}

require_file "$XLSX"
require_file "$REFERENCE_METADATA"
require_file "$FULL_X3D_DIR/metadata.json"
require_file "$SHORT_WINDOW_DIR/metadata.json"
require_file "$SHORT_WINDOW_DIR/windowed_skeleton_uniform13.npz"

cd "$CLIPGCN_ROOT"

if [[ "${REBUILD_SPECIAL:-0}" == "1" || ! -f "$SPECIAL_WINDOW_DIR/metadata.json" ]]; then
  log_step "Building start-anchor raw windows for A35/A38/A46/A47/A48"
  "$PYTHON" tools/build_windowed_clipgcn_dataset.py \
    --src-root "$CLIPGCN_ROOT/data" \
    --reference-metadata "$REFERENCE_METADATA" \
    --out-dir "$SPECIAL_WINDOW_DIR" \
    --window-seconds 2 \
    --stride-seconds 1 \
    --frames 13 \
    --resize-device "$RESIZE_DEVICE" \
    --gpu-batch-size "$GPU_BATCH_SIZE" \
    --decode-workers "$DECODE_WORKERS" \
    --video-timeout "$VIDEO_TIMEOUT" \
    --include-action A035 \
    --include-action A038 \
    --include-action A046 \
    --include-action A047 \
    --include-action A048 \
    --anchor-first-frame-actions A035 \
    --anchor-first-frame-actions A038 \
    --anchor-first-frame-actions A046 \
    --anchor-first-frame-actions A047 \
    --anchor-first-frame-actions A048 \
    --overwrite
else
  log_step "Start-anchor raw window cache already exists; skipping"
fi

log_step "Building X3D/CTR-GCN action-detail raw training data"
"$PYTHON" tools/build_action_detail_raw_training_data.py \
  --xlsx "$XLSX" \
  --src-root "$CLIPGCN_ROOT/data" \
  --full-x3d-dir "$FULL_X3D_DIR" \
  --short-window-dir "$SHORT_WINDOW_DIR" \
  --special-window-dir "$SPECIAL_WINDOW_DIR" \
  --x3d-out-dir "$X3D_OUT_DIR" \
  --ctrgcn-out "$CTRGCN_OUT" \
  --overwrite

log_step "Raw training datasets are ready"
echo "X3D data: $X3D_OUT_DIR"
echo "CTR-GCN data: $CTRGCN_OUT"
