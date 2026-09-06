#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/workspace
CLIPGCN_ROOT=${CLIPGCN_ROOT:-"$ROOT/CLIPGCN"}
X3D_ROOT=${X3D_ROOT:-"$ROOT/X3D"}
CTRGCN_ROOT=${CTRGCN_ROOT:-"$ROOT/CTR-GCN"}
YOLO_ROOT=${YOLO_ROOT:-"$ROOT/yolov5"}

PYTHON=${PYTHON:-/root/miniconda3/envs/clipgcn/bin/python}
WINDOW_DIR=${WINDOW_DIR:-"$CLIPGCN_ROOT/data/windowed_2s1s_13f"}
TRIMODAL_DIR=${TRIMODAL_DIR:-"$CLIPGCN_ROOT/data/contrastive_windowed_2s1s_13f"}
ZSL_ROOT=${ZSL_ROOT:-"$CLIPGCN_ROOT/data/contrastive_windowed_zsl_splits"}
X3D_FEATURE_DIR=${X3D_FEATURE_DIR:-"$X3D_ROOT/outputs/x3d-s_clipgcn_windowed_2s1s_13f"}
POSE_FEATURE_DIR=${POSE_FEATURE_DIR:-"$CTRGCN_ROOT/data/etri/windowed_2s1s_13f_features"}
OBJECT_DIR=${OBJECT_DIR:-"$X3D_ROOT/data/clipgcn_windowed_2s1s_13f"}

X3D_CONFIG=${X3D_CONFIG:-"$X3D_ROOT/configs/x3d-s_clipgcn_windowed_2s1s_13f.yaml"}
X3D_CHECKPOINT=${X3D_CHECKPOINT:-"$X3D_ROOT/outputs/x3d-s_clipgcn_windowed_2s1s_13f/model_final.pth"}
CTRGCN_CONFIG=${CTRGCN_CONFIG:-"$CTRGCN_ROOT/config/etri-p1-p230/ctrgcn_joint_windowed_2s1s_13f.yaml"}
CTRGCN_WEIGHTS=${CTRGCN_WEIGHTS:?Set CTRGCN_WEIGHTS to the trained windowed CTR-GCN .pt file}

X3D_DEVICE=${X3D_DEVICE:-cuda:1}
POSE_DEVICE=${POSE_DEVICE:-cuda:1}
YOLO_DEVICE=${YOLO_DEVICE:-1}
X3D_BATCH_SIZE=${X3D_BATCH_SIZE:-64}
POSE_BATCH_SIZE=${POSE_BATCH_SIZE:-128}
YOLO_BATCH_SIZE=${YOLO_BATCH_SIZE:-64}
NUM_WORKERS=${NUM_WORKERS:-8}
UNSEEN_COUNT=${UNSEEN_COUNT:-5}
RUN_YOLO=${RUN_YOLO:-0}
RUN_ZSL=${RUN_ZSL:-1}
ZSL_SEED=${ZSL_SEED:-20260615}
SKIP_EXISTING=${SKIP_EXISTING:-1}

log_step() {
  printf '\n[%s] %s\n' "$(date '+%F %T')" "$*"
}

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Required file is missing: $1" >&2
    exit 1
  fi
}

require_file "$WINDOW_DIR/metadata.json"
require_file "$WINDOW_DIR/windowed_skeleton_uniform13.npz"
require_file "$X3D_CONFIG"
require_file "$X3D_CHECKPOINT"
require_file "$CTRGCN_CONFIG"
require_file "$CTRGCN_WEIGHTS"

mkdir -p "$X3D_FEATURE_DIR" "$POSE_FEATURE_DIR" "$OBJECT_DIR" "$ZSL_ROOT"

object_rs_maps_are_ready() {
  "$PYTHON" - "$OBJECT_DIR" <<'PY'
import sys
from pathlib import Path

import numpy as np

object_dir = Path(sys.argv[1])
for split in ("train", "val", "test"):
    path = object_dir / f"{split}_frame7_yolov5m_objects.npy"
    metadata_path = object_dir / f"{split}_frame7_yolov5m_objects.rs_metadata.json"
    if not path.exists() or not metadata_path.exists():
        raise SystemExit(1)
    try:
        array = np.load(path, mmap_mode="r")
    except Exception:
        raise SystemExit(1)
    if array.ndim != 4 or array.shape[1:] != (50, 6, 6):
        raise SystemExit(1)
raise SystemExit(0)
PY
}

for split in train val test; do
  x3d_output="$X3D_FEATURE_DIR/${split}_res5_model_007000.npy"
  x3d_labels="${x3d_output%.npy}.labels.npy"
  x3d_valid="${x3d_output%.npy}.valid_indices.npy"
  if [[ "$SKIP_EXISTING" == "1" && -f "$x3d_output" && -f "$x3d_labels" && -f "$x3d_valid" ]]; then
    log_step "Skipping existing X3D features for $split"
  else
    log_step "Extracting X3D features for $split with $X3D_CHECKPOINT"
    cd "$X3D_ROOT"
    "$PYTHON" tools/extract_x3d_features.py \
      --config "$X3D_CONFIG" \
      --checkpoint "$X3D_CHECKPOINT" \
      --data-dir "$WINDOW_DIR" \
      --split "$split" \
      --output "$x3d_output" \
      --layer s5 \
      --batch-size "$X3D_BATCH_SIZE" \
      --num-workers "$NUM_WORKERS" \
      --device "$X3D_DEVICE" \
      --dtype float16 \
      --layout btchw \
      --spatial-size 6 \
      --num-channels 192 \
      --save-sidecars
  fi

  if [[ "$RUN_YOLO" == "1" ]]; then
    yolo_output="$OBJECT_DIR/${split}_frame7_yolov5m_objects.npy"
    yolo_metadata="${yolo_output%.npy}.metadata.json"
    if [[ "$SKIP_EXISTING" == "1" && -f "$yolo_output" && -f "$yolo_metadata" ]]; then
      log_step "Skipping existing YOLO object detections for $split"
    else
      log_step "Extracting YOLO object detections for $split"
      cd "$YOLO_ROOT"
      "$PYTHON" extract_frame7_objects.py \
        --data-dir "$WINDOW_DIR" \
        --split "$split" \
        --output "$yolo_output" \
        --device "$YOLO_DEVICE" \
        --batch-size "$YOLO_BATCH_SIZE"
    fi
  fi

  pose_output="$POSE_FEATURE_DIR/${split}_l4_raw_NMCTV.npy"
  pose_labels="${pose_output%.npy}_labels.npy"
  pose_metadata="${pose_output%.npy}_metadata.json"
  if [[ "$SKIP_EXISTING" == "1" && -f "$pose_output" && -f "$pose_labels" && -f "$pose_metadata" ]]; then
    log_step "Skipping existing CTR-GCN pose features for $split"
  else
    log_step "Extracting CTR-GCN pose features for $split with $CTRGCN_WEIGHTS"
    cd "$CTRGCN_ROOT"
    "$PYTHON" tools/extract_ctrgcn_pose_feature_map.py \
      --config "$CTRGCN_CONFIG" \
      --weights "$CTRGCN_WEIGHTS" \
      --data "$WINDOW_DIR/windowed_skeleton_uniform13.npz" \
      --split "$split" \
      --hook-layer l4 \
      --feature-mode raw \
      --raw-layout NMCTV \
      --output "$pose_output" \
      --batch-size "$POSE_BATCH_SIZE" \
      --num-worker "$NUM_WORKERS" \
      --device "$POSE_DEVICE" \
      --window-size 13 \
      --dtype float32 \
      --pin-memory
  fi
done

if [[ "$RUN_YOLO" == "1" ]]; then
  if [[ "$SKIP_EXISTING" == "1" ]] && object_rs_maps_are_ready; then
    log_step "Skipping object RS conversion; maps are already valid"
  else
    log_step "Converting refreshed object detections into RS maps"
    cd "$CLIPGCN_ROOT"
    "$PYTHON" tools/build_object_rs_maps.py \
      --paths \
        "$OBJECT_DIR/train_frame7_yolov5m_objects.npy" \
        "$OBJECT_DIR/val_frame7_yolov5m_objects.npy" \
        "$OBJECT_DIR/test_frame7_yolov5m_objects.npy" \
      --backup-dir "$OBJECT_DIR/object_dict_backups_before_rs" \
      --overwrite
  fi
else
  for split in train val test; do
    require_file "$OBJECT_DIR/${split}_frame7_yolov5m_objects.npy"
  done
fi

log_step "Assembling aligned windowed trimodal data"
cd "$CLIPGCN_ROOT"
"$PYTHON" tools/assemble_windowed_trimodal_data.py \
  --window-dir "$WINDOW_DIR" \
  --x3d-output-dir "$X3D_FEATURE_DIR" \
  --pose-output-dir "$POSE_FEATURE_DIR" \
  --object-dir "$OBJECT_DIR" \
  --output-dir "$TRIMODAL_DIR" \
  --overwrite

if [[ "$RUN_ZSL" == "1" ]]; then
  log_step "Building 50/5 zero-shot split"
  "$PYTHON" tools/build_windowed_zsl_splits.py \
    --input-dir "$TRIMODAL_DIR" \
    --output-root "$ZSL_ROOT" \
    --seed "$ZSL_SEED" \
    --unseen-count "$UNSEEN_COUNT" \
    --overwrite
fi

log_step "Feature refresh complete"
