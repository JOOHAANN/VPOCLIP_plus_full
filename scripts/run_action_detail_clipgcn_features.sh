#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/workspace
CLIPGCN_ROOT=${CLIPGCN_ROOT:-"$ROOT/CLIPGCN"}
X3D_ROOT=${X3D_ROOT:-"$ROOT/X3D"}
CTRGCN_ROOT=${CTRGCN_ROOT:-"$ROOT/CTR-GCN"}
YOLO_ROOT=${YOLO_ROOT:-"$ROOT/yolov5"}

PYTHON=${PYTHON:-/root/miniconda3/envs/clipgcn/bin/python}
ACTION_DIR=${ACTION_DIR:-"$X3D_ROOT/data/clipgcn_action_detail_raw_13f_zsl_50_5"}
CTRGCN_DATA=${CTRGCN_DATA:-"$CTRGCN_ROOT/data/etri/ETRI_P1_P230_action_detail_raw_13f_zsl_50_5.npz"}
TRIMODAL_DIR=${TRIMODAL_DIR:-"$CLIPGCN_ROOT/data/contrastive_action_detail_raw_13f_zsl_50_5"}
X3D_FEATURE_DIR=${X3D_FEATURE_DIR:-"$X3D_ROOT/outputs/x3d-s_clipgcn_action_detail_raw_13f_zsl_50_5_features"}
POSE_FEATURE_DIR=${POSE_FEATURE_DIR:-"$CTRGCN_ROOT/data/etri/action_detail_raw_13f_zsl_50_5_features"}
OBJECT_DIR=${OBJECT_DIR:-"$X3D_ROOT/data/clipgcn_action_detail_raw_13f_zsl_50_5_objects"}

X3D_CONFIG=${X3D_CONFIG:-"$X3D_ROOT/configs/x3d-s_clipgcn_action_detail_raw_13f_zsl_50_5.yaml"}
X3D_CHECKPOINT=${X3D_CHECKPOINT:?Set X3D_CHECKPOINT to the trained X3D .pth file}
CTRGCN_CONFIG=${CTRGCN_CONFIG:-"$CTRGCN_ROOT/config/etri-p1-p230/ctrgcn_joint_action_detail_raw_13f_zsl_50_5_pretrain_aug.yaml"}
CTRGCN_WEIGHTS=${CTRGCN_WEIGHTS:?Set CTRGCN_WEIGHTS to the trained CTR-GCN .pt file}
YOLO_WEIGHTS=${YOLO_WEIGHTS:-"$YOLO_ROOT/weights/yolov5m.pt"}

SPLITS=${SPLITS:-"train val test test_seen"}
X3D_DEVICE=${X3D_DEVICE:-cuda:1}
POSE_DEVICE=${POSE_DEVICE:-cuda:1}
YOLO_DEVICE=${YOLO_DEVICE:-1}
X3D_BATCH_SIZE=${X3D_BATCH_SIZE:-64}
POSE_BATCH_SIZE=${POSE_BATCH_SIZE:-128}
YOLO_BATCH_SIZE=${YOLO_BATCH_SIZE:-64}
NUM_WORKERS=${NUM_WORKERS:-8}
SKIP_EXISTING=${SKIP_EXISTING:-1}
RUN_YOLO=${RUN_YOLO:-1}

log_step() {
  printf '\n[%s] %s\n' "$(date '+%F %T')" "$*"
}

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Required file is missing: $1" >&2
    exit 1
  fi
}

object_rs_map_ready() {
  local path=$1
  "$PYTHON" - "$path" <<'PY'
import sys
import numpy as np
path = sys.argv[1]
array = np.load(path, mmap_mode="r")
raise SystemExit(0 if array.ndim == 4 and array.shape[1:] == (50, 6, 6) else 1)
PY
}

require_file "$ACTION_DIR/metadata.json"
require_file "$CTRGCN_DATA"
require_file "$X3D_CONFIG"
require_file "$X3D_CHECKPOINT"
require_file "$CTRGCN_CONFIG"
require_file "$CTRGCN_WEIGHTS"
require_file "$YOLO_WEIGHTS"

mkdir -p "$X3D_FEATURE_DIR" "$POSE_FEATURE_DIR" "$OBJECT_DIR" "$TRIMODAL_DIR"

log_step "Preparing joint-xy arrays for CLIPGCN alignment"
cd "$CLIPGCN_ROOT"
"$PYTHON" tools/build_action_detail_joint_xy.py \
  --target-dir "$ACTION_DIR" \
  --splits $SPLITS

for split in $SPLITS; do
  x3d_output="$X3D_FEATURE_DIR/${split}_res5_model_007000.npy"
  x3d_labels="${x3d_output%.npy}.labels.npy"
  x3d_valid="${x3d_output%.npy}.valid_indices.npy"
  if [[ "$SKIP_EXISTING" == "1" && -f "$x3d_output" && -f "$x3d_labels" && -f "$x3d_valid" ]]; then
    log_step "Skipping existing X3D features for $split"
  else
    log_step "Extracting X3D features for $split"
    cd "$X3D_ROOT"
    "$PYTHON" tools/extract_x3d_features.py \
      --config "$X3D_CONFIG" \
      --checkpoint "$X3D_CHECKPOINT" \
      --data-dir "$ACTION_DIR" \
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
        --data-dir "$ACTION_DIR" \
        --weights "$YOLO_WEIGHTS" \
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
    log_step "Extracting CTR-GCN pose features for $split"
    cd "$CTRGCN_ROOT"
    "$PYTHON" tools/extract_ctrgcn_pose_feature_map.py \
      --config "$CTRGCN_CONFIG" \
      --weights "$CTRGCN_WEIGHTS" \
      --data "$CTRGCN_DATA" \
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

object_paths=()
for split in $SPLITS; do
  object_paths+=("$OBJECT_DIR/${split}_frame7_yolov5m_objects.npy")
done

if [[ "$RUN_YOLO" == "1" ]]; then
  all_rs_ready=1
  for path in "${object_paths[@]}"; do
    if [[ ! -f "$path" ]] || ! object_rs_map_ready "$path"; then
      all_rs_ready=0
      break
    fi
  done
  if [[ "$SKIP_EXISTING" == "1" && "$all_rs_ready" == "1" ]]; then
    log_step "Skipping object RS conversion; maps are already valid"
  else
    log_step "Converting YOLO detections into RS maps"
    cd "$CLIPGCN_ROOT"
    "$PYTHON" tools/build_object_rs_maps.py \
      --paths "${object_paths[@]}" \
      --backup-dir "$OBJECT_DIR/object_dict_backups_before_rs" \
      --overwrite
  fi
else
  for path in "${object_paths[@]}"; do
    require_file "$path"
  done
fi

log_step "Assembling aligned action-detail trimodal data for CLIPGCN"
cd "$CLIPGCN_ROOT"
"$PYTHON" tools/assemble_windowed_trimodal_data.py \
  --window-dir "$ACTION_DIR" \
  --x3d-output-dir "$X3D_FEATURE_DIR" \
  --pose-output-dir "$POSE_FEATURE_DIR" \
  --object-dir "$OBJECT_DIR" \
  --output-dir "$TRIMODAL_DIR" \
  --splits $SPLITS \
  --overwrite

log_step "Action-detail CLIPGCN feature preparation complete: $TRIMODAL_DIR"
