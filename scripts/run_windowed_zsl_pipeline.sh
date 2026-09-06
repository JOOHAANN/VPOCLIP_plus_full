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

X3D_CONFIG=${X3D_CONFIG:-"$X3D_ROOT/configs/x3d-s_clipgcn_tensor_cross_subject_70_10_20_182.yaml"}
X3D_CHECKPOINT=${X3D_CHECKPOINT:-"$X3D_ROOT/outputs/x3d-s_clipgcn_tensor_cs_70_10_20_182/model_007000.pth"}
CTRGCN_CONFIG=${CTRGCN_CONFIG:-"$CTRGCN_ROOT/config/etri-p1-p230/ctrgcn_joint_raw_13.yaml"}
CTRGCN_WEIGHTS=${CTRGCN_WEIGHTS:-"$CTRGCN_ROOT/work_dir/etri_p1_p230_13frames/xsub/ctrgcn_joint_raw/runs-50-2700.pt"}

WINDOW_SECONDS=${WINDOW_SECONDS:-2.0}
STRIDE_SECONDS=${STRIDE_SECONDS:-1.0}
FRAMES=${FRAMES:-13}
X3D_DEVICE=${X3D_DEVICE:-cuda:0}
POSE_DEVICE=${POSE_DEVICE:-cuda:0}
YOLO_DEVICE=${YOLO_DEVICE:-0}
RESIZE_DEVICE=${RESIZE_DEVICE:-cuda:0}
X3D_BATCH_SIZE=${X3D_BATCH_SIZE:-64}
POSE_BATCH_SIZE=${POSE_BATCH_SIZE:-128}
YOLO_BATCH_SIZE=${YOLO_BATCH_SIZE:-64}
NUM_WORKERS=${NUM_WORKERS:-8}
DECODE_WORKERS=${DECODE_WORKERS:-8}
UNSEEN_COUNT=${UNSEEN_COUNT:-5}
TARGET_SEEN_COUNT=${TARGET_SEEN_COUNT:-}
RUN_TRAIN=${RUN_TRAIN:-0}
TRAIN_CONFIG=${TRAIN_CONFIG:-"$CLIPGCN_ROOT/config_windowed_50_5.yaml"}

log_step() {
  printf '\n[%s] %s\n' "$(date '+%F %T')" "$*"
}

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Required file is missing: $1" >&2
    exit 1
  fi
}

log_step "Checking Python environment: $PYTHON"
"$PYTHON" - <<'PY'
import cv2
import numpy
import torch
print("deps ok:", "numpy", numpy.__version__, "torch", torch.__version__, "cv2", cv2.__version__)
PY

require_file "$X3D_CONFIG"
require_file "$X3D_CHECKPOINT"
require_file "$CTRGCN_CONFIG"
require_file "$CTRGCN_WEIGHTS"

if [[ -n "$TARGET_SEEN_COUNT" ]]; then
  "$PYTHON" - "$X3D_ROOT/data/clipgcn_tensor_cs_70_10_20/metadata.json" "$TARGET_SEEN_COUNT" "$UNSEEN_COUNT" <<'PY'
import json
import sys
from pathlib import Path

metadata = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
class_count = len(metadata["classes"])
seen_count = int(sys.argv[2])
unseen_count = int(sys.argv[3])
if seen_count + unseen_count > class_count:
    raise SystemExit(
        f"Requested {seen_count}/{unseen_count}, but this source has only {class_count} valid classes. "
        "A000 is excluded; with 5 unseen classes the valid split is 50/5."
    )
PY
fi

mkdir -p "$X3D_FEATURE_DIR" "$POSE_FEATURE_DIR" "$OBJECT_DIR" "$ZSL_ROOT"

log_step "Building windowed tensor/skeleton/joint-xy cache"
cd "$CLIPGCN_ROOT"
"$PYTHON" tools/build_windowed_clipgcn_dataset.py \
  --src-root "$CLIPGCN_ROOT/data" \
  --reference-metadata "$X3D_ROOT/data/clipgcn_tensor_cs_70_10_20/metadata.json" \
  --out-dir "$WINDOW_DIR" \
  --window-seconds "$WINDOW_SECONDS" \
  --stride-seconds "$STRIDE_SECONDS" \
  --frames "$FRAMES" \
  --resize-device "$RESIZE_DEVICE" \
  --gpu-batch-size "$YOLO_BATCH_SIZE" \
  --decode-workers "$DECODE_WORKERS" \
  --overwrite

for split in train val test; do
  log_step "Extracting X3D features for $split"
  cd "$X3D_ROOT"
  "$PYTHON" tools/extract_x3d_features.py \
    --config "$X3D_CONFIG" \
    --checkpoint "$X3D_CHECKPOINT" \
    --data-dir "$WINDOW_DIR" \
    --split "$split" \
    --output "$X3D_FEATURE_DIR/${split}_res5_model_007000.npy" \
    --layer s5 \
    --batch-size "$X3D_BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --device "$X3D_DEVICE" \
    --dtype float16 \
    --layout btchw \
    --spatial-size 6 \
    --num-channels 192 \
    --save-sidecars

  log_step "Extracting YOLO object detections for $split"
  cd "$YOLO_ROOT"
  "$PYTHON" extract_frame7_objects.py \
    --data-dir "$WINDOW_DIR" \
    --split "$split" \
    --output "$OBJECT_DIR/${split}_frame7_yolov5m_objects.npy" \
    --device "$YOLO_DEVICE" \
    --batch-size "$YOLO_BATCH_SIZE"

  log_step "Extracting CTR-GCN pose features for $split"
  cd "$CTRGCN_ROOT"
  "$PYTHON" tools/extract_ctrgcn_pose_feature_map.py \
    --config "$CTRGCN_CONFIG" \
    --weights "$CTRGCN_WEIGHTS" \
    --data "$WINDOW_DIR/windowed_skeleton_uniform13.npz" \
    --split "$split" \
    --hook-layer l4 \
    --feature-mode raw \
    --raw-layout NMCTV \
    --output "$POSE_FEATURE_DIR/${split}_l4_raw_NMCTV.npy" \
    --batch-size "$POSE_BATCH_SIZE" \
    --num-worker "$NUM_WORKERS" \
    --device "$POSE_DEVICE" \
    --window-size "$FRAMES" \
    --dtype float32 \
    --pin-memory
done

log_step "Converting object detections into RS maps"
cd "$CLIPGCN_ROOT"
"$PYTHON" tools/build_object_rs_maps.py \
  --paths \
    "$OBJECT_DIR/train_frame7_yolov5m_objects.npy" \
    "$OBJECT_DIR/val_frame7_yolov5m_objects.npy" \
    "$OBJECT_DIR/test_frame7_yolov5m_objects.npy" \
  --backup-dir "$OBJECT_DIR/object_dict_backups_before_rs" \
  --overwrite

log_step "Assembling aligned windowed trimodal data"
"$PYTHON" tools/assemble_windowed_trimodal_data.py \
  --window-dir "$WINDOW_DIR" \
  --x3d-output-dir "$X3D_FEATURE_DIR" \
  --pose-output-dir "$POSE_FEATURE_DIR" \
  --object-dir "$OBJECT_DIR" \
  --output-dir "$TRIMODAL_DIR" \
  --overwrite

zsl_args=(--input-dir "$TRIMODAL_DIR" --output-root "$ZSL_ROOT" --unseen-count "$UNSEEN_COUNT" --overwrite)
if [[ -n "$TARGET_SEEN_COUNT" ]]; then
  zsl_args+=(--seen-count "$TARGET_SEEN_COUNT" --split-name "${TARGET_SEEN_COUNT}_${UNSEEN_COUNT}")
fi

log_step "Building zero-shot split"
"$PYTHON" tools/build_windowed_zsl_splits.py "${zsl_args[@]}"

if [[ "$RUN_TRAIN" == "1" ]]; then
  log_step "Training CLIPGCN with $TRAIN_CONFIG"
  "$PYTHON" train.py --config "$TRAIN_CONFIG"
fi

log_step "Pipeline complete"
