#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/home/youhan/ws
VPO_ROOT="$ROOT/VPOCLIP_plus_full"
X3D_ROOT="$ROOT/X3D_full"
YOLO_ROOT="$ROOT/yolov5_full"
CTR_ROOT="$ROOT/CTR-GCN_17_full"
PYTHON=/home/youhan/.conda/envs/clipgcn/bin/python

CACHE="$X3D_ROOT/data/etri_rgb_c001_cs_45_25_10_20"
X3D_CONFIG="$X3D_ROOT/configs/x3d-s_etri_rgb_cs_45_25_10_20_zsl50_5_182.yaml"
X3D_WEIGHT="$X3D_ROOT/outputs/x3d-s_clipgcn_tensor_cs_45_25_10_20_zsl50_5_182/model_007000.pth"
YOLO_WEIGHT="$YOLO_ROOT/best.pt"
CTR_CONFIG="$CTR_ROOT/config/etri-coco17/ctrgcn_joint_coco17_13.yaml"
CTR_WEIGHT="$CTR_ROOT/work_dir/etri_coco17/ctrgcn_joint_coco17_13_cs45_zsl50_5/runs-42-504.pt"
POSE_NPZ="$CTR_ROOT/data/etri_coco17/ETRI_CS_45_25_10_20_rtmpose_coco17_13.npz"
SUBJECT_MANIFEST="$ROOT/ETRI_subject_split_45_25_10_20.json"

RAW="$VPO_ROOT/data/current_45_25_10_20_features"
X3D_OUT="$RAW/x3d"
OBJECT_OUT="$RAW/object"
POSE_OUT="$RAW/pose"
ZSL_OUT="$VPO_ROOT/data/contrastive_cs_45_25_10_20_zsl_50_5"
CLOSED_OUT="$VPO_ROOT/data/contrastive_cs_45_25_10_20_55_0"
STATE="$VPO_ROOT/logs/vpoclip_45_25_10_20_20260906/state"
LOG="$VPO_ROOT/logs/vpoclip_45_25_10_20_20260906/pipeline.log"

mkdir -p "$X3D_OUT" "$OBJECT_OUT" "$POSE_OUT" "$ZSL_OUT" "$CLOSED_OUT" "$STATE" "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
export PATH=/home/youhan/.conda/envs/clipgcn/bin:$PATH
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export PYTHONUNBUFFERED=1

step() { printf '\n[%s] %s\n' "$(date -Is)" "$*"; }
need() { [[ -f "$1" ]] || { echo "Missing required file: $1"; exit 1; }; }
for path in "$X3D_CONFIG" "$X3D_WEIGHT" "$YOLO_WEIGHT" "$CTR_CONFIG" "$CTR_WEIGHT" "$POSE_NPZ" "$SUBJECT_MANIFEST"; do need "$path"; done

step "Hardware and environment"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
"$PYTHON" -c 'import torch, numpy, pandas, clip; print("torch", torch.__version__, "CUDA", torch.cuda.is_available())'

for split in train val test; do
  if [[ ! -f "$STATE/x3d_$split.done" ]]; then
    step "X3D model_007000 inference: $split"
    cd "$X3D_ROOT"
    "$PYTHON" tools/extract_x3d_features.py \
      --config "$X3D_CONFIG" --checkpoint "$X3D_WEIGHT" --data-dir "$CACHE" \
      --split "$split" --output "$X3D_OUT/${split}_video.npy" --layer s5 \
      --batch-size 512 --num-workers 12 --device cuda:0 --dtype float16 \
      --layout btchw --spatial-size 6 --save-sidecars
    touch "$STATE/x3d_$split.done"
  fi
done

for split in train val test; do
  if [[ ! -f "$STATE/yolo_$split.done" ]]; then
    step "YOLOv5 best.pt inference: $split"
    cd "$YOLO_ROOT"
    "$PYTHON" extract_frame7_objects.py --data-dir "$CACHE" --weights "$YOLO_WEIGHT" \
      --split "$split" --output "$OBJECT_OUT/${split}_object.npy" \
      --device 0 --batch-size 256 --imgsz 640
    touch "$STATE/yolo_$split.done"
  fi
done

if [[ ! -f "$STATE/object_rs.done" ]]; then
  step "Convert YOLO detections to VPOCLIP 50x6x6 RS maps"
  cd "$VPO_ROOT"
  "$PYTHON" tools/build_object_rs_maps.py --paths \
    "$OBJECT_OUT/train_object.npy" "$OBJECT_OUT/val_object.npy" "$OBJECT_OUT/test_object.npy" \
    --backup-dir "$OBJECT_OUT/detection_backups" --overwrite
  touch "$STATE/object_rs.done"
fi

for split in train val test; do
  if [[ ! -f "$STATE/pose_$split.done" ]]; then
    step "CTR-GCN runs-42-504 inference: $split"
    cd "$VPO_ROOT"
    "$PYTHON" tools/extract_current_ctrgcn_features.py \
      --ctrgcn-root "$CTR_ROOT" --config "$CTR_CONFIG" --weights "$CTR_WEIGHT" \
      --npz "$POSE_NPZ" --split "$split" --output-dir "$POSE_OUT" \
      --batch-size 1024 --window-size 64 --device cuda:0
    touch "$STATE/pose_$split.done"
  fi
done

if [[ ! -f "$STATE/assembled.done" ]]; then
  step "Assemble subject-disjoint strict 50/5 ZSL and 55/0 datasets"
  cd "$VPO_ROOT"
  "$PYTHON" tools/assemble_current_45_25_10_20.py \
    --cache-dir "$CACHE" --x3d-dir "$X3D_OUT" --pose-dir "$POSE_OUT" \
    --object-dir "$OBJECT_OUT" --zsl-dir "$ZSL_OUT" --closed-dir "$CLOSED_OUT" \
    --manifest "$SUBJECT_MANIFEST"
  touch "$STATE/assembled.done"
fi

cd "$VPO_ROOT"
if [[ ! -f "$STATE/final_aug.done" ]]; then
  step "Train ZSL stage 1: final_aug"
  "$PYTHON" train.py --config config_final_aug.yaml
  touch "$STATE/final_aug.done"
fi

ZSL_LAST=$("$PYTHON" -c 'import json; print(json.load(open("work_dir/final_aug/latest_run.json"))["last_model"])')
need "$ZSL_LAST"
if [[ ! -f "$STATE/final_aug_entropy_single_h2.done" ]]; then
  step "Train ZSL stage 2: final_aug_entropy_single_h2 from current last_model.pth"
  "$PYTHON" train.py --config config_final_aug_entropy_single_h2.yaml --init-checkpoint "$ZSL_LAST"
  touch "$STATE/final_aug_entropy_single_h2.done"
fi

if [[ ! -f "$STATE/final55_aug.done" ]]; then
  step "Train closed-set stage 1: final55_aug"
  "$PYTHON" train.py --config config_final55_aug.yaml
  touch "$STATE/final55_aug.done"
fi

CLOSED_LAST=$("$PYTHON" -c 'import json; print(json.load(open("work_dir/final55_aug/latest_run.json"))["last_model"])')
need "$CLOSED_LAST"
if [[ ! -f "$STATE/final55_aug_entropy_h5_full300_score.done" ]]; then
  step "Train closed-set stage 2: final55_aug_entropy_h5_full300_score from current last_model.pth"
  "$PYTHON" train.py --config config_final55_aug_entropy_h5_full300_score.yaml --init-checkpoint "$CLOSED_LAST"
  touch "$STATE/final55_aug_entropy_h5_full300_score.done"
fi

step "Pipeline complete"
