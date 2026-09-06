#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 45_10|40_15" >&2
  exit 2
fi

split="$1"
case "$split" in
  45_10|40_15) ;;
  *) echo "unsupported split: $split" >&2; exit 2 ;;
esac

x3d_root=/workspace/X3D
vpo_root=/workspace/VPOCLIP_plus
tensor_dir="$x3d_root/data/clipgcn_tensor_cs_70_10_20_zsl_${split}_coco17"
object_dir="${tensor_dir}_objects"
x3d_run="$x3d_root/outputs/x3d-s_clipgcn_tensor_cs_70_10_20_zsl_${split}_coco17_182"
feature_dir="${x3d_run}_features"
trimodal_dir="$vpo_root/data/contrastive_cs_70_10_20_zsl_${split}_coco17_current"
pid_file="$x3d_root/logs/x3d_${split}_coco17_full12000.pid"

echo "[$(date -Is)] waiting for full 12,000-iteration X3D checkpoint: $x3d_run/model_final.pth"
while [[ ! -s "$x3d_run/model_final.pth" ]]; do
  pid=$(<"$pid_file")
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "X3D process $pid exited without model_final.pth" >&2
    exit 1
  fi
  sleep 60
done
sleep 10

mkdir -p "$feature_dir"
cd "$x3d_root"
for data_split in train val test test_seen; do
  echo "[$(date -Is)] extracting X3D S5 features: $split/$data_split"
  /root/miniconda3/envs/clipgcn/bin/python tools/extract_x3d_features.py \
    --config configs/x3d-s_clipgcn_tensor_cross_subject_70_10_20_zsl_50_5_182.yaml \
    --checkpoint "$x3d_run/model_final.pth" \
    --data-dir "$tensor_dir" \
    --split "$data_split" \
    --output "$feature_dir/${data_split}_res5_model_final.npy" \
    --layer s5 --spatial-size 6 --num-channels 192 \
    --batch-size 32 --num-workers 4 --device cuda --save-sidecars
done

cd "$vpo_root"
echo "[$(date -Is)] assembling X3D/object features with RTMPose sample alignment"
/root/miniconda3/envs/clipgcn/bin/python tools/assemble_coco17_cs_split.py \
  --tensor-dir "$tensor_dir" \
  --feature-dir "$feature_dir" \
  --object-dir "$object_dir" \
  --output-dir "$trimodal_dir"

echo "[$(date -Is)] extracting COCO-17 CTR-GCN features"
/root/miniconda3/envs/clipgcn/bin/python tools/build_coco17_pose_features_cs50_5.py \
  --split-dir "$trimodal_dir" --batch-size 256

echo "[$(date -Is)] starting full 300-epoch, eight-prompt VPOCLIP training"
/root/miniconda3/envs/clipgcn/bin/python train.py --config "config_final${split%%_*}_aug_coco17.yaml"

echo "[$(date -Is)] starting matched lambda-H=2 anchored entropy adaptation"
/root/miniconda3/envs/clipgcn/bin/python train.py --config "config_final${split%%_*}_aug_coco17_entropy_h2.yaml"
echo "[$(date -Is)] pipeline complete: $split"
