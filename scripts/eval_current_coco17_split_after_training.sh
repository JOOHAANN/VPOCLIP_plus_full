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

root=/workspace/VPOCLIP_plus
stem="${split%%_*}"
pipeline_pid_file="$root/work_dir/pipeline_${split}_coco17.pid"
pipeline_log="$root/work_dir/pipeline_${split}_coco17.log"
config="config_final${stem}_aug_coco17_entropy_h2.yaml"
split_dir="data/contrastive_cs_70_10_20_zsl_${split}_coco17_current"
run_dir="work_dir/final${stem}_aug_coco17_entropy_h2/lambdaH2_anchor100"
checkpoint="$run_dir/last_model.pth"

pipeline_pid=$(<"$pipeline_pid_file")
echo "[$(date -Is)] waiting for training pipeline PID $pipeline_pid"
while kill -0 "$pipeline_pid" 2>/dev/null; do sleep 60; done
if ! grep -q "pipeline complete: $split" "$pipeline_log"; then
  echo "training pipeline exited without a completion marker" >&2
  exit 1
fi
test -s "$checkpoint"

cd "$root"
echo "[$(date -Is)] evaluating standard ZSL"
/root/miniconda3/envs/clipgcn/bin/python test.py \
  --config "$config" --checkpoint "$checkpoint" \
  --split-dir "$split_dir" --class-split-dir "$split_dir" \
  --prefix trimodal_test --protocol zsl \
  --output "$run_dir/zsl_test_results.json"

echo "[$(date -Is)] evaluating seen clips against all 55 prototypes"
/root/miniconda3/envs/clipgcn/bin/python test.py \
  --config "$config" --checkpoint "$checkpoint" \
  --split-dir "$split_dir" --class-split-dir "$split_dir" \
  --prefix trimodal_test_seen --protocol gzsl_seen \
  --output "$run_dir/seen_all55_test_results.json"

echo "[$(date -Is)] building class centroids"
/root/miniconda3/envs/clipgcn/bin/python tools/build_class_centroids.py \
  --config "$config" --checkpoint "$checkpoint" \
  --split-dir "$split_dir" --output "$run_dir/class_centroids.npz"

echo "[$(date -Is)] evaluating honest validation-calibrated GZSL gate"
/root/miniconda3/envs/clipgcn/bin/python tools/eval_gzsl_gate.py \
  --config "$config" --checkpoint "$checkpoint" \
  --centroids "$run_dir/class_centroids.npz" \
  --split-dir "$split_dir" --class-split-dir "$split_dir" \
  --output "$run_dir/gate_tau_calibration.json"
echo "[$(date -Is)] evaluation complete: $split"
