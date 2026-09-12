#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT=/home/youhan/ws/VPOCLIP_plus_full
PYTHON=/home/youhan/.conda/envs/clipgcn/bin/python
CONFIG=$PROJECT/rl/handoff_configs/trajectory_g1_pseudo_selected_stage_b_raw.yaml
CACHE=$PROJECT/data/rl_raw_new50_5_group1_g1_pseudo_selected_stage_b/cache
TRACK=$PROJECT/data/trajectory_g1_pseudo_selected_stage_b/object_tracks_full
POLICY_ROOT=$PROJECT/work_dir/trajectory_g1_pseudo_selected_stage_b_2d
EVAL_ROOT=$PROJECT/work_dir/trajectory_g1_pseudo_selected_stage_b_2d/evaluation_test_30seeds
LOG_ROOT=$PROJECT/logs/trajectory_g1_pseudo_selected_stage_b_2d

cd "$PROJECT"
mkdir -p "$LOG_ROOT" "$POLICY_ROOT" "$EVAL_ROOT" "$TRACK"
exec > >(tee -a "$LOG_ROOT/pipeline.log") 2>&1

echo "PIPELINE_START $(date -Is)"
echo "MODE original_2d_trajectory_fusion"
echo "VPOCLIP_CHECKPOINT $PROJECT/work_dir/g1_pseudo_selected_stage_b/last_model.pth"
echo "CACHE_ROOT $CACHE"

for split in dqn_train val test; do
  if [[ -f "$CACHE/$split/metadata.json" && -f "$CACHE/$split/logits.npy" ]]; then
    echo "CACHE_EXISTS $split"
  else
    echo "CACHE_BUILD_START $split"
    "$PYTHON" -u -m rl.build_cache --config "$CONFIG" --split "$split" \
      2>&1 | tee -a "$LOG_ROOT/cache_${split}.log"
    echo "CACHE_BUILD_DONE $split"
  fi
done

if [[ ! -e "$CACHE/seen_test" ]]; then
  ln -s "$CACHE/test" "$CACHE/seen_test"
fi

# Keep the existing object-track arrays read-only and expose them through a
# separate compatibility root for the new cache's seen_test alias.
for split in dqn_train val test; do
  if [[ ! -e "$TRACK/$split" ]]; then
    ln -s "$PROJECT/data/trajectory_latest_vpoclip_new50_5/object_tracks_full/$split" "$TRACK/$split"
  fi
done
if [[ ! -e "$TRACK/seen_test" ]]; then
  ln -s "$TRACK/test" "$TRACK/seen_test"
fi

echo "CACHE_COUNTS"
for split in dqn_train val test; do
  "$PYTHON" - "$CACHE/$split/metadata.json" <<'PY'
import json, sys
p = sys.argv[1]
payload = json.load(open(p, encoding='utf-8'))
print(p, len(payload.get('episodes', [])))
PY
done

COMMON=(
  --cache-root "$CACHE"
  --track-root "$TRACK"
  --epochs 40
  --batch-size 16384
  --updates-per-epoch 64
  --learning-rate 1e-4
  --seeds 20260909 20260910 20260911
  --unseen-classes 14 15 25 39 46
  --pseudo-unseen-classes 0 2 9 10 11 17 26 34 49 50
)

for variant in v4 v5 v6 object_only geometry_only; do
  output="$POLICY_ROOT/$variant"
  if [[ -f "$output/training_summary.json" && -f "$output/summary.json" ]]; then
    echo "TRAIN_EXISTS $variant"
  else
    echo "TRAIN_START $variant"
    "$PYTHON" -u -m rl.train_trajectory_g1_pseudo_selected_stage_b_2d \
      --variant "$variant" --output-root "$output" "${COMMON[@]}" \
      2>&1 | tee -a "$LOG_ROOT/train_${variant}.log"
    echo "TRAIN_DONE $variant"
  fi

  eval_output="$EVAL_ROOT/$variant"
  if [[ -f "$eval_output/summary.json" ]]; then
    echo "EVAL_EXISTS $variant"
  else
    echo "EVAL_START $variant"
    "$PYTHON" -u -m rl.evaluate_trajectory_variants_new_split \
      --variant "$variant" \
      --cache-root "$CACHE" \
      --track-root "$TRACK" \
      --policy-root "$output" \
      --output-root "$eval_output" \
      --class-bank 14 15 25 39 46 \
      --expected-episodes 271 \
      --eval-seeds 20260920 20260921 20260922 20260923 20260924 20260925 20260926 20260927 20260928 20260929 20260930 20260931 20260932 20260933 20260934 20260935 20260936 20260937 20260938 20260939 20260940 20260941 20260942 20260943 20260944 20260945 20260946 20260947 20260948 20260949 \
      2>&1 | tee -a "$LOG_ROOT/eval_${variant}.log"
    echo "EVAL_DONE $variant"
  fi
done

echo "PIPELINE_COMPLETE $(date -Is)"
