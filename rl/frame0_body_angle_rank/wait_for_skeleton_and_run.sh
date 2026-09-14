#!/usr/bin/env bash
set -Eeuo pipefail

# The ETRI skeleton archives are access-controlled and are intentionally not
# guessed or replaced with the old optical-axis CSV.  This driver waits for
# the user/admin to place the two extracted subject ranges, then continues
# automatically in the same tmux session.
ROOT="/home/youhan/ws/VPOCLIP_plus_full"
SKELETON_ROOT="/home/youhan/ws/ETRI-Activity3D-Skeleton"
LOG_DIR="$ROOT/logs/frame0_body_angle_rank_v1_v6"
LOG_FILE="$LOG_DIR/pipeline.log"

mkdir -p "$LOG_DIR"
cd "$ROOT"

echo "FRAME0_ANGLE_PIPELINE_WAITING_FOR_SKELETON" | tee -a "$LOG_FILE"
while true; do
  first="$SKELETON_ROOT/Skeleton(P001-P050)/P001-P050/A001_P001_G001_C001.csv"
  second="$SKELETON_ROOT/Skeleton(P051-P100)/P051-P100/A001_P051_G001_C001.csv"
  if [[ -f "$first" && -f "$second" ]]; then
    echo "FRAME0_ANGLE_SKELETON_ROOT_READY" | tee -a "$LOG_FILE"
    break
  fi
  first_state="missing"
  second_state="missing"
  [[ -f "$first" ]] && first_state="ok"
  [[ -f "$second" ]] && second_state="ok"
  echo "FRAME0_ANGLE_WAITING first=$first_state second=$second_state" | tee -a "$LOG_FILE"
  sleep 60
done

python3 -m rl.frame0_body_angle_rank.build_ranked_cache 2>&1 | tee -a "$LOG_FILE"
bash rl/frame0_body_angle_rank/run_all.sh 2>&1 | tee -a "$LOG_FILE"
echo "FRAME0_ANGLE_PIPELINE_COMPLETE" | tee -a "$LOG_FILE"
