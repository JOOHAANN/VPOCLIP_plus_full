#!/usr/bin/env bash
set -Eeuo pipefail

CLIPGCN_ROOT=${CLIPGCN_ROOT:-/workspace/CLIPGCN}
LOG_DIR=${LOG_DIR:-"$CLIPGCN_ROOT/work_dir/action_detail_raw_training_data/logs"}
mkdir -p "$LOG_DIR"

RUN_ID=${RUN_ID:-$(date '+%Y%m%d_%H%M%S')}
LOG_FILE="$LOG_DIR/action_detail_raw_${RUN_ID}.log"

setsid bash -lc "cd '$CLIPGCN_ROOT' && exec '$CLIPGCN_ROOT/scripts/run_action_detail_raw_training_data.sh'" \
  >"$LOG_FILE" 2>&1 < /dev/null &
PID=$!

echo "$PID" >"$LOG_DIR/action_detail_raw_${RUN_ID}.pid"
echo "Started action-detail raw training-data build"
echo "PID: $PID"
echo "Log: $LOG_FILE"
