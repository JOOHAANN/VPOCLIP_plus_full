#!/usr/bin/env bash
set -Eeuo pipefail

CLIPGCN_ROOT=${CLIPGCN_ROOT:-/workspace/CLIPGCN}
RUN_ID=${RUN_ID:-$(date '+%Y%m%d_%H%M%S')}
LOG_DIR=${LOG_DIR:-"$CLIPGCN_ROOT/work_dir/windowed_zsl_pipeline/logs"}
PID_DIR=${PID_DIR:-"$CLIPGCN_ROOT/work_dir/windowed_zsl_pipeline"}
mkdir -p "$LOG_DIR" "$PID_DIR"

LOG_FILE="$LOG_DIR/windowed_zsl_${RUN_ID}.log"
PID_FILE="$PID_DIR/windowed_zsl_${RUN_ID}.pid"

nohup bash "$CLIPGCN_ROOT/scripts/run_windowed_zsl_pipeline.sh" >"$LOG_FILE" 2>&1 &
pid=$!
echo "$pid" >"$PID_FILE"

echo "started pid=$pid"
echo "log=$LOG_FILE"
echo "pid_file=$PID_FILE"
echo "follow: tail -f $LOG_FILE"
