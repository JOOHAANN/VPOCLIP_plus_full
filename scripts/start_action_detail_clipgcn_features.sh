#!/usr/bin/env bash
set -Eeuo pipefail

CLIPGCN_ROOT=${CLIPGCN_ROOT:-/workspace/CLIPGCN}
LOG_DIR=${LOG_DIR:-"$CLIPGCN_ROOT/work_dir/action_detail_clipgcn_features/logs"}
mkdir -p "$LOG_DIR"

stamp=$(date '+%Y%m%d_%H%M%S')
log_file="$LOG_DIR/action_detail_clipgcn_features_${stamp}.log"
pid_file="$LOG_DIR/action_detail_clipgcn_features_${stamp}.pid"

cd "$CLIPGCN_ROOT"
nohup ./scripts/run_action_detail_clipgcn_features.sh >"$log_file" 2>&1 &
pid=$!
echo "$pid" >"$pid_file"

echo "Started action-detail CLIPGCN feature preparation"
echo "PID: $pid"
echo "Log: $log_file"
echo "PID file: $pid_file"
