#!/usr/bin/env bash
set -Eeuo pipefail
cd /home/youhan/ws/VPOCLIP_plus_full
export PATH=/home/youhan/.conda/envs/clipgcn/bin:$PATH
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
trap 'echo PIPELINE_FAILED_LINE_$LINENO' ERR
python3 -m rl.frame0_body_angle_rank.extract_required
python3 -m rl.frame0_body_angle_rank.build_ranked_cache
bash rl/frame0_body_angle_rank/run_all.sh
