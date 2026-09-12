#!/usr/bin/env bash
set -euo pipefail
cd /home/youhan/ws/VPOCLIP_plus_full
PY=/home/youhan/.conda/envs/clipgcn/bin/python
CACHE=/home/youhan/ws/VPOCLIP_plus_full/data/rl_candidate_rank_v6_new50_5/cache
OUT=/home/youhan/ws/VPOCLIP_plus_full/work_dir/published_selectors_v1/mflstm_a2c_old_split
ANCHOR=/home/youhan/ws/VPOCLIP_plus_full/work_dir/published_selectors_v1/v5_old_weighted_anchor
GATE=/home/youhan/ws/VPOCLIP_plus_full/work_dir/weighted_fusion_policy_variant_comparison_v1/gates/v7_old_split/seed_20260909/best.pt

exec "$PY" -u work_dir/published_selectors_v1/run.py \
  --model mflstm_a2c \
  --cache-root "$CACHE" \
  --output-root "$OUT" \
  --anchor-root "$ANCHOR" \
  --anchor-name v5_old_weighted_anchor \
  --gate-checkpoint "$GATE" \
  --true-unseen 0 2 26 34 50 \
  --seen-classes 1 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 27 28 29 30 31 32 33 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 51 52 53 54 \
  --pseudo-unseen 5 6 14 22 25 38 42 44 45 51 \
  --train-episodes 2734 \
  --eval-seeds 20260920 20260921 20260922 20260923 20260924 20260925 20260926 20260927 20260928 20260929 20260930 20260931 20260932 20260933 20260934 20260935 20260936 20260937 20260938 20260939 20260940 20260941 20260942 20260943 20260944 20260945 20260946 20260947 20260948 20260949 \
  --eval-moves 1
