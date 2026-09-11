#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/home/youhan/ws/VPOCLIP_plus_full
PY=/home/youhan/.conda/envs/clipgcn/bin/python
RUN_ROOT_NAME="${RUN_ROOT_NAME:-new50_5_groups_20260909}"
GROUP_NAME="${GROUP_NAME:-group1}"
GROUP="$ROOT/logs/$RUN_ROOT_NAME/$GROUP_NAME"
DATA="$ROOT/data/new50_5_${GROUP_NAME}_zsl"
CFG="$GROUP/vpoclip_stage_b.yaml"
MODEL_VARIANT="${MODEL_VARIANT:-last}"
if [[ "$MODEL_VARIANT" != "last" && "$MODEL_VARIANT" != "best" ]]; then
  echo "MODEL_VARIANT must be last or best" >&2
  exit 2
fi
CKPT="$ROOT/work_dir/new50_5_${GROUP_NAME}/recipe_stage_b/${MODEL_VARIANT}_model.pth"
OUT="$GROUP/evaluation_${MODEL_VARIANT}"
LOG="$GROUP/evaluation_${MODEL_VARIANT}.log"

mkdir -p "$OUT"
cd "$ROOT"
# The concurrent second-group X3D job occupies the GPU.  These two tests are
# lightweight fusion-head inference, so run them on CPU to avoid contending
# for CUDA memory and to leave the training throughput unchanged.
export CUDA_VISIBLE_DEVICES=""
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1
export RUN_ROOT_NAME GROUP_NAME MODEL_VARIANT

exec > >(tee -a "$LOG") 2>&1
test -s "$CKPT"
test -s "$DATA/metadata.json"

"$PY" test.py --config "$CFG" --checkpoint "$CKPT" \
  --split-dir "$DATA" --class-split-dir "$DATA" \
  --prefix trimodal_test --sample-scope all --candidate-scope unseen \
  --batch-size 512 --num-workers 8 \
  --output "$OUT/unseen5_zsl_last.json"

"$PY" test.py --config "$CFG" --checkpoint "$CKPT" \
  --split-dir "$DATA" --class-split-dir "$DATA" \
  --prefix trimodal_test_seen --sample-scope all --candidate-scope seen \
  --batch-size 512 --num-workers 8 \
  --output "$OUT/seen50_closed_last.json"

"$PY" - <<'PY'
import json
import os
from pathlib import Path

variant = os.environ["MODEL_VARIANT"]
run_root = os.environ.get("RUN_ROOT_NAME", "new50_5_groups_20260909")
group_name = os.environ.get("GROUP_NAME", "group1")
out = Path(f"/home/youhan/ws/VPOCLIP_plus_full/logs/{run_root}/{group_name}/evaluation_{variant}")
files = {
    "unseen5_zsl": out / "unseen5_zsl_last.json",
    "seen50_closed": out / "seen50_closed_last.json",
}
summary = {}
for name, path in files.items():
    metrics = json.loads(path.read_text(encoding="utf-8"))["metrics"]
    summary[name] = {
        key: metrics.get(key)
        for key in ("top1_acc", "macro_top1_acc", "top5_acc", "num_samples", "candidate_labels")
    }
summary["checkpoint"] = f"/home/youhan/ws/VPOCLIP_plus_full/work_dir/new50_5_{group_name}/recipe_stage_b/{variant}_model.pth"
(out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY
