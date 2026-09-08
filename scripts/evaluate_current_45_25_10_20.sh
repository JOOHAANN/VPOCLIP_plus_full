#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/home/youhan/ws/VPOCLIP_plus_full
PYTHON=/home/youhan/.conda/envs/clipgcn/bin/python
ZSL="$ROOT/work_dir/final_aug_entropy_single_h2/final_best_single_lambdaH2_anchor100/last_model.pth"
CLOSED="$ROOT/work_dir/final55_aug_entropy_h5_full300_score/final55_h5_8prompt_300_acc_minus_h/last_model.pth"
LOG="$ROOT/logs/vpoclip_45_25_10_20_20260906/evaluation.log"
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
export PATH=/home/youhan/.conda/envs/clipgcn/bin:$PATH
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16
cd "$ROOT"

test -f "$ZSL" && test -f "$CLOSED"

"$PYTHON" test.py --config config_final_aug_entropy_single_h2.yaml \
  --checkpoint "$ZSL" \
  --split-dir data/contrastive_cs_45_25_10_20_zsl_50_5 \
  --class-split-dir data/contrastive_cs_45_25_10_20_zsl_50_5 \
  --prefix trimodal_test --protocol zsl --batch-size 256 --num-workers 12 \
  --output work_dir/eval_zsl_last.json

"$PYTHON" test.py --config config_final_aug_entropy_single_h2.yaml \
  --checkpoint "$ZSL" \
  --split-dir data/contrastive_cs_45_25_10_20_zsl_50_5 \
  --class-split-dir data/contrastive_cs_45_25_10_20_zsl_50_5 \
  --prefix trimodal_test --protocol gzsl_unseen --batch-size 256 --num-workers 12 \
  --output work_dir/eval_gzsl_unseen_last.json

"$PYTHON" test.py --config config_final_aug_entropy_single_h2.yaml \
  --checkpoint "$ZSL" \
  --split-dir data/contrastive_cs_45_25_10_20_zsl_50_5 \
  --class-split-dir data/contrastive_cs_45_25_10_20_zsl_50_5 \
  --prefix trimodal_test_seen --protocol gzsl_seen --batch-size 256 --num-workers 12 \
  --output work_dir/eval_gzsl_seen_last.json

"$PYTHON" test.py --config config_final55_aug_entropy_h5_full300_score.yaml \
  --checkpoint "$CLOSED" \
  --split-dir data/contrastive_cs_45_25_10_20_55_0 \
  --class-split-dir data/contrastive_cs_45_25_10_20_55_0 \
  --prefix test --sample-scope all --candidate-scope all \
  --batch-size 256 --num-workers 12 --output work_dir/eval_closed55_last.json

"$PYTHON" - <<'PY'
import json
from pathlib import Path
root = Path("work_dir")
files = {
    "zsl_50_5_unseen": root / "eval_zsl_last.json",
    "gzsl_unseen_55_candidates": root / "eval_gzsl_unseen_last.json",
    "gzsl_seen_55_candidates": root / "eval_gzsl_seen_last.json",
    "closed_set_55_0": root / "eval_closed55_last.json",
}
metrics = {}
for name, path in files.items():
    payload = json.loads(path.read_text())
    metrics[name] = payload["metrics"]
    print(name, "top1=", metrics[name]["top1_acc"], "macro=", metrics[name]["macro_top1_acc"], "n=", metrics[name]["num_samples"])
seen = metrics["gzsl_seen_55_candidates"]["top1_acc"]
unseen = metrics["gzsl_unseen_55_candidates"]["top1_acc"]
metrics["gzsl_harmonic_mean"] = 2 * seen * unseen / (seen + unseen) if seen + unseen else 0.0
out = root / "eval_summary_45_25_10_20.json"
out.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
print("gzsl_harmonic_mean=", metrics["gzsl_harmonic_mean"])
print("summary=", out)
PY
