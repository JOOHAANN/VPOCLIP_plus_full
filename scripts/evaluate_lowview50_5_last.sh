#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/home/youhan/ws/VPOCLIP_plus_full
PY=/home/youhan/.conda/envs/clipgcn/bin/python
CFG="$ROOT/logs/allviews_lowview50_5_20260907/final_aug_entropy_single_h2.yaml"
CKPT="$ROOT/work_dir/allviews_lowview50_5_final_aug_entropy_single_h2/allviews_lowview50_5_20260907/last_model.pth"
DATA="$ROOT/data/allviews_cs45_lowview50_5_zsl"
LOG="$ROOT/logs/allviews_lowview50_5_20260907/eval_last_model.log"
cd "$ROOT"
export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
mkdir -p "$(dirname "$LOG")"
: > "$LOG"
run() {
  "$PY" test.py --config "$CFG" --checkpoint "$CKPT" --split-dir "$DATA" \
    --class-split-dir "$DATA" --batch-size 256 --num-workers 12 "$@" 2>&1 | tee -a "$LOG"
}
run --prefix trimodal_test --protocol zsl --output work_dir/allviews_lowview50_5_eval_zsl_last.json
run --prefix trimodal_test_seen --sample-scope all --candidate-scope seen --output work_dir/allviews_lowview50_5_eval_seen50_last.json
run --prefix trimodal_test --protocol gzsl_unseen --output work_dir/allviews_lowview50_5_eval_gzsl_unseen_last.json
run --prefix trimodal_test_seen --protocol gzsl_seen --output work_dir/allviews_lowview50_5_eval_gzsl_seen_last.json
"$PY" - <<'PY' | tee -a "$LOG"
import json
from pathlib import Path
root = Path("work_dir")
files = {"zsl_unseen5": root/"allviews_lowview50_5_eval_zsl_last.json",
         "seen50_only": root/"allviews_lowview50_5_eval_seen50_last.json",
         "gzsl_unseen55": root/"allviews_lowview50_5_eval_gzsl_unseen_last.json",
         "gzsl_seen55": root/"allviews_lowview50_5_eval_gzsl_seen_last.json"}
metrics = {}
for key, path in files.items():
    value = json.loads(path.read_text())["metrics"]
    metrics[key] = value
    print(key, "top1=", value["top1_acc"], "macro=", value["macro_top1_acc"], "n=", value["num_samples"])
seen = metrics["gzsl_seen55"]["top1_acc"]
unseen = metrics["gzsl_unseen55"]["top1_acc"]
metrics["gzsl_harmonic_mean"] = 2*seen*unseen/(seen+unseen) if seen+unseen else 0.0
out = root/"allviews_lowview50_5_eval_summary.json"
out.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
print("gzsl_H=", metrics["gzsl_harmonic_mean"])
print("summary=", out)
PY
