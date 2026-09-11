#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/home/youhan/ws/VPOCLIP_plus_full
PY=/home/youhan/.conda/envs/clipgcn/bin/python
CFG=$ROOT/configs/eval_legacy50_5.yaml
CKPT=$ROOT/work_dir/final_aug_entropy_single_h2/final_best_single_lambdaH2_anchor100/last_model.pth
DATA=$ROOT/data/contrastive_cs_45_25_10_20_zsl_50_5
OUT=$ROOT/work_dir/legacy50_5_eval
LOG=$ROOT/logs/legacy50_5_eval_20260909.log

cd "$ROOT"
export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
mkdir -p "$OUT" "$(dirname "$LOG")"
: > "$LOG"

run() {
  "$PY" test.py --config "$CFG" --checkpoint "$CKPT" \
    --split-dir "$DATA" --class-split-dir "$DATA" \
    --batch-size 256 --num-workers 12 "$@" 2>&1 | tee -a "$LOG"
}

# Strict five-way / fifty-way scores and the joint 55-class GZSL diagnostic.
run --prefix trimodal_test_seen --sample-scope all --candidate-scope seen \
    --output "$OUT/seen50_only_last.json"
run --prefix trimodal_test --sample-scope all --candidate-scope unseen \
    --output "$OUT/unseen5_only_last.json"
run --prefix trimodal_test_seen --protocol gzsl_seen \
    --output "$OUT/gzsl_seen55_last.json"
run --prefix trimodal_test --protocol gzsl_unseen \
    --output "$OUT/gzsl_unseen55_last.json"

"$PY" - <<'PY' | tee -a "$LOG"
import json
from pathlib import Path

root = Path("/home/youhan/ws/VPOCLIP_plus_full/work_dir/legacy50_5_eval")
files = {
    "seen50_only": root / "seen50_only_last.json",
    "unseen5_only": root / "unseen5_only_last.json",
    "gzsl_seen55": root / "gzsl_seen55_last.json",
    "gzsl_unseen55": root / "gzsl_unseen55_last.json",
}
summary = {}
for name, path in files.items():
    metrics = json.loads(path.read_text())["metrics"]
    summary[name] = metrics
    print(name, "top1=", metrics["top1_acc"], "macro=", metrics["macro_top1_acc"], "n=", metrics["num_samples"])
seen = summary["gzsl_seen55"]["top1_acc"]
unseen = summary["gzsl_unseen55"]["top1_acc"]
summary["gzsl_harmonic_mean"] = 2 * seen * unseen / (seen + unseen) if seen + unseen else 0.0
out = root / "summary.json"
out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
print("gzsl_H=", summary["gzsl_harmonic_mean"])
print("summary=", out)
PY
