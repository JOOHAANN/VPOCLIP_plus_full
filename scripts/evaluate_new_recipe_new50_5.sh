#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/home/youhan/ws/VPOCLIP_plus_full
PY=/home/youhan/.conda/envs/clipgcn/bin/python
CFG="$ROOT/configs/vpoclip_new50_5_recipe_stage_b.yaml"
CKPT="$ROOT/work_dir/allviews_lowview50_5_recipe_stage_b/last_model.pth"
DATA="$ROOT/data/allviews_cs45_lowview50_5_zsl"
OUT="$ROOT/work_dir/allviews_lowview50_5_recipe_eval"
LOG="$ROOT/logs/vpoclip_new50_5_recipe_20260909/evaluation.log"
mkdir -p "$OUT" "$(dirname "$LOG")"
cd "$ROOT"
export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
exec > >(tee -a "$LOG") 2>&1
test -s "$CKPT"
run() {
  "$PY" test.py --config "$CFG" --checkpoint "$CKPT" --split-dir "$DATA" \
    --class-split-dir "$DATA" --batch-size 256 --num-workers 12 "$@"
}
run --prefix trimodal_test_seen --sample-scope all --candidate-scope seen \
  --output "$OUT/seen50_only_last.json"
run --prefix trimodal_test --sample-scope all --candidate-scope unseen \
  --output "$OUT/unseen5_only_last.json"
run --prefix trimodal_test_seen --protocol gzsl_seen \
  --output "$OUT/gzsl_seen55_last.json"
run --prefix trimodal_test --protocol gzsl_unseen \
  --output "$OUT/gzsl_unseen55_last.json"
"$PY" - <<'PY'
import json
from pathlib import Path
root = Path("/home/youhan/ws/VPOCLIP_plus_full/work_dir/allviews_lowview50_5_recipe_eval")
files = {"seen50_only": root/"seen50_only_last.json",
         "unseen5_only": root/"unseen5_only_last.json",
         "gzsl_seen55": root/"gzsl_seen55_last.json",
         "gzsl_unseen55": root/"gzsl_unseen55_last.json"}
summary = {}
for key, path in files.items():
    m = json.loads(path.read_text())["metrics"]
    summary[key] = {name: m.get(name) for name in
                    ("top1_acc", "macro_top1_acc", "top5_acc", "num_samples", "candidate_labels")}
    print(key, summary[key])
s = summary["gzsl_seen55"]["top1_acc"]
u = summary["gzsl_unseen55"]["top1_acc"]
summary["gzsl_harmonic_mean"] = 2*s*u/(s+u) if s+u else 0.0
(root/"summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
print("gzsl_harmonic_mean", summary["gzsl_harmonic_mean"])
PY
