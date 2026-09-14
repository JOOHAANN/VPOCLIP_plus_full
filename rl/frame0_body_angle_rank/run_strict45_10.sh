#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/home/youhan/ws/VPOCLIP_plus_full"
PYTHON="/home/youhan/.conda/envs/clipgcn/bin/python"
PREP="rl.frame0_body_angle_rank.prepare_strict45_10_inputs"
CONFIG="$ROOT/work_dir/frame0_body_angle_rank_strict45_10/build_cache.yaml"
CACHE="$ROOT/work_dir/frame0_body_angle_rank_strict45_10/cache"
TRACK6="$ROOT/work_dir/frame0_body_angle_rank_strict45_10/object_tracks_v3_bbox_size"
TRACK4="$ROOT/work_dir/frame0_body_angle_rank_strict45_10/object_tracks_full"
OUT="$ROOT/work_dir/frame0_body_angle_rank_strict45_10_v1_v6"
LOG="$ROOT/logs/frame0_body_angle_rank_strict45_10_v1_v6"
UNSEEN=(1 7 14 15 18 25 39 46 52 54)
PSEUDO=(8 27 28 35 50)

mkdir -p "$LOG" "$OUT"
cd "$ROOT"

"$PYTHON" -u -m "$PREP" 2>&1 | tee "$LOG/01_prepare.log"

for split in dqn_train val test; do
  if [[ -f "$CACHE/$split/metadata.json" ]]; then
    echo "STRICT45_10_SKIP_CACHE $split"
    continue
  fi
  echo "STRICT45_10_BUILD_CACHE $split"
  "$PYTHON" -u - <<PY 2>&1 | tee "$LOG/02_cache_${split}.log"
from pathlib import Path
from rl.build_cache import export_cache, load_config
config = load_config("$CONFIG")
manifest = Path("$CACHE/$split/manifest.jsonl")
export_cache(config, manifest)
print("STRICT45_10_CACHE_COMPLETE", "$split", flush=True)
PY
done

# The ranked cache keeps the exact frame-0 anatomical yaw as a sidecar for
# v4.  ``export_cache`` preserves it in metadata, but the trainer expects the
# compact per-episode array in each split directory.  Keep the unit explicit:
# this array is degrees; true_yaw_v4 converts it to radians before sin/cos.
for split in dqn_train val test; do
  "$PYTHON" -u - <<PY 2>&1 | tee "$LOG/02b_body_yaw_${split}.log"
from pathlib import Path
import json
import math
import numpy as np

root = Path("$CACHE") / "$split"
metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
episodes = metadata["episodes"]
yaw = np.full((len(episodes), 4), np.nan, dtype=np.float32)
for row_index, episode in enumerate(episodes):
    for rank, view in enumerate(episode.get("views", [])):
        value = view.get("body_forward_yaw_in_camera_deg")
        if value is None:
            continue
        value = float(value)
        if math.isfinite(value) and 0 <= rank < 4:
            yaw[row_index, rank] = value
np.save(root / "body_yaw_deg.npy", yaw)
print("STRICT45_10_BODY_YAW_COMPLETE", "$split", list(yaw.shape), "finite", int(np.isfinite(yaw).sum()), flush=True)
PY
done

ln -sfn test "$CACHE/seen_test"

if [[ ! -f "$TRACK6/complete.json" ]]; then
  echo "STRICT45_10_BUILD_TRACKS"
  "$PYTHON" -u -m rl.build_object_position_tracks \
    --cache-root "$CACHE" \
    --output-root "$TRACK6" \
    --splits dqn_train val test \
    --yolo-repo /home/youhan/ws/yolov5_full \
    --yolo-weights /home/youhan/ws/yolov5_full/best.pt \
    --decode-workers 12 \
    --video-batch 16 \
    --frame-batch 128 \
    --include-box-size 2>&1 | tee "$LOG/03_tracks_v3.log"
fi

if [[ ! -f "$TRACK4/complete.json" ]]; then
  "$PYTHON" -u - <<PY 2>&1 | tee "$LOG/04_tracks_c4.log"
from pathlib import Path
import json
import numpy as np
src = Path("$TRACK6")
dst = Path("$TRACK4")
dst.mkdir(parents=True, exist_ok=True)
for split in ("dqn_train", "val", "test"):
    arr = np.load(src / split / "object_position_track.npy", mmap_mode="r")
    out_dir = dst / split
    out_dir.mkdir(parents=True, exist_ok=True)
    out = np.lib.format.open_memmap(out_dir / "object_position_track.npy", mode="w+", dtype=np.float16, shape=(arr.shape[0], arr.shape[1], arr.shape[2], arr.shape[3], 4))
    out[...] = np.asarray(arr[..., (0, 1, 2, 5)], dtype=np.float16)
    out.flush()
    (out_dir / "complete.json").write_text(json.dumps({"split": split, "source": str(src / split / "object_position_track.npy"), "shape": list(out.shape), "fields": ["presence", "x", "y", "confidence"]}, indent=2))
(dst / "complete.json").write_text(json.dumps({"source": str(src), "channels": 4}, indent=2))
print("STRICT45_10_TRACKS_C4_COMPLETE", flush=True)
PY
fi

ln -sfn test "$TRACK6/seen_test"
ln -sfn test "$TRACK4/seen_test"

variants=(v1 v2_legacy v3 v4 v5 v6 object_only geometry_only)
for variant in "${variants[@]}"; do
  if [[ -f "$OUT/$variant/complete.json" ]]; then
    echo "STRICT45_10_SKIP_VARIANT $variant"
    continue
  fi
  echo "STRICT45_10_TRAIN_VARIANT $variant"
  "$PYTHON" -u -m rl.frame0_body_angle_rank.train_ranked_variant \
    --variant "$variant" \
    --cache-root "$CACHE" \
    --track-root "$TRACK4" \
    --track-root-v3 "$TRACK6" \
    --output-root "$OUT/$variant" \
    --epochs 40 \
    --batch-size 16384 \
    --updates-per-epoch 64 \
    --learning-rate 1e-4 \
    --seeds 20260909 20260910 20260911 \
    --memory-fraction 0.16 \
    --unseen-classes "${UNSEEN[@]}" \
    --pseudo-unseen-classes "${PSEUDO[@]}" \
    2>&1 | tee "$LOG/05_train_${variant}.log"
done

echo "STRICT45_10_ALL_VARIANTS_COMPLETE"
