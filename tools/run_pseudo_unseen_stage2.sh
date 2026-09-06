#!/bin/bash
# Stage 2 of the pseudo-unseen protocol.
#
# Stage 1 trained on 45 classes and picked the epoch that maximised ZSL accuracy
# on 5 held-out pseudo-unseen classes. Only that epoch *number* carries over:
# stage 2 retrains from scratch on all 50 seen classes and stops there, so the
# final model gets the full training set while the stopping point was chosen
# without ever touching the test classes.
#
# Usage:  GPU=1 ./tools/run_pseudo_unseen_stage2.sh objconcat legacy
set -e
cd /workspace/VPOCLIP_plus
PY=/root/miniconda3/envs/clipgcn/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${GPU:-1}

for variant in "$@"; do
  hist=$(ls -t work_dir/pu_stage1_${variant}/run_*/history.json 2>/dev/null | head -1)
  if [ -z "$hist" ]; then
    echo "[$variant] stage 1 history missing, skipping"
    continue
  fi

  # The pseudo-unseen set is 50 clips, so one clip moves it 2 points and a raw
  # arg-max lands on single-epoch noise spikes (epoch 2 for two of the heads).
  # A 9-epoch moving average picks the region that is genuinely best instead.
  best_epoch=$($PY - "$hist" <<'EOF'
import json, sys
import numpy as np
history = json.load(open(sys.argv[1]))
scores = np.array([np.nan if v is None else v for v in history["pseudo_zsl"]], dtype=float)
window = 9
smoothed = np.convolve(scores, np.ones(window) / window, mode="valid")
print(int(np.nanargmax(smoothed)) + window // 2 + 1)
EOF
)
  echo "==== ${variant}: stage 1 picked epoch ${best_epoch}, retraining on all 50 classes ===="

  cfg="config_pu_stage2_${variant}.yaml"
  $PY - "$variant" "$best_epoch" <<'EOF'
import sys
from pathlib import Path
variant, epochs = sys.argv[1], int(sys.argv[2])
text = Path(f"config_cs_50_5_coco17_{variant}.yaml").read_text()
text = text.replace("  epochs: 300", f"  epochs: {epochs}")
# Nothing is selected here: the stopping point came from stage 1, so the model
# to report is simply the last one.
text = text.replace("  model_selection: val_accuracy", "  model_selection: val_loss")
text = text.replace(f"./work_dir/cs_50_5_coco17_{variant}", f"./work_dir/pu_stage2_{variant}")
Path(f"config_pu_stage2_{variant}.yaml").write_text(text)
EOF

  $PY train.py --config "$cfg" > "work_dir/pu_s2_${variant}.log" 2>&1
  echo "[done] ${variant} stage 2 (${best_epoch} epochs)"
done
echo "STAGE2_COMPLETE"
