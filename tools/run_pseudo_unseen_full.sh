#!/bin/bash
# Full pseudo-unseen selection protocol, end to end.
#
#   groups 1-4  each hold out 5 of the 50 seen classes and track ZSL accuracy on
#               the ~637 clips of those classes that training therefore drops
#   aggregate   turns the four best epochs into one stopping point via the
#               earliest-within-one-standard-error rule
#   final       retrains on all 50 classes and stops there
#
# The real unseen classes are never read, so the resulting number is reportable.
set -e
cd /workspace/VPOCLIP_plus
PY=/root/miniconda3/envs/clipgcn/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID
VARIANT=${VARIANT:-objconcat}

# Groups 1 and 2 are already running; wait for them, then run 3 and 4.
while pgrep -f "config_pug[12]_${VARIANT}" > /dev/null; do sleep 60; done
echo "=== groups 1-2 done ==="

CUDA_VISIBLE_DEVICES=1 $PY train.py --config "config_pug3_${VARIANT}.yaml" \
  > "work_dir/pug3.log" 2>&1 &
pid3=$!
CUDA_VISIBLE_DEVICES=2 $PY train.py --config "config_pug4_${VARIANT}.yaml" \
  > "work_dir/pug4.log" 2>&1 &
pid4=$!
wait $pid3 $pid4
echo "=== groups 3-4 done ==="

$PY tools/aggregate_pseudo_unseen_epochs.py --groups 1 2 3 4 --variant "$VARIANT" \
  | tee work_dir/pseudo_unseen_epochs.txt
epoch=$(grep "recommended stopping epoch" work_dir/pseudo_unseen_epochs.txt | awk '{print $NF}')
echo "=== retraining on all 50 classes for ${epoch} epochs ==="

$PY - "$VARIANT" "$epoch" <<'EOF'
import sys
from pathlib import Path
variant, epochs = sys.argv[1], int(sys.argv[2])
text = Path(f"config_cs_50_5_coco17_{variant}.yaml").read_text()
text = text.replace("  epochs: 300", f"  epochs: {epochs}")
# The stopping point came from the pseudo-unseen groups, so there is nothing
# left to select here: the model to report is the last one.
text = text.replace("  model_selection: val_accuracy", "  model_selection: val_loss")
text = text.replace(f"./work_dir/cs_50_5_coco17_{variant}", f"./work_dir/pufinal_{variant}")
Path(f"config_pufinal_{variant}.yaml").write_text(text)
EOF

CUDA_VISIBLE_DEVICES=1 $PY train.py --config "config_pufinal_${VARIANT}.yaml" \
  > "work_dir/pufinal_${VARIANT}.log" 2>&1
echo "=== final model trained ==="

run=$(ls -dt work_dir/pufinal_${VARIANT}/run_*/ | head -1)
CUDA_VISIBLE_DEVICES=1 $PY test.py --config "config_pufinal_${VARIANT}.yaml" \
  --checkpoint "${run}last_model.pth" \
  --split-dir ./data/contrastive_cs_70_10_20_zsl_50_5 \
  --class-split-dir ./data/contrastive_zsl_splits/50_5 \
  --prefix trimodal_test --protocol zsl \
  --output "work_dir/pufinal_${VARIANT}/zsl_test.json" > /dev/null 2>&1
$PY -c "
import json
m = json.load(open('work_dir/pufinal_${VARIANT}/zsl_test.json'))['metrics']
print(f\"FINAL ZSL top1 = {m['top1_acc'] * 100:.2f}%  (n={m['num_samples']})\")
"
echo "PROTOCOL_COMPLETE"
