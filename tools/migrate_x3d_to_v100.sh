#!/bin/bash
# Move the Smarthome X3D fine-tune from the P100 to a V100 once the object
# extraction frees one up. The P100 has no tensor cores, so mixed precision buys
# it nothing: it reports a 17 hour ETA against roughly 5 on a V100. Training
# resumes from the last saved checkpoint, so only the current interval is lost.
set -e
cd /workspace/X3D
PY=/root/miniconda3/envs/clipgcn/bin/python
LOG=/workspace/VPOCLIP_plus/work_dir

while pgrep -f "extract_smarthome_objects" > /dev/null; do sleep 60; done
echo "=== object extraction finished, migrating X3D to GPU 2 ==="

for pid in $(pgrep -f "train.py -cfg configs/x3d-s_smarthome_182.yaml"); do
  kill "$pid" 2>/dev/null || true
done
sleep 10

$PY - <<'EOF'
from pathlib import Path
path = Path("configs/x3d-s_smarthome_182.yaml")
path.write_text(path.read_text().replace("RESUME: False", "RESUME: True"))
print("RESUME enabled")
EOF

export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2
$PY tools/train.py -cfg configs/x3d-s_smarthome_182.yaml >> "$LOG/sh_x3d_train.log" 2>&1
echo "X3D_DONE"
