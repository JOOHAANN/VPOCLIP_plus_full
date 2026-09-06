"""Run GZSL evaluation (S / U / H) for the zero-shot splits.

For each split (default 50_5 45_10 40_15) this driver invokes test.py twice:
  U: unseen samples vs ALL candidates  (--protocol gzsl_unseen)
  S: seen test samples vs ALL candidates (--protocol gzsl_seen)
with the unseen prototype scores multiplied by --scale (calibrated stacking),
then reports S, U and the harmonic mean H = 2SU/(S+U).

Usage:
  cd /workspace/CLIPGCN
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
    /root/miniconda3/envs/clipgcn/bin/python tools/run_gzsl_eval.py --scale 1.03
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def latest_checkpoint(split):
    base = ROOT / f"work_dir/clipgcn_contrastive_{split}"
    runs = sorted((d for d in base.glob("run_*") if (d / "best_model.pth").exists()),
                  key=lambda d: d.name)
    if not runs:
        raise FileNotFoundError(f"No run with best_model.pth under {base}")
    return runs[-1] / "best_model.pth"


def run_test(split, protocol, checkpoint, scale, out_json):
    if protocol == "gzsl_unseen":
        split_dir, prefix = f"data/contrastive_zsl_splits/{split}", "unseen"
    else:
        split_dir, prefix = f"data/contrastive_test_seen_splits/{split}", "seen"
    cmd = [
        PYTHON, "test.py",
        "--config", f"config_{split}.yaml",
        "--checkpoint", str(checkpoint),
        "--split-dir", split_dir,
        "--prefix", prefix,
        "--class-split-dir", f"data/contrastive_zsl_splits/{split}",
        "--protocol", protocol,
        "--unseen-score-scale", str(scale),
        "--output", str(out_json),
    ]
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout[-2000:])
        print(result.stderr[-2000:])
        raise RuntimeError(f"test.py failed: {split} {protocol}")
    metrics = json.loads(Path(out_json).read_text())["metrics"]
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="+", default=["50_5", "45_10", "40_15"])
    parser.add_argument("--scale", type=float, default=1.03,
                        help="unseen-score-scale (calibrated stacking coefficient)")
    args = parser.parse_args()

    print(f"{'split':8s} {'scale':>6s} {'S top1':>8s} {'U top1':>8s} {'H':>8s} {'U top5':>8s}")
    print("-" * 52)
    for split in args.splits:
        ckpt = latest_checkpoint(split)
        out_dir = ckpt.parent
        m_u = run_test(split, "gzsl_unseen", ckpt, args.scale,
                       out_dir / f"gzsl_unseen_scale{args.scale}.json")
        m_s = run_test(split, "gzsl_seen", ckpt, args.scale,
                       out_dir / f"gzsl_seen_scale{args.scale}.json")
        S, U = m_s["top1_acc"], m_u["top1_acc"]
        H = 2 * S * U / (S + U) if (S + U) > 0 else 0.0
        u5 = m_u.get("top5_acc")
        print(f"{split:8s} {args.scale:6.2f} {S*100:7.2f}% {U*100:7.2f}% {H*100:7.2f}% "
              f"{(u5*100 if u5 is not None else float('nan')):7.2f}%")
        print(f"         ckpt: {ckpt.relative_to(ROOT)}")
    print("\nH = 2SU/(S+U); S=seen测试样本@全候选, U=unseen样本@全候选, unseen分数×scale")


if __name__ == "__main__":
    main()
