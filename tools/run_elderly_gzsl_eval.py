"""Elderly-scenario GZSL evaluation on a per-split subset of classes.

For each split (50_5 / 45_10 / 40_15):
  seen candidates   = 13 classes drawn from THAT split's official seen set.
                      Preference order: classes currently marked `seen` in
                      class_selection.yaml (the elderly action picks); any that
                      collide with the split's unseen set (or exceed 13) are
                      dropped, and the list is topped up randomly (seeded).
  unseen candidates = exactly that split's official unseen set (5/10/15).
  everything else   = excluded (neither samples nor candidates).

Then runs test.py twice per split (gzsl_seen / gzsl_unseen) with calibrated
stacking and reports S, U, H.

Usage:
  cd /workspace/CLIPGCN
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
    /root/miniconda3/envs/clipgcn/bin/python tools/run_elderly_gzsl_eval.py --scale 1.18
  # 只看选类结果不跑测试: 加 --dry-run
"""
import argparse
import json
import random
import re
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
N_SEEN = 13


def load_action_names():
    sys.path.insert(0, str(ROOT))
    from model import _read_xlsx_rows

    rows = _read_xlsx_rows(str(ROOT / "ntu55_global_descriptions.xlsx"))
    return {int(r["ID"]) - 1: r["action_description"] for r in rows}


def load_preferred_seen():
    with open(ROOT / "class_selection.yaml", encoding="utf-8") as f:
        classes = yaml.safe_load(f)["classes"]
    preferred = []
    for code, state in classes.items():
        if str(state).strip().lower() == "seen":
            preferred.append(int(re.fullmatch(r"A(\d+)", str(code)).group(1)) - 1)
    return sorted(preferred)


def pick_seen(preferred, split_seen, split_unseen, rng):
    split_seen = set(split_seen)
    chosen = [c for c in preferred if c in split_seen]
    dropped = [c for c in preferred if c not in split_seen]
    if len(chosen) > N_SEEN:
        rng.shuffle(chosen)
        chosen = chosen[:N_SEEN]
    elif len(chosen) < N_SEEN:
        pool = sorted(split_seen - set(chosen))
        rng.shuffle(pool)
        chosen += pool[: N_SEEN - len(chosen)]
    assert len(chosen) == N_SEEN
    assert not set(chosen) & set(split_unseen), "seen picks collide with unseen!"
    return sorted(chosen), dropped


def latest_checkpoint(split):
    base = ROOT / f"work_dir/clipgcn_contrastive_{split}"
    runs = sorted((d for d in base.glob("run_*") if (d / "best_model.pth").exists()),
                  key=lambda d: d.name)
    if not runs:
        raise FileNotFoundError(f"No run with best_model.pth under {base}")
    return runs[-1] / "best_model.pth"


def run_test(split, protocol, checkpoint, class_dir, scale, out_json):
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
        "--class-split-dir", str(class_dir),
        "--protocol", protocol,
        "--unseen-score-scale", str(scale),
        "--output", str(out_json),
    ]
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout[-2000:])
        print(result.stderr[-2000:])
        raise RuntimeError(f"test.py failed: {split} {protocol}")
    return json.loads(Path(out_json).read_text())["metrics"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits", nargs="+", default=["50_5", "45_10", "40_15"])
    parser.add_argument("--scale", type=float, default=1.18,
                        help="unseen-score-scale (calibrated stacking)")
    parser.add_argument("--seed", type=int, default=20260712, help="随机补位的种子(可复现)")
    parser.add_argument("--dry-run", action="store_true", help="只打印选类结果, 不跑测试")
    args = parser.parse_args()

    names = load_action_names()
    preferred = load_preferred_seen()
    print(f"class_selection.yaml 里标 seen 的偏好类({len(preferred)}个): {preferred}\n")

    results = []
    for split in args.splits:
        meta = json.loads((ROOT / f"data/contrastive_zsl_splits/{split}/metadata.json").read_text())
        split_seen, split_unseen = meta["seen_classes"], meta["unseen_classes"]

        rng = random.Random(args.seed + hash(split) % 1000)
        chosen, dropped = pick_seen(preferred, split_seen, split_unseen, rng)

        print(f"===== {split}: 13 seen + {len(split_unseen)} unseen =====")
        if dropped:
            print(f"  (偏好类中不在该划分seen集而被替换的: {dropped})")
        for c in chosen:
            print(f"  seen   label={c:2d}  {names[c]}")
        for c in split_unseen:
            print(f"  unseen label={c:2d}  {names[c]}")

        class_dir = ROOT / f"data/class_selection_elderly/{split}"
        class_dir.mkdir(parents=True, exist_ok=True)
        (class_dir / "metadata.json").write_text(json.dumps({
            "description": f"elderly-scenario subset for {split}: 13 seen + official unseen",
            "seed": args.seed,
            "seen_classes": chosen,
            "unseen_classes": sorted(split_unseen),
        }, indent=2))
        print(f"  metadata -> {class_dir / 'metadata.json'}\n")

        if args.dry_run:
            continue

        ckpt = latest_checkpoint(split)
        out_dir = ckpt.parent
        m_u = run_test(split, "gzsl_unseen", ckpt, class_dir, args.scale,
                       out_dir / f"elderly_gzsl_unseen_scale{args.scale}.json")
        m_s = run_test(split, "gzsl_seen", ckpt, class_dir, args.scale,
                       out_dir / f"elderly_gzsl_seen_scale{args.scale}.json")
        S, U = m_s["top1_acc"], m_u["top1_acc"]
        H = 2 * S * U / (S + U) if (S + U) > 0 else 0.0
        results.append((split, S, U, H, m_u.get("top5_acc")))

    if results:
        print(f"\n{'split':8s} {'scale':>6s} {'S top1':>8s} {'U top1':>8s} {'H':>8s} {'U top5':>8s}")
        print("-" * 52)
        for split, S, U, H, u5 in results:
            print(f"{split:8s} {args.scale:6.2f} {S*100:7.2f}% {U*100:7.2f}% {H*100:7.2f}% "
                  f"{(u5*100 if u5 is not None else float('nan')):7.2f}%")


if __name__ == "__main__":
    main()
