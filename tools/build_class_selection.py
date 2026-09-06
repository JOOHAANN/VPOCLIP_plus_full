"""Compile class_selection.yaml into a metadata.json that test.py understands.

Each class in class_selection.yaml is one of:
    seen   - goes into metadata seen_classes
    unseen - goes into metadata unseen_classes
    off    - excluded from both (never a sample, never a candidate)

(true/false are accepted as aliases of seen/off for backward compatibility.)

Then point test.py at the output dir, e.g. standard GZSL protocol:
    S: --class-split-dir data/class_selection --sample-scope seen   --candidate-scope all
    U: --class-split-dir data/class_selection --sample-scope unseen --candidate-scope all
"""
import argparse
import json
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
VALID = {"seen", "unseen", "off"}


def normalize(value):
    if value is True:
        return "seen"
    if value is False:
        return "off"
    value = str(value).strip().lower()
    if value not in VALID:
        raise ValueError(f"class state must be seen/unseen/off, got: {value!r}")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", default=str(ROOT / "class_selection.yaml"))
    parser.add_argument("--output-dir", default=str(ROOT / "data/class_selection"))
    args = parser.parse_args()

    with open(args.selection, "r", encoding="utf-8") as f:
        selection = yaml.safe_load(f)["classes"]

    groups = {"seen": [], "unseen": [], "off": []}
    for code, state in selection.items():
        m = re.fullmatch(r"A(\d+)", str(code))
        if m is None:
            raise ValueError(f"Bad class code: {code} (expected A001..A055)")
        label = int(m.group(1)) - 1  # xlsx ID=1..55 -> dataset label 0..54
        if not 0 <= label <= 54:
            raise ValueError(f"{code}: label {label} out of range 0..54")
        groups[normalize(state)].append(label)

    total = sum(len(v) for v in groups.values())
    if total != 55:
        raise ValueError(f"Expected 55 classes, got {total}")
    if not groups["seen"] and not groups["unseen"]:
        raise ValueError("All classes are off — nothing to test.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "description": "Custom class roles compiled from class_selection.yaml",
        "source": str(args.selection),
        "seen_classes": sorted(groups["seen"]),
        "unseen_classes": sorted(groups["unseen"]),
        "off_classes": sorted(groups["off"]),
        "seen_class_count": len(groups["seen"]),
        "unseen_class_count": len(groups["unseen"]),
        "off_class_count": len(groups["off"]),
    }
    out_path = out_dir / "metadata.json"
    out_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"seen={len(groups['seen'])}  unseen={len(groups['unseen'])}  off={len(groups['off'])}  -> {out_path}")
    if groups["unseen"]:
        print("unseen labels:", sorted(groups["unseen"]))
    if groups["off"]:
        print("off labels:", sorted(groups["off"]))


if __name__ == "__main__":
    main()
