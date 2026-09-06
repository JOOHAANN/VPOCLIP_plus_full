"""Build cross-subject and zero-shot splits for Toyota Smarthome.

Filenames encode everything needed: Cook.Cut_p02_r01_v14_c06.mp4 is class
Cook.Cut, subject 2, repetition 1, video 14, camera 6.

Subjects are partitioned 70/10/20 as in the ETRI pipeline, so the two datasets
are evaluated under the same protocol. Unseen classes are drawn from those with
enough test clips to measure - the tail of this dataset is very thin (Cutbread
has 22 clips in total), and a five-clip test class would be meaningless.

Usage:
  cd /workspace/VPOCLIP_plus
  python tools/build_smarthome_splits.py
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

NAME = re.compile(r"^(?P<action>.+?)_p(?P<subject>\d+)_r(?P<rep>\d+)_v(?P<video>\d+)_c(?P<cam>\d+)\.mp4$")


def parse_args():
    parser = argparse.ArgumentParser(description="Toyota Smarthome split builder.")
    parser.add_argument("--video-dir", default="data/toyota_smarthome/mp4")
    parser.add_argument("--output-dir", default="data/smarthome_splits")
    parser.add_argument("--num-unseen", type=int, default=5)
    parser.add_argument("--min-clips", type=int, default=100,
                        help="A class must have at least this many clips to be eligible as unseen.")
    parser.add_argument("--seed", type=int, default=20260729)
    return parser.parse_args()


def main():
    args = parse_args()
    clips = []
    for path in sorted(Path(args.video_dir).glob("*.mp4")):
        match = NAME.match(path.name)
        if not match:
            raise ValueError(f"Unparsable filename: {path.name}")
        clips.append({
            "file": path.name,
            "action": match["action"],
            "subject": int(match["subject"]),
            "camera": int(match["cam"]),
        })
    print(f"clips: {len(clips)}")

    actions = sorted({c["action"] for c in clips})
    action_to_label = {a: i for i, a in enumerate(actions)}
    subjects = sorted({c["subject"] for c in clips})
    print(f"classes: {len(actions)}   subjects: {len(subjects)}")

    # 70/10/20 over subjects, matching the ETRI protocol.
    rng = np.random.default_rng(args.seed)
    shuffled = rng.permutation(subjects)
    n_train = round(len(subjects) * 0.7)
    n_val = max(1, round(len(subjects) * 0.1))
    split_subjects = {
        "train": sorted(shuffled[:n_train].tolist()),
        "val": sorted(shuffled[n_train:n_train + n_val].tolist()),
        "test": sorted(shuffled[n_train + n_val:].tolist()),
    }
    for name, group in split_subjects.items():
        print(f"  {name:5s}: {len(group)} subjects {group}")

    subject_split = {s: name for name, group in split_subjects.items() for s in group}
    for clip in clips:
        clip["split"] = subject_split[clip["subject"]]
        clip["label"] = action_to_label[clip["action"]]

    # Unseen classes: sampled from those that keep a usable test set.
    test_counts = Counter(c["action"] for c in clips if c["split"] == "test")
    eligible = sorted(a for a in actions if test_counts[a] >= args.min_clips // 5)
    unseen = sorted(rng.choice(eligible, args.num_unseen, replace=False).tolist())
    seen = [a for a in actions if a not in unseen]
    print(f"\nunseen classes ({len(unseen)}):")
    for a in unseen:
        print(f"  {a:26s} total {sum(1 for c in clips if c['action'] == a):5d}  test {test_counts[a]:4d}")
    print(f"seen classes: {len(seen)}")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "dataset": "toyota_smarthome",
        "actions": actions,
        "action_to_label": action_to_label,
        "seen_classes": sorted(action_to_label[a] for a in seen),
        "unseen_classes": sorted(action_to_label[a] for a in unseen),
        "seen_actions": seen,
        "unseen_actions": unseen,
        "split_subjects": split_subjects,
        "seed": args.seed,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))

    # Manifests: one per split, plus the ZSL train/test views.
    unseen_labels = set(metadata["unseen_classes"])
    groups = defaultdict(list)
    for clip in clips:
        groups[clip["split"]].append(clip)
        if clip["split"] == "train" and clip["label"] not in unseen_labels:
            groups["zsl_train"].append(clip)
        if clip["split"] == "test":
            groups["zsl_test_unseen" if clip["label"] in unseen_labels else "zsl_test_seen"].append(clip)

    print()
    for name, group in sorted(groups.items()):
        np.save(output / f"{name}_sample_names.npy",
                np.array([c["file"] for c in group], dtype=object))
        np.save(output / f"{name}_labels.npy",
                np.array([c["label"] for c in group], dtype=np.int64))
        classes = len({c["label"] for c in group})
        people = len({c["subject"] for c in group})
        print(f"  {name:17s} {len(group):6d} clips  {classes:3d} classes  {people:3d} subjects")
    print(f"\nsaved to {output}")


if __name__ == "__main__":
    main()
