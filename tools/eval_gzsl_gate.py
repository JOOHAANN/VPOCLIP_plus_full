"""Generalised zero-shot evaluation with a z-score gate in front of the classifier.

The trained model scores every clip against all 55 text prototypes at once, so a
seen class and an unseen class compete on the same scale. Seen classes were
optimised against and win almost every time, which is why plain joint arg-max
recognises hardly any unseen clip.

This compares that baseline against a two-stage decision that leaves the network
untouched:

    e = normalize(encode_visual(...))
    k = argmax over the 50 seen prototypes
    z = (1 - cos(e, C_k) - D_k) / sigma_k
    output = argmax over the 5 unseen prototypes   if z > tau
             k                                     otherwise

so the arg-max never mixes the two label sets: the gate decides which set to use.
tau is calibrated on the validation split (seen classes only, never the test
set); the best-on-test tau is also reported as an upper bound, clearly labelled.

Usage:
  cd /workspace/VPOCLIP_plus
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
    python tools/eval_gzsl_gate.py \
      --config config_final_aug.yaml \
      --checkpoint work_dir/final_aug/run_20260728_225739/last_model.pth \
      --centroids work_dir/final_aug/run_20260728_225739/class_centroids.npz
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as Data

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import l2_normalize
from test import build_text_bank, load_model, load_split_classes, logits_to_cosine
from train import TrimodalContrastiveDataset, get_device, get_path_from_config, load_config, move_batch_to_device


def parse_args():
    parser = argparse.ArgumentParser(description="GZSL with a z-score gate.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--centroids", required=True)
    parser.add_argument("--split-dir", default="./data/contrastive_cs_70_10_20_zsl_50_5")
    parser.add_argument("--class-split-dir", default="./data/contrastive_zsl_splits/50_5")
    parser.add_argument("--val-prefix", default="trimodal_val", help="Seen-only split used to calibrate tau.")
    parser.add_argument("--seen-prefix", default="trimodal_test_seen")
    parser.add_argument("--unseen-prefix", default="trimodal_test")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output", default="work_dir/gzsl_gate_comparison.json")
    return parser.parse_args()


@torch.no_grad()
def encode_split(model, loader, device, use_amp):
    """Cosine scores against every prototype, embeddings and labels."""

    cosines, embeddings, labels = [], [], []
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            features = model.encode_visual(
                batch["video"], batch["pose"], batch["object"], batch["joint_xy"]
            )
            logits = model(batch["video"], batch["pose"], batch["object"], batch["joint_xy"])
        cosines.append(logits_to_cosine(model, logits.float()).cpu())
        embeddings.append(l2_normalize(features.float()).cpu())
        labels.append(batch["label"].cpu())
    return torch.cat(cosines), torch.cat(embeddings), torch.cat(labels)


def zscores(embeddings, predicted, centroids):
    """(d(e, C_k) - D_k) / sigma_k for the class each clip was assigned to.

    Classes without a centroid cannot be scored; they get -inf so the gate keeps
    them rather than rejecting on missing calibration data.
    """

    out = np.full(len(embeddings), -np.inf, dtype=np.float32)
    for position, label in enumerate(predicted.tolist()):
        slot = centroids["index"].get(int(label))
        if slot is None:
            continue
        distance = 1.0 - float(embeddings[position] @ centroids["vectors"][slot])
        out[position] = (distance - centroids["mean"][slot]) / centroids["std"][slot]
    return out


def harmonic(seen_accuracy, unseen_accuracy):
    if seen_accuracy + unseen_accuracy == 0:
        return 0.0
    return 2 * seen_accuracy * unseen_accuracy / (seen_accuracy + unseen_accuracy)


def gated_predictions(scores, tau):
    """Seen label when the gate accepts, best unseen label when it rejects."""

    return np.where(scores["z"] > tau, scores["best_unseen"], scores["best_seen"])


def summarise(seen, unseen, tau=None):
    """Accuracies over the joint label space, plus their harmonic mean."""

    seen_accuracy = float((gated_predictions(seen, tau) == seen["labels"]).mean()) * 100 \
        if tau is not None else float((seen["joint"] == seen["labels"]).mean()) * 100
    unseen_accuracy = float((gated_predictions(unseen, tau) == unseen["labels"]).mean()) * 100 \
        if tau is not None else float((unseen["joint"] == unseen["labels"]).mean()) * 100
    return {
        "gzsl_seen": seen_accuracy,
        "gzsl_unseen": unseen_accuracy,
        "harmonic": harmonic(seen_accuracy, unseen_accuracy),
    }


def main():
    args = parse_args()
    config_path = str(Path(args.config).resolve())
    config = load_config(config_path)
    device = get_device(config["runtime"].get("device"))
    use_amp = bool(config["runtime"].get("amp", False)) and device.type == "cuda"

    stats = np.load(args.centroids)
    centroids = {
        "index": {int(c): i for i, c in enumerate(stats["classes"].tolist())},
        "vectors": torch.as_tensor(stats["centroids"], dtype=torch.float32),
        "mean": stats["mean_distance"],
        "std": stats["std_distance"],
    }

    class_split_dir = get_path_from_config(config_path, args.class_split_dir)
    all_labels = [int(v) for v in load_split_classes(class_split_dir, "all")]
    seen_labels = [int(v) for v in load_split_classes(class_split_dir, "seen")]
    unseen_labels = [int(v) for v in load_split_classes(class_split_dir, "unseen")]

    model = load_model(config, config_path, device, args.checkpoint)
    candidate_labels, _ = build_text_bank(model, config, config_path, all_labels)
    candidate_labels = [int(v) for v in candidate_labels]
    print(f"prototypes: {len(candidate_labels)} = {len(seen_labels)} seen + {len(unseen_labels)} unseen")

    seen_columns = [candidate_labels.index(v) for v in seen_labels]
    unseen_columns = [candidate_labels.index(v) for v in unseen_labels]
    candidate_array = np.array(candidate_labels)

    pose_suffix = config["data"]["train"].get("pose_suffix", "")
    split_dir = get_path_from_config(config_path, args.split_dir)

    splits = {}
    for tag, prefix in (("val", args.val_prefix), ("seen", args.seen_prefix), ("unseen", args.unseen_prefix)):
        dataset = TrimodalContrastiveDataset(split_dir, prefix=prefix, mmap=True, pose_suffix=pose_suffix)
        loader = Data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        cosine, embeddings, labels = encode_split(model, loader, device, use_amp)
        best_seen = candidate_array[np.array(seen_columns)[cosine[:, seen_columns].argmax(dim=1).numpy()]]
        splits[tag] = {
            "labels": labels.numpy(),
            "joint": candidate_array[cosine.argmax(dim=1).numpy()],
            "best_seen": best_seen,
            "best_unseen": candidate_array[np.array(unseen_columns)[cosine[:, unseen_columns].argmax(dim=1).numpy()]],
            "z": zscores(embeddings, best_seen, centroids),
            "cosine": cosine.numpy(),
        }
        print(f"{tag:7s}: {len(dataset):5d} clips")

    results = {}

    # Ceilings: what each group scores when the label set is already known, so
    # the gate has nothing left to get wrong.
    results["ceiling_closed_set"] = {
        "gzsl_seen": float((splits["seen"]["best_seen"] == splits["seen"]["labels"]).mean()) * 100,
        "gzsl_unseen": float((splits["unseen"]["best_unseen"] == splits["unseen"]["labels"]).mean()) * 100,
    }
    results["ceiling_closed_set"]["harmonic"] = harmonic(
        results["ceiling_closed_set"]["gzsl_seen"], results["ceiling_closed_set"]["gzsl_unseen"]
    )

    # Baseline: one arg-max over all 55 prototypes, exactly as test.py does today.
    results["baseline_joint_argmax"] = summarise(splits["seen"], splits["unseen"])

    # The usual GZSL fix: penalise every seen score by a constant before the
    # joint arg-max. Given the best gamma the test set allows, so it is scored
    # against the gate's oracle row rather than its calibrated ones.
    seen_mask = np.isin(candidate_array, seen_labels)
    best_stacking = None
    for gamma in np.linspace(0.0, 0.5, 501):
        entry = {}
        for tag in ("seen", "unseen"):
            adjusted = splits[tag]["cosine"].copy()
            adjusted[:, seen_mask] -= gamma
            predicted = candidate_array[adjusted.argmax(axis=1)]
            entry[f"gzsl_{'seen' if tag == 'seen' else 'unseen'}"] = \
                float((predicted == splits[tag]["labels"]).mean()) * 100
        entry["harmonic"] = harmonic(entry["gzsl_seen"], entry["gzsl_unseen"])
        entry["gamma"] = float(gamma)
        if best_stacking is None or entry["harmonic"] > best_stacking["harmonic"]:
            best_stacking = entry
    results["calibrated_stacking_oracle_gamma"] = best_stacking

    # Honest tau: rejects a fixed fraction of the validation clips, which hold
    # seen classes only, so the test set plays no part in choosing it.
    finite = splits["val"]["z"][np.isfinite(splits["val"]["z"])]
    for reject_rate in (0.01, 0.05, 0.10, 0.20, 0.30):
        tau = float(np.quantile(finite, 1 - reject_rate))
        entry = summarise(splits["seen"], splits["unseen"], tau)
        entry["tau"] = tau
        entry["val_reject_rate"] = reject_rate * 100
        results[f"zscore_gate_val{int(reject_rate * 100)}"] = entry

    # Upper bound: the tau that maximises H on the test set itself. Not a usable
    # operating point, only a ceiling for what the gate could deliver.
    grid = np.unique(np.concatenate([splits["seen"]["z"], splits["unseen"]["z"]]))
    grid = grid[np.isfinite(grid)]
    best = max(
        (summarise(splits["seen"], splits["unseen"], float(t)) | {"tau": float(t)} for t in grid),
        key=lambda entry: entry["harmonic"],
    )
    results["zscore_gate_oracle_tau"] = best

    print(f"\n{'scheme':>28s} {'tau':>8s} {'GZSL-S':>8s} {'GZSL-U':>8s} {'H':>8s}")
    print("-" * 66)
    for name, entry in results.items():
        knob = entry.get("tau", entry.get("gamma"))
        tau = f"{knob:8.3f}" if knob is not None else " " * 8
        print(f"{name:>28s} {tau} {entry['gzsl_seen']:7.2f}% {entry['gzsl_unseen']:7.2f}% "
              f"{entry['harmonic']:7.2f}%")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps({
        "checkpoint": args.checkpoint,
        "centroids": args.centroids,
        "results": results,
    }, indent=2))
    print(f"\nsaved: {args.output}")


if __name__ == "__main__":
    main()
