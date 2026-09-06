"""Compare the open-set rejection rules on a trained ETRI checkpoint.

Seen-class clips should be accepted, unseen-class clips rejected, so the three
rules are scored as binary detectors over the pooled test set:

  entropy    softmax entropy over the prototype scores, one global threshold
  prototype  raw cosine to the best prototype, one global threshold
  zscore     (d(e, C_k) - D_k) / sigma_k, one global tau

Only the third normalises by the predicted class's own spread, which is what
lets a single threshold mean the same thing for a tight class and a loose one.
Reported per rule: AUROC, FPR at 95% TPR and the accuracy at the threshold
calibrated for 95% TPR on the seen clips.

Usage:
  cd /workspace/VPOCLIP_plus
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
    python tools/compare_rejection_rules.py \
      --config config_final_aug.yaml --checkpoint <last_model.pth> \
      --centroids <class_centroids.npz>
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as Data

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gating import normalized_entropy
from model import l2_normalize
from test import build_text_bank, load_model, load_split_classes, logits_to_cosine
from train import TrimodalContrastiveDataset, get_device, get_path_from_config, load_config, move_batch_to_device


def parse_args():
    parser = argparse.ArgumentParser(description="Rejection rule comparison.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--centroids", required=True)
    parser.add_argument("--split-dir", default="./data/contrastive_cs_70_10_20_zsl_50_5")
    parser.add_argument("--class-split-dir", default="./data/contrastive_zsl_splits/50_5")
    parser.add_argument("--seen-prefix", default="trimodal_test_seen")
    parser.add_argument("--unseen-prefix", default="trimodal_test")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output", default="work_dir/rejection_rule_comparison.json")
    return parser.parse_args()


@torch.no_grad()
def score_split(model, loader, device, use_amp, centroids, candidate_labels):
    """Per-clip novelty scores under each rule (higher means more novel)."""

    entropies, negative_cosines, zscores = [], [], []
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            features = model.encode_visual(
                batch["video"], batch["pose"], batch["object"], batch["joint_xy"]
            )
            logits = model(batch["video"], batch["pose"], batch["object"], batch["joint_xy"])
        embeddings = l2_normalize(features.float())
        cosine = logits_to_cosine(model, logits.float())

        for row, embedding in zip(cosine, embeddings):
            best = int(torch.argmax(row))
            label = int(candidate_labels[best])
            entropies.append(normalized_entropy(row.cpu(), 0.1))
            # Negated so that, like the others, larger means more novel.
            negative_cosines.append(-float(row[best]))

            if label in centroids["index"]:
                slot = centroids["index"][label]
                distance = 1.0 - float(embedding.cpu() @ centroids["vectors"][slot])
                zscores.append((distance - centroids["mean"][slot]) / centroids["std"][slot])
            else:
                # No centroid for this class: fall back to the plain distance so
                # the clip still gets a score instead of being dropped.
                zscores.append(float("nan"))
    return np.array(entropies), np.array(negative_cosines), np.array(zscores)


def detection_metrics(seen_scores, unseen_scores):
    """AUROC, FPR@95TPR and the accuracy at that operating point."""

    seen = seen_scores[np.isfinite(seen_scores)]
    unseen = unseen_scores[np.isfinite(unseen_scores)]
    if len(seen) == 0 or len(unseen) == 0:
        return {}

    # AUROC via the rank-sum identity; "positive" is an unseen clip.
    combined = np.concatenate([unseen, seen])
    ranks = combined.argsort().argsort() + 1
    positive_rank_sum = ranks[: len(unseen)].sum()
    auroc = (positive_rank_sum - len(unseen) * (len(unseen) + 1) / 2) / (len(unseen) * len(seen))

    # Threshold that keeps 95% of the seen clips accepted.
    threshold = float(np.quantile(seen, 0.95))
    rejected_unseen = float((unseen > threshold).mean())
    accepted_seen = float((seen <= threshold).mean())
    return {
        "auroc": float(auroc) * 100,
        "fpr_at_95tpr": (1 - rejected_unseen) * 100,
        "threshold_at_95": threshold,
        "unseen_rejected": rejected_unseen * 100,
        "seen_accepted": accepted_seen * 100,
        "balanced_accuracy": (rejected_unseen + accepted_seen) / 2 * 100,
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
    print(f"centroids for {len(centroids['index'])} classes")

    candidate_labels = load_split_classes(get_path_from_config(config_path, args.class_split_dir), "all")
    model = load_model(config, config_path, device, args.checkpoint)
    candidate_labels, _texts = build_text_bank(model, config, config_path, candidate_labels)
    candidate_labels = [int(label) for label in candidate_labels]

    pose_suffix = config["data"]["train"].get("pose_suffix", "")
    split_dir = get_path_from_config(config_path, args.split_dir)
    scores = {}
    for tag, prefix in (("seen", args.seen_prefix), ("unseen", args.unseen_prefix)):
        dataset = TrimodalContrastiveDataset(split_dir, prefix=prefix, mmap=True, pose_suffix=pose_suffix)
        loader = Data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        scores[tag] = score_split(model, loader, device, use_amp, centroids, candidate_labels)
        print(f"{tag}: {len(dataset)} clips")

    results = {}
    print(f"\n{'rule':>10s} {'AUROC':>8s} {'FPR@95':>8s} {'unseen rej':>11s} {'seen acc':>9s} {'balanced':>9s}")
    print("-" * 62)
    for index, rule in enumerate(("entropy", "prototype", "zscore")):
        metrics = detection_metrics(scores["seen"][index], scores["unseen"][index])
        results[rule] = metrics
        print(f"{rule:>10s} {metrics['auroc']:7.2f}% {metrics['fpr_at_95tpr']:7.2f}% "
              f"{metrics['unseen_rejected']:10.2f}% {metrics['seen_accepted']:8.2f}% "
              f"{metrics['balanced_accuracy']:8.2f}%")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps({
        "checkpoint": args.checkpoint,
        "centroids": args.centroids,
        "results": results,
    }, indent=2))
    print(f"\nsaved: {args.output}")


if __name__ == "__main__":
    main()
