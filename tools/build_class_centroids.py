"""Compute per-class centroid statistics for the z-score rejection rule.

The current rule compares a raw cosine against one global threshold, which
assumes every class is equally tight around its prototype. They are not: a
visually consistent class like "reading a book" clusters far more tightly than
"walking", so a single cutoff is simultaneously too strict for one and too
loose for the other.

For each class k this stores three numbers computed on the training split:

  C_k     the centroid, i.e. the mean of the L2-normalised visual embeddings
  D_k     the mean distance of that class's samples to C_k
  sigma_k the standard deviation of those distances

At runtime only d(e, C_k) is new, and the decision becomes

  z_k = (d(e, C_k) - D_k) / sigma_k > tau  ->  unknown

which puts every class on the same scale, so tau can be one global constant.

Usage:
  cd /workspace/VPOCLIP_plus
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
    python tools/build_class_centroids.py \
      --config config_final_aug.yaml \
      --checkpoint work_dir/final_aug/run_*/last_model.pth
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
from test import load_model
from train import TrimodalContrastiveDataset, get_device, get_path_from_config, load_config, move_batch_to_device


def parse_args():
    parser = argparse.ArgumentParser(description="Per-class centroid statistics.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split-dir", default=None, help="Defaults to the config's training split.")
    parser.add_argument("--prefix", default=None)
    parser.add_argument("--pose-suffix", default=None)
    parser.add_argument("--output", default=None, help="Defaults to <run dir>/class_centroids.npz")
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


@torch.no_grad()
def collect_embeddings(model, loader, device, use_amp):
    """L2-normalised visual embeddings and their labels, in loader order."""

    embeddings = []
    labels = []
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            features = model.encode_visual(
                batch["video"], batch["pose"], batch["object"], batch["joint_xy"]
            )
        embeddings.append(l2_normalize(features.float()).cpu())
        labels.append(batch["label"].cpu())
    return torch.cat(embeddings), torch.cat(labels)


def main():
    args = parse_args()
    config_path = str(Path(args.config).resolve())
    config = load_config(config_path)
    device = get_device(config["runtime"].get("device"))

    train_config = config["data"]["train"]
    split_dir = args.split_dir or train_config["data_dir"]
    dataset = TrimodalContrastiveDataset(
        data_dir=get_path_from_config(config_path, split_dir),
        prefix=args.prefix or train_config.get("prefix", "trimodal_train"),
        mmap=True,
        pose_suffix=args.pose_suffix if args.pose_suffix is not None
        else train_config.get("pose_suffix", ""),
    )
    loader = Data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=config["data"]["dataloader"].get("num_workers", 4),
                            pin_memory=True)
    print(f"clips: {len(dataset)}")

    model = load_model(config, config_path, device, args.checkpoint)
    use_amp = bool(config["runtime"].get("amp", False)) and device.type == "cuda"
    embeddings, labels = collect_embeddings(model, loader, device, use_amp)
    print(f"embeddings: {tuple(embeddings.shape)}")

    classes = sorted(set(int(v) for v in labels.tolist()))
    centroids, means, deviations, counts = [], [], [], []
    for label in classes:
        member = embeddings[labels == label]
        centroid = l2_normalize(member.mean(dim=0, keepdim=True))[0]
        # Unit vectors, so the cosine distance is 1 - dot.
        distances = 1.0 - member @ centroid
        centroids.append(centroid)
        means.append(float(distances.mean()))
        # A one-sample class would give sigma = nan; fall back to the pooled value later.
        deviations.append(float(distances.std(unbiased=False)))
        counts.append(int(member.shape[0]))

    deviations = np.array(deviations, dtype=np.float32)
    floor = float(np.median(deviations[deviations > 0])) * 0.1 if (deviations > 0).any() else 1e-3
    tightened = int((deviations < floor).sum())
    if tightened:
        print(f"raised sigma to the {floor:.4f} floor for {tightened} narrow classes")
    deviations = np.maximum(deviations, floor)

    output = args.output
    if output is None:
        output = str(Path(args.checkpoint).parent / "class_centroids.npz")
    np.savez(
        output,
        classes=np.array(classes, dtype=np.int64),
        centroids=torch.stack(centroids).numpy().astype(np.float32),
        mean_distance=np.array(means, dtype=np.float32),
        std_distance=deviations,
        counts=np.array(counts, dtype=np.int64),
    )

    print(f"\n{'class':>6s} {'n':>6s} {'D_k':>8s} {'sigma_k':>8s}")
    order = np.argsort(means)
    for index in list(order[:3]) + list(order[-3:]):
        print(f"{classes[index]:6d} {counts[index]:6d} {means[index]:8.4f} {deviations[index]:8.4f}")
    print(f"\nD_k spans {min(means):.4f} to {max(means):.4f} - the spread a single "
          f"global cosine threshold has to cover")
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
