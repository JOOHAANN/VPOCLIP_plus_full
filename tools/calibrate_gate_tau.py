"""Pick the z-score gate's tau without ever looking at the test set.

tau cannot be read off a table: z depends on how tightly the run happened to
pack its classes, and on this embedding space it lands around 10-170 rather
than the textbook 2-3. What does transfer between runs is the *rejection rate*
- the fraction of familiar clips the gate is willing to sacrifice - so that is
what gets calibrated here, exactly as the pseudo-unseen protocol transfers an
epoch number rather than a weight.

Stage 1 runs on the four augg models. Each was trained with five of the fifty
classes held out, so for that model those five are genuinely unseen and an
honest H can be measured. Sweeping the rejection rate gives the r that maximises
H, once per group.

Stage 2 takes the median r over the groups and turns it into a tau for the
final_aug run through its own validation quantile, then reports what that costs
on the real test set. The test set is read once, at the end, to report a number
- never to choose one.

Usage:
  cd /workspace/VPOCLIP_plus
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 \
    python tools/calibrate_gate_tau.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as Data
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import l2_normalize
from test import build_text_bank, load_model, load_split_classes, logits_to_cosine
from train import TrimodalContrastiveDataset, get_device, get_path_from_config, load_config, move_batch_to_device

RATE_GRID = np.concatenate([np.arange(0.005, 0.10, 0.005), np.arange(0.10, 0.61, 0.01)])


def parse_args():
    parser = argparse.ArgumentParser(description="Honest tau calibration for the z-score gate.")
    parser.add_argument("--groups", nargs="+", default=["augg1", "augg2", "augg3", "augg4"])
    parser.add_argument("--final-config", default="config_final_aug.yaml")
    parser.add_argument("--final-checkpoint",
                        default="work_dir/final_aug/run_20260728_225739/last_model.pth")
    parser.add_argument("--final-centroids",
                        default="work_dir/final_aug/run_20260728_225739/class_centroids.npz")
    parser.add_argument("--split-dir", default="./data/contrastive_cs_70_10_20_zsl_50_5")
    parser.add_argument("--class-split-dir", default="./data/contrastive_zsl_splits/50_5")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output", default="work_dir/gate_tau_calibration.json")
    return parser.parse_args()


@torch.no_grad()
def encode_split(model, split_dir, prefix, pose_suffix, device, use_amp, batch_size):
    dataset = TrimodalContrastiveDataset(split_dir, prefix=prefix, mmap=True, pose_suffix=pose_suffix)
    loader = Data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
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
    return torch.cat(cosines).numpy(), torch.cat(embeddings).numpy(), torch.cat(labels).numpy()


def class_statistics(embeddings, labels, classes):
    """Centroid, mean distance and spread per class, with the usual sigma floor."""

    vectors, means, deviations = [], [], []
    for label in classes:
        member = embeddings[labels == label]
        centroid = member.mean(axis=0)
        centroid /= max(np.linalg.norm(centroid), 1e-12)
        distances = 1.0 - member @ centroid
        vectors.append(centroid)
        means.append(distances.mean())
        deviations.append(distances.std())
    deviations = np.array(deviations, dtype=np.float32)
    positive = deviations[deviations > 0]
    floor = float(np.median(positive)) * 0.1 if len(positive) else 1e-3
    return {
        "index": {int(c): i for i, c in enumerate(classes)},
        "vectors": np.stack(vectors),
        "mean": np.array(means, dtype=np.float32),
        "std": np.maximum(deviations, floor),
    }


def zscores(embeddings, predicted, stats):
    out = np.full(len(embeddings), -np.inf, dtype=np.float32)
    for position, label in enumerate(predicted.tolist()):
        slot = stats["index"].get(int(label))
        if slot is not None:
            distance = 1.0 - float(embeddings[position] @ stats["vectors"][slot])
            out[position] = (distance - stats["mean"][slot]) / stats["std"][slot]
    return out


def harmonic(seen_accuracy, unseen_accuracy):
    if seen_accuracy + unseen_accuracy == 0:
        return 0.0
    return 2 * seen_accuracy * unseen_accuracy / (seen_accuracy + unseen_accuracy)


def gate_scores(pool, tau):
    predicted = np.where(pool["z"] > tau, pool["best_unseen"], pool["best_seen"])
    return float((predicted == pool["labels"]).mean()) * 100


def sweep(seen_pool, unseen_pool, reference_z, rates=RATE_GRID):
    """H at each rejection rate; tau comes from a quantile of the seen reference."""

    finite = reference_z[np.isfinite(reference_z)]
    curve = []
    for rate in rates:
        tau = float(np.quantile(finite, 1 - rate))
        seen_accuracy = gate_scores(seen_pool, tau)
        unseen_accuracy = gate_scores(unseen_pool, tau)
        curve.append({
            "rate": float(rate),
            "tau": tau,
            "gzsl_seen": seen_accuracy,
            "gzsl_unseen": unseen_accuracy,
            "harmonic": harmonic(seen_accuracy, unseen_accuracy),
        })
    return curve


def make_pool(cosine, embeddings, labels, mask, seen_columns, unseen_columns, candidate_array, stats):
    best_seen = candidate_array[np.array(seen_columns)[cosine[mask][:, seen_columns].argmax(axis=1)]]
    return {
        "labels": labels[mask],
        "best_seen": best_seen,
        "best_unseen": candidate_array[np.array(unseen_columns)[cosine[mask][:, unseen_columns].argmax(axis=1)]],
        "z": zscores(embeddings[mask], best_seen, stats),
    }


def calibrate_group(name, args, device):
    """Best rejection rate for one pseudo-unseen model."""

    config_path = str(Path(f"config_{name}.yaml").resolve())
    config = load_config(config_path)
    with open(config_path) as handle:
        pseudo = [int(v) for v in yaml.safe_load(handle)["data"]["train"]["pseudo_unseen"]]

    checkpoint = sorted(Path(f"work_dir/{name}").glob("run_*/last_model.pth"))[-1]
    use_amp = bool(config["runtime"].get("amp", False)) and device.type == "cuda"

    class_split_dir = get_path_from_config(config_path, args.class_split_dir)
    seen50 = [int(v) for v in load_split_classes(class_split_dir, "seen")]
    seen45 = [c for c in seen50 if c not in pseudo]

    model = load_model(config, config_path, device, str(checkpoint))
    candidate_labels, _ = build_text_bank(model, config, config_path, seen50)
    candidate_labels = [int(v) for v in candidate_labels]
    candidate_array = np.array(candidate_labels)
    seen_columns = [candidate_labels.index(v) for v in seen45]
    unseen_columns = [candidate_labels.index(v) for v in pseudo]

    pose_suffix = config["data"]["train"].get("pose_suffix", "")
    split_dir = get_path_from_config(config_path, args.split_dir)
    encoded = {
        prefix: encode_split(model, split_dir, prefix, pose_suffix, device, use_amp, args.batch_size)
        for prefix in ("trimodal_train", "trimodal_val")
    }

    train_cosine, train_embeddings, train_labels = encoded["trimodal_train"]
    val_cosine, val_embeddings, val_labels = encoded["trimodal_val"]
    stats = class_statistics(train_embeddings, train_labels, seen45)

    # Familiar clips come from the held-out validation subjects; the pseudo-unseen
    # pool takes every clip of the five withheld classes, from both files.
    seen_pool = make_pool(val_cosine, val_embeddings, val_labels, np.isin(val_labels, seen45),
                          seen_columns, unseen_columns, candidate_array, stats)
    unseen_cosine = np.concatenate([train_cosine, val_cosine])
    unseen_embeddings = np.concatenate([train_embeddings, val_embeddings])
    unseen_labels = np.concatenate([train_labels, val_labels])
    unseen_pool = make_pool(unseen_cosine, unseen_embeddings, unseen_labels,
                            np.isin(unseen_labels, pseudo),
                            seen_columns, unseen_columns, candidate_array, stats)

    curve = sweep(seen_pool, unseen_pool, seen_pool["z"])
    best = max(curve, key=lambda entry: entry["harmonic"])
    print(f"{name}: pseudo-unseen {pseudo}  seen {len(seen_pool['labels'])} / "
          f"unseen {len(unseen_pool['labels'])} clips  ->  best rate {best['rate']*100:.1f}% "
          f"(tau {best['tau']:.2f}, H {best['harmonic']:.2f}%)")
    del model
    torch.cuda.empty_cache()
    return {"pseudo_unseen": pseudo, "best": best, "curve": curve}


def main():
    args = parse_args()
    device = get_device(None)

    print("=== stage 1: rejection rate on the pseudo-unseen models ===")
    groups = {name: calibrate_group(name, args, device) for name in args.groups}
    rates = [groups[name]["best"]["rate"] for name in args.groups]
    chosen_rate = float(np.median(rates))
    print(f"\nrates {[f'{r*100:.1f}%' for r in rates]}  ->  median {chosen_rate*100:.1f}%")

    print("\n=== stage 2: apply that rate to final_aug ===")
    config_path = str(Path(args.final_config).resolve())
    config = load_config(config_path)
    use_amp = bool(config["runtime"].get("amp", False)) and device.type == "cuda"

    stored = np.load(args.final_centroids)
    stats = {
        "index": {int(c): i for i, c in enumerate(stored["classes"].tolist())},
        "vectors": stored["centroids"],
        "mean": stored["mean_distance"],
        "std": stored["std_distance"],
    }

    class_split_dir = get_path_from_config(config_path, args.class_split_dir)
    all_labels = [int(v) for v in load_split_classes(class_split_dir, "all")]
    seen_labels = [int(v) for v in load_split_classes(class_split_dir, "seen")]
    unseen_labels = [int(v) for v in load_split_classes(class_split_dir, "unseen")]

    model = load_model(config, config_path, device, args.final_checkpoint)
    candidate_labels, _ = build_text_bank(model, config, config_path, all_labels)
    candidate_labels = [int(v) for v in candidate_labels]
    candidate_array = np.array(candidate_labels)
    seen_columns = [candidate_labels.index(v) for v in seen_labels]
    unseen_columns = [candidate_labels.index(v) for v in unseen_labels]

    pose_suffix = config["data"]["train"].get("pose_suffix", "")
    split_dir = get_path_from_config(config_path, args.split_dir)
    pools = {}
    for tag, prefix in (("val", "trimodal_val"), ("seen", "trimodal_test_seen"), ("unseen", "trimodal_test")):
        cosine, embeddings, labels = encode_split(
            model, split_dir, prefix, pose_suffix, device, use_amp, args.batch_size
        )
        pools[tag] = make_pool(cosine, embeddings, labels, np.ones(len(labels), dtype=bool),
                               seen_columns, unseen_columns, candidate_array, stats)

    curve = sweep(pools["seen"], pools["unseen"], pools["val"]["z"])
    lookup = {round(entry["rate"], 4): entry for entry in curve}
    transferred = lookup[round(min(RATE_GRID, key=lambda r: abs(r - chosen_rate)), 4)]
    oracle = max(curve, key=lambda entry: entry["harmonic"])

    print(f"\n{'source of tau':>34s} {'rate':>7s} {'tau':>8s} {'GZSL-S':>8s} {'GZSL-U':>8s} {'H':>8s}")
    print("-" * 76)
    for title, entry in (("transferred from pseudo-unseen", transferred), ("oracle on test (upper bound)", oracle)):
        print(f"{title:>34s} {entry['rate']*100:6.1f}% {entry['tau']:8.2f} {entry['gzsl_seen']:7.2f}% "
              f"{entry['gzsl_unseen']:7.2f}% {entry['harmonic']:7.2f}%")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps({
        "groups": {name: {"pseudo_unseen": value["pseudo_unseen"], "best": value["best"]}
                   for name, value in groups.items()},
        "chosen_rate": chosen_rate,
        "transferred": transferred,
        "oracle": oracle,
        "final_curve": curve,
    }, indent=2))
    print(f"\nsaved: {args.output}")


if __name__ == "__main__":
    main()
