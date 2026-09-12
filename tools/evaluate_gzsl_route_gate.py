"""Evaluate a GZSL route gate and the routed action accuracy.

This is distinct from a standalone unknown detector.  For every clip the gate
chooses a label bank first (50 seen or 5 unseen prototypes), then the VPOCLIP
argmax is taken only inside that bank.  Thresholds are calibrated on the two
pseudo-unseen groups; group1's real unseen classes are read only for the final
report.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as Data

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model import build_model_from_config, l2_normalize  # noqa: E402
from train import (  # noqa: E402
    TrimodalContrastiveDataset,
    get_device,
    get_path_from_config,
    load_config,
    move_batch_to_device,
    prepare_action_text_bank,
)
from train_skeleton_unknown_gate import (  # noqa: E402
    PSEUDO_UNSEEN,
    TRUE_UNSEEN,
    UnknownGateMLP,
    extract_skeleton_features,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gate-bundle", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


@torch.inference_mode()
def extract_vpo(model, data_dir, prefix, device, batch_size, workers):
    dataset = TrimodalContrastiveDataset(data_dir, prefix=prefix, mmap=True, pose_suffix="_coco17")
    kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": workers,
        "persistent_workers": False,
        "pin_memory": device.type == "cuda",
    }
    if workers > 0:
        kwargs["prefetch_factor"] = 2
    loader = Data.DataLoader(dataset, **kwargs)
    use_amp = device.type == "cuda"
    text = l2_normalize(model.text_features.float())
    all_cosines, all_labels = [], []
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            visual = l2_normalize(
                model.encode_visual(batch["video"], batch["pose"], batch["object"], batch["joint_xy"]).float()
            )
            cosine = visual @ text.t()
        all_cosines.append(cosine.float().cpu())
        all_labels.append(batch["label"].cpu())
    cosine = torch.cat(all_cosines).numpy().astype(np.float32, copy=False)
    labels = torch.cat(all_labels).numpy().astype(np.int64, copy=False)
    scale = float(model.logit_scale.detach().float().exp().clamp(max=100).cpu())
    probability = torch.softmax(torch.from_numpy(cosine / max(scale, 1e-6) / 0.1), dim=1).numpy()
    sorted_cos = np.sort(cosine, axis=1)
    entropy = -(probability * np.log(np.maximum(probability, 1e-8))).sum(axis=1)
    summary = np.stack(
        [
            cosine.max(axis=1), cosine.mean(axis=1), cosine.std(axis=1),
            sorted_cos[:, -1] - sorted_cos[:, -2],
            sorted_cos[:, -1] - sorted_cos[:, -5], entropy,
            probability.max(axis=1),
        ], axis=1,
    ).astype(np.float32)
    return cosine, summary, labels


def auc(labels, scores):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    positive = labels == 1
    negative = labels == 0
    if not positive.any() or not negative.any():
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    return float((ranks[positive].sum() - positive.sum() * (positive.sum() + 1) / 2) /
                 (positive.sum() * negative.sum()))


def route_stats(labels, unknown_mask, best_seen, best_unseen, true_seen, true_unseen):
    labels = np.asarray(labels)
    unknown_mask = np.asarray(unknown_mask, dtype=bool)
    prediction = np.where(unknown_mask, best_unseen, best_seen)
    seen_mask = np.isin(labels, true_seen)
    unseen_mask = np.isin(labels, true_unseen)
    seen_route = (~unknown_mask[seen_mask]).mean() if seen_mask.any() else float("nan")
    unseen_route = unknown_mask[unseen_mask].mean() if unseen_mask.any() else float("nan")
    seen_conditional = (best_seen[seen_mask] == labels[seen_mask]).mean() if seen_mask.any() else float("nan")
    unseen_conditional = (best_unseen[unseen_mask] == labels[unseen_mask]).mean() if unseen_mask.any() else float("nan")
    seen_routed = (prediction[seen_mask] == labels[seen_mask]).mean() if seen_mask.any() else float("nan")
    unseen_routed = (prediction[unseen_mask] == labels[unseen_mask]).mean() if unseen_mask.any() else float("nan")
    harmonic = 2 * seen_routed * unseen_routed / max(seen_routed + unseen_routed, 1e-12)
    return {
        "num_seen": int(seen_mask.sum()),
        "num_unseen": int(unseen_mask.sum()),
        "seen_route_acceptance": float(seen_route),
        "unseen_route_recall": float(unseen_route),
        "conditional_seen_accuracy": float(seen_conditional),
        "conditional_unseen_accuracy": float(unseen_conditional),
        "routed_seen_accuracy": float(seen_routed),
        "routed_unseen_accuracy": float(unseen_routed),
        "routed_harmonic": float(harmonic),
    }


def best_seen_unseen(cosine, labels_order, seen, unseen):
    seen_columns = [labels_order.index(int(label)) for label in seen]
    unseen_columns = [labels_order.index(int(label)) for label in unseen]
    labels_array = np.asarray(labels_order)
    return (
        labels_array[np.asarray(seen_columns)[cosine[:, seen_columns].argmax(axis=1)]],
        labels_array[np.asarray(unseen_columns)[cosine[:, unseen_columns].argmax(axis=1)]],
    )


def choose_threshold(scores, labels):
    labels = np.asarray(labels, dtype=np.int64)
    grid = np.unique(np.quantile(scores, np.linspace(0.001, 0.999, 999)))
    best = None
    for threshold in grid:
        predicted = scores >= threshold
        tpr = predicted[labels == 1].mean()
        tnr = (~predicted[labels == 0]).mean()
        h = 2 * tpr * tnr / max(tpr + tnr, 1e-12)
        row = {"threshold": float(threshold), "unknown_recall": float(tpr),
               "known_acceptance": float(tnr), "harmonic_route": float(h)}
        if best is None or (row["harmonic_route"], row["known_acceptance"]) > (best["harmonic_route"], best["known_acceptance"]):
            best = row
    best["auc"] = auc(labels, scores)
    return best


def main():
    args = parse_args()
    config_path = str(Path(args.config).resolve())
    config = load_config(config_path)
    device = get_device(config.get("runtime", {}).get("device", "cuda:0"))
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    gate_bundle = torch.load(args.gate_bundle, map_location="cpu", weights_only=False)
    gate = UnknownGateMLP(int(gate_bundle["input_dim"])).to(device)
    gate.load_state_dict(gate_bundle["state_dict"])
    gate.eval()
    mean = gate_bundle["mean"].numpy().astype(np.float32)
    std = gate_bundle["std"].numpy().astype(np.float32)
    stored_threshold = float(gate_bundle["threshold"])

    model = build_model_from_config(
        config,
        device=device,
        download_root=get_path_from_config(config_path, config["model"]["text_encoder"].get("download_root")),
    )
    prepare_action_text_bank(model, config, config_path)
    try:
        state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()
    labels_order = [int(v) for v in model.text_label_ids.detach().cpu().tolist()]
    seen = [v for v in labels_order if v not in TRUE_UNSEEN]
    unseen = list(TRUE_UNSEEN)

    split_data = {}
    for prefix in ("trimodal_val", "trimodal_test_seen", "trimodal_test"):
        print(f"extracting {prefix}", flush=True)
        skeleton, labels = extract_skeleton_features(args.data_dir, prefix)
        cosine, vpo_summary, vpo_labels = extract_vpo(model, args.data_dir, prefix, device, args.batch_size, args.workers)
        if not np.array_equal(labels, vpo_labels):
            raise RuntimeError(f"label order mismatch for {prefix}")
        x = np.concatenate((skeleton, vpo_summary), axis=1)
        x = np.clip((x - mean) / np.maximum(std, 1e-5), -12.0, 12.0).astype(np.float32)
        with torch.inference_mode():
            probability = []
            for start in range(0, len(x), 8192):
                probability.append(torch.sigmoid(gate(torch.from_numpy(x[start:start + 8192]).to(device))).cpu())
        probability = torch.cat(probability).numpy()
        best_seen, best_unseen = best_seen_unseen(cosine, labels_order, seen, unseen)
        split_data[prefix] = {"labels": labels, "cosine": cosine, "probability": probability,
                              "best_seen": best_seen, "best_unseen": best_unseen}

    val = split_data["trimodal_val"]
    pseudo_label = np.isin(val["labels"], PSEUDO_UNSEEN).astype(np.int64)
    # Stored threshold was selected on the same pseudo validation protocol by
    # train_skeleton_unknown_gate.py; recompute it as a consistency check.
    recalibrated = choose_threshold(val["probability"], pseudo_label)
    thresholds = {
        "stored_gate_threshold": stored_threshold,
        "recalibrated_pseudo_val": recalibrated,
    }
    test_seen = split_data["trimodal_test_seen"]
    test_unseen = split_data["trimodal_test"]
    combined_labels = np.concatenate((test_seen["labels"], test_unseen["labels"]))
    combined_probability = np.concatenate((test_seen["probability"], test_unseen["probability"]))
    combined_best_seen = np.concatenate((test_seen["best_seen"], test_unseen["best_seen"]))
    combined_best_unseen = np.concatenate((test_seen["best_unseen"], test_unseen["best_unseen"]))

    results = {"calibration": thresholds, "conditional_ceiling": {
        "seen": float((test_seen["best_seen"] == test_seen["labels"]).mean()),
        "unseen": float((test_unseen["best_unseen"] == test_unseen["labels"]).mean()),
    }}
    for name, threshold in (("stored_gate", stored_threshold), ("recalibrated_gate", recalibrated["threshold"])):
        results[name] = {
            "validation_pseudo": route_stats(
                val["labels"], val["probability"] >= threshold, val["best_seen"], val["best_unseen"],
                [label for label in labels_order if label not in PSEUDO_UNSEEN], PSEUDO_UNSEEN,
            ),
            "test_seen_and_unseen": {
                "seen": route_stats(test_seen["labels"], test_seen["probability"] >= threshold,
                                     test_seen["best_seen"], test_seen["best_unseen"], seen, unseen),
                "unseen": route_stats(test_unseen["labels"], test_unseen["probability"] >= threshold,
                                       test_unseen["best_seen"], test_unseen["best_unseen"], seen, unseen),
            },
            "combined_gzsl": route_stats(
                combined_labels, combined_probability >= threshold,
                combined_best_seen, combined_best_unseen, seen, unseen,
            ),
        }

    # A lightweight confidence-only route is included as a diagnostic.  Its
    # threshold is also learned on pseudo classes, never on true unseen clips.
    val_seen_cols = [labels_order.index(label) for label in labels_order if label not in PSEUDO_UNSEEN]
    val_pseudo_cols = [labels_order.index(label) for label in PSEUDO_UNSEEN]
    delta_val = val["cosine"][:, val_pseudo_cols].max(axis=1) - val["cosine"][:, val_seen_cols].max(axis=1)
    delta_threshold = choose_threshold(delta_val, pseudo_label)
    for prefix in ("trimodal_test_seen", "trimodal_test"):
        row = split_data[prefix]
        delta = row["cosine"][:, [labels_order.index(label) for label in PSEUDO_UNSEEN]].max(axis=1) - row["cosine"][:, [labels_order.index(label) for label in labels_order if label not in PSEUDO_UNSEEN]].max(axis=1)
        # `delta` is only a route diagnostic; actual GZSL uses the true 5-class
        # bank, so apply its positive score as the unknown branch.
        row.setdefault("delta", delta)
    results["confidence_delta_route"] = {
        "pseudo_calibration": delta_threshold,
        "test_seen_and_unseen": {
            "seen": route_stats(test_seen["labels"], test_seen["delta"] >= delta_threshold["threshold"],
                                 test_seen["best_seen"], test_seen["best_unseen"], seen, unseen),
            "unseen": route_stats(test_unseen["labels"], test_unseen["delta"] >= delta_threshold["threshold"],
                                   test_unseen["best_seen"], test_unseen["best_unseen"], seen, unseen),
        },
    }
    combined_delta = np.concatenate((test_seen["delta"], test_unseen["delta"]))
    results["confidence_delta_route"]["combined_gzsl"] = route_stats(
        combined_labels, combined_delta >= delta_threshold["threshold"],
        combined_best_seen, combined_best_unseen, seen, unseen,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"checkpoint": str(Path(args.checkpoint).resolve()),
                                  "gate_bundle": str(Path(args.gate_bundle).resolve()),
                                  "true_unseen_held_out": TRUE_UNSEEN,
                                  "pseudo_unknown": PSEUDO_UNSEEN,
                                  **results}, indent=2))
    print(json.dumps(results, indent=2), flush=True)
    print(f"saved: {output}", flush=True)


if __name__ == "__main__":
    main()
