"""Analyze class-bank and skeleton novelty scores for the GZSL route gate.

This diagnostic never tunes on group1's true unseen labels.  It calibrates each
score on pseudo-unseen classes and reports the held-out group1 test only as a
final diagnostic.  It is intentionally separate from training the deployable
gate so accidental test-set threshold selection is difficult.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
TOOLS = ROOT / "tools"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TOOLS))

from evaluate_gzsl_route_gate import extract_vpo  # noqa: E402
from train import (  # noqa: E402
    get_device,
    get_path_from_config,
    load_config,
    prepare_action_text_bank,
)
from train_skeleton_unknown_gate import (  # noqa: E402
    PSEUDO_UNSEEN,
    TRUE_UNSEEN,
    extract_skeleton_features,
)
from model import build_model_from_config  # noqa: E402


def auc(labels, scores):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    positive = labels == 1
    negative = labels == 0
    if not positive.any() or not negative.any():
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    return float((ranks[positive].sum() - positive.sum() * (positive.sum() + 1) / 2)
                 / (positive.sum() * negative.sum()))


def threshold_by_harmonic(labels, scores):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    grid = np.unique(np.quantile(scores, np.linspace(0.001, 0.999, 999)))
    best = None
    for threshold in grid:
        unknown = scores >= threshold
        tpr = float(unknown[labels == 1].mean())
        tnr = float((~unknown[labels == 0]).mean())
        harmonic = 2 * tpr * tnr / max(tpr + tnr, 1e-12)
        row = {"threshold": float(threshold), "unknown_recall": tpr,
               "known_acceptance": tnr, "harmonic_route": float(harmonic)}
        if best is None or (row["harmonic_route"], row["known_acceptance"]) > (
                best["harmonic_route"], best["known_acceptance"]):
            best = row
    best["auc"] = auc(labels, scores)
    return best


def threshold_for_known_acceptance(labels, scores, acceptance=0.95):
    """Set an unknown-high threshold from known validation clips only."""
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    known_scores = scores[labels == 0]
    # Unknown is score >= threshold; retaining `acceptance` of known clips
    # means the threshold is the (1-acceptance) quantile of known scores.
    threshold = float(np.quantile(known_scores, 1.0 - acceptance))
    unknown = scores >= threshold
    return {
        "threshold": threshold,
        "target_known_acceptance": float(acceptance),
        "pseudo_unknown_recall": float(unknown[labels == 1].mean()),
        "pseudo_known_acceptance": float((~unknown[labels == 0]).mean()),
        "auc": auc(labels, scores),
    }


def bank(cosine, labels_order, classes, class_mean=None, class_std=None):
    indices = [labels_order.index(int(label)) for label in classes]
    values = cosine[:, indices]
    if class_mean is not None:
        values = (values - class_mean[indices]) / class_std[indices]
    return values.max(axis=1)


def row_standardize(cosine):
    """Standardize class scores per clip so text-bank offsets cancel."""
    mean = cosine.mean(axis=1, keepdims=True)
    std = np.maximum(cosine.std(axis=1, keepdims=True), 1e-5)
    return (cosine - mean) / std


def logsumexp_bank(cosine, labels_order, classes):
    indices = [labels_order.index(int(label)) for label in classes]
    values = cosine[:, indices]
    high = values.max(axis=1, keepdims=True)
    return (high[:, 0] + np.log(np.exp(values - high).sum(axis=1))).astype(np.float32)


def run_route(score_seen, score_unseen, threshold,
              seen_seen_pred, seen_unseen_pred,
              unseen_seen_pred, unseen_unseen_pred,
              seen_labels, unseen_labels):
    seen_unknown = score_seen >= threshold
    unseen_unknown = score_unseen >= threshold
    pred_seen = np.where(seen_unknown, seen_unseen_pred, seen_seen_pred)
    pred_unseen = np.where(unseen_unknown, unseen_unseen_pred, unseen_seen_pred)
    seen_accuracy = float((pred_seen == seen_labels).mean())
    unseen_accuracy = float((pred_unseen == unseen_labels).mean())
    harmonic = 2 * seen_accuracy * unseen_accuracy / max(seen_accuracy + unseen_accuracy, 1e-12)
    return {
        "seen_route_acceptance": float((~seen_unknown).mean()),
        "unseen_route_recall": float(unseen_unknown.mean()),
        "routed_seen_accuracy": seen_accuracy,
        "routed_unseen_accuracy": unseen_accuracy,
        "routed_harmonic": float(harmonic),
        "seen_conditional_accuracy": float((seen_seen_pred == seen_labels).mean()),
        "unseen_conditional_accuracy": float((unseen_unseen_pred == unseen_labels).mean()),
    }


def main():
    config_path = str((ROOT / "config_g1_full_stage_b.yaml").resolve())
    checkpoint = ROOT / "work_dir/g1_pseudo_selected_stage_b/last_model.pth"
    data_dir = ROOT / "data/new50_5_group1_zsl"
    output = ROOT / "work_dir/g1_skeleton_unknown_gate/route_candidates.json"
    config = load_config(config_path)
    device = get_device(config.get("runtime", {}).get("device", "cuda:0"))
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    model = build_model_from_config(
        config, device=device,
        download_root=get_path_from_config(config_path, config["model"]["text_encoder"].get("download_root")),
    )
    prepare_action_text_bank(model, config, config_path)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=False)
    model.eval()
    labels_order = [int(v) for v in model.text_label_ids.detach().cpu().tolist()]
    seen = [label for label in labels_order if label not in TRUE_UNSEEN]
    data = {}
    for prefix in ("trimodal_train", "trimodal_val", "trimodal_test_seen", "trimodal_test"):
        print(f"extracting {prefix}", flush=True)
        skeleton, labels = extract_skeleton_features(data_dir, prefix)
        cosine, _, cosine_labels = extract_vpo(model, data_dir, prefix, device, 2048, 8)
        if not np.array_equal(labels, cosine_labels):
            raise RuntimeError(f"label order mismatch in {prefix}")
        data[prefix] = {"skeleton": skeleton, "cosine": cosine, "labels": labels}

    train = data["trimodal_train"]
    val = data["trimodal_val"]
    seen_test = data["trimodal_test_seen"]
    unseen_test = data["trimodal_test"]
    train_mean = np.zeros(len(labels_order), dtype=np.float32)
    train_std = np.ones(len(labels_order), dtype=np.float32)
    for label in labels_order:
        values = train["cosine"][train["labels"] == label, labels_order.index(label)]
        if values.size:
            train_mean[labels_order.index(label)] = values.mean()
            train_std[labels_order.index(label)] = max(values.std(), 1e-3)

    known_pseudo = [label for label in seen if label not in PSEUDO_UNSEEN]
    pseudo_y = np.isin(val["labels"], PSEUDO_UNSEEN).astype(np.int64)
    seen_columns = [labels_order.index(x) for x in seen]
    unseen_columns = [labels_order.index(x) for x in TRUE_UNSEEN]
    seen_seen_pred = np.asarray(seen)[seen_test["cosine"][:, seen_columns].argmax(axis=1)]
    seen_unseen_pred = np.asarray(TRUE_UNSEEN)[seen_test["cosine"][:, unseen_columns].argmax(axis=1)]
    unseen_seen_pred = np.asarray(seen)[unseen_test["cosine"][:, seen_columns].argmax(axis=1)]
    unseen_unseen_pred = np.asarray(TRUE_UNSEEN)[unseen_test["cosine"][:, unseen_columns].argmax(axis=1)]
    # Candidate scores are unknown-high. Pseudo calibration uses only pseudo
    # classes and the remaining actual seen classes; true unseen columns are
    # never used to set a threshold.
    candidates = {}
    for normalized in (False, True):
        pseudo_score = bank(val["cosine"], labels_order, PSEUDO_UNSEEN,
                            train_mean if normalized else None, train_std if normalized else None)
        pseudo_score -= bank(val["cosine"], labels_order, known_pseudo,
                             train_mean if normalized else None, train_std if normalized else None)
        threshold_row = threshold_by_harmonic(pseudo_y, pseudo_score)
        seen_score = bank(seen_test["cosine"], labels_order, TRUE_UNSEEN,
                          train_mean if normalized else None, train_std if normalized else None)
        seen_score -= bank(seen_test["cosine"], labels_order, seen,
                           train_mean if normalized else None, train_std if normalized else None)
        unseen_score = bank(unseen_test["cosine"], labels_order, TRUE_UNSEEN,
                            train_mean if normalized else None, train_std if normalized else None)
        unseen_score -= bank(unseen_test["cosine"], labels_order, seen,
                             train_mean if normalized else None, train_std if normalized else None)
        candidates[f"cosine_bank_delta_normalized_{normalized}"] = {
            "pseudo_calibration": threshold_row,
            "test": run_route(seen_score, unseen_score, threshold_row["threshold"],
                               seen_seen_pred, seen_unseen_pred,
                               unseen_seen_pred, unseen_unseen_pred,
                               seen_test["labels"], unseen_test["labels"]),
        }
        known_calibration = {}
        for acceptance in (0.90, 0.95, 0.98):
            known_threshold = threshold_for_known_acceptance(
                np.zeros(len(val["labels"]), dtype=np.int64),
                bank(val["cosine"], labels_order, TRUE_UNSEEN,
                     train_mean if normalized else None, train_std if normalized else None)
                - bank(val["cosine"], labels_order, seen,
                       train_mean if normalized else None, train_std if normalized else None),
                acceptance,
            )
            # The helper is deliberately fed a known-only calibration vector;
            # attach the same pseudo recall separately for readability.
            known_threshold["pseudo_unknown_recall"] = float(
                (pseudo_score >= known_threshold["threshold"])[pseudo_y == 1].mean()
            )
            known_calibration[str(acceptance)] = {
                "calibration": known_threshold,
                "test": run_route(seen_score, unseen_score, known_threshold["threshold"],
                                   seen_seen_pred, seen_unseen_pred,
                                   unseen_seen_pred, unseen_unseen_pred,
                                   seen_test["labels"], unseen_test["labels"]),
            }
        candidates[f"cosine_bank_delta_normalized_{normalized}"]["known_only_calibration"] = known_calibration

    # Per-clip score normalization removes an additive/multiplicative offset
    # caused by the text-bank geometry.  Unlike train-set class z-scores, this
    # does not require statistics for the held-out unseen classes.
    val_row = row_standardize(val["cosine"])
    seen_row = row_standardize(seen_test["cosine"])
    unseen_row = row_standardize(unseen_test["cosine"])
    pseudo_score = bank(val_row, labels_order, PSEUDO_UNSEEN) - bank(val_row, labels_order, known_pseudo)
    threshold_row = threshold_by_harmonic(pseudo_y, pseudo_score)
    seen_score = bank(seen_row, labels_order, TRUE_UNSEEN) - bank(seen_row, labels_order, seen)
    unseen_score = bank(unseen_row, labels_order, TRUE_UNSEEN) - bank(unseen_row, labels_order, seen)
    candidates["cosine_bank_delta_per_clip_z"] = {
        "pseudo_calibration": threshold_row,
        "test": run_route(seen_score, unseen_score, threshold_row["threshold"],
                           seen_seen_pred, seen_unseen_pred,
                           unseen_seen_pred, unseen_unseen_pred,
                           seen_test["labels"], unseen_test["labels"]),
    }

    # Log-sum-exp over each branch is a smoother alternative to max cosine.
    # Correct for the different number of labels in each branch before
    # calibrating on pseudo classes.
    pseudo_score = (logsumexp_bank(val["cosine"], labels_order, PSEUDO_UNSEEN)
                    - logsumexp_bank(val["cosine"], labels_order, known_pseudo)
                    + np.log(len(known_pseudo) / len(PSEUDO_UNSEEN)))
    threshold_row = threshold_by_harmonic(pseudo_y, pseudo_score)
    seen_score = (logsumexp_bank(seen_test["cosine"], labels_order, TRUE_UNSEEN)
                  - logsumexp_bank(seen_test["cosine"], labels_order, seen)
                  + np.log(len(seen) / len(TRUE_UNSEEN)))
    unseen_score = (logsumexp_bank(unseen_test["cosine"], labels_order, TRUE_UNSEEN)
                    - logsumexp_bank(unseen_test["cosine"], labels_order, seen)
                    + np.log(len(seen) / len(TRUE_UNSEEN)))
    candidates["cosine_branch_logsumexp"] = {
        "pseudo_calibration": threshold_row,
        "test": run_route(seen_score, unseen_score, threshold_row["threshold"],
                           seen_seen_pred, seen_unseen_pred,
                           unseen_seen_pred, unseen_unseen_pred,
                           seen_test["labels"], unseen_test["labels"]),
    }

    # A class-agnostic energy score over the actual seen bank. It is useful as
    # a baseline, but its threshold is still learned only using pseudo groups.
    pseudo_seen_energy = np.logaddexp.reduce(val["cosine"][:, [labels_order.index(x) for x in known_pseudo]], axis=1)
    seen_energy = np.logaddexp.reduce(seen_test["cosine"][:, [labels_order.index(x) for x in seen]], axis=1)
    unseen_energy = np.logaddexp.reduce(unseen_test["cosine"][:, [labels_order.index(x) for x in seen]], axis=1)
    energy_threshold = threshold_by_harmonic(pseudo_y, -pseudo_seen_energy)
    candidates["negative_seen_energy"] = {
        "pseudo_calibration": energy_threshold,
        "test": run_route(-seen_energy, -unseen_energy, energy_threshold["threshold"],
                           seen_seen_pred, seen_unseen_pred,
                           unseen_seen_pred, unseen_unseen_pred,
                           seen_test["labels"], unseen_test["labels"]),
    }
    known_energy_calibration = {}
    val_seen_energy = np.logaddexp.reduce(val["cosine"][:, [labels_order.index(x) for x in seen]], axis=1)
    for acceptance in (0.90, 0.95, 0.98):
        known_threshold = threshold_for_known_acceptance(
            np.zeros(len(val_seen_energy), dtype=np.int64), -val_seen_energy, acceptance
        )
        known_threshold["pseudo_unknown_recall"] = float(
            ((-pseudo_seen_energy) >= known_threshold["threshold"])[pseudo_y == 1].mean()
        )
        known_energy_calibration[str(acceptance)] = {
            "calibration": known_threshold,
            "test": run_route(-seen_energy, -unseen_energy, known_threshold["threshold"],
                               seen_seen_pred, seen_unseen_pred,
                               unseen_seen_pred, unseen_unseen_pred,
                               seen_test["labels"], unseen_test["labels"]),
        }
    candidates["negative_seen_energy"]["known_only_calibration"] = known_energy_calibration

    # Nearest class-centroid skeleton novelty; all centroids are learned from
    # train clips and pseudo thresholding is performed on val only.
    x_train = train["skeleton"]
    feature_std = np.maximum(x_train.std(axis=0), 1e-3)
    centroids = np.stack([x_train[train["labels"] == label].mean(axis=0) for label in seen])
    centroids = centroids / feature_std

    def skeleton_distance(x):
        z = x / feature_std
        result = []
        for chunk in np.array_split(z, max(1, int(np.ceil(len(z) / 512)))):
            distances = ((chunk[:, None, :] - centroids[None, :, :]) ** 2).mean(axis=2)
            result.append(np.sqrt(distances.min(axis=1)))
        return np.concatenate(result)

    pseudo_distance = skeleton_distance(val["skeleton"])
    skeleton_threshold = threshold_by_harmonic(pseudo_y, pseudo_distance)
    candidates["skeleton_nearest_centroid"] = {
        "pseudo_calibration": skeleton_threshold,
        "test": run_route(skeleton_distance(seen_test["skeleton"]), skeleton_distance(unseen_test["skeleton"]),
                           skeleton_threshold["threshold"],
                           seen_seen_pred, seen_unseen_pred,
                           unseen_seen_pred, unseen_unseen_pred,
                           seen_test["labels"], unseen_test["labels"]),
    }
    skeleton_known_calibration = {}
    for acceptance in (0.90, 0.95, 0.98):
        known_threshold = threshold_for_known_acceptance(
            np.zeros(len(val["labels"]), dtype=np.int64),
            skeleton_distance(val["skeleton"]), acceptance,
        )
        known_threshold["pseudo_unknown_recall"] = float(
            (pseudo_distance >= known_threshold["threshold"])[pseudo_y == 1].mean()
        )
        skeleton_known_calibration[str(acceptance)] = {
            "calibration": known_threshold,
            "test": run_route(skeleton_distance(seen_test["skeleton"]), skeleton_distance(unseen_test["skeleton"]),
                               known_threshold["threshold"],
                               seen_seen_pred, seen_unseen_pred,
                               unseen_seen_pred, unseen_unseen_pred,
                               seen_test["labels"], unseen_test["labels"]),
        }
    candidates["skeleton_nearest_centroid"]["known_only_calibration"] = skeleton_known_calibration

    result = {
        "checkpoint": str(checkpoint.resolve()),
        "pseudo_unknown": PSEUDO_UNSEEN,
        "true_unseen_held_out": TRUE_UNSEEN,
        "candidates": candidates,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=True))
    print(json.dumps(result, indent=2, allow_nan=True), flush=True)
    print(f"saved: {output}", flush=True)


if __name__ == "__main__":
    main()
