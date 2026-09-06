import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as Data

from model import apply_action_text_bank, build_model_from_config
from train import (
    TrimodalContrastiveDataset,
    get_device,
    get_path_from_config,
    load_config,
    move_batch_to_device,
    print_device_info,
    progress_bar,
)


def load_split_classes(split_dir, scope):
    metadata_path = Path(split_dir) / "metadata.json"
    if not metadata_path.exists():
        return None

    with open(metadata_path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    if scope == "unseen":
        return metadata["unseen_classes"]
    if scope == "seen":
        return metadata["seen_classes"]
    if scope == "all":
        return sorted(metadata["seen_classes"] + metadata["unseen_classes"])
    raise ValueError(f"Unsupported candidate scope: {scope}")


def load_split_metadata(split_dir):
    metadata_path = Path(split_dir) / "metadata.json"
    if not metadata_path.exists():
        return {}

    with open(metadata_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_latest_run_info(config, config_path):
    output_config = config.get("outputs", {})
    work_dir = output_config.get("work_dir")
    if not work_dir:
        return None

    latest_path = os.path.join(get_path_from_config(config_path, work_dir), "latest_run.json")
    if not os.path.exists(latest_path):
        return None

    with open(latest_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def build_text_bank(model, config, config_path, candidate_labels):
    text_config = config["data"]["text"]
    return apply_action_text_bank(
        model,
        text_config,
        get_path_from_config(config_path, text_config["xlsx"]),
        candidate_labels,
    )


def load_model(config, config_path, device, checkpoint_path):
    model = build_model_from_config(
        config,
        device=device,
        download_root=get_path_from_config(config_path, config["model"]["text_encoder"].get("download_root")),
    )

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    try:
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    # Periodic snapshots omit the frozen CLIP encoder, which build_model already
    # rebuilt; a full checkpoint still has to match exactly.
    is_snapshot = not any(k.startswith("text_encoder") for k in state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or (not is_snapshot and missing):
        raise RuntimeError(f"Checkpoint mismatch: missing={missing[:3]} unexpected={unexpected[:3]}")
    if is_snapshot:
        print(f"Loaded fusion-head snapshot ({len(state)} tensors)")
    model.to(device)
    model.eval()
    return model


def logits_to_cosine(model, logits):
    """Undo the learned temperature to recover cosine similarities in [-1,1]."""

    logit_scale = getattr(model, "logit_scale", None)
    if logit_scale is None:
        raise AttributeError("Model does not expose logit_scale, cannot recover cosine scores.")

    scale = logit_scale.exp().clamp(max=100).to(device=logits.device, dtype=logits.dtype)
    return (logits / scale).clamp(-1.0, 1.0)


def logits_to_unit_cosine_scores(model, logits):
    return (logits_to_cosine(model, logits) + 1.0) * 0.5


def build_label_mask(candidate_labels, selected_labels, device):
    selected = {int(label) for label in selected_labels}
    return torch.as_tensor(
        [int(label) in selected for label in candidate_labels],
        dtype=torch.bool,
        device=device,
    )


def apply_unseen_score_scale(model, logits, candidate_labels, unseen_labels, unseen_score_scale):
    if unseen_score_scale == 1.0 or not unseen_labels:
        return logits, False

    unseen_mask = build_label_mask(candidate_labels, unseen_labels, logits.device)
    if not torch.any(unseen_mask):
        return logits, False

    # Use non-negative confidence-like scores before scaling unseen classes.
    scaled_scores = logits_to_unit_cosine_scores(model, logits)
    scaled_scores[:, unseen_mask] *= unseen_score_scale
    return scaled_scores, True


def evaluate(
    model,
    data_loader,
    candidate_labels,
    device,
    use_amp=False,
    unseen_labels=None,
    unseen_score_scale=1.0,
    entropy_temperature=0.1,
):
    candidate_tensor = torch.as_tensor(candidate_labels, dtype=torch.long, device=device)
    unseen_labels = unseen_labels or []
    total = 0
    correct1 = 0
    correct5 = 0
    loss_sum = 0.0
    entropy_sum = 0.0
    inference_time_seconds = 0.0
    predictions = []
    used_unseen_score_scale = False
    per_class = {
        int(label): {
            "num_samples": 0,
            "top1_correct": 0,
            "top5_correct": 0,
        }
        for label in candidate_labels
    }

    with torch.no_grad():
        for batch in progress_bar(data_loader, "Test"):
            batch = move_batch_to_device(batch, device)

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start_time = time.perf_counter()
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(
                    batch["video"],
                    batch["pose"],
                    batch["object"],
                    batch["joint_xy"],
                )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            inference_time_seconds += time.perf_counter() - start_time

            prediction_scores, batch_used_scale = apply_unseen_score_scale(
                model,
                logits,
                candidate_labels,
                unseen_labels=unseen_labels,
                unseen_score_scale=unseen_score_scale,
            )
            used_unseen_score_scale = used_unseen_score_scale or batch_used_scale

            pred_indices = torch.argmax(prediction_scores, dim=1)
            pred_labels = candidate_tensor[pred_indices]
            labels = batch["label"].long()
            targets = model.build_target_indices(labels)
            loss = F.cross_entropy(logits.float(), targets)
            entropy_logits = logits_to_cosine(model, logits).float() / entropy_temperature
            log_probabilities = F.log_softmax(entropy_logits, dim=1)
            probabilities = log_probabilities.exp()
            batch_entropy = -(probabilities * log_probabilities).sum(dim=1)
            if logits.shape[1] > 1:
                batch_entropy = batch_entropy / np.log(logits.shape[1])
            top1_correct_mask = pred_labels == labels
            correct1 += int(top1_correct_mask.sum().item())

            top5_correct_mask = None
            if prediction_scores.shape[1] >= 5:
                top5_indices = torch.topk(prediction_scores, k=5, dim=1).indices
                top5_labels = candidate_tensor[top5_indices]
                top5_correct_mask = (top5_labels == labels[:, None]).any(dim=1)
                correct5 += int(top5_correct_mask.sum().item())

            batch_size = labels.shape[0]
            total += batch_size
            loss_sum += float(loss.detach().cpu()) * batch_size
            entropy_sum += float(batch_entropy.sum().detach().cpu())
            label_list = labels.detach().cpu().tolist()
            top1_correct_list = top1_correct_mask.detach().cpu().tolist()
            top5_correct_list = (
                top5_correct_mask.detach().cpu().tolist()
                if top5_correct_mask is not None
                else [False] * batch_size
            )
            for label, is_top1_correct, is_top5_correct in zip(label_list, top1_correct_list, top5_correct_list):
                label = int(label)
                if label not in per_class:
                    per_class[label] = {
                        "num_samples": 0,
                        "top1_correct": 0,
                        "top5_correct": 0,
                    }
                per_class[label]["num_samples"] += 1
                per_class[label]["top1_correct"] += int(is_top1_correct)
                per_class[label]["top5_correct"] += int(is_top5_correct)
            predictions.extend(
                {
                    "label": int(label),
                    "pred": int(pred),
                }
                for label, pred in zip(labels.detach().cpu().tolist(), pred_labels.detach().cpu().tolist())
            )

    per_class_accuracy = {}
    for label in sorted(per_class):
        stats = per_class[label]
        num_samples = stats["num_samples"]
        per_class_accuracy[str(label)] = {
            "num_samples": num_samples,
            "top1_correct": stats["top1_correct"],
            "top1_acc": stats["top1_correct"] / num_samples if num_samples else None,
            "top5_correct": stats["top5_correct"] if len(candidate_labels) >= 5 else None,
            "top5_acc": (
                stats["top5_correct"] / num_samples
                if len(candidate_labels) >= 5 and num_samples
                else None
            ),
        }
    observed_class_acc = [
        value["top1_acc"]
        for value in per_class_accuracy.values()
        if value["num_samples"] > 0 and value["top1_acc"] is not None
    ]

    metrics = {
        "num_samples": total,
        "loss": loss_sum / max(total, 1),
        "mean_normalized_entropy": entropy_sum / max(total, 1),
        "entropy_temperature": float(entropy_temperature),
        "top1_acc": correct1 / max(total, 1),
        "top5_acc": correct5 / max(total, 1) if len(candidate_labels) >= 5 else None,
        "macro_top1_acc": sum(observed_class_acc) / len(observed_class_acc) if observed_class_acc else None,
        "per_class_accuracy": per_class_accuracy,
        "candidate_labels": [int(label) for label in candidate_labels],
        "score_calibration": {
            "enabled": used_unseen_score_scale,
            "method": "unit_cosine_unseen_scale" if used_unseen_score_scale else "none",
            "unseen_labels": [int(label) for label in unseen_labels],
            "unseen_score_scale": float(unseen_score_scale),
        },
        "inference_time_seconds": inference_time_seconds,
        "avg_inference_time_seconds_per_sample": inference_time_seconds / max(total, 1),
        "avg_inference_time_ms_per_sample": (inference_time_seconds / max(total, 1)) * 1000.0,
        "samples_per_second": total / inference_time_seconds if inference_time_seconds > 0 else None,
    }
    return metrics, predictions


def build_eval_dataset(split_dir, prefixes, zero_streams=(), pose_suffix=""):
    datasets = [
        TrimodalContrastiveDataset(
            data_dir=split_dir,
            prefix=prefix,
            mmap=True,
            zero_streams=zero_streams,
            pose_suffix=pose_suffix,
        )
        for prefix in prefixes
    ]
    if len(datasets) == 1:
        return datasets[0]
    return Data.ConcatDataset(datasets)


def get_dataset_labels(dataset):
    if isinstance(dataset, Data.Subset):
        labels = get_dataset_labels(dataset.dataset)
        return labels[np.asarray(dataset.indices, dtype=np.int64)]
    if isinstance(dataset, Data.ConcatDataset):
        return np.concatenate([get_dataset_labels(child) for child in dataset.datasets])
    if hasattr(dataset, "labels"):
        return np.asarray(dataset.labels)
    raise TypeError(f"Cannot extract labels from dataset type: {type(dataset).__name__}")


def filter_dataset_by_sample_scope(dataset, sample_scope, split_metadata):
    if sample_scope == "all":
        return dataset

    class_key = f"{sample_scope}_classes"
    if class_key not in split_metadata:
        raise KeyError(
            f"class metadata does not define {class_key}; "
            "pass --class-split-dir pointing to the 50/5 split metadata."
        )

    allowed = {int(label) for label in split_metadata[class_key]}
    labels = get_dataset_labels(dataset)
    indices = [index for index, label in enumerate(labels) if int(label) in allowed]
    if not indices:
        raise ValueError(f"--sample-scope {sample_scope} selected zero samples.")
    return Data.Subset(dataset, indices)


def validate_candidate_coverage(dataset, candidate_labels, candidate_scope):
    if candidate_labels is None:
        return

    labels = get_dataset_labels(dataset)
    candidate_set = {int(label) for label in candidate_labels}
    outside = sorted({int(label) for label in np.unique(labels) if int(label) not in candidate_set})
    if outside:
        count = int(sum(int(label) not in candidate_set for label in labels))
        raise ValueError(
            f"{count} evaluated samples have labels outside --candidate-scope {candidate_scope}: {outside}. "
            "Use --sample-scope to filter samples, or use --candidate-scope all."
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate CLIPGCN zero-shot action recognition.")
    parser.add_argument("--config", default="config.yaml", help="Path to CLIPGCN YAML config.")
    parser.add_argument("--checkpoint", default=None, help="Model checkpoint. Defaults to outputs.best_model.")
    parser.add_argument(
        "--split-dir",
        default=None,
        help="Directory containing <prefix>_video.npy, <prefix>_pose.npy, etc. Defaults to data.train.data_dir.",
    )
    parser.add_argument(
        "--class-split-dir",
        default=None,
        help="Directory whose metadata.json defines seen/unseen classes. Defaults to --split-dir.",
    )
    parser.add_argument("--prefix", default="unseen", help="Dataset prefix, usually unseen.")
    parser.add_argument(
        "--prefixes",
        nargs="+",
        default=None,
        help="Evaluate one or more prefixes concatenated together. Overrides --prefix.",
    )
    parser.add_argument(
        "--sample-scope",
        choices=["all", "seen", "unseen"],
        default="all",
        help="Optionally filter evaluated samples by seen/unseen class metadata.",
    )
    parser.add_argument(
        "--candidate-scope",
        choices=["unseen", "seen", "all"],
        default="unseen",
        help="Which text labels are valid predictions. Use unseen for standard ZSL accuracy.",
    )
    parser.add_argument(
        "--pose-suffix",
        default=None,
        help="Skeleton feature variant, e.g. _coco17. Defaults to the config's data.train.pose_suffix.",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    parser.add_argument(
        "--unseen-score-scale",
        type=float,
        default=1.0,
        help="Multiplier applied to unseen class confidence scores before top-k prediction.",
    )
    parser.add_argument(
        "--zero-streams",
        nargs="*",
        choices=("video", "pose", "object"),
        default=None,
        help="Ablation: zero these input streams at eval. Overrides data.zero_streams. "
        "Must match how the checkpoint was trained.",
    )
    parser.add_argument(
        "--protocol",
        choices=("zsl", "gzsl_seen", "gzsl_unseen"),
        default=None,
        help="Preset for sample/candidate scopes. zsl: unseen samples vs unseen candidates. "
        "gzsl_unseen: unseen samples vs ALL candidates. gzsl_seen: seen samples vs ALL candidates. "
        "Overrides --sample-scope/--candidate-scope.",
    )
    args = parser.parse_args()
    if args.protocol == "zsl":
        args.sample_scope, args.candidate_scope = "unseen", "unseen"
    elif args.protocol == "gzsl_unseen":
        args.sample_scope, args.candidate_scope = "unseen", "all"
    elif args.protocol == "gzsl_seen":
        args.sample_scope, args.candidate_scope = "seen", "all"
    return args


def main():
    args = parse_args()
    config_path = os.path.abspath(args.config)
    config = load_config(config_path)
    device = get_device(config["runtime"].get("device"))
    print_device_info(device)

    latest_run = load_latest_run_info(config, config_path)
    if args.checkpoint:
        checkpoint_path = get_path_from_config(config_path, args.checkpoint)
    elif latest_run and latest_run.get("best_model"):
        checkpoint_path = latest_run["best_model"]
    else:
        checkpoint_path = get_path_from_config(config_path, config["outputs"]["best_model"])
    split_dir = args.split_dir or get_path_from_config(config_path, config["data"]["train"]["data_dir"])
    split_dir = get_path_from_config(config_path, split_dir)
    class_split_dir = args.class_split_dir or split_dir
    class_split_dir = get_path_from_config(config_path, class_split_dir)

    if args.unseen_score_scale <= 0:
        raise ValueError("--unseen-score-scale must be positive.")

    split_metadata = load_split_metadata(class_split_dir)
    candidate_labels = load_split_classes(class_split_dir, args.candidate_scope)
    model = load_model(config, config_path, device, checkpoint_path)
    candidate_labels, _texts = build_text_bank(model, config, config_path, candidate_labels)

    prefixes = args.prefixes or [args.prefix]
    if args.zero_streams is not None:
        zero_streams = list(args.zero_streams)
    else:
        zero_streams = config.get("data", {}).get("zero_streams") or []
    if zero_streams:
        print(f"[ablation] zero_streams = {sorted(zero_streams)}")
    # The skeleton backbone is part of the checkpoint's identity, so the eval
    # data must use the same one the config trained on.
    pose_suffix = args.pose_suffix
    if pose_suffix is None:
        pose_suffix = config["data"]["train"].get("pose_suffix", "")
    if pose_suffix:
        print(f"Using pose features with suffix {pose_suffix!r}")
    dataset = build_eval_dataset(split_dir, prefixes, zero_streams=zero_streams, pose_suffix=pose_suffix)
    dataset = filter_dataset_by_sample_scope(dataset, args.sample_scope, split_metadata)
    validate_candidate_coverage(dataset, candidate_labels, args.candidate_scope)
    loader_config = config["data"]["dataloader"]
    data_loader = Data.DataLoader(
        dataset,
        batch_size=args.batch_size or loader_config["batch_size"],
        shuffle=False,
        num_workers=args.num_workers if args.num_workers is not None else loader_config.get("num_workers", 4),
        pin_memory=loader_config.get("pin_memory", True),
        drop_last=False,
    )

    use_amp = bool(config["runtime"].get("amp", False)) and device.type == "cuda"
    loss_config = config.get("train", {}).get("loss") or {}
    entropy_temperature = float(loss_config.get("entropy_temperature", 0.1))
    if entropy_temperature <= 0:
        raise ValueError("train.loss.entropy_temperature must be positive.")
    unseen_labels = split_metadata.get("unseen_classes", [])
    if args.unseen_score_scale != 1.0 and not unseen_labels:
        print("Warning: --unseen-score-scale was set, but class metadata has no unseen_classes.")
    metrics, predictions = evaluate(
        model,
        data_loader,
        candidate_labels,
        device,
        use_amp=use_amp,
        unseen_labels=unseen_labels,
        unseen_score_scale=args.unseen_score_scale,
        entropy_temperature=entropy_temperature,
    )

    print("Action recognition results")
    print(f"  checkpoint: {checkpoint_path}")
    print(f"  split_dir: {split_dir}")
    print(f"  class_split_dir: {class_split_dir}")
    print(f"  prefixes: {prefixes}")
    print(f"  sample_scope: {args.sample_scope}")
    print(f"  candidate_scope: {args.candidate_scope}")
    print(f"  candidate_labels: {metrics['candidate_labels']}")
    print(f"  score_calibration: {metrics['score_calibration']}")
    print(f"  num_samples: {metrics['num_samples']}")
    print(f"  loss: {metrics['loss']:.4f}")
    print(
        f"  mean_normalized_entropy: {metrics['mean_normalized_entropy']:.4f} "
        f"(temperature={metrics['entropy_temperature']:.4g})"
    )
    print(f"  top1_acc: {metrics['top1_acc']:.4f}")
    if metrics["macro_top1_acc"] is not None:
        print(f"  macro_top1_acc: {metrics['macro_top1_acc']:.4f}")
    if metrics["top5_acc"] is not None:
        print(f"  top5_acc: {metrics['top5_acc']:.4f}")
    print("  per_class_top1_acc:")
    for label, stats in metrics["per_class_accuracy"].items():
        if stats["num_samples"] == 0:
            continue
        print(
            f"    class {label}: "
            f"{stats['top1_acc']:.4f} "
            f"({stats['top1_correct']}/{stats['num_samples']})"
        )
    print(f"  total_inference_time: {metrics['inference_time_seconds']:.4f}s")
    print(f"  avg_inference_time: {metrics['avg_inference_time_ms_per_sample']:.4f} ms/sample")
    if metrics["samples_per_second"] is not None:
        print(f"  throughput: {metrics['samples_per_second']:.2f} samples/s")

    output_config_path = args.output
    if output_config_path is None and latest_run and latest_run.get("test_results"):
        output_config_path = latest_run["test_results"]
    if output_config_path is None:
        output_config_path = config.get("outputs", {}).get("test_results")
    if output_config_path:
        output_path = get_path_from_config(config_path, output_config_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "metrics": metrics,
                    "predictions": predictions,
                    "prefixes": prefixes,
                    "sample_scope": args.sample_scope,
                    "split_dir": split_dir,
                    "class_split_dir": class_split_dir,
                    "checkpoint": checkpoint_path,
                },
                handle,
                indent=2,
                ensure_ascii=False,
            )
        print(f"Saved test results to {output_path}")


if __name__ == "__main__":
    main()
