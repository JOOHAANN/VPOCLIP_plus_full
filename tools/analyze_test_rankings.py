#!/usr/bin/env python
"""Analyze CLIPGCN candidate rankings with optional unseen score scaling."""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as Data


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from test import build_text_bank, load_model, load_split_classes, load_split_metadata  # noqa: E402
from train import (  # noqa: E402
    TrimodalContrastiveDataset,
    get_device,
    get_path_from_config,
    load_config,
    move_batch_to_device,
    print_device_info,
    progress_bar,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split-dir", default=None)
    parser.add_argument("--class-split-dir", default=None)
    parser.add_argument("--prefix", default="unseen")
    parser.add_argument("--candidate-scope", choices=["unseen", "seen", "all"], default="unseen")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--unseen-score-scale", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--group-top-k", type=int, default=5)
    parser.add_argument(
        "--output",
        default="./work_dir/ranking_analysis.json",
        help="JSON file for detailed per-sample and per-class ranking diagnostics.",
    )
    return parser.parse_args()


def load_latest_run_info(config, config_path):
    work_dir = config.get("outputs", {}).get("work_dir")
    if not work_dir:
        return None
    latest_path = Path(get_path_from_config(config_path, work_dir)) / "latest_run.json"
    if not latest_path.exists():
        return None
    return json.loads(latest_path.read_text(encoding="utf-8"))


def recover_unit_cosine_scores(model, logits):
    scale = model.logit_scale.exp().clamp(max=100).to(device=logits.device, dtype=logits.dtype)
    cosine_scores = (logits / scale).clamp(-1.0, 1.0)
    unit_scores = (cosine_scores + 1.0) * 0.5
    return cosine_scores, unit_scores


def label_mask(candidate_labels, selected_labels, device):
    selected = {int(label) for label in selected_labels}
    return torch.as_tensor([int(label) in selected for label in candidate_labels], dtype=torch.bool, device=device)


def top_candidates(scores, candidate_tensor, k, allowed_mask=None):
    if allowed_mask is not None:
        candidate_indices = torch.nonzero(allowed_mask, as_tuple=False).flatten()
        if candidate_indices.numel() == 0:
            return []
        scoped_scores = scores[candidate_indices]
        local_k = min(k, scoped_scores.numel())
        values, local_indices = torch.topk(scoped_scores, k=local_k)
        indices = candidate_indices[local_indices]
    else:
        local_k = min(k, scores.numel())
        values, indices = torch.topk(scores, k=local_k)

    return [
        {
            "rank": rank + 1,
            "global_rank": int(torch.sum(scores > value).detach().cpu()) + 1,
            "label": int(candidate_tensor[index].detach().cpu()),
            "candidate_index": int(index.detach().cpu()),
            "score": float(value.detach().cpu()),
        }
        for rank, (index, value) in enumerate(zip(indices, values))
    ]


def summarize_values(values):
    if not values:
        return {"count": 0, "mean": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": float(np.mean(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def main():
    args = parse_args()
    if args.unseen_score_scale <= 0:
        raise ValueError("--unseen-score-scale must be positive.")

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
    class_split_dir = get_path_from_config(config_path, args.class_split_dir or split_dir)
    split_metadata = load_split_metadata(class_split_dir)
    seen_labels = [int(label) for label in split_metadata.get("seen_classes", [])]
    unseen_labels = [int(label) for label in split_metadata.get("unseen_classes", [])]

    candidate_labels = load_split_classes(class_split_dir, args.candidate_scope)
    model = load_model(config, config_path, device, checkpoint_path)
    candidate_labels, _texts = build_text_bank(model, config, config_path, candidate_labels)
    candidate_tensor = torch.as_tensor(candidate_labels, dtype=torch.long, device=device)
    seen_mask = label_mask(candidate_labels, seen_labels, device)
    unseen_mask = label_mask(candidate_labels, unseen_labels, device)

    dataset = TrimodalContrastiveDataset(data_dir=split_dir, prefix=args.prefix, mmap=True)
    sample_names = dataset.sample_names
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
    samples = []
    unseen_label_set = set(unseen_labels)
    class_stats = defaultdict(
        lambda: {
            "num_samples": 0,
            "top1_correct": 0,
            "top5_correct": 0,
            "top1_winning_raw_cosine": [],
            "top1_winning_unit_cosine": [],
            "top1_winning_calibrated_score": [],
            "correct_top1_raw_cosine": [],
            "correct_top1_unit_cosine": [],
            "correct_top1_calibrated_score": [],
            "top1_pred_counts": Counter(),
            "top5_label_counts": Counter(),
        }
    )

    global_index = 0
    with torch.no_grad():
        for batch in progress_bar(data_loader, "Analyze"):
            batch = move_batch_to_device(batch, device)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(batch["video"], batch["pose"], batch["object"], batch["joint_xy"])

            raw_cosine, unit_cosine = recover_unit_cosine_scores(model, logits)
            calibrated_scores = unit_cosine.clone()
            if args.unseen_score_scale != 1.0 and torch.any(unseen_mask):
                calibrated_scores[:, unseen_mask] *= args.unseen_score_scale

            top_indices = torch.argmax(calibrated_scores, dim=1)
            top_labels = candidate_tensor[top_indices]
            labels = batch["label"].long()
            top5_indices = torch.topk(calibrated_scores, k=min(args.top_k, calibrated_scores.shape[1]), dim=1).indices
            top5_labels = candidate_tensor[top5_indices]

            for row in range(labels.shape[0]):
                true_label = int(labels[row].detach().cpu())
                pred_label = int(top_labels[row].detach().cpu())
                pred_index = int(top_indices[row].detach().cpu())
                top5_label_list = [int(label) for label in top5_labels[row].detach().cpu().tolist()]
                sample_name = str(sample_names[global_index]) if sample_names is not None else None

                top_all = top_candidates(calibrated_scores[row], candidate_tensor, args.top_k)
                top_seen = top_candidates(calibrated_scores[row], candidate_tensor, args.group_top_k, allowed_mask=seen_mask)
                top_unseen = top_candidates(
                    calibrated_scores[row],
                    candidate_tensor,
                    args.group_top_k,
                    allowed_mask=unseen_mask,
                )

                record = {
                    "index": global_index,
                    "sample_name": sample_name,
                    "label": true_label,
                    "label_is_unseen": true_label in unseen_label_set,
                    "pred": pred_label,
                    "pred_is_unseen": pred_label in unseen_label_set,
                    "correct_top1": pred_label == true_label,
                    "top1": {
                        "label": pred_label,
                        "raw_cosine": float(raw_cosine[row, pred_index].detach().cpu()),
                        "unit_cosine": float(unit_cosine[row, pred_index].detach().cpu()),
                        "calibrated_score": float(calibrated_scores[row, pred_index].detach().cpu()),
                        "is_unseen": pred_label in unseen_label_set,
                    },
                    "top_all": top_all,
                    "top_seen": top_seen,
                    "top_unseen": top_unseen,
                }
                samples.append(record)

                stats = class_stats[true_label]
                stats["num_samples"] += 1
                stats["top1_correct"] += int(record["correct_top1"])
                stats["top5_correct"] += int(true_label in top5_label_list)
                stats["top1_pred_counts"][pred_label] += 1
                stats["top5_label_counts"].update(top5_label_list)
                stats["top1_winning_raw_cosine"].append(record["top1"]["raw_cosine"])
                stats["top1_winning_unit_cosine"].append(record["top1"]["unit_cosine"])
                stats["top1_winning_calibrated_score"].append(record["top1"]["calibrated_score"])
                if record["correct_top1"]:
                    stats["correct_top1_raw_cosine"].append(record["top1"]["raw_cosine"])
                    stats["correct_top1_unit_cosine"].append(record["top1"]["unit_cosine"])
                    stats["correct_top1_calibrated_score"].append(record["top1"]["calibrated_score"])

                global_index += 1

    per_class = {}
    for label in sorted(class_stats):
        stats = class_stats[label]
        num_samples = stats["num_samples"]
        per_class[str(label)] = {
            "num_samples": num_samples,
            "top1_acc": stats["top1_correct"] / num_samples if num_samples else None,
            "top5_acc": stats["top5_correct"] / num_samples if num_samples else None,
            "top1_pred_counts": {str(k): int(v) for k, v in stats["top1_pred_counts"].most_common()},
            "top5_label_counts": {str(k): int(v) for k, v in stats["top5_label_counts"].most_common()},
            "top1_winning_raw_cosine": summarize_values(stats["top1_winning_raw_cosine"]),
            "top1_winning_unit_cosine": summarize_values(stats["top1_winning_unit_cosine"]),
            "top1_winning_calibrated_score": summarize_values(stats["top1_winning_calibrated_score"]),
            "correct_top1_raw_cosine": summarize_values(stats["correct_top1_raw_cosine"]),
            "correct_top1_unit_cosine": summarize_values(stats["correct_top1_unit_cosine"]),
            "correct_top1_calibrated_score": summarize_values(stats["correct_top1_calibrated_score"]),
        }

    output = {
        "config": {
            "checkpoint": checkpoint_path,
            "split_dir": split_dir,
            "class_split_dir": class_split_dir,
            "prefix": args.prefix,
            "candidate_scope": args.candidate_scope,
            "candidate_labels": [int(label) for label in candidate_labels],
            "seen_labels": seen_labels,
            "unseen_labels": unseen_labels,
            "unseen_score_scale": args.unseen_score_scale,
            "top_k": args.top_k,
            "group_top_k": args.group_top_k,
        },
        "summary": {
            "num_samples": len(samples),
            "top1_acc": sum(int(item["correct_top1"]) for item in samples) / max(len(samples), 1),
        },
        "per_class": per_class,
        "samples": samples,
    }

    output_path = Path(get_path_from_config(config_path, args.output))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved ranking analysis to {output_path}")
    print(f"num_samples: {output['summary']['num_samples']}")
    print(f"top1_acc: {output['summary']['top1_acc']:.4f}")
    for label in unseen_labels:
        if str(label) in per_class:
            stats = per_class[str(label)]
            correct_scores = stats["correct_top1_calibrated_score"]
            print(
                f"unseen class {label}: top1_acc={stats['top1_acc']:.4f}, "
                f"correct winning calibrated score mean={correct_scores['mean']}"
            )


if __name__ == "__main__":
    main()
