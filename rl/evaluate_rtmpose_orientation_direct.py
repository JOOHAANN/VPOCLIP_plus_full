"""Directly compare MLP orientation with the 3-D skeleton orientation labels.

This evaluator intentionally does not report action-recognition accuracy.  It
compares, for every test sample,

    predicted_yaw_from_RTMPose - yaw_from_3D_skeleton

using circular angular error.  It can evaluate the single checkpoint and the
30 seed checkpoints produced by ``train_rtmpose_orientation_mlp.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from train_rtmpose_orientation_mlp import BIN_CENTERS_DEG, OrientationMLP


def wrap_degrees(value: np.ndarray) -> np.ndarray:
    return (np.asarray(value, dtype=np.float64) + 180.0) % 360.0 - 180.0


def direct_metrics(pred_deg: np.ndarray, target_deg: np.ndarray) -> dict[str, Any]:
    signed = wrap_degrees(pred_deg - target_deg)
    absolute = np.abs(signed)
    return {
        "n": int(len(absolute)),
        "mean_abs_error_deg": float(absolute.mean()),
        "median_abs_error_deg": float(np.median(absolute)),
        "p90_abs_error_deg": float(np.percentile(absolute, 90)),
        "p95_abs_error_deg": float(np.percentile(absolute, 95)),
        "rmse_circular_deg": float(np.sqrt(np.mean(signed**2))),
        "signed_bias_deg": float(signed.mean()),
        "within_5_deg": float(np.mean(absolute <= 5.0)),
        "within_10_deg": float(np.mean(absolute <= 10.0)),
        "within_20_deg": float(np.mean(absolute <= 20.0)),
        "within_30_deg": float(np.mean(absolute <= 30.0)),
        "within_45_deg": float(np.mean(absolute <= 45.0)),
        "within_90_deg": float(np.mean(absolute <= 90.0)),
    }


@torch.no_grad()
def predict_checkpoint(
    checkpoint: Path,
    x: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model = OrientationMLP().to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    model.eval()
    loader = DataLoader(TensorDataset(torch.from_numpy(x)), batch_size=batch_size, shuffle=False)
    centers = torch.as_tensor(BIN_CENTERS_DEG, dtype=torch.float32, device=device)
    unit = torch.stack(
        (torch.cos(torch.deg2rad(centers)), torch.sin(torch.deg2rad(centers))), dim=-1
    )
    soft_yaw: list[np.ndarray] = []
    class_yaw: list[np.ndarray] = []
    for (batch,) in loader:
        logits = model(batch.to(device))
        probability = torch.softmax(logits, dim=-1)
        expected = probability @ unit
        soft = torch.rad2deg(torch.atan2(expected[:, 1], expected[:, 0]))
        hard = centers[logits.argmax(-1)]
        soft_yaw.append(soft.cpu().numpy())
        class_yaw.append(hard.cpu().numpy())
    return np.concatenate(soft_yaw), np.concatenate(class_yaw)


def mean_ci(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean())
    sd = float(array.std(ddof=1))
    half = float(1.96 * sd / math.sqrt(len(array)))
    return {
        "mean": mean,
        "sd": sd,
        "ci95_low": mean - half,
        "ci95_high": mean + half,
        "n_seeds": int(len(array)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(
            "/home/youhan/ws/VPOCLIP_plus_full/work_dir/rtmpose_orientation_mlp/"
            "orientation_test_frame6.npz"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "/home/youhan/ws/VPOCLIP_plus_full/work_dir/rtmpose_orientation_mlp/best.pt"
        ),
    )
    parser.add_argument(
        "--seed-root",
        type=Path,
        default=Path(
            "/home/youhan/ws/VPOCLIP_plus_full/work_dir/rtmpose_orientation_mlp/seed_runs"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/home/youhan/ws/VPOCLIP_plus_full/work_dir/rtmpose_orientation_mlp/"
            "direct_orientation_comparison_30seed.json"
        ),
    )
    parser.add_argument(
        "--pairs-output",
        type=Path,
        default=Path(
            "/home/youhan/ws/VPOCLIP_plus_full/work_dir/rtmpose_orientation_mlp/"
            "test_orientation_pairs.csv"
        ),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args()
    device = torch.device(args.device)
    with np.load(args.data) as data:
        x = np.asarray(data["x"], dtype=np.float32)
        target = np.asarray(data["angle_deg"], dtype=np.float64)

    soft, hard = predict_checkpoint(args.checkpoint, x, args.batch_size, device)
    single = {
        "checkpoint": str(args.checkpoint),
        "soft_expected_yaw": direct_metrics(soft, target),
        "argmax_bin_center_yaw": direct_metrics(hard, target),
    }

    seed_rows = []
    for checkpoint in sorted(args.seed_root.glob("seed_*/best.pt")):
        seed_soft, seed_hard = predict_checkpoint(checkpoint, x, args.batch_size, device)
        seed_rows.append(
            {
                "seed": checkpoint.parent.name,
                "soft_expected_yaw": direct_metrics(seed_soft, target),
                "argmax_bin_center_yaw": direct_metrics(seed_hard, target),
            }
        )
    if not seed_rows:
        raise RuntimeError(f"no seed checkpoints found under {args.seed_root}")

    metric_names = (
        "mean_abs_error_deg",
        "median_abs_error_deg",
        "p90_abs_error_deg",
        "p95_abs_error_deg",
        "rmse_circular_deg",
        "signed_bias_deg",
        "within_5_deg",
        "within_10_deg",
        "within_20_deg",
        "within_30_deg",
        "within_45_deg",
        "within_90_deg",
    )
    seed_summary = {}
    for branch in ("soft_expected_yaw", "argmax_bin_center_yaw"):
        seed_summary[branch] = {
            metric: mean_ci([row[branch][metric] for row in seed_rows])
            for metric in metric_names
        }

    # Show concrete pairs from the representative checkpoint: closest, median,
    # and largest circular discrepancies.  The index refers to the test cache
    # row; the cache audit records the source split and frame convention.
    signed = wrap_degrees(soft - target)
    absolute = np.abs(signed)
    candidate_indices = {
        "smallest_error": int(np.argmin(absolute)),
        "median_error": int(np.argsort(absolute)[len(absolute) // 2]),
        "largest_error": int(np.argmax(absolute)),
    }
    examples = {}
    for name, index in candidate_indices.items():
        examples[name] = {
            "test_cache_index": index,
            "skeleton_yaw_deg": float(target[index]),
            "mlp_yaw_deg": float(soft[index]),
            "signed_difference_deg": float(signed[index]),
            "absolute_difference_deg": float(absolute[index]),
        }

    args.pairs_output.parent.mkdir(parents=True, exist_ok=True)
    with args.pairs_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "test_cache_index",
                "skeleton_yaw_deg",
                "mlp_yaw_deg",
                "signed_difference_deg",
                "absolute_difference_deg",
                "argmax_bin_center_yaw_deg",
            ]
        )
        for index in range(len(target)):
            writer.writerow(
                [
                    index,
                    float(target[index]),
                    float(soft[index]),
                    float(signed[index]),
                    float(absolute[index]),
                    float(hard[index]),
                ]
            )

    result = {
        "comparison": "MLP predicted orientation versus 3-D skeleton computed orientation",
        "target_definition": "3-D anatomical body-forward yaw in the source camera frame",
        "prediction_definition": "circular expectation of the eight MLP output-bin centers",
        "test_data": str(args.data),
        "single_checkpoint": single,
        "seed_level_summary": seed_summary,
        "seed_count": len(seed_rows),
        "pairs_output": str(args.pairs_output),
        "representative_examples": examples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
