"""Experiment A: a tiny shared gating network for weighted logit fusion."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from . import fusion_weighted_common_v1 as common


class LightweightGate(nn.Module):
    """D -> 16 -> 1 shared scorer, initialized to equal-mean fusion."""

    def __init__(self, input_dim: int = 13) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.network = nn.Sequential(nn.Linear(input_dim, 16), nn.ReLU(), nn.Linear(16, 1))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


def validation_score(raw: object, gate: LightweightGate, seed: int) -> float:
    bank = common.class_bank(common.PSEUDO_UNSEEN, raw.device)
    eligible = common.eligible_episodes(raw, common.PSEUDO_UNSEEN, minimum_views=2)
    starts = torch.zeros_like(eligible)
    values = []
    for set_name in ("Random-2", "Preferred-Orthogonal-2", "Random-3", "Preferred-Orthogonal-3"):
        k = 2 if set_name.endswith("-2") else 3
        keep = raw.valid[eligible].sum(-1) >= k
        episodes = eligible[keep]
        fixed_starts = torch.zeros_like(episodes)
        path_episodes, paths = common.path_candidates(raw, episodes, fixed_starts, set_name, seed)
        if len(path_episodes):
            metrics = common.path_metrics(raw, path_episodes, paths, bank, "lightweight_gating", gate)
            values.append(metrics["correct"].float().mean())
    return float(torch.stack(values).mean()) if values else 0.0


def train_gate(
    raw: object,
    output_root: Path,
    seed: int,
    epochs: int,
    batch_size: int,
    updates_per_epoch: int,
    learning_rate: float,
) -> dict[str, object]:
    run = output_root / f"seed_{seed}"
    run.mkdir(parents=True, exist_ok=True)
    complete = run / "complete.json"
    if complete.exists():
        return json.loads(complete.read_text(encoding="utf-8"))
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = raw.device
    features = common.attach_sensor_features(raw)
    pool = common.eligible_episodes(raw, common.SEEN_CLASSES, minimum_views=4)
    generator = torch.Generator(device=device).manual_seed(seed)
    gate = LightweightGate(13).to(device)
    optimizer = torch.optim.AdamW(gate.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=learning_rate * 0.05)
    bank = common.class_bank(common.SEEN_CLASSES, device)
    best_score = -float("inf")
    best_epoch = 0
    for epoch in range(1, epochs + 1):
        gate.train()
        losses = []
        started = time.monotonic()
        for _ in range(updates_per_epoch):
            ids = torch.randint(len(pool), (batch_size,), device=device, generator=generator)
            episodes = pool[ids]
            order = torch.rand((batch_size, 4), device=device, generator=generator).argsort(-1)
            k = int(torch.randint(2, 5, (), device=device, generator=generator))
            paths = order[:, :k]
            inputs, components = common.path_inputs(raw, episodes, paths)
            weights = torch.softmax(gate(inputs).squeeze(-1), dim=-1)
            fused = (weights[..., None] * raw.logits[episodes[:, None], paths]).sum(1)
            scores = fused.index_select(-1, bank)
            local = torch.searchsorted(bank, raw.labels[episodes])
            loss = F.cross_entropy(scores, local)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gate.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        score = validation_score(raw, gate, seed + epoch)
        payload = {
            "model": gate.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "seed": seed,
            "validation_score": score,
            "best_score": best_score,
            "best_epoch": best_epoch,
        }
        torch.save(payload, run / "last.pt")
        if score > best_score:
            best_score, best_epoch = score, epoch
            payload["best_score"] = best_score
            payload["best_epoch"] = best_epoch
            torch.save(payload, run / "best.pt")
        row = {
            "epoch": epoch,
            "seed": seed,
            "loss": float(np.mean(losses)),
            "validation_score": score,
            "best_score": best_score,
            "best_epoch": best_epoch,
            "seconds": time.monotonic() - started,
            "training_episodes": int(len(pool)),
            "input_dim": 13,
        }
        with (run / "train.log").open("a", encoding="utf-8", buffering=1) as handle:
            handle.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
    result = {
        "seed": seed,
        "checkpoint": str(run / "best.pt"),
        "best_score": best_score,
        "best_epoch": best_epoch,
        "training_episodes": int(len(pool)),
        "input_dim": 13,
    }
    common.dump(complete, result)
    return result


def load_gate(checkpoint: Path, device: torch.device) -> LightweightGate:
    gate = LightweightGate(13).to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    gate.load_state_dict(payload["model"], strict=True)
    gate.eval()
    return gate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--updates-per-epoch", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--train-seeds", type=int, nargs="+", default=common.TRAIN_SEEDS)
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=common.EVAL_SEEDS)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda:0")
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    args.output_root.mkdir(parents=True, exist_ok=True)
    train_raw = common.load_raw(args.cache_root, "dqn_train", device)
    val_raw = common.load_raw(args.cache_root, "val", device)
    summaries = []
    for seed in args.train_seeds:
        summaries.append(train_gate(train_raw, args.output_root / "models", seed, args.epochs, args.batch_size, args.updates_per_epoch, args.learning_rate))
    chosen = max(summaries, key=lambda row: float(row["best_score"]))
    checkpoint = Path(str(chosen["checkpoint"]))
    gate = load_gate(checkpoint, device)
    metadata = {
        "experiment": "fusion_lightweight_gating_v1",
        "cache_root": str(args.cache_root),
        "training_classes": common.SEEN_CLASSES,
        "pseudo_unseen_selection_classes": common.PSEUDO_UNSEEN,
        "true_unseen_test_only": common.TRUE_UNSEEN,
        "train_seeds": args.train_seeds,
        "eval_seeds": args.eval_seeds,
        "selected_checkpoint": str(checkpoint),
        "input_dim": 13,
        "input_fields": [
            "normalized entropy", "top1-top2 margin", "max probability",
            "13-frame pose visibility mean/std/switch-rate",
            "13-frame pose edge-visibility mean/std",
            "sample-specific azimuth sin/cos", "cached body-yaw confidence",
            "relative angular diversity", "cross-view JS divergence",
        ],
        "omitted_unavailable_fields": ["elevation sin/cos", "per-frame logits entropy_std", "top1_switch_rate from logits", "logit_variance_mean", "margin_std"],
        "recognizer_frozen": True,
        "fusion": "weighted sum of frozen per-view classification logits",
        "anchor_preserved": True,
    }
    common.dump(args.output_root / "config_resolved.json", metadata)
    del train_raw, val_raw
    torch.cuda.empty_cache()
    results = common.evaluate_all(args.cache_root, args.output_root, gate, args.eval_seeds)
    summary = {**metadata, "selection_runs": summaries, "results": results}
    common.dump(args.output_root / "summary.json", summary)
    print("LIGHTWEIGHT_GATING_EXPERIMENT_COMPLETE", flush=True)


if __name__ == "__main__":
    main()

