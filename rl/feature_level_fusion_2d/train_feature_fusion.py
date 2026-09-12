"""Train the isolated 512-D feature fusion module on the seen split."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .. import multistep_g18_view_policy_v1 as cache_api
from .feature_fusion import FeatureFusion512, fuse_sequence, infer_logit_scale, normalized_text


SEEN_CLASSES = sorted(set(range(55)) - {14, 15, 25, 39, 46})
PSEUDO_UNSEEN = [0, 2, 9, 10, 11, 17, 26, 34, 49, 50]


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    tmp.replace(path)


@torch.no_grad()
def validation_score(raw: object, fusion: FeatureFusion512, classes: list[int]) -> float:
    bank = torch.as_tensor(classes, device=raw.device, dtype=torch.long)
    eligible = torch.isin(raw.labels, bank) & (raw.valid.sum(-1) >= 2)
    episodes = eligible.nonzero().flatten()
    if not len(episodes):
        return 0.0
    scale = infer_logit_scale(raw)
    # Use the same physical start convention as the active-view experiment:
    # all valid pairs starting from view 0, plus all valid unordered pairs.
    values: list[torch.Tensor] = []
    for first in range(4):
        for second in range(first + 1, 4):
            keep = eligible & raw.valid[:, first] & raw.valid[:, second]
            ids = keep.nonzero().flatten()
            if not len(ids):
                continue
            paths = torch.as_tensor(
                [[first, second]] * len(ids), device=raw.device, dtype=torch.long
            )
            scores = float(scale) * (
                fuse_sequence(fusion, raw.z[ids[:, None], paths])
                @ normalized_text(raw).t()
            )
            restricted = scores.index_select(-1, bank)
            local = torch.searchsorted(bank, raw.labels[ids])
            values.append(restricted.argmax(-1).eq(local).float().mean())
    return float(torch.stack(values).mean()) if values else 0.0


def train_seed(
    seed: int,
    train_raw: object,
    val_raw: object,
    output_root: Path,
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
    device = train_raw.device
    generator = torch.Generator(device=device).manual_seed(seed)
    fusion = FeatureFusion512().to(device)
    optimizer = torch.optim.AdamW(fusion.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=learning_rate * 0.05)
    seen_bank = torch.as_tensor(SEEN_CLASSES, device=device, dtype=torch.long)
    eligible = (train_raw.valid.sum(-1) == 4) & torch.isin(train_raw.labels, seen_bank)
    pool = eligible.nonzero().flatten()
    if not len(pool):
        raise RuntimeError("no complete seen-class training episodes for feature fusion")
    scale = infer_logit_scale(train_raw)
    best_score = -float("inf")
    best_epoch = 0
    log_path = run / "train.log"
    for epoch in range(1, epochs + 1):
        fusion.train()
        started = time.monotonic()
        losses: list[float] = []
        for _ in range(updates_per_epoch):
            ids = torch.randint(len(pool), (batch_size,), device=device, generator=generator)
            episodes = pool[ids]
            order = torch.rand((batch_size, 4), device=device, generator=generator).argsort(-1)
            z = train_raw.z[episodes[:, None], order[:, :3]]
            pair = fuse_sequence(fusion, z[:, :2])
            triple = fuse_sequence(fusion, z[:, :3])
            use_triple = torch.rand((batch_size,), device=device, generator=generator) > 0.5
            fused = torch.where(use_triple[:, None], triple, pair)
            scores = float(scale) * (fused @ normalized_text(train_raw).t())
            logits = scores.index_select(-1, seen_bank)
            local = torch.searchsorted(seen_bank, train_raw.labels[episodes])
            loss = F.cross_entropy(logits, local)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(fusion.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        score = validation_score(val_raw, fusion, PSEUDO_UNSEEN)
        row = {
            "epoch": epoch,
            "seed": seed,
            "loss": float(np.mean(losses)),
            "validation_pair_top1": score,
            "seconds": time.monotonic() - started,
            "training_episodes": int(len(pool)),
        }
        with log_path.open("a", encoding="utf-8", buffering=1) as handle:
            handle.write(json.dumps(row) + "\n")
        payload = {
            "model": fusion.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "seed": seed,
            "logit_scale": scale,
            "validation_pair_top1": score,
            "best_score": best_score,
            "best_epoch": best_epoch,
        }
        torch.save(payload, run / "last.pt")
        if score > best_score:
            best_score, best_epoch = score, epoch
            payload["best_score"] = best_score
            payload["best_epoch"] = best_epoch
            torch.save(payload, run / "best.pt")
        print(json.dumps(row), flush=True)
    result = {
        "seed": seed,
        "best_score": best_score,
        "best_epoch": best_epoch,
        "checkpoint": str(run / "best.pt"),
        "training_episodes": int(len(pool)),
        "logit_scale": scale,
    }
    dump(complete, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32768)
    parser.add_argument("--updates-per-epoch", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260909, 20260910, 20260911])
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the feature-fusion experiment")
    device = torch.device("cuda:0")
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    train_raw = cache_api.GPUCache(args.cache_root / "dqn_train", device)
    val_raw = cache_api.GPUCache(args.cache_root / "val", device)
    cache_api.attach_view_valid(train_raw, args.cache_root, "dqn_train")
    cache_api.attach_view_valid(val_raw, args.cache_root, "val")
    args.output_root.mkdir(parents=True, exist_ok=True)
    dump(args.output_root / "config_resolved.json", {
        "cache_root": str(args.cache_root),
        "output_root": str(args.output_root),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "updates_per_epoch": args.updates_per_epoch,
        "learning_rate": args.learning_rate,
        "seeds": args.seeds,
        "training_classes": SEEN_CLASSES,
        "validation_classes": PSEUDO_UNSEEN,
        "fusion": "FeatureFusion512(mean + residual(absolute_difference))",
        "recognizer_features": "normalized cached z [512] -> fusion -> normalized text cosine",
    })
    summaries = []
    for seed in args.seeds:
        summaries.append(train_seed(seed, train_raw, val_raw, args.output_root, args.epochs, args.batch_size, args.updates_per_epoch, args.learning_rate))
        torch.cuda.empty_cache()
    dump(args.output_root / "training_summary.json", summaries)
    print("FEATURE_FUSION_TRAIN_COMPLETE", flush=True)


if __name__ == "__main__":
    main()

