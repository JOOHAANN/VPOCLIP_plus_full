"""Train and evaluate supervised semantic-conditioned NBV selectors.

This is the first SC-NBV stage from the implementation guide.  It is a
contextual-bandit/utility-ranking experiment, not a replacement of the
legacy DDQN files.  Hyperparameters are selected on policy-held-out seen
classes; the five true unseen classes are evaluated only after selection.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from .sc_nbv import (
    FUTURE_LOGIT_DIM,
    NUM_VIEWS,
    RawViewData,
    SCNBVData,
    SemanticConditionedViewUtilityNet,
    make_selector_model,
    causal_batch,
    evaluate_policy,
    permute_candidates,
    sc_nbv_loss,
    seed_all,
    validation_score,
)


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    tmp.replace(path)


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def _raw_split_path(raw_root: Path, split_name: str) -> Path:
    """Resolve filtered seen-test evaluation to the shared raw test cache."""
    path = Path(raw_root) / split_name
    if not path.exists() and split_name == "seen_test":
        return Path(raw_root) / "test"
    return path


def _training_sampler(
    data: SCNBVData,
    tie_threshold: float,
    priority_fraction: float,
    class_balanced: bool = False,
    four_view_only: bool = False,
    allowed_start_views: list[int] | None = None,
    utility_override: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    valid = data.data["valid"]
    utility = data.data["utility"] if utility_override is None else utility_override
    if utility.shape != data.data["utility"].shape:
        raise ValueError("utility_override must have the same shape as cached utility")
    candidate_count = data.data["candidate_count"]
    best = utility.masked_fill(~valid, -torch.inf).max(-1).values
    worst = utility.masked_fill(~valid, torch.inf).min(-1).values
    spread = best - worst
    informative = (candidate_count > 1) & (spread > float(tie_threshold))
    usable = candidate_count > 0
    if four_view_only:
        # candidate_count excludes the current view, so exactly three legal
        # alternatives means the original recording has all four low views.
        usable &= candidate_count == (NUM_VIEWS - 1)
    if allowed_start_views is not None:
        allowed = torch.as_tensor(
            [int(view) for view in allowed_start_views],
            device=data.device,
            dtype=data.data["start_views"].dtype,
        )
        if allowed.numel() == 0 or (allowed < 0).any() or (allowed >= NUM_VIEWS).any():
            raise ValueError("allowed_start_views must contain valid view indices")
        usable &= torch.isin(data.data["start_views"], allowed)
    indices = usable.nonzero().flatten()
    if class_balanced:
        labels = data.data["labels"][indices]
        counts = torch.bincount(labels, minlength=FUTURE_LOGIT_DIM).float()
        base_weights = counts[labels].clamp_min(1.0).reciprocal()
    else:
        base_weights = torch.ones(len(indices), device=data.device, dtype=torch.float32)
    uniform = base_weights / base_weights.sum().clamp_min(1e-12)
    uniform /= uniform.sum().clamp_min(1e-12)
    selected = informative[indices]
    priority = base_weights * selected.float()
    if priority.sum() > 0:
        priority /= priority.sum()
        probability = (1.0 - float(priority_fraction)) * uniform + float(priority_fraction) * priority
    else:
        probability = uniform
    stats = {
        "contexts": int(len(indices)),
        "all_contexts": int(len(data.data["z"])),
        "four_view_only": bool(four_view_only),
        "four_view_contexts": int((candidate_count == (NUM_VIEWS - 1)).sum()),
        "allowed_start_views": None if allowed_start_views is None else [int(x) for x in allowed_start_views],
        "start_view_contexts": int(usable.sum()),
        "informative_contexts": int(selected.sum()),
        "single_candidate_contexts": int((candidate_count[indices] <= 1).sum()),
        "class_balanced": bool(class_balanced),
    }
    return indices, probability, stats


def _utility_stats(
    data: SCNBVData,
    utility_override: torch.Tensor | None = None,
) -> tuple[float, float]:
    utility = data.data["utility"] if utility_override is None else utility_override
    values = utility[data.data["valid"]]
    mean = float(values.mean())
    std = max(float(values.std()), 1e-4)
    return mean, std


def _full_class_mask(
    labels: torch.Tensor,
    class_pool: torch.Tensor,
) -> torch.Tensor:
    """Build one fixed seen-class task mask for every training context."""

    mask = torch.zeros(
        (labels.shape[0], FUTURE_LOGIT_DIM),
        dtype=torch.bool,
        device=labels.device,
    )
    mask[:, class_pool] = True
    return mask


def _resolve_task_bank_pool(
    spec: Any,
    cfg: Mapping[str, Any],
    train_class_pool: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Resolve the pool used only for dynamic utility-bank sampling.

    Episode labels still come exclusively from the policy-training cache. A
    broader pool changes the open-vocabulary competitors used to construct
    the offline margin, but no class ID is exposed to the causal selector.
    Historical configurations default to the policy-train pool.
    """

    if spec is None:
        return train_class_pool
    if isinstance(spec, str):
        key = spec.strip().lower()
        if key in {"train", "policy_train", "policy-train"}:
            values = cfg["split"]["train_classes"]
        elif key in {"seen", "all_seen", "all-seen"}:
            values = cfg["split"].get("seen_classes", cfg["split"]["train_classes"])
        elif key in {"all", "open_vocab", "open-vocab"}:
            values = list(range(FUTURE_LOGIT_DIM))
        elif key in {"train_plus_pseudo", "train+pseudo", "seen_without_unseen"}:
            values = list(cfg["split"]["train_classes"]) + list(
                cfg["split"].get("pseudo_unseen_classes", [])
            )
        else:
            raise ValueError(f"unknown task_bank_class_pool: {spec}")
    else:
        values = spec
    pool = torch.as_tensor(
        sorted({int(value) for value in values}), device=device, dtype=torch.long
    )
    if pool.numel() < 2:
        raise ValueError("task_bank_class_pool must contain at least two classes")
    if (pool < 0).any() or (pool >= FUTURE_LOGIT_DIM).any():
        raise ValueError("task_bank_class_pool contains an invalid class ID")
    return pool


def _task_bank_mask(task_bank: torch.Tensor) -> torch.Tensor:
    """Convert cached global class IDs into a per-row legal-class mask."""

    if task_bank.ndim != 2:
        raise ValueError("task_bank must have shape [batch, bank_size]")
    mask = torch.zeros(
        (task_bank.shape[0], FUTURE_LOGIT_DIM),
        dtype=torch.bool,
        device=task_bank.device,
    )
    mask.scatter_(1, task_bank.long(), True)
    return mask


def _resampled_task_bank_features(
    batch: Mapping[str, torch.Tensor],
    row_bank: torch.Tensor,
    prototypes: torch.Tensor,
    top_k: int,
    top_prototype_count: int,
    temperature: float,
) -> dict[str, torch.Tensor]:
    """Build a causal state summary for a freshly sampled 5-way bank.

    The cache contains one task bank for reproducible offline labels, but a
    single bank per recording can become an accidental training identifier.
    This helper lets a training batch draw a new seen-only bank while keeping
    the state summary, current-logit summary, and offline utility target on the
    exact same bank.  It intentionally returns no class IDs to the selector.
    """

    if row_bank.ndim != 2 or row_bank.shape[0] != batch["z"].shape[0]:
        raise ValueError("row_bank must have shape [batch, bank_size]")
    if not 1 <= int(top_k) <= row_bank.shape[1]:
        raise ValueError("top_k must fit the sampled task bank")
    if not 1 <= int(top_prototype_count) <= int(top_k):
        raise ValueError("top_prototype_count must fit the sampled task bank")
    # These numpy helpers are used only for their well-tested definition of
    # the class-ID-free summaries.  The batch is small enough that the CPU
    # round trip is avoided by implementing the same operations in torch.
    z = F.normalize(batch["z"].float(), dim=-1)
    text = F.normalize(prototypes.float(), dim=-1)
    row_text = text[row_bank.long()]
    similarity = torch.einsum("bd,bkd->bk", z, row_text)
    probability = F.softmax(similarity / max(float(temperature), 1e-4), dim=-1)
    order = probability.argsort(dim=-1, descending=True)[:, :int(top_k)]
    top_probability = probability.gather(1, order)
    top_probability = top_probability / top_probability.sum(-1, keepdim=True).clamp_min(1e-8)
    top_text = row_text.gather(
        1, order[..., None].expand(-1, -1, row_text.shape[-1])
    )
    semantic = F.normalize(
        (top_probability[..., None] * top_text).sum(1), dim=-1
    )
    entropy = -(probability * probability.clamp_min(1e-8).log()).sum(-1)
    entropy = entropy / math.log(float(row_bank.shape[1]))
    top_gap = top_probability[:, 0] - top_probability[:, 1] if top_k >= 2 else top_probability[:, 0]
    evidence = torch.cat(
        (
            top_probability,
            entropy[:, None],
            top_gap[:, None],
            probability.amax(-1, keepdim=True),
        ),
        dim=-1,
    )
    selected = top_text[:, :int(top_prototype_count)]
    if int(top_prototype_count) < int(top_k):
        selected = selected[:, :int(top_prototype_count)]
    bank_top_semantic = selected.reshape(selected.shape[0], -1)
    # ``bank_top_probability`` is always five-wide in the cache contract.
    bank_top_probability = top_probability[:, :int(top_prototype_count)]
    if bank_top_probability.shape[1] < int(top_prototype_count):
        bank_top_probability = F.pad(
            bank_top_probability,
            (0, int(top_prototype_count) - bank_top_probability.shape[1]),
        )
    current_scores = batch["current_logits"].float().gather(1, row_bank.long())
    bank_current_top5 = current_scores.sort(dim=-1, descending=True).values[:, :5]
    if bank_current_top5.shape[1] < 5:
        bank_current_top5 = F.pad(bank_current_top5, (0, 5 - bank_current_top5.shape[1]), value=-32.0)
    return {
        "bank_semantic": semantic,
        "bank_evidence": evidence,
        "bank_top_semantic": bank_top_semantic,
        "bank_top_probability": bank_top_probability,
        "bank_current_top5": bank_current_top5,
    }


def _transition_auxiliary_loss(
    prediction: torch.Tensor,
    batch: Mapping[str, torch.Tensor],
    state_weight: torch.Tensor,
    class_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Train-only causal transition target from privileged future logits.

    The selector never receives ``future_logits``.  The auxiliary head learns
    a candidate-conditioned residual from current to future VPOCLIP logits,
    which supplies a denser visual-transition signal to the shared skeleton /
    geometry representation.  When a five-way task bank is present, only its
    columns are supervised so the head is not encouraged to memorize unused
    class IDs.
    """

    required = {"current_logits", "future_logits"}
    missing = required.difference(batch)
    if missing:
        raise KeyError(f"missing auxiliary target fields: {sorted(missing)}")
    target = (
        batch["future_logits"] - batch["current_logits"][:, None, :]
    ) / 10.0
    error = F.smooth_l1_loss(prediction, target, reduction="none")
    mask = batch["valid"].float()[:, :, None]
    task_bank = batch.get("task_bank")
    if class_mask is not None:
        mask = mask * class_mask.float()[:, None, :]
    elif task_bank is not None:
        class_mask = torch.zeros(
            (*task_bank.shape[:1], FUTURE_LOGIT_DIM),
            dtype=torch.float32,
            device=task_bank.device,
        )
        class_mask.scatter_(1, task_bank.long(), 1.0)
        mask = mask * class_mask[:, None, :]
    mask = mask * state_weight[:, None, None]
    return (error * mask).sum() / mask.sum().clamp_min(1.0)


def _sample_task_masks(
    labels: torch.Tensor,
    class_pool: torch.Tensor,
    generator: torch.Generator,
    task_size: int = 5,
    full_fraction: float = 0.25,
) -> torch.Tensor:
    """Sample five-way tasks while always retaining the training GT class."""

    if not 2 <= task_size <= int(class_pool.numel()):
        raise ValueError("task_size must be between 2 and the class-pool size")
    random_scores = torch.rand(
        (labels.shape[0], class_pool.numel()),
        device=labels.device,
        generator=generator,
    )
    is_label = class_pool[None, :] == labels[:, None]
    random_scores = random_scores.masked_fill(is_label, 2.0)
    selected = random_scores.topk(task_size, dim=-1).indices
    five_way = torch.zeros(
        (labels.shape[0], FUTURE_LOGIT_DIM),
        dtype=torch.bool,
        device=labels.device,
    )
    five_way.scatter_(1, class_pool[selected], True)
    full = torch.rand(
        (labels.shape[0], 1), device=labels.device, generator=generator
    ) < float(full_fraction)
    full_way = torch.zeros_like(five_way)
    full_way[:, class_pool] = True
    return torch.where(full, full_way, five_way)


def _utility_from_task_mask(
    current_logits: torch.Tensor,
    future_logits: torch.Tensor,
    labels: torch.Tensor,
    valid: torch.Tensor,
    class_mask: torch.Tensor,
    mode: str,
    temperature: float = 2.0,
) -> torch.Tensor:
    """Create causal offline utility targets for a sampled seen-only task.

    ``delta_margin`` and ``correctness_margin`` are the original privileged
    GT-margin targets.  The label-free modes are an additional transfer
    experiment: they use only the current recognizer distribution and the
    privileged future logits to form a training target.  The GT label is not
    consulted in those branches, so the selector is trained to prefer a
    confidence-reducing / complementary next view rather than a seen-class
    identity-specific action.
    """

    if mode not in {
        "delta_margin",
        "correctness_margin",
        "entropy_gain",
        "js_divergence",
        "entropy_js",
    }:
        raise ValueError(f"unknown dynamic utility mode: {mode}")
    fused = 0.5 * (current_logits[:, None, :] + future_logits)
    masked_fused = fused.masked_fill(~class_mask[:, None, :], -torch.inf)
    if mode in {"entropy_gain", "js_divergence", "entropy_js"}:
        # Logits are from a frozen recognizer and are not calibrated
        # probabilities.  A moderate temperature prevents one overconfident
        # current class from making the label-free target effectively binary.
        temp = max(float(temperature), 1e-4)
        masked_current = current_logits.masked_fill(~class_mask, -torch.inf)
        current_probability = F.softmax(masked_current / temp, dim=-1)
        fused_probability = F.softmax(masked_fused / temp, dim=-1)
        current_log_probability = current_probability.clamp_min(1e-8).log()
        fused_log_probability = fused_probability.clamp_min(1e-8).log()
        current_entropy = -(
            current_probability * current_log_probability
        ).sum(-1)
        fused_entropy = -(
            fused_probability * fused_log_probability
        ).sum(-1)
        entropy_gain = current_entropy[:, None] - fused_entropy
        if mode == "entropy_gain":
            utility = entropy_gain
        else:
            midpoint = 0.5 * (current_probability[:, None, :] + fused_probability)
            log_midpoint = midpoint.clamp_min(1e-8).log()
            js = 0.5 * (
                current_probability[:, None, :]
                * (current_log_probability[:, None, :] - log_midpoint)
            ).sum(-1)
            js = js + 0.5 * (
                fused_probability * (fused_log_probability - log_midpoint)
            ).sum(-1)
            utility = js if mode == "js_divergence" else entropy_gain + 0.5 * js
        return utility.masked_fill(~valid, 0.0)
    true = fused.gather(-1, labels[:, None, None].expand(-1, NUM_VIEWS, 1)).squeeze(-1)
    rivals = class_mask.clone()
    rivals.scatter_(1, labels[:, None], False)
    rival = masked_fused.masked_fill(~rivals[:, None, :], -torch.inf).amax(-1)
    margin = true - rival
    if mode == "correctness_margin":
        utility = masked_fused.argmax(-1).eq(labels[:, None]).float()
        utility = utility + 0.20 * torch.tanh(margin / 10.0)
    else:
        current_scores = current_logits.masked_fill(~class_mask, -torch.inf)
        current_true = current_logits.gather(-1, labels[:, None]).squeeze(-1)
        current_rival = current_scores.masked_fill(~rivals, -torch.inf).amax(-1)
        utility = margin - (current_true - current_rival)[:, None]
    return utility.masked_fill(~valid, 0.0)


def _short_metrics(report: Mapping[str, Any], criterion: str = "top1_gain") -> dict[str, Any]:
    metrics = report["metrics"]
    return {
        "sc_nbv_top1": metrics["sc_nbv"]["top1"],
        "random_expected_top1": metrics["random_expected"]["top1"],
        "gain_vs_random_expected": validation_score(report, criterion=criterion),
        "sc_nbv_mean_margin_gain": metrics["sc_nbv"]["mean_margin_gain"],
        "sc_nbv_mean_oracle_regret": metrics["sc_nbv"]["mean_oracle_regret"],
        "sc_nbv_oracle_hit_top1": metrics["sc_nbv"]["oracle_hit_top1"],
    }


def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    best_score: float,
    best_epoch: int,
    variant: str,
    seed: int,
    config: Mapping[str, Any],
    utility_mean: float,
    utility_std: float,
    sampler_generator: torch.Generator,
) -> None:
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": int(epoch),
        "best_score": float(best_score),
        "best_epoch": int(best_epoch),
        "variant": variant,
        "seed": int(seed),
        "config": dict(config),
        "utility_mean": float(utility_mean),
        "utility_std": float(utility_std),
        "sampler_state": sampler_generator.get_state(),
        "cpu_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        "causal_inputs_only": True,
        "future_view_features_in_model": False,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def train_seed(
    variant: str,
    variant_config: Mapping[str, Any],
    train: SCNBVData,
    pseudo: SCNBVData,
    raw_pseudo: RawViewData,
    cfg: Mapping[str, Any],
    seed: int,
    out: Path,
    device: torch.device,
) -> dict[str, Any]:
    run = out / f"seed_{seed}"
    run.mkdir(parents=True, exist_ok=True)
    complete_path = run / "complete.json"
    if complete_path.exists():
        return json.loads(complete_path.read_text(encoding="utf-8"))
    seed_all(seed)
    generator = torch.Generator(device=device).manual_seed(seed)
    model = make_selector_model(variant_config).to(device)
    learning_rate = float(variant_config.get("learning_rate", 2e-4))
    weight_decay = float(variant_config.get("weight_decay", 0.01))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    epochs = int(cfg["train"]["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=learning_rate * 0.05)
    tie_threshold = float(variant_config.get("tie_threshold", cfg["data"]["tie_threshold"]))
    validation_metric = str(cfg["train"].get("validation_metric", "top1_gain"))
    rank_lambda = float(variant_config.get("rank_lambda", 0.5))
    rank_base_margin = float(variant_config.get("rank_base_margin", cfg["train"]["rank_base_margin"]))
    rank_utility_scale = float(variant_config.get("rank_utility_scale", cfg["train"]["rank_utility_scale"]))
    auxiliary_enabled = bool(variant_config.get("auxiliary_transition", False))
    auxiliary_weight = float(variant_config.get("auxiliary_weight", 0.0))
    dynamic_tasks = bool(variant_config.get("dynamic_task_banks", False))
    resample_task_banks = bool(variant_config.get("resample_task_banks", False))
    dynamic_utility = str(variant_config.get("utility_mode", "delta_margin"))
    utility_temperature = float(variant_config.get("utility_temperature", 2.0))
    centered_targets = bool(variant_config.get("centered_targets", False))
    loss_mode = str(variant_config.get("loss_mode", "huber"))
    listwise_temperature = float(variant_config.get("listwise_temperature", 0.20))
    oracle_ce_weight = float(variant_config.get("oracle_ce_weight", 0.0))
    oracle_ce_min_gap = float(variant_config.get("oracle_ce_min_gap", tie_threshold))
    task_size = int(variant_config.get("task_bank_size", 5))
    task_full_fraction = float(variant_config.get("task_full_fraction", 0.25))
    top_k = int(cfg["data"].get("top_k", 5))
    top_prototype_count = int(cfg["data"].get("top_prototype_count", 5))
    if resample_task_banks and not dynamic_tasks:
        raise ValueError("resample_task_banks requires dynamic_task_banks")
    if resample_task_banks and task_full_fraction != 0.0:
        raise ValueError("resample_task_banks currently requires task_full_fraction=0")
    if resample_task_banks and task_size != 5:
        raise ValueError("resample_task_banks currently requires a five-way task bank")
    if resample_task_banks and raw_pseudo.prototypes is None:
        raise FileNotFoundError(
            f"resample_task_banks requires text_prototypes.npy in {raw_pseudo.root}"
        )
    train_class_pool = torch.as_tensor(
        cfg["split"]["train_classes"], device=device, dtype=torch.long
    )
    task_bank_class_pool = _resolve_task_bank_pool(
        variant_config.get("task_bank_class_pool", "train"),
        cfg,
        train_class_pool,
        device,
    )
    if not torch.isin(train.data["labels"], train_class_pool).all():
        raise ValueError("training cache contains labels outside split.train_classes")
    if not torch.isin(train.data["labels"], task_bank_class_pool).all():
        raise ValueError(
            "task_bank_class_pool must contain every policy-training label"
        )
    static_utility_mode = variant_config.get("static_utility_mode")
    static_utility: torch.Tensor | None = None
    if static_utility_mode is not None:
        static_utility_mode = str(static_utility_mode)
        if not {"current_logits", "future_logits"}.issubset(train.data):
            raise KeyError(
                "static_utility_mode requires train cache current_logits and future_logits"
            )
        static_mask = _full_class_mask(train.data["labels"], train_class_pool)
        static_utility = _utility_from_task_mask(
            train.data["current_logits"],
            train.data["future_logits"],
            train.data["labels"],
            train.data["valid"],
            static_mask,
            static_utility_mode,
            temperature=utility_temperature,
        )
    utility_mean, utility_std = _utility_stats(train, static_utility)
    indices, probabilities, sample_stats = _training_sampler(
        train,
        tie_threshold,
        float(cfg["train"].get("priority_fraction", 0.5)),
        class_balanced=bool(variant_config.get("class_balanced", False)),
        four_view_only=bool(variant_config.get("four_view_only", cfg["train"].get("four_view_only", False))),
        allowed_start_views=(
            [int(view) for view in variant_config["training_start_views"]]
            if variant_config.get("training_start_views") is not None else None
        ),
        utility_override=static_utility,
    )
    batch_size = int(cfg["train"]["batch_size"])
    batches = int(cfg["train"].get("batches_per_epoch", 0))
    if batches <= 0:
        batches = max(1, math.ceil(len(indices) / batch_size))
    audit = {
        "format": "sc_nbv_training_audit_v1",
        "variant": variant,
        "seed": int(seed),
        "variant_config": dict(variant_config),
        "train_cache": str(train.root),
        "pseudo_unseen_cache": str(pseudo.root),
        "train_contexts": int(train.num_contexts),
        "pseudo_unseen_contexts": int(pseudo.num_contexts),
        "sample_stats": sample_stats,
        "utility_mean": utility_mean,
        "utility_std": utility_std,
        "loss": ("listwise_soft_target" if loss_mode == "listwise" else "Huber(delta_margin)" )
        + " + rank_lambda * pairwise_margin_ranking"
        + (" + separated-oracle-action cross_entropy" if oracle_ce_weight > 0.0 else "")
        + (" + auxiliary future-logit transition" if auxiliary_enabled else ""),
        "loss_mode": loss_mode,
        "listwise_temperature": listwise_temperature,
        "oracle_ce_weight": oracle_ce_weight,
        "oracle_ce_min_gap": oracle_ce_min_gap,
        "tie_threshold": tie_threshold,
        "rank_lambda": rank_lambda,
        "future_view_features_in_model": False,
        "ground_truth_used_only_for_offline_utility": True,
        "auxiliary_transition": auxiliary_enabled,
        "auxiliary_weight": auxiliary_weight,
        "class_balanced": bool(variant_config.get("class_balanced", False)),
        "dynamic_task_banks": dynamic_tasks,
        "resample_task_banks": resample_task_banks,
        "task_bank_class_pool": task_bank_class_pool.detach().cpu().tolist(),
        "dynamic_utility": dynamic_utility,
        "utility_temperature": utility_temperature,
        "static_utility_mode": static_utility_mode,
        "training_utility": (
            "fixed seen-bank correctness + margin"
            if static_utility_mode == "correctness_margin"
            else "fixed seen-bank " + str(static_utility_mode)
            if static_utility_mode is not None
            else "cache delta_margin"
        ),
        "centered_targets": centered_targets,
        "class_split": {
            "train": cfg["split"]["train_classes"],
            "pseudo_unseen": cfg["split"]["pseudo_unseen_classes"],
            "true_unseen": cfg["split"]["unseen_classes"],
        },
    }
    dump(run / "audit.json", audit)
    logs: list[dict[str, Any]] = []
    best_score = -float("inf")
    best_epoch = 0
    for epoch in range(1, epochs + 1):
        started = time.monotonic()
        model.train()
        losses: list[torch.Tensor] = []
        hubers: list[float] = []
        rankings: list[float] = []
        listwises: list[float] = []
        oracle_ces: list[float] = []
        auxiliaries: list[float] = []
        for _ in range(batches):
            sample_ids = torch.multinomial(probabilities, batch_size, replacement=True, generator=generator)
            batch = train.batch(indices[sample_ids])
            batch = permute_candidates(batch, generator)
            if dynamic_tasks:
                if resample_task_banks:
                    # Draw a fresh five-way open-vocabulary task for every
                    # update; the episode GT label is still from train only.
                    # Rebuild all bank-conditioned summaries from that same
                    # bank so the causal state and offline target stay
                    # aligned.  The row bank itself is training metadata and
                    # is excluded by causal_batch().
                    dynamic_mask = _sample_task_masks(
                        batch["labels"],
                        task_bank_class_pool,
                        generator,
                        task_size=task_size,
                        full_fraction=0.0,
                    )
                    row_bank = dynamic_mask.nonzero(as_tuple=False)[:, 1].reshape(
                        batch["labels"].shape[0], task_size
                    )
                    batch.update(
                        _resampled_task_bank_features(
                            batch,
                            row_bank,
                            raw_pseudo.prototypes,
                            top_k=top_k,
                            top_prototype_count=top_prototype_count,
                            temperature=float(
                                cfg["data"].get("semantic_temperature", 0.07)
                            ),
                        )
                    )
                elif "task_bank" in batch:
                    # The cache builder stores one sampled 5-way bank per
                    # episode.  Use that exact bank for both the semantic
                    # state summary and the offline utility label.
                    dynamic_mask = _task_bank_mask(batch["task_bank"])
                else:
                    dynamic_mask = _sample_task_masks(
                        batch["labels"],
                        task_bank_class_pool,
                        generator,
                        task_size=task_size,
                        full_fraction=task_full_fraction,
                    )
                utility_target = _utility_from_task_mask(
                    batch["current_logits"],
                    batch["future_logits"],
                    batch["labels"],
                    batch["valid"],
                    dynamic_mask,
                    dynamic_utility,
                    temperature=utility_temperature,
                )
            else:
                dynamic_mask = None
                if static_utility_mode is not None:
                    # Recompute after candidate-slot permutation so the
                    # candidate targets stay aligned with the randomized
                    # action slots.  This remains an offline label only.
                    static_mask = _full_class_mask(batch["labels"], train_class_pool)
                    utility_target = _utility_from_task_mask(
                        batch["current_logits"],
                        batch["future_logits"],
                        batch["labels"],
                        batch["valid"],
                        static_mask,
                        static_utility_mode,
                        temperature=utility_temperature,
                    )
                else:
                    utility_target = batch["utility"]
            if str(variant_config.get("architecture", "semantic_mlp")) == "factorized_skeleton":
                output = model(causal_batch(batch), return_aux=auxiliary_enabled)
            else:
                output = model(causal_batch(batch))
            if auxiliary_enabled:
                prediction, transition_prediction = output
            else:
                prediction = output
                transition_prediction = None
            loss, parts = sc_nbv_loss(
                prediction,
                utility_target,
                batch["valid"],
                batch["state_weight"],
                utility_mean,
                utility_std,
                rank_lambda=rank_lambda,
                tie_threshold=tie_threshold,
                base_margin=rank_base_margin,
                utility_margin_scale=rank_utility_scale,
                centered_targets=centered_targets,
                loss_mode=loss_mode,
                listwise_temperature=listwise_temperature,
                oracle_ce_weight=oracle_ce_weight,
                oracle_ce_min_gap=oracle_ce_min_gap,
            )
            if auxiliary_enabled:
                aux_loss = _transition_auxiliary_loss(
                    transition_prediction,
                    batch,
                    batch["state_weight"],
                    class_mask=dynamic_mask,
                )
                loss = loss + auxiliary_weight * aux_loss
                auxiliaries.append(float(aux_loss.detach()))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(loss.detach())
            hubers.append(parts["huber"])
            rankings.append(parts["ranking"])
            listwises.append(parts.get("listwise", 0.0))
            oracle_ces.append(parts.get("oracle_ce", 0.0))
        scheduler.step()
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
        val_report = evaluate_policy(
            model,
            pseudo,
            raw_pseudo,
            cfg["split"]["pseudo_unseen_classes"],
            protocol=str(cfg["train"].get("validation_protocol", "fixed0")),
            seed=int(cfg["evaluation"].get("random_seed", 20260909)),
        )
        score = validation_score(
            val_report,
            criterion=validation_metric,
        )
        is_best = score > best_score
        if is_best:
            best_score = score
            best_epoch = epoch
        _save_checkpoint(
            run / "last.pt", model, optimizer, scheduler, epoch, best_score, best_epoch,
            variant, seed, variant_config, utility_mean, utility_std, generator,
        )
        if is_best:
            _save_checkpoint(
                run / "best.pt", model, optimizer, scheduler, epoch, best_score, best_epoch,
                variant, seed, variant_config, utility_mean, utility_std, generator,
            )
        row = {
            "variant": variant,
            "seed": int(seed),
            "epoch": epoch,
            "loss": float(torch.stack(losses).mean()),
            "huber": float(np.mean(hubers)),
            "ranking": float(np.mean(rankings)),
            "listwise": float(np.mean(listwises)),
            "oracle_ce": float(np.mean(oracle_ces)),
            "auxiliary": float(np.mean(auxiliaries)) if auxiliaries else 0.0,
            "seconds": time.monotonic() - started,
            "batches": batches,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "validation": _short_metrics(val_report, criterion=validation_metric),
            "selection_score": score,
            "validation_metric": validation_metric,
            "best_score": best_score,
            "best_epoch": best_epoch,
        }
        logs.append(row)
        with (run / "train.log").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
    result = {
        "variant": variant,
        "seed": int(seed),
        "best_score": float(best_score),
        "best_epoch": int(best_epoch),
        "path": str(run),
        "sample_stats": sample_stats,
    }
    dump(complete_path, result)
    del model, optimizer, scheduler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def _load_model(checkpoint: Path, variant_config: Mapping[str, Any], device: torch.device) -> torch.nn.Module:
    model = make_selector_model(variant_config).to(device)
    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(saved["model"])
    model.eval()
    return model


def _reference_metrics(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    metrics = value.get("metrics", {})
    fixed = metrics.get("test_fixed0")
    if not fixed:
        return None
    return {
        "source": str(path),
        "description": "previous g18 reference on the same frozen VPOCLIP cache",
        "fixed0": {
            "single": fixed.get("single", {}).get("top1"),
            "fixed": fixed.get("fixed", {}).get("top1"),
            "random": fixed.get("random", {}).get("top1"),
            "policy": fixed.get("policy", {}).get("top1"),
            "oracle": fixed.get("oracle", {}).get("top1"),
            "random_exact": fixed.get("random_exact", {}).get("top1"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="rl/handoff_configs/sc_nbv_current_v1.yaml")
    parser.add_argument("--force", action="store_true", help="ignore existing complete files")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    cfg_path = _resolve(root, args.config)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    torch.set_num_threads(int(cfg.get("runtime", {}).get("num_threads", 4)))
    torch.set_float32_matmul_precision(str(cfg.get("runtime", {}).get("matmul_precision", "high")))
    device = torch.device(cfg["runtime"]["device"])
    cache_root = _resolve(root, cfg["sc_cache"])
    output_root = _resolve(root, cfg["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    dump(output_root / "config_resolved.json", cfg)
    train = SCNBVData(cache_root / "train", device)
    pseudo = SCNBVData(cache_root / "pseudo_unseen", device)
    # Validate the loaded cache, not just its declared configuration. Older
    # bundles can contain pseudo-unseen classes in the policy-training pool.
    actual_train = set(train.data["labels"].unique().cpu().tolist())
    actual_pseudo = set(pseudo.data["labels"].unique().cpu().tolist())
    overlap = sorted(actual_train & actual_pseudo)
    if overlap:
        raise ValueError(f"training/pseudo-unseen cache class overlap: {overlap}")
    for name, actual, expected in (
        ("train", actual_train, cfg["split"]["train_classes"]),
        ("pseudo_unseen", actual_pseudo, cfg["split"]["pseudo_unseen_classes"]),
    ):
        if not actual.issubset(set(expected)):
            raise ValueError(f"{name} cache labels disagree with configured class split")
        leaked = sorted(actual & set(cfg["split"]["unseen_classes"]))
        if leaked:
            raise ValueError(f"true-unseen labels in {name} cache: {leaked}")
    raw_pseudo = RawViewData(_raw_split_path(_resolve(root, cfg["raw_cache"]), "val"), device)
    seeds = [int(seed) for seed in cfg["runtime"]["seeds"]]
    all_results: dict[str, Any] = {
        "format": "sc_nbv_experiment_v1",
        "config": str(cfg_path),
        "recognizer_checkpoint": cfg["recognizer"]["checkpoint"],
        "raw_cache": cfg["raw_cache"],
        "sc_cache": cfg["sc_cache"],
        "protocol": {
            "train": "policy-train seen classes only",
            "selection": "pseudo-unseen held-out seen classes only",
            "final_test": "five real unseen classes after selection",
            "primary_start": "fixed0",
            "secondary_start": "random_start",
            "future_view_features_in_model": False,
        },
        "variants": {},
    }
    for variant, variant_config in cfg["train"]["variants"].items():
        variant_out = output_root / variant
        variant_out.mkdir(parents=True, exist_ok=True)
        records = []
        for seed in seeds:
            if args.force and (variant_out / f"seed_{seed}" / "complete.json").exists():
                # A force run writes into the same version only when explicitly requested;
                # keeping old files would make the result ambiguous, so use a fresh suffix.
                raise RuntimeError("--force requires a fresh output_root to avoid mixed checkpoints")
            print("SC_NBV_TRAIN_START", variant, seed, flush=True)
            records.append(train_seed(variant, variant_config, train, pseudo, raw_pseudo, cfg, seed, variant_out, device))
        validation_mean = float(np.mean([record["best_score"] for record in records]))
        validation_min = float(np.min([record["best_score"] for record in records]))
        selection = {
            "variant": variant,
            "seeds": records,
            "mean_pseudo_unseen_gain": validation_mean,
            "min_pseudo_unseen_gain": validation_min,
            "criterion": f"mean fixed0 pseudo-unseen {cfg['train'].get('validation_metric', 'top1_gain')} gain over expected random",
            "test_was_not_used_for_selection": True,
        }
        dump(variant_out / "selection.json", selection)
        final_reports: dict[str, Any] = {}
        for split_name, raw_split, class_key in (
            ("seen_test", "seen_test", "seen_classes"),
            ("unseen", "test", "unseen_classes"),
        ):
            context = SCNBVData(cache_root / split_name, device)
            raw = RawViewData(
                _raw_split_path(_resolve(root, cfg["raw_cache"]), raw_split), device
            )
            split_reports = {}
            for record in records:
                model = _load_model(Path(record["path"]) / "best.pt", variant_config, device)
                per_seed = {}
                for protocol in cfg["evaluation"]["protocols"]:
                    report = evaluate_policy(
                        model,
                        context,
                        raw,
                        cfg["split"][class_key],
                        protocol=protocol,
                        seed=int(cfg["evaluation"].get("random_seed", 20260909)),
                    )
                    report_path = variant_out / f"evaluation_{split_name}_{protocol}_seed_{record['seed']}.json"
                    dump(report_path, report)
                    per_seed[protocol] = report["metrics"]
                split_reports[str(record["seed"])] = per_seed
                del model
            final_reports[split_name] = split_reports
            del context, raw
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        result = {
            "selection": selection,
            "validation": {
                "mean_gain": validation_mean,
                "min_gain": validation_min,
            },
            "final_evaluation": final_reports,
        }
        dump(variant_out / "summary.json", result)
        all_results["variants"][variant] = result
        print("SC_NBV_VARIANT_COMPLETE", variant, json.dumps(result["validation"]), flush=True)
    ranked = sorted(
        all_results["variants"].items(),
        key=lambda item: item[1]["validation"]["mean_gain"],
        reverse=True,
    )
    all_results["selected_variant"] = ranked[0][0] if ranked else None
    all_results["selection"] = {
        "criterion": "pseudo-unseen validation only",
        "ranked": [(name, value["validation"]) for name, value in ranked],
        "test_was_not_used": True,
    }
    all_results["reference_g18"] = _reference_metrics(_resolve(root, cfg["evaluation"]["reference_g18_summary"]))
    dump(output_root / "summary.json", all_results)
    print(json.dumps(all_results, indent=2), flush=True)


if __name__ == "__main__":
    main()
