import argparse
import copy
import json
import os
import random
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as Data
from torch.utils.data import Dataset, Subset
from torch.utils.data import WeightedRandomSampler

from model import apply_action_text_bank, build_model_from_config, load_action_descriptions

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def load_config(config_path):
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("Please install PyYAML first: pip install pyyaml") from exc

    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    # Small experiment configs can inherit the full baseline and override only
    # the settings under study. Existing standalone configs are unchanged.
    base_config = config.pop("base_config", None)
    if base_config:
        base_path = get_path_from_config(config_path, base_config)
        if os.path.abspath(base_path) == os.path.abspath(config_path):
            raise ValueError("A config cannot inherit itself.")
        base = load_config(base_path)

        def merge(target, overrides):
            for key, value in overrides.items():
                if isinstance(value, dict) and isinstance(target.get(key), dict):
                    merge(target[key], value)
                else:
                    target[key] = value
            return target

        config = merge(base, config)
    return config


def get_path_from_config(config_path, path):
    if path is None or os.path.isabs(str(path)):
        return path
    config_dir = os.path.dirname(os.path.abspath(config_path))
    return os.path.join(config_dir, str(path))


def get_device(device_name=None):
    if device_name is None:
        device_name = "cuda:0" if torch.cuda.is_available() else "cpu"

    if str(device_name).startswith("cuda"):
        if not torch.cuda.is_available():
            print("CUDA is not available, using CPU instead.")
            return torch.device("cpu")
        if ":" in str(device_name):
            gpu_id = int(str(device_name).split(":")[1])
            if gpu_id >= torch.cuda.device_count():
                raise ValueError(f"GPU {gpu_id} does not exist. Available GPU count: {torch.cuda.device_count()}")
    return torch.device(device_name)


def print_device_info(device):
    if torch.cuda.is_available():
        print("Available GPUs:")
        for gpu_id in range(torch.cuda.device_count()):
            print(f"  cuda:{gpu_id} - {torch.cuda.get_device_name(gpu_id)}")
    print(f"Using device: {device}")


def progress_bar(data_loader, desc):
    if tqdm is None:
        print(f"{desc}...")
        return data_loader
    return tqdm(data_loader, desc=desc, leave=False)


class TrimodalContrastiveDataset(Dataset):
    def __init__(self, data_dir, prefix="trimodal_train", mmap=True, zero_streams=(), pose_suffix="",
                 augment=None):
        self.data_dir = Path(data_dir)
        self.prefix = prefix
        # 消融: data.zero_streams 指定要置零的模态 (video/pose/object)。
        # pose 置零时连带 joint_xy, 因为 joint_xy 只服务于 pose 分支的空间放置。
        self.zero_streams = set(zero_streams or [])
        unknown = self.zero_streams - {"video", "pose", "object"}
        if unknown:
            raise ValueError(f"zero_streams only supports video/pose/object, got: {sorted(unknown)}")
        # Training-time regularisation, all off by default so existing configs
        # keep their behaviour. train_acc hits 1.0 by epoch ~7-43 on 5872 clips,
        # so the features are memorised long before the schedule ends.
        augment = augment or {}
        self.noise_std = float(augment.get("noise_std", 0.0))
        self.time_jitter = float(augment.get("time_jitter", 0.0))
        self.modality_dropout = float(augment.get("modality_dropout", 0.0))
        self.augment_enabled = bool(augment.get("enabled", False))

        mmap_mode = "r" if mmap else None

        self.video = np.load(self.data_dir / f"{prefix}_video.npy", mmap_mode=mmap_mode, allow_pickle=False)
        # pose_suffix picks an alternative skeleton backbone, e.g. "_coco17"
        # for the RTMPose features that match what the robot runs.
        self.pose = np.load(self.data_dir / f"{prefix}_pose{pose_suffix}.npy", mmap_mode=mmap_mode, allow_pickle=False)
        self.object = np.load(self.data_dir / f"{prefix}_object.npy", mmap_mode=mmap_mode, allow_pickle=False)
        self.joint_xy = np.load(self.data_dir / f"{prefix}_joint_xy{pose_suffix}.npy", mmap_mode=mmap_mode, allow_pickle=False)
        self.labels = np.load(self.data_dir / f"{prefix}_labels.npy", mmap_mode=mmap_mode, allow_pickle=False)

        sample_path = self.data_dir / f"{prefix}_sample_names.npy"
        self.sample_names = np.load(sample_path, allow_pickle=True) if sample_path.exists() else None

        lengths = {len(self.video), len(self.pose), len(self.object), len(self.joint_xy), len(self.labels)}
        if len(lengths) != 1:
            raise ValueError(
                "Modality lengths are not aligned: "
                f"video={len(self.video)}, pose={len(self.pose)}, object={len(self.object)}, "
                f"joint_xy={len(self.joint_xy)}, labels={len(self.labels)}"
            )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        item = {
            "video": torch.from_numpy(np.array(self.video[index], copy=True)).float(),
            "pose": torch.from_numpy(np.array(self.pose[index], copy=True)).float(),
            "object": torch.from_numpy(np.array(self.object[index], copy=True)).float(),
            "joint_xy": torch.from_numpy(np.array(self.joint_xy[index], copy=True)).float(),
            "label": torch.as_tensor(int(self.labels[index]), dtype=torch.long),
        }
        for stream in self.zero_streams:
            item[stream] = torch.zeros_like(item[stream])
        if "pose" in self.zero_streams:
            item["joint_xy"] = torch.zeros_like(item["joint_xy"])

        if self.augment_enabled:
            item = self._augment(item)
        return item

    def _augment(self, item):
        if self.noise_std > 0:
            for stream in ("video", "pose", "object"):
                item[stream] = item[stream] + torch.randn_like(item[stream]) * self.noise_std

        if self.time_jitter > 0 and torch.rand(1).item() < self.time_jitter:
            # Roll the 13-frame axis by one step; video is [T,C,H,W] and pose is
            # [M,C,T,V], so the time axis differs per stream.
            shift = int(torch.randint(-1, 2, (1,)).item())
            if shift:
                item["video"] = torch.roll(item["video"], shift, dims=0)
                item["pose"] = torch.roll(item["pose"], shift, dims=2)
                item["joint_xy"] = torch.roll(item["joint_xy"], shift, dims=0)

        if self.modality_dropout > 0:
            # Drop at most one stream, so a clip never loses everything. This is
            # also what the robot faces when the camera is occluded or the pose
            # estimator loses the person.
            if torch.rand(1).item() < self.modality_dropout:
                stream = ("video", "pose", "object")[int(torch.randint(0, 3, (1,)).item())]
                item[stream] = torch.zeros_like(item[stream])
                if stream == "pose":
                    item["joint_xy"] = torch.zeros_like(item["joint_xy"])
        return item


def stratified_split_indices(labels, val_fraction, seed):
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    train_indices = []
    val_indices = []

    for label in sorted(np.unique(labels).tolist()):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        val_count = int(round(len(indices) * val_fraction))
        if val_fraction > 0 and len(indices) > 1:
            val_count = min(max(val_count, 1), len(indices) - 1)
        val_indices.extend(indices[:val_count].tolist())
        train_indices.extend(indices[val_count:].tolist())

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


def dataset_labels(dataset):
    if isinstance(dataset, Subset):
        base_labels = dataset_labels(dataset.dataset)
        return np.asarray(base_labels)[np.asarray(dataset.indices, dtype=np.int64)]
    if hasattr(dataset, "labels"):
        return np.asarray(dataset.labels)
    raise TypeError(f"Cannot extract labels from dataset type: {type(dataset).__name__}")


def build_class_balanced_sampler(dataset):
    labels = dataset_labels(dataset).astype(np.int64)
    classes, counts = np.unique(labels, return_counts=True)
    class_weights = {int(label): 1.0 / float(count) for label, count in zip(classes, counts)}
    sample_weights = np.asarray([class_weights[int(label)] for label in labels], dtype=np.float64)
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )


def build_dataloaders(config, config_path):
    data_config = config["data"]
    loader_config = data_config["dataloader"]
    train_config = data_config["train"]
    val_config = data_config.get("val", {})
    split_config = data_config.get("validation_split", {})

    zero_streams = data_config.get("zero_streams") or []
    if zero_streams:
        print(f"[ablation] zero_streams = {sorted(zero_streams)}")

    train_dataset = TrimodalContrastiveDataset(
        data_dir=get_path_from_config(config_path, train_config["data_dir"]),
        prefix=train_config.get("prefix", "trimodal_train"),
        mmap=train_config.get("mmap", True),
        zero_streams=zero_streams,
        pose_suffix=train_config.get("pose_suffix", ""),
        augment=data_config.get("augment"),
    )

    if val_config.get("data_dir"):
        val_dataset = TrimodalContrastiveDataset(
            data_dir=get_path_from_config(config_path, val_config["data_dir"]),
            prefix=val_config.get("prefix", "trimodal_train"),
            mmap=val_config.get("mmap", True),
            zero_streams=zero_streams,
            pose_suffix=val_config.get("pose_suffix", ""),
        )
    else:
        val_fraction = float(split_config.get("fraction", 0.1))
        train_indices, val_indices = stratified_split_indices(
            train_dataset.labels,
            val_fraction=val_fraction,
            seed=int(split_config.get("seed", 20260616)),
        )
        val_dataset = Subset(train_dataset, val_indices)
        train_dataset = Subset(train_dataset, train_indices)

    # Pseudo-unseen protocol: drop these classes' training clips but leave their
    # text prototypes in the contrastive bank, exactly matching the condition of
    # the real unseen classes at test time.
    # Two ways to get a held-out-class validation set:
    #   data.pseudo_val  - an explicit source, e.g. the real unseen classes taken
    #                      from validation subjects (diagnostic use)
    #   train.pseudo_unseen - carve classes out of the training set itself
    pseudo_config = data_config.get("pseudo_val") or {}
    pseudo_unseen = [int(c) for c in (train_config.get("pseudo_unseen") or [])]
    pseudo_val_dataset = None

    if pseudo_unseen:
        train_labels = dataset_labels(train_dataset)
        keep = ~np.isin(train_labels, pseudo_unseen)
        print(f"[pseudo-unseen] {sorted(pseudo_unseen)}: dropped "
              f"{int((~keep).sum())}/{len(train_labels)} training clips")
        train_dataset = Subset(train_dataset, np.flatnonzero(keep).tolist())

    if pseudo_config:
        pseudo_classes = [int(c) for c in pseudo_config["classes"]]
        parts = []
        for entry in pseudo_config["sources"]:
            source = TrimodalContrastiveDataset(
                data_dir=get_path_from_config(config_path, entry["data_dir"]),
                prefix=entry.get("prefix", "val"),
                mmap=entry.get("mmap", True),
                zero_streams=zero_streams,
                pose_suffix=entry.get("pose_suffix", ""),
            )
            keep = np.flatnonzero(np.isin(dataset_labels(source), pseudo_classes)).tolist()
            if not keep:
                raise ValueError(f"No clips of classes {pseudo_classes} in {entry['data_dir']}")
            parts.append(Subset(source, keep))
            print(f"[pseudo-val] {len(keep)} clips from "
                  f"{entry['data_dir']}/{entry.get('prefix', 'val')}")
        pseudo_val_dataset = parts[0] if len(parts) == 1 else Data.ConcatDataset(parts)
        print(f"[pseudo-val] total {len(pseudo_val_dataset)} clips of classes {sorted(pseudo_classes)}")
    elif pseudo_unseen:
        val_labels = dataset_labels(val_dataset)
        pseudo_idx = np.flatnonzero(np.isin(val_labels, pseudo_unseen)).tolist()
        if not pseudo_idx:
            raise ValueError("No validation clips of the pseudo-unseen classes.")
        pseudo_val_dataset = Subset(val_dataset, pseudo_idx)
        print(f"[pseudo-unseen] validation clips: {len(pseudo_idx)}")

    train_sampler = None
    train_shuffle = loader_config.get("train_shuffle", True)
    if loader_config.get("class_balanced_sampling", False):
        train_sampler = build_class_balanced_sampler(train_dataset)
        train_shuffle = False
        print("Using class-balanced sampling for the training loader.")

    num_workers = int(loader_config.get("num_workers", 4))
    persistent_workers = bool(loader_config.get("persistent_workers", False)) and num_workers > 0
    train_loader = Data.DataLoader(
        train_dataset,
        batch_size=loader_config["batch_size"],
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_memory=loader_config.get("pin_memory", True),
        drop_last=loader_config.get("drop_last", True),
    )
    pseudo_val_loader = None
    if pseudo_val_dataset is not None:
        pseudo_val_loader = Data.DataLoader(
            pseudo_val_dataset,
            batch_size=loader_config["batch_size"],
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            pin_memory=loader_config.get("pin_memory", True),
            drop_last=False,
        )

    val_loader = Data.DataLoader(
        val_dataset,
        batch_size=loader_config["batch_size"],
        shuffle=loader_config.get("val_shuffle", False),
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_memory=loader_config.get("pin_memory", True),
        drop_last=False,
    )
    return train_loader, val_loader, pseudo_val_loader


def build_optimizer(model, optimizer_config):
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    name = optimizer_config["name"].lower()
    if name == "adam":
        return torch.optim.Adam(
            params,
            lr=optimizer_config["lr"],
            weight_decay=optimizer_config.get("weight_decay", 0.0),
        )
    if name == "adamw":
        return torch.optim.AdamW(
            params,
            lr=optimizer_config["lr"],
            weight_decay=optimizer_config.get("weight_decay", 0.0),
        )
    raise ValueError(f"Unsupported optimizer: {optimizer_config['name']}")


def build_scheduler(optimizer, train_config, epochs):
    """Learning-rate schedule, cosine by default.

    A constant LR let the pooling reducers bottom out around epoch 20-25 and
    then drift into overfitting for the remaining 200+ epochs. Decaying to a
    small LR lets them keep refining instead.
    """

    name = str(train_config.get("scheduler", "cosine")).lower()
    if name in {"none", "constant"}:
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=float(train_config.get("min_lr", 1e-6)),
        )
    raise ValueError(f"Unsupported scheduler: {name}")


def move_batch_to_device(batch, device):
    return {
        "video": batch["video"].to(device, non_blocking=True),
        "pose": batch["pose"].to(device, non_blocking=True),
        "object": batch["object"].to(device, non_blocking=True),
        "joint_xy": batch["joint_xy"].to(device, non_blocking=True),
        "label": batch["label"].to(device, non_blocking=True),
    }


def resolve_loss_config(train_config):
    """Validate the optional entropy-minimisation loss configuration.

    Missing ``train.loss`` means the historical cross-entropy-only objective,
    so every existing config keeps exactly the same behaviour.
    """

    loss_config = dict(train_config.get("loss") or {})
    mode = str(loss_config.get("mode", "cross_entropy")).lower()
    aliases = {
        "ce": "cross_entropy",
        "ce_entropy": "cross_entropy_entropy",
        "entropy": "cross_entropy_entropy",
    }
    mode = aliases.get(mode, mode)
    if mode not in {"cross_entropy", "cross_entropy_entropy"}:
        raise ValueError(
            "train.loss.mode must be cross_entropy or cross_entropy_entropy, "
            f"got {mode}"
        )

    loss_config["mode"] = mode
    loss_config["entropy_weight"] = float(loss_config.get("entropy_weight", 0.0))
    loss_config["entropy_temperature"] = float(loss_config.get("entropy_temperature", 0.1))
    loss_config["entropy_start_epoch"] = int(loss_config.get("entropy_start_epoch", 0))
    loss_config["entropy_warmup_epochs"] = int(loss_config.get("entropy_warmup_epochs", 0))
    loss_config["entropy_scope"] = str(loss_config.get("entropy_scope", "all")).lower()
    loss_config["entropy_normalize"] = bool(loss_config.get("entropy_normalize", True))
    loss_config["entropy_correct_only"] = bool(loss_config.get("entropy_correct_only", False))
    loss_config["scale_independent"] = bool(loss_config.get("scale_independent", True))
    loss_config["anchor_weight"] = float(loss_config.get("anchor_weight", 0.0))
    loss_config["anchor_scope"] = str(loss_config.get("anchor_scope", "visual")).lower()

    if loss_config["entropy_weight"] < 0:
        raise ValueError("train.loss.entropy_weight must be non-negative.")
    if loss_config["entropy_temperature"] <= 0:
        raise ValueError("train.loss.entropy_temperature must be positive.")
    if loss_config["anchor_weight"] < 0:
        raise ValueError("train.loss.anchor_weight must be non-negative.")
    if loss_config["entropy_start_epoch"] < 0 or loss_config["entropy_warmup_epochs"] < 0:
        raise ValueError("Entropy start/warmup epochs must be non-negative.")
    if loss_config["entropy_scope"] not in {"all", "seen"}:
        raise ValueError("train.loss.entropy_scope must be all or seen.")
    if loss_config["anchor_scope"] not in {"all", "visual"}:
        raise ValueError("train.loss.anchor_scope must be all or visual.")
    return loss_config


def entropy_weight_at_epoch(loss_config, epoch):
    if loss_config["mode"] != "cross_entropy_entropy":
        return 0.0
    target_weight = loss_config["entropy_weight"]
    start_epoch = loss_config["entropy_start_epoch"]
    if epoch < start_epoch:
        return 0.0
    warmup_epochs = loss_config["entropy_warmup_epochs"]
    if warmup_epochs <= 0:
        return target_weight
    progress = min((epoch - start_epoch + 1) / warmup_epochs, 1.0)
    return target_weight * progress


def predictive_entropy_loss(
    model,
    logits,
    targets,
    loss_config,
    candidate_columns=None,
    correct_only=None,
):
    """Mean conditional entropy H(p(y|x)) for the configured candidates.

    By default the learnable CLIP logit scale is divided back out before the
    fixed entropy temperature is applied. Otherwise the network can minimise H
    merely by increasing that scalar rather than separating visual features in
    cosine space.
    """

    entropy_logits = logits
    entropy_targets = targets
    if candidate_columns is not None:
        entropy_logits = entropy_logits.index_select(1, candidate_columns)
        target_lookup = torch.full(
            (logits.shape[1],), -1, dtype=torch.long, device=logits.device
        )
        target_lookup[candidate_columns] = torch.arange(
            candidate_columns.numel(), device=logits.device
        )
        entropy_targets = target_lookup[targets]
        if torch.any(entropy_targets < 0):
            raise ValueError("A target label is outside the entropy candidate scope.")

    if loss_config["scale_independent"] and hasattr(model, "logit_scale"):
        # Keep the denominator in the graph: logits/exp(logit_scale) is then
        # mathematically independent of logit_scale, including its gradient.
        scale = model.logit_scale.exp().clamp(max=100)
        entropy_logits = entropy_logits / scale
    entropy_logits = entropy_logits.float() / loss_config["entropy_temperature"]
    log_probabilities = F.log_softmax(entropy_logits, dim=1)
    probabilities = log_probabilities.exp()
    per_sample = -(probabilities * log_probabilities).sum(dim=1)
    if loss_config["entropy_normalize"] and entropy_logits.shape[1] > 1:
        per_sample = per_sample / np.log(entropy_logits.shape[1])

    if correct_only is None:
        correct_only = loss_config["entropy_correct_only"]
    if correct_only:
        correct = entropy_logits.detach().argmax(dim=1).eq(entropy_targets)
        if not torch.any(correct):
            return entropy_logits.sum() * 0.0
        per_sample = per_sample[correct]
    return per_sample.mean()


def model_selection_score(
    selection,
    *,
    val_loss,
    val_acc,
    val_predictive_entropy,
    pseudo_acc,
):
    """Return a higher-is-better checkpoint score for a validation result."""

    if val_loss is None:
        return None
    if selection == "pseudo_zsl":
        return pseudo_acc
    if selection == "val_accuracy":
        return val_acc
    if selection == "val_accuracy_minus_entropy":
        return val_acc - val_predictive_entropy
    return -val_loss


def build_anchor_parameters(model, loss_config):
    """Snapshot trainable parameters for optional L2-SP regularisation."""

    if loss_config["anchor_weight"] == 0:
        return None
    selected = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if loss_config["anchor_scope"] == "visual" and not name.startswith("visual_encoder."):
            continue
        selected[name] = parameter.detach().clone()
    if not selected:
        raise ValueError("No trainable parameters match train.loss.anchor_scope.")
    return selected


def parameter_anchor_loss(model, anchor_parameters):
    """Relative squared L2 distance from the initial checkpoint parameters."""

    if not anchor_parameters:
        if hasattr(model, "logit_scale"):
            return model.logit_scale * 0.0
        return next(model.parameters()).reshape(-1)[0] * 0.0
    numerator = None
    denominator = None
    current = dict(model.named_parameters())
    for name, reference in anchor_parameters.items():
        parameter = current[name]
        squared_drift = (parameter - reference).float().square().sum()
        squared_reference = reference.float().square().sum()
        numerator = squared_drift if numerator is None else numerator + squared_drift
        denominator = squared_reference if denominator is None else denominator + squared_reference
    return numerator / denominator.clamp_min(1e-12)


def run_one_epoch(
    model,
    data_loader,
    optimizer,
    scaler,
    device,
    use_amp,
    train,
    epoch,
    num_epochs,
    grad_clip_norm=None,
    loss_config=None,
    entropy_columns=None,
    anchor_parameters=None,
):
    model.train(train)
    if train and getattr(model, "_freeze_batchnorm", False):
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()
    if getattr(model, "_trainable_scope", "all") == "logit_scale_only":
        # Frozen BatchNorm statistics and dropout masks make this a true scalar
        # calibration: cosine geometry and therefore every argmax stay fixed.
        model.visual_encoder.eval()
        model.text_encoder.eval()
    total_loss = 0.0
    total_ce_loss = 0.0
    total_entropy = 0.0
    total_predictive_entropy = 0.0
    total_anchor = 0.0
    total_correct = 0
    total_samples = 0
    loss_config = loss_config or resolve_loss_config({})
    entropy_weight = entropy_weight_at_epoch(loss_config, epoch)
    mode = "Train" if train else "Val"
    bar = progress_bar(data_loader, f"{mode} epoch {epoch + 1}/{num_epochs}")

    for batch in bar:
        batch = move_batch_to_device(batch, device)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                # Same computation as model.contrastive_loss, unrolled so the
                # logits are available for the accuracy that selects the model.
                logits = model(
                    batch["video"],
                    batch["pose"],
                    batch["object"],
                    batch["joint_xy"],
                )
                targets = model.build_target_indices(batch["label"])
                ce_loss = F.cross_entropy(logits, targets)
                if loss_config["mode"] == "cross_entropy_entropy":
                    entropy_loss = predictive_entropy_loss(
                        model,
                        logits,
                        targets,
                        loss_config,
                        candidate_columns=entropy_columns,
                    )
                else:
                    entropy_loss = logits.sum() * 0.0
                if (loss_config["mode"] == "cross_entropy_entropy"
                        and not loss_config["entropy_correct_only"]):
                    predictive_entropy = entropy_loss
                else:
                    predictive_entropy = predictive_entropy_loss(
                        model,
                        logits,
                        targets,
                        loss_config,
                        candidate_columns=entropy_columns,
                        correct_only=False,
                    )
                anchor_loss = parameter_anchor_loss(model, anchor_parameters)
                loss = (
                    ce_loss
                    + entropy_weight * entropy_loss
                    + loss_config["anchor_weight"] * anchor_loss
                )

        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite loss at epoch {epoch + 1} during {mode.lower()}: {float(loss.detach().cpu())}"
            )

        if train:
            scaler.scale(loss).backward()
            if grad_clip_norm is not None and grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()

        batch_size = batch["label"].shape[0]
        total_loss += float(loss.detach().cpu()) * batch_size
        total_ce_loss += float(ce_loss.detach().cpu()) * batch_size
        total_entropy += float(entropy_loss.detach().cpu()) * batch_size
        total_predictive_entropy += float(predictive_entropy.detach().cpu()) * batch_size
        total_anchor += float(anchor_loss.detach().cpu()) * batch_size
        total_correct += int((logits.detach().argmax(dim=1) == targets).sum())
        total_samples += batch_size

        if tqdm is not None:
            bar.set_postfix(
                loss=total_loss / max(total_samples, 1),
                ce=total_ce_loss / max(total_samples, 1),
                entropy=total_entropy / max(total_samples, 1),
                pred_entropy=total_predictive_entropy / max(total_samples, 1),
                anchor=total_anchor / max(total_samples, 1),
                acc=total_correct / max(total_samples, 1),
            )

    denominator = max(total_samples, 1)
    return (
        total_loss / denominator,
        total_correct / denominator,
        total_ce_loss / denominator,
        total_entropy / denominator,
        total_predictive_entropy / denominator,
        total_anchor / denominator,
        entropy_weight,
    )


def trainable_state_dict(model):
    """Weights worth saving: everything except the frozen CLIP text encoder.

    That encoder is 151M of the 152.7M parameters and never changes, so keeping
    it in every periodic checkpoint would cost 343 MB a piece instead of 5.5 MB.
    It is rebuilt from clip.load() when the checkpoint is loaded.
    """

    return {k: v for k, v in model.state_dict().items() if not k.startswith("text_encoder")}


@torch.no_grad()
def update_batchnorm(model, data_loader, device, max_batches=40):
    """Recompute BatchNorm running statistics for averaged weights.

    Averaging weights leaves the running mean/var belonging to whichever epoch
    happened to write them last, which does not match the averaged network.
    """

    bn_layers = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    if not bn_layers:
        return
    momenta = [(m, m.momentum) for m in bn_layers]
    for module in bn_layers:
        module.reset_running_stats()
        module.momentum = None
    model.train()
    for index, batch in enumerate(data_loader):
        if index >= max_batches:
            break
        batch = move_batch_to_device(batch, device)
        model(batch["video"], batch["pose"], batch["object"], batch["joint_xy"])
    for module, momentum in momenta:
        module.momentum = momentum
    model.eval()


def evaluate_pseudo_zsl(model, data_loader, pseudo_labels, device, use_amp):
    """ZSL top-1 on the held-out pseudo-unseen classes.

    Candidates are restricted to the pseudo-unseen prototypes, so this asks the
    same question the real test does - can the model place a class it never saw
    a clip of - and can therefore be used to pick the epoch without touching the
    test set.
    """

    model.eval()
    bank = model.text_label_ids.detach().cpu().tolist()
    columns = torch.as_tensor([bank.index(int(l)) for l in pseudo_labels], device=device)
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in data_loader:
            batch = move_batch_to_device(batch, device)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(batch["video"], batch["pose"], batch["object"], batch["joint_xy"])
            predicted = columns[logits[:, columns].argmax(dim=1)]
            correct += int((predicted == model.build_target_indices(batch["label"])).sum())
            total += batch["label"].shape[0]
    return correct / max(total, 1)


def prepare_action_text_bank(model, config, config_path):
    text_config = config["data"]["text"]
    xlsx_path = get_path_from_config(config_path, text_config["xlsx"])
    labels, texts, records = load_action_descriptions(
        xlsx_path=xlsx_path,
        text_column=text_config.get("text_column", "global_description"),
        id_column=text_config.get("id_column", "ID"),
        label_offset=text_config.get("label_offset", 1),
        prompt_template=text_config.get("prompt_template", "{global_description}"),
    )
    labels, texts = apply_action_text_bank(model, text_config, xlsx_path)

    output_path = text_config.get("cache_output")
    if output_path:
        output_path = get_path_from_config(config_path, output_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save(
            {
                "embeddings": model.text_features.detach().cpu(),
                "labels": labels,
                "texts": texts,
                "records": records,
            },
            output_path,
        )
        with open(os.path.splitext(output_path)[0] + ".json", "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "output": output_path,
                    "shape": list(model.text_features.shape),
                    "labels": labels,
                    "texts": texts,
                },
                handle,
                indent=2,
                ensure_ascii=False,
            )
    print(f"Loaded {len(texts)} action descriptions from {xlsx_path}")


def load_initial_checkpoint(model, checkpoint_path, device):
    """Initialise a run from either a full model or a fusion-head snapshot."""

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Initial checkpoint does not exist: {checkpoint_path}")
    try:
        state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    partial = not any(key.startswith("text_encoder") for key in state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected or (not partial and missing):
        raise RuntimeError(
            f"Initial checkpoint mismatch: missing={missing[:3]} unexpected={unexpected[:3]}"
        )
    kind = "fusion-head snapshot" if partial else "full model"
    print(f"Initialised from {kind}: {checkpoint_path}")


def prepare_output_paths(config, config_path):
    output_config = config["outputs"]
    base_work_dir = get_path_from_config(config_path, output_config["work_dir"])

    if output_config.get("auto_run_dir", False):
        run_name = output_config.get("run_name")
        if not run_name:
            run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")

        work_dir = os.path.join(base_work_dir, run_name)
        if os.path.exists(work_dir):
            suffix = 1
            while os.path.exists(f"{work_dir}_{suffix:02d}"):
                suffix += 1
            work_dir = f"{work_dir}_{suffix:02d}"

        os.makedirs(work_dir, exist_ok=False)
        output_config["work_dir"] = work_dir
        output_config["best_model"] = os.path.join(work_dir, "best_model.pth")
        output_config["last_model"] = os.path.join(work_dir, "last_model.pth")
        output_config["history"] = os.path.join(work_dir, "history.json")
        output_config["train_curve"] = os.path.join(work_dir, "train_curve.png")
        output_config["test_results"] = os.path.join(work_dir, "unseen_test_results.json")

        latest_path = os.path.join(base_work_dir, "latest_run.json")
        os.makedirs(base_work_dir, exist_ok=True)
        with open(latest_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "work_dir": work_dir,
                    "best_model": output_config["best_model"],
                    "last_model": output_config["last_model"],
                    "history": output_config["history"],
                    "train_curve": output_config["train_curve"],
                    "test_results": output_config["test_results"],
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                },
                handle,
                indent=2,
                ensure_ascii=False,
            )

        shutil.copy2(config_path, os.path.join(work_dir, "source_config.yaml"))
        with open(os.path.join(work_dir, "resolved_config.json"), "w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2, ensure_ascii=False)
    else:
        os.makedirs(base_work_dir, exist_ok=True)

    return output_config["work_dir"]


def train_model(model, train_loader, val_loader, config, config_path, device, pseudo_val_loader=None):
    train_config = config["train"]
    output_config = config["outputs"]
    trainable_scope = str(train_config.get("trainable_scope", "all")).lower()
    if trainable_scope == "all":
        pass
    elif trainable_scope == "logit_scale_only":
        for parameter in model.parameters():
            parameter.requires_grad = False
        model.logit_scale.requires_grad = True
    else:
        raise ValueError("train.trainable_scope must be all or logit_scale_only.")
    model._trainable_scope = trainable_scope
    model._freeze_batchnorm = bool(train_config.get("freeze_batchnorm", False))
    optimizer = build_optimizer(model, train_config["optimizer"])
    loss_config = resolve_loss_config(train_config)
    if loss_config["anchor_weight"] > 0 and not train_config.get("init_checkpoint"):
        raise ValueError("train.loss.anchor_weight requires train.init_checkpoint.")
    anchor_parameters = build_anchor_parameters(model, loss_config)

    entropy_columns = None
    if (loss_config["mode"] == "cross_entropy_entropy"
            and loss_config["entropy_scope"] == "seen"):
        seen_labels = sorted(int(label) for label in np.unique(dataset_labels(train_loader.dataset)))
        bank_labels = model.text_label_ids.detach().cpu().tolist()
        missing = sorted(set(seen_labels) - set(bank_labels))
        if missing:
            raise ValueError(f"Seen labels missing from the text bank: {missing}")
        entropy_columns = torch.as_tensor(
            [bank_labels.index(label) for label in seen_labels],
            dtype=torch.long,
            device=device,
        )
    print(
        "Loss: "
        f"mode={loss_config['mode']}, "
        f"entropy_weight={loss_config['entropy_weight']:.6g}, "
        f"temperature={loss_config['entropy_temperature']:.6g}, "
        f"scope={loss_config['entropy_scope']}, "
        f"correct_only={loss_config['entropy_correct_only']}, "
        f"scale_independent={loss_config['scale_independent']}, "
        f"anchor_weight={loss_config['anchor_weight']:.6g}, "
        f"anchor_scope={loss_config['anchor_scope']}"
    )
    print(f"Trainable scope: {trainable_scope}")
    print(f"Freeze BatchNorm statistics: {model._freeze_batchnorm}")

    epochs = int(train_config["epochs"])
    val_interval = int(train_config.get("val_interval", 1))
    if val_interval <= 0:
        raise ValueError(f"val_interval must be positive, got {val_interval}")
    use_amp = bool(config["runtime"].get("amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    grad_clip_norm = train_config.get("grad_clip_norm", None)
    early_stopping_patience = train_config.get("early_stopping_patience")
    early_stopping_min_delta = float(train_config.get("early_stopping_min_delta", 0.0))
    scheduler = build_scheduler(optimizer, train_config, epochs)

    # Val loss is dominated by a few hard clips and kept picking checkpoints
    # that were worse on test, so accuracy is the default selection metric.
    selection = str(train_config.get("model_selection", "val_accuracy")).lower()
    valid_selections = {
        "val_accuracy",
        "val_loss",
        "pseudo_zsl",
        "val_accuracy_minus_entropy",
    }
    if selection not in valid_selections:
        choices = ", ".join(sorted(valid_selections))
        raise ValueError(f"model_selection must be one of {choices}, got {selection}")
    # Classes come from whichever pseudo-validation source the config used:
    # an explicit data.pseudo_val block, or classes carved out of the train set.
    pseudo_source = config["data"].get("pseudo_val") or {}
    pseudo_unseen = [
        int(c) for c in (pseudo_source.get("classes")
                         or config["data"]["train"].get("pseudo_unseen") or [])
    ]
    if selection == "pseudo_zsl" and pseudo_val_loader is None:
        raise ValueError("model_selection=pseudo_zsl needs data.train.pseudo_unseen to be set.")

    # Stochastic weight averaging over the tail of training. The per-epoch curve
    # is flat there, so its arg-max jumps around between runs; averaging the
    # plateau is steadier than betting on any single epoch.
    swa_start = train_config.get("swa_start")
    # Snapshot every epoch by default: stripped of the frozen CLIP encoder a
    # snapshot is 5.5 MB, so a whole 300-epoch run costs ~1.6 GB and any later
    # SWA range, ensemble or selection rule can be tried without retraining.
    checkpoint_every = train_config.get("checkpoint_every", 1)
    save_every = int(train_config.get("save_every", 1))
    if save_every <= 0:
        raise ValueError("train.save_every must be positive.")
    swa_state = None
    swa_count = 0

    best_model_wts = copy.deepcopy(model.state_dict())
    best_score = float("-inf")
    best_epoch = -1
    history = {
        "epoch": [],
        "train_loss": [],
        "train_ce_loss": [],
        "train_entropy": [],
        "train_predictive_entropy": [],
        "train_anchor": [],
        "entropy_weight": [],
        "train_acc": [],
        "val_loss": [],
        "val_ce_loss": [],
        "val_entropy": [],
        "val_predictive_entropy": [],
        "val_anchor": [],
        "val_acc": [],
        "pseudo_zsl": [],
        "selection_score": [],
        "lr": [],
    }
    start_time = time.time()

    work_dir = get_path_from_config(config_path, output_config["work_dir"])
    os.makedirs(work_dir, exist_ok=True)
    best_path = get_path_from_config(config_path, output_config["best_model"])
    last_path = get_path_from_config(config_path, output_config["last_model"])
    history_path = get_path_from_config(config_path, output_config["history"])
    os.makedirs(os.path.dirname(best_path), exist_ok=True)
    os.makedirs(os.path.dirname(last_path), exist_ok=True)
    if early_stopping_patience is not None:
        early_stopping_patience = int(early_stopping_patience)
        if early_stopping_patience <= 0:
            raise ValueError("early_stopping_patience must be positive when set.")
    epochs_without_improvement = 0

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        print("-" * 10)

        current_lr = optimizer.param_groups[0]["lr"]
        (
            train_loss,
            train_acc,
            train_ce,
            train_entropy,
            train_predictive_entropy,
            train_anchor,
            entropy_weight,
        ) = run_one_epoch(
            model, train_loader, optimizer, scaler, device, use_amp, True, epoch, epochs,
            grad_clip_norm, loss_config, entropy_columns, anchor_parameters
        )
        should_validate = (epoch + 1) % val_interval == 0 or epoch + 1 == epochs
        if should_validate:
            (
                val_loss,
                val_acc,
                val_ce,
                val_entropy,
                val_predictive_entropy,
                val_anchor,
                _,
            ) = run_one_epoch(
                model, val_loader, optimizer, scaler, device, use_amp, False, epoch, epochs,
                None, loss_config, entropy_columns, anchor_parameters
            )
        else:
            val_loss, val_acc = None, None
            val_ce, val_entropy = None, None
            val_predictive_entropy = None
            val_anchor = None

        pseudo_acc = None
        if pseudo_val_loader is not None and should_validate:
            pseudo_acc = evaluate_pseudo_zsl(model, pseudo_val_loader, pseudo_unseen, device, use_amp)

        if scheduler is not None:
            scheduler.step()

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_ce_loss"].append(train_ce)
        history["train_entropy"].append(train_entropy)
        history["train_predictive_entropy"].append(train_predictive_entropy)
        history["train_anchor"].append(train_anchor)
        history["entropy_weight"].append(entropy_weight)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_ce_loss"].append(val_ce)
        history["val_entropy"].append(val_entropy)
        history["val_predictive_entropy"].append(val_predictive_entropy)
        history["val_anchor"].append(val_anchor)
        history["val_acc"].append(val_acc)
        history["pseudo_zsl"].append(pseudo_acc)
        print(
            f"lr: {current_lr:.2e}  train_loss: {train_loss:.4f}  "
            f"ce: {train_ce:.4f}  H: {train_entropy:.4f}  "
            f"H_all: {train_predictive_entropy:.4f}  "
            f"anchor: {train_anchor:.6g}  lambda_H: {entropy_weight:.6g}  "
            f"train_acc: {train_acc:.4f}"
        )
        if val_loss is None:
            print(f"val: skipped (val_interval={val_interval})")
        else:
            line = (
                f"val_loss: {val_loss:.4f}  ce: {val_ce:.4f}  "
                f"H: {val_entropy:.4f}  H_all: {val_predictive_entropy:.4f}  "
                f"anchor: {val_anchor:.6g}  val_acc: {val_acc:.4f}"
            )
            if pseudo_acc is not None:
                line += f"  pseudo_zsl: {pseudo_acc:.4f}"
            print(line)

        score = model_selection_score(
            selection,
            val_loss=val_loss,
            val_acc=val_acc,
            val_predictive_entropy=val_predictive_entropy,
            pseudo_acc=pseudo_acc,
        )
        history["selection_score"].append(score)
        history["lr"].append(current_lr)
        if score is not None:
            print(f"selection_score ({selection}): {score:.4f}")
        if score is not None and score > best_score + early_stopping_min_delta:
            best_score = score
            best_epoch = epoch
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, best_path)
            epochs_without_improvement = 0
            print(f"Best model updated at epoch {epoch + 1} ({selection}): {best_path}")
        elif score is not None and early_stopping_patience is not None:
            epochs_without_improvement += 1

        if checkpoint_every and (epoch + 1) % checkpoint_every == 0:
            snapshot_dir = os.path.join(os.path.dirname(best_path), "snapshots")
            os.makedirs(snapshot_dir, exist_ok=True)
            torch.save(trainable_state_dict(model),
                       os.path.join(snapshot_dir, f"epoch_{epoch + 1:04d}.pth"))

        if swa_start is not None and (epoch + 1) >= int(swa_start):
            current = {k: v.detach().float().cpu() for k, v in model.state_dict().items()}
            if swa_state is None:
                swa_state = current
            else:
                for key in swa_state:
                    swa_state[key] += (current[key] - swa_state[key]) / (swa_count + 1)
            swa_count += 1

        if (epoch + 1) % save_every == 0 or epoch + 1 == epochs:
            torch.save(model.state_dict(), last_path)
            with open(history_path, "w", encoding="utf-8") as handle:
                json.dump(history, handle, indent=2)

        elapsed = time.time() - start_time
        print("Training Time {:.0f}m {:.0f}s".format(elapsed // 60, elapsed % 60))

        if early_stopping_patience is not None and epochs_without_improvement >= early_stopping_patience:
            print(
                f"Early stopping at epoch {epoch + 1}: "
                f"no {selection} improvement for {early_stopping_patience} validation checks."
            )
            break

    torch.save(best_model_wts, best_path)
    torch.save(model.state_dict(), last_path)

    with open(history_path, "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    if swa_state is not None:
        swa_path = os.path.join(os.path.dirname(best_path), "swa_model.pth")
        model.load_state_dict({k: v.to(device) for k, v in swa_state.items()})
        update_batchnorm(model, train_loader, device)
        torch.save(model.state_dict(), swa_path)
        print(f"SWA over {swa_count} epochs (from {swa_start}) saved to {swa_path}")

    model.load_state_dict(best_model_wts)
    reported_best_score = -best_score if selection == "val_loss" else best_score
    print(
        f"Best model saved to {best_path} "
        f"(epoch {best_epoch + 1}, {selection}={reported_best_score:.4f})"
    )
    print(f"Last model saved to {last_path}")
    return history


def plot_loss(history, save_path=None, show=False):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed, skip plotting.")
        return

    def series(key):
        return [np.nan if value is None else value for value in history.get(key, [])]

    _figure, (loss_axis, acc_axis) = plt.subplots(1, 2, figsize=(11, 4))
    loss_axis.plot(history["epoch"], history["train_loss"], "r-", label="train_loss")
    loss_axis.plot(history["epoch"], series("val_loss"), "b-", label="val_loss")
    loss_axis.set_xlabel("Epoch")
    loss_axis.set_ylabel("Loss")
    if "train_ce_loss" in history:
        loss_axis.plot(history["epoch"], series("train_ce_loss"), "r--", alpha=0.65, label="train_ce")
    loss_axis.set_title("Training objective")
    loss_axis.legend()

    acc_axis.plot(history["epoch"], series("train_acc"), "r-", label="train_acc")
    acc_axis.plot(history["epoch"], series("val_acc"), "b-", label="val_acc")
    val_acc = series("val_acc")
    if np.any(np.isfinite(val_acc)):
        best = int(np.nanargmax(val_acc))
        acc_axis.axvline(history["epoch"][best], color="green", linestyle="--",
                         label=f"best val_acc @ep{history['epoch'][best]}")
    acc_axis.set_xlabel("Epoch")
    acc_axis.set_ylabel("Top-1 accuracy")
    acc_axis.set_title("Accuracy (model selection)")
    acc_axis.legend()
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200)
        print(f"Training curve saved to {save_path}")
    if show:
        plt.show()
    plt.close()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Train CLIPGCN contrastive model.")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config file.")
    parser.add_argument(
        "--zero-streams",
        nargs="*",
        choices=("video", "pose", "object"),
        default=None,
        help="Ablation: zero these input streams during training/val. Overrides data.zero_streams. "
        "Also suffixes the run dir, e.g. --zero-streams pose object -> run_*_zero-pose-object.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override runtime.seed. Repeat runs with different seeds to separate a real "
        "architecture effect from run-to-run variance.",
    )
    parser.add_argument("--run-name", default=None, help="Override outputs.run_name.")
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = os.path.abspath(args.config)
    config = load_config(config_path)
    if args.zero_streams is not None:
        config.setdefault("data", {})["zero_streams"] = list(args.zero_streams)
        suffix = "_zero-" + "-".join(sorted(args.zero_streams)) if args.zero_streams else ""
        if suffix:
            outputs = config.setdefault("outputs", {})
            run_name = outputs.get("run_name") or datetime.now().strftime("run_%Y%m%d_%H%M%S")
            outputs["run_name"] = run_name + suffix
    if args.seed is not None:
        config["runtime"]["seed"] = args.seed
    if args.run_name:
        config.setdefault("outputs", {})["run_name"] = args.run_name
    set_seed(int(config["runtime"].get("seed", 20260616)))
    print(f"Seed: {config['runtime'].get('seed')}")
    run_dir = prepare_output_paths(config, config_path)
    print(f"Run output directory: {run_dir}")

    device = get_device(config["runtime"].get("device"))
    print_device_info(device)

    model = build_model_from_config(
        config,
        device=device,
        download_root=get_path_from_config(config_path, config["model"]["text_encoder"].get("download_root")),
    )
    fusion_config = config["model"].get("fusion", {})
    trainable = sum(p.numel() for p in model.visual_encoder.parameters() if p.requires_grad)
    print(
        f"Fusion head: reducer={fusion_config.get('reducer', 'attention_pool')}, "
        f"object_fusion={fusion_config.get('object_fusion', 'cross_attention')}, "
        f"trainable params={trainable / 1e6:.2f}M"
    )
    prepare_action_text_bank(model, config, config_path)
    model = model.to(device)

    init_checkpoint = config["train"].get("init_checkpoint")
    if init_checkpoint:
        load_initial_checkpoint(
            model,
            get_path_from_config(config_path, init_checkpoint),
            device,
        )

    train_loader, val_loader, pseudo_val_loader = build_dataloaders(config, config_path)
    history = train_model(model, train_loader, val_loader, config, config_path, device, pseudo_val_loader)

    curve_path = get_path_from_config(config_path, config["outputs"].get("train_curve"))
    if curve_path:
        plot_loss(history, save_path=curve_path, show=not config["runtime"].get("no_show", True))


if __name__ == "__main__":
    main()
