"""Frozen VPOCLIP adapter used only to export per-view embeddings and logits."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Mapping, Optional

import torch
import torch.nn.functional as F
from torch import nn

from .contracts import TensorDict


class VPOCLIPAdapter:
    """Own the frozen recognizer and expose a cache-oriented batch interface."""

    def __init__(self, model: nn.Module, device: torch.device) -> None:
        self.model = model.to(device)
        self.device = device
        self.freeze()

    def freeze(self) -> None:
        """Put the recognizer in evaluation mode and disable all gradients."""

        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def encode_all_views(self, batch: Mapping[str, torch.Tensor]) -> TensorDict:
        """Return ``z[B,V,512]`` and ``logits[B,V,55]`` without gradients."""
        video = batch["video"]
        if video.ndim < 2:
            raise ValueError("video input must have leading [B,V] dimensions")
        bsz, views = video.shape[:2]
        def flatten(value: torch.Tensor) -> torch.Tensor:
            return value.reshape(bsz * views, *value.shape[2:])
        with torch.inference_mode():
            visual = self.model.encode_visual(
                flatten(video),
                flatten(batch["pose"]),
                flatten(batch.get("object", batch.get("object_map"))),
                flatten(batch["joint_xy"]),
            )
            z = F.normalize(visual.float(), dim=-1)
            text = F.normalize(self.model.text_features.float(), dim=-1)
            scale = self.model.logit_scale.exp().clamp(max=100)
            logits = (scale * z @ text.t()).reshape(bsz, views, -1)
        return {"z": z.reshape(bsz, views, -1), "logits": logits}

    def assert_frozen(self, reference: Optional[Dict[str, torch.Tensor]] = None) -> None:
        """Check gradient flags and optionally compare against a weight snapshot."""

        for parameter in self.model.parameters():
            if parameter.requires_grad:
                raise AssertionError("VPOCLIP parameter unexpectedly requires gradients")
        if reference is not None:
            current = self.model.state_dict()
            for key, value in reference.items():
                if key not in current or not torch.equal(current[key].detach().cpu(), value.detach().cpu()):
                    raise AssertionError(f"VPOCLIP parameter changed: {key}")

    @classmethod
    def from_config(cls, config_path: str, checkpoint_path: str, device: str) -> "VPOCLIPAdapter":
        """Build the existing model, text bank, and checkpoint exactly as test.py does."""

        import yaml
        root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(root))
        from model import apply_action_text_bank, build_model_from_config

        cfg_path = Path(config_path).resolve()
        config = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        def resolve(value):
            if value is None:
                return None
            path = Path(value)
            return path if path.is_absolute() else (cfg_path.parent / path).resolve()
        model = build_model_from_config(
            config,
            device=torch.device(device),
            download_root=str(resolve(config["model"]["text_encoder"].get("download_root"))) if config["model"]["text_encoder"].get("download_root") else None,
        )
        ckpt = resolve(checkpoint_path)
        try:
            state = torch.load(ckpt, map_location=device, weights_only=True)
        except Exception:
            state = torch.load(ckpt, map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=True)
        text_cfg = config["data"]["text"]
        xlsx = resolve(text_cfg["xlsx"])
        apply_action_text_bank(model, text_cfg, str(xlsx))
        return cls(model, torch.device(device))


# Implementation guide
# 1. Load the recognizer through model.build_model_from_config and populate the
#    same eight-prompt text bank used by the selected final_aug checkpoint.
# 2. Load checkpoints with the same full-model/fusion-head rules as test.py.
# 3. Flatten every [B,V,...] input to [B*V,...], call encode_visual once under
#    torch.inference_mode(), compute scaled cosine logits, then restore B and V.
# 4. Never add an RL projection to z or modify the 512-D VPOCLIP semantic space.
# 5. Snapshot state_dict before RL training and verify bitwise equality afterward.
