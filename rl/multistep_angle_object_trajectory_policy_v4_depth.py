"""v4 + first-frame per-object relative depth.

This is an isolated successor of ``multistep_angle_object_trajectory_policy_v4``.
The original v4 files are not modified.  It keeps v4's causal person/object
trajectory and torso-orientation inputs, and adds one static 50-D depth vector
for each observed camera view.  The depth vector is the first-frame MiDaS
relative-depth cache joined by the exact ``sample_name`` in the trajectory
metadata, so camera IDs are never used as an ordering assumption.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from . import multistep_angle_object_trajectory_policy_v1 as v1
from . import multistep_angle_object_trajectory_policy_v4 as v4


ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = ROOT / "data" / "trajectory_latest_vpoclip_new50_5" / "cache_compat"
TRACK_ROOT = ROOT / "data" / "trajectory_latest_vpoclip_new50_5" / "object_tracks_full"
DEPTH_ROOT = ROOT / "data" / "new50_5_group1_zsl_depth_first_frame_fp32"
OUTPUT_ROOT = ROOT / "work_dir" / "multistep_angle_object_trajectory_policy_v4_depth"
_OUTPUT_ROOT_OVERRIDE = os.environ.get("V4_DEPTH_OUTPUT_ROOT")
if _OUTPUT_ROOT_OVERRIDE:
    OUTPUT_ROOT = Path(_OUTPUT_ROOT_OVERRIDE)

V4_CHECKPOINT = (
    ROOT
    / "work_dir"
    / "multistep_angle_object_trajectory_policy_v4_body_orientation"
    / "seed_20260909"
    / "best.pt"
)

DEPTH_SPLITS = ("dqn_train", "train", "val", "test_seen", "test")
DEPTH_DIM = 50
DEPTH_HIDDEN = 32


def _load_depth_for_trajectory_split(
    trajectory_root: Path,
    depth_root: Path,
    split: str,
    expected_episodes: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Join depth rows to trajectory views by exact sample name."""

    # A trajectory split is not identical to the VPOCLIP ZSL split: for
    # example, trajectory ``seen_test`` still contains rows for the five true
    # unseen actions.  Build one exact-name index over all available depth
    # exports instead of inferring membership from the split name.
    lookup: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    available_depth_splits: list[str] = []
    for depth_split in DEPTH_SPLITS:
        depth_dir = depth_root / depth_split
        names_path = depth_dir / "sample_names.npy"
        values_path = depth_dir / "depth_rel_fp32.npy"
        valid_path = depth_dir / "depth_valid_uint8.npy"
        if not (names_path.exists() and values_path.exists() and valid_path.exists()):
            continue
        depth_names = np.load(names_path, allow_pickle=True)
        depth_values = np.load(values_path, mmap_mode="r", allow_pickle=False)
        depth_valid = np.load(valid_path, mmap_mode="r", allow_pickle=False)
        if depth_values.shape != (len(depth_names), DEPTH_DIM):
            raise ValueError(f"unexpected depth shape {depth_values.shape} for {depth_dir}")
        for index, name in enumerate(depth_names.tolist()):
            lookup[str(name)] = (np.asarray(depth_values[index]), np.asarray(depth_valid[index]))
        available_depth_splits.append(depth_split)

    metadata = json.loads((trajectory_root / split / "metadata.json").read_text(encoding="utf-8"))
    episodes = metadata["episodes"]
    if len(episodes) != expected_episodes:
        raise ValueError(f"trajectory episode count changed: {len(episodes)} != {expected_episodes}")
    values = np.zeros((len(episodes), v1.NUM_VIEWS, DEPTH_DIM), dtype=np.float32)
    valid = np.zeros((len(episodes), v1.NUM_VIEWS, DEPTH_DIM), dtype=np.uint8)
    missing: list[str] = []
    views_joined = 0
    for episode_index, episode in enumerate(episodes):
        for view_index, view in enumerate(episode.get("views", [])[: v1.NUM_VIEWS]):
            sample_name = str(view["sample_name"])
            row = lookup.get(sample_name)
            if row is None:
                missing.append(sample_name)
                continue
            values[episode_index, view_index] = row[0]
            valid[episode_index, view_index] = row[1]
            views_joined += 1
    if missing:
        preview = missing[:5]
        raise FileNotFoundError(f"{len(missing)} trajectory views missing from depth cache, e.g. {preview}")
    audit = {
        "trajectory_split": split,
        "depth_splits": available_depth_splits,
        "episodes": len(episodes),
        "views_joined": views_joined,
        "depth_rows": int(sum(len(np.load(depth_root / item / "sample_names.npy", allow_pickle=True)) for item in available_depth_splits)),
        "depth_source": str(depth_root),
        "join_key": "episode.views[*].sample_name == depth sample_names.npy",
    }
    return values, valid, audit


def attach_depth_tracks(raw: Any, trajectory_root: Path, depth_root: Path, split: str) -> dict[str, Any]:
    values, valid, audit = _load_depth_for_trajectory_split(
        trajectory_root, depth_root, split, raw.num_episodes
    )
    raw.depth = torch.from_numpy(values).to(raw.device, dtype=torch.float32)
    raw.depth_valid = torch.from_numpy(valid.astype(bool)).to(raw.device, dtype=torch.bool)
    return audit


def build_trajectory_state_v4_depth(
    raw: Any,
    episodes: torch.Tensor,
    history: torch.Tensor,
    history_count: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Build v4 state and mean-pool the static depth vector over observations."""

    state = v4.build_trajectory_state_v4(raw, episodes, history, history_count)
    episodes = episodes.long()
    history = history.long()
    count = history_count.long().clamp(1, history.shape[1])
    positions = torch.arange(history.shape[1], device=episodes.device)[None, :]
    observed = (positions < count[:, None]).float()
    denominator = count.float().clamp_min(1.0)[:, None]
    observed_depth = raw.depth[episodes[:, None], history]
    depth = (observed_depth * observed[..., None]).sum(1) / denominator
    state["depth_prior"] = depth
    return state


class BodyOrientationDepthTrajectoryPolicy(v4.BodyOrientationTrajectoryPolicy):
    """v4 policy plus a compact 50-slot first-frame depth branch."""

    def __init__(self) -> None:
        super().__init__()
        self.depth = nn.Sequential(
            nn.Linear(DEPTH_DIM, DEPTH_HIDDEN),
            nn.LayerNorm(DEPTH_HIDDEN),
            nn.GELU(),
        )
        # v4 context width is 32 + 64 + 16 = 112; append 32 depth features.
        self.context = nn.Sequential(
            nn.Linear(32 + 64 + 16 + DEPTH_HIDDEN, 64),
            nn.LayerNorm(64),
            nn.GELU(),
        )
        self._initialize_matching_v4_weights()

    def _initialize_matching_v4_weights(self) -> None:
        """Transfer v4 weights; start the new depth branch as a zero residual."""

        if not V4_CHECKPOINT.exists():
            return
        saved = torch.load(V4_CHECKPOINT, map_location="cpu", weights_only=False)
        old = saved.get("online", saved)
        current = self.state_dict()
        with torch.no_grad():
            for name, value in old.items():
                if name not in current:
                    continue
                if name == "context.0.weight":
                    current[name][:, : value.shape[1]].copy_(value)
                    current[name][:, value.shape[1] :].zero_()
                elif current[name].shape == value.shape:
                    current[name].copy_(value)
            # Make the initial depth contribution exactly zero before learning.
            current["depth.0.weight"].normal_(mean=0.0, std=0.02)
            current["depth.0.bias"].zero_()
        self.load_state_dict(current, strict=True)

    def forward(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        person = self.person_summary(
            self.person_temporal(state["person_trajectory"].transpose(1, 2)).flatten(1)
        )
        obj = state["object_trajectory"].reshape(
            state["object_trajectory"].shape[0],
            v1.NUM_FRAMES,
            v1.NUM_OBJECT_CLASSES * v1.OBJECT_CHANNELS,
        )
        obj = self.object_summary(self.object_temporal(obj.transpose(1, 2)).flatten(1))
        angle = self.angle(state["current_relative_angle"])
        depth = self.depth(state["depth_prior"])
        context = self.context(torch.cat((person, obj, angle, depth), -1))
        candidate = self.candidate(v1.angle_candidate_features(state["candidate"]))
        expanded = context[:, None].expand(-1, v1.NUM_VIEWS, -1)
        q = (self.context_to_action(context)[:, None] * candidate).sum(-1) / (64.0 ** 0.5)
        return q + 0.20 * self.residual(torch.cat((expanded, candidate), -1)).squeeze(-1)


def _attach_all(raw: Any, split: str) -> None:
    # v1's original object track attachment remains unchanged.
    v1.attach_object_tracks(raw, TRACK_ROOT, split)
    audit = attach_depth_tracks(raw, CACHE_ROOT, DEPTH_ROOT, split)
    print("DEPTH_JOIN", json.dumps(audit, ensure_ascii=False), flush=True)


def _rewrite_metadata(join_audits: list[dict[str, Any]]) -> None:
    summary_path = OUTPUT_ROOT / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["variant"] = "multistep_angle_object_trajectory_policy_v4_depth"
        summary["base_comparison"] = "multistep_angle_object_trajectory_policy_v4_body_orientation"
        summary["policy_inputs"] = [
            "sample-specific human/camera relative angle",
            "causal 13-frame person center/velocity trajectory",
            "causal 13-frame torso-axis orientation sin/cos trajectory",
            "causal 13-frame per-class object presence/position/confidence trajectory",
            "object position relative to person center",
            "first-frame 50-slot MiDaS relative depth prior, pooled over observed views",
        ]
        summary["depth_definition"] = {
            "shape_per_view": [50],
            "dtype": "float32",
            "normalization": "per-frame 5th-95th percentile to [0,1]",
            "join": "exact sample_name; no camera-ID ordering assumption",
            "source": str(DEPTH_ROOT),
        }
        summary["depth_join_audits"] = join_audits
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    for audit_path in OUTPUT_ROOT.glob("seed_*/audit.json"):
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        audit["variant"] = "v4_depth"
        audit["base_variant"] = "v4_body_orientation"
        audit["depth_input"] = {
            "shape_per_view": [50],
            "dtype": "float32",
            "source": str(DEPTH_ROOT),
            "frame": "first decoded frame at 640x480",
        }
        audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUTPUT_ROOT / "depth_join_audits.json").write_text(
        json.dumps(join_audits, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> None:
    # Patch only the imported v1 runner's extension points.  The historical v1
    # and v4 files remain untouched and retain their original output folders.
    v1.CACHE_ROOT = CACHE_ROOT
    v1.TRACK_ROOT = TRACK_ROOT
    v1.OUTPUT_ROOT = OUTPUT_ROOT
    v1.build_trajectory_state = build_trajectory_state_v4_depth
    v1.AngleObjectTrajectoryPolicy = BodyOrientationDepthTrajectoryPolicy

    audits: list[dict[str, Any]] = []
    original_attach_object_tracks = v1.attach_object_tracks

    def attach_with_audit(raw: Any, track_root: Path, split: str) -> None:
        original_attach_object_tracks(raw, track_root, split)
        audit = attach_depth_tracks(raw, CACHE_ROOT, DEPTH_ROOT, split)
        audits.append(audit)
        print("DEPTH_JOIN", json.dumps(audit, ensure_ascii=False), flush=True)

    v1.attach_object_tracks = attach_with_audit
    v1.main()
    _rewrite_metadata(audits)


if __name__ == "__main__":
    main()
