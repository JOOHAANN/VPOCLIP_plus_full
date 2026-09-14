#!/usr/bin/env python3
"""Run the final 50/5 recognizer refit after pseudo-unseen selection.

The pseudo-unseen classes [1, 7, 14, 15, 18] are no longer held out here.
Only [25, 39, 46, 52, 54] remain unseen.  The driver reuses the established
50/5 backbone and VPOCLIP recipe, while writing to isolated final-refit
directories.  The active-view/DDQN stage is run separately after its logits
and geometry cache have been built.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


VPO = Path("/home/youhan/ws/VPOCLIP_plus_full")
RL = VPO / "rl"
sys.path.insert(0, str(VPO / "tools"))

import run_new50_5_groups_pipeline as pipeline  # noqa: E402


FINAL_UNSEEN = [25, 39, 46, 52, 54]
PSEUDO_READDED = [1, 7, 14, 15, 18]
RUN_ROOT = VPO / "logs/final50_5_pseudo_refit_20260914"


def main() -> None:
    # Keep the same canonical subject split and the same recipes selected in
    # the strict 45/5/5 stage.  Only the final class partition changes.
    pipeline.RUN_ROOT = RUN_ROOT
    pipeline.RAW_CACHE = Path("/home/youhan/ws/X3D_full/data/etri_rgb_allviews_cs_45_25_10_20")
    pipeline.POSE_CACHE = Path("/home/youhan/ws/CTR-GCN_17_full/data/etri_coco17/allviews_cs45")
    pipeline.POSE_NPZ = pipeline.POSE_CACHE / "ETRI_55_CS_rtmpose_coco17_13.npz"
    pipeline.SUBJECT_MANIFEST = Path("/home/youhan/ws/ETRI_subject_split_45_25_10_20.json")
    pipeline.OBJECT_DIR = VPO / "data/final50_5_pseudo_refit_object"
    # This directory is a large shared extraction cache.  It is deliberately
    # reused so the refit does not create a second 24-GiB feature copy.
    pipeline.FEATURE_DIR = VPO / "data/new50_5_groups_shared_features"
    pipeline.ZSL_ROOT = VPO / "data"
    pipeline.X3D_TEMPLATE = Path(
        "/home/youhan/ws/X3D_full/configs/x3d-s_etri_rgb_cs_45_25_10_20_zsl50_5_182.yaml"
    )
    pipeline.CTR_TEMPLATE = Path(
        "/home/youhan/ws/CTR-GCN_17_full/config/etri-coco17/ctrgcn_joint_coco17_13.yaml"
    )
    pipeline.VPO_STAGE_A_TEMPLATE = VPO / "configs/vpoclip_new50_5_recipe_stage_a.yaml"
    pipeline.VPO_STAGE_B_TEMPLATE = VPO / "configs/vpoclip_new50_5_recipe_stage_b.yaml"

    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "protocol": "final_50_5_after_pseudo_unseen_selection",
        "pseudo_unseen_used_for_selection": PSEUDO_READDED,
        "pseudo_unseen_readded_to_seen": PSEUDO_READDED,
        "final_seen_classes": sorted(set(range(55)) - set(FINAL_UNSEEN)),
        "final_true_unseen_classes": FINAL_UNSEEN,
        "recognizer_training_classes": sorted(set(range(55)) - set(FINAL_UNSEEN)),
        "hyperparameters_source": str(
            VPO / "work_dir/strict45_5_sequential_reward_search/selected_reward.json"
        ),
        "active_policy_hyperparameters_source": str(
            VPO / "work_dir/strict45_5_sequential_accuracy_first/config_resolved.json"
        ),
        "note": "No final true-unseen sample or label is used during this refit.",
    }
    (RUN_ROOT / "final_refit_protocol.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    pipeline.run_group({"name": "final", "unseen_zero_based": FINAL_UNSEEN})
    (RUN_ROOT / "complete.json").write_text(
        json.dumps({**manifest, "status": "complete"}, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
