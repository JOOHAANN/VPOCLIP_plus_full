"""v6: v1 with every object-related input removed.

The person trajectory, relative human/camera angle and candidate geometry are
kept.  YOLO presence, coordinates, confidence, human-object relations and all
50 object slots are absent from the state.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import multistep_angle_object_trajectory_policy_v1 as v1
from .multistep_angle_object_trajectory_policy_v5_no_object import (
    PersonGeometryTrajectoryPolicy,
    attach_no_object_tracks,
    build_trajectory_state_v5,
)


ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = ROOT / "data" / "trajectory_latest_vpoclip_new50_5" / "cache_compat"
OUTPUT_ROOT = ROOT / "work_dir" / "multistep_angle_object_trajectory_policy_v6_no_object_group1"


def _rewrite_metadata() -> None:
    summary_path = OUTPUT_ROOT / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary.update(
            {
                "variant": "multistep_angle_object_trajectory_policy_v6_no_object",
                "base_comparison": "multistep_angle_object_trajectory_policy_v1",
                "policy_inputs": [
                    "sample-specific human/camera relative angle",
                    "causal 13-frame person center/velocity trajectory",
                    "candidate-view geometry and reachability",
                ],
                "object_features": "removed_entirely",
                "object_classes_in_state": 0,
                "object_channels_in_state": 0,
                "formal_unseen_split": [14, 15, 25, 39, 46],
            }
        )
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    for audit_path in OUTPUT_ROOT.glob("seed_*/audit.json"):
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        audit.update(
            {
                "variant": "v6_no_object",
                "policy_inputs_only": [
                    "sample-specific human/camera relative angle",
                    "causal 13-frame person center/velocity trajectory",
                    "candidate-view geometry and reachability",
                ],
                "object_features": "removed_entirely",
                "object_classes_in_state": 0,
                "object_channels_in_state": 0,
                "policy_inputs_excluded": [
                    "RGB/video embedding",
                    "VPOCLIP z",
                    "skeleton embedding",
                    "semantic belief",
                    "current logits",
                    "class ID",
                    "future-view features",
                    "all YOLO object features",
                    "all 50 object categories",
                ],
            }
        )
        audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")


def main() -> None:
    v1.CACHE_ROOT = CACHE_ROOT
    v1.TRACK_ROOT = ROOT / "data" / "no_object_tracks_not_used"
    v1.OUTPUT_ROOT = OUTPUT_ROOT
    v1.attach_object_tracks = attach_no_object_tracks
    v1.build_trajectory_state = build_trajectory_state_v5
    v1.AngleObjectTrajectoryPolicy = PersonGeometryTrajectoryPolicy

    original_dump = v1.dump

    def dump_v6(path: Path, value: Any) -> None:
        if path.name == "audit.json":
            value = dict(value)
            value["object_features"] = "removed_entirely"
            value["object_classes_in_state"] = 0
            value["object_channels_in_state"] = 0
        original_dump(path, value)

    v1.dump = dump_v6
    v1.main()
    _rewrite_metadata()


if __name__ == "__main__":
    main()
