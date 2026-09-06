"""CLI scaffold for constructing aligned multi-view VPOCLIP caches."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict


def build_episode_manifest(config: Dict[str, Any]) -> Path:
    """Create one row per synchronized person/action/repetition/window episode."""

    raise NotImplementedError("Implement dataset-specific multi-view alignment here.")


def export_cache(config: Dict[str, Any], manifest_path: Path) -> Path:
    """Run the frozen adapter and write all cache arrays plus metadata."""

    raise NotImplementedError("Implement batched cache export and atomic finalization here.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an aligned VPOCLIP active-view cache.")
    parser.add_argument("--config", default="rl/config_rl.yaml")
    parser.add_argument("--split", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Load YAML, build the manifest, export arrays, and run cache validation."""

    raise NotImplementedError("Wire configuration and cache construction here.")


if __name__ == "__main__":
    main()


# Implementation guide
# 1. Build and inspect the episode manifest before running expensive inference.
# 2. Reject groups whose views differ in person, action, repetition, scene, or
#    temporal window; never pad an episode with unrelated videos.
# 3. Export z/logits in a single flattened B*V pass through VPOCLIPAdapter.
# 4. Save labels and target_columns separately and record class-bank ordering.
# 5. Write to a temporary directory, validate all shapes/checksums, then rename.
