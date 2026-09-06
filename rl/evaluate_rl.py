"""Label-free policy evaluation and multi-view baseline comparison."""

from __future__ import annotations

import argparse
from typing import Any, Dict


class Evaluator:
    """Compare learned view selection with reproducible three-view baselines."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config

    def run(self, split: str) -> Dict[str, Any]:
        """Evaluate single, random, fixed, DQN, and oracle policies."""

        raise NotImplementedError("Implement split evaluation and metric aggregation here.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate active multi-view VPOCLIP.")
    parser.add_argument("--config", default="rl/config_rl.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("seen_val", "pseudo_unseen_val", "unseen_test"), required=True)
    return parser.parse_args()


def main() -> None:
    """Run evaluation and write JSON predictions, metrics, and selected views."""

    raise NotImplementedError("Wire checkpoint loading and Evaluator.run here.")


if __name__ == "__main__":
    main()


# Implementation guide
# 1. Disable gradients and optimizer creation for every evaluation split.
# 2. During real unseen inference, choose views without labels and reveal labels
#    only after all predictions are frozen for metric computation.
# 3. Report final Top-1/Top-5, A-to-ABC gain, movement cost, selected-view
#    histogram, entropy/margin changes, and per-class accuracy.
# 4. Compare single A, random three views, fixed uniform views, DQN, and an
#    explicitly labeled oracle upper bound; never present oracle as deployable.
