"""Label-free policy evaluation and multi-view baseline comparison."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml

from .contracts import CacheShape
from .double_dqn_agent import DoubleDQNAgent
from .logit_fusion import LogitFusion
from .multiview_cache import MultiViewCache
from .networks import DuelingQNetwork
from .state_builder import StateBuilder


class Evaluator:
    """Compare learned view selection with reproducible three-view baselines."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config

    def run(self, split: str) -> Dict[str, Any]:
        """Evaluate deployable policies and a clearly labeled oracle upper bound."""
        split = {"seen_val": "val", "pseudo_unseen_val": "val", "unseen_test": "test"}.get(split, split)
        config_path = Path(self.config["_config_path"])
        root_value = Path(self.config["cache"]["root"])
        root = (config_path.parent / root_value).resolve() / split
        meta = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
        cache = MultiViewCache(root, CacheShape(len(meta["episodes"]), 4), mmap=True).open()
        device = torch.device(str(self.config["runtime"].get("device", "cuda:0")))
        prototypes = torch.from_numpy(np.load(root / "text_prototypes.npy")).float().to(device)
        fusion = LogitFusion(55, 5)
        class_ids = self.config.get("evaluation", {}).get("class_ids")
        class_tensor = None if class_ids is None else torch.as_tensor(class_ids, device=device)
        builder = StateBuilder(cache, fusion, prototypes, 4, max_selected_views=2)
        network = DuelingQNetwork(4, self.config["state"]).to(device)
        checkpoint = torch.load(self.config["_checkpoint_path"], map_location=device, weights_only=False)
        network.load_state_dict(checkpoint.get("online", checkpoint), strict=True)
        network.eval()

        def fused(episode: int, views: tuple[int, ...]):
            logits = torch.stack([cache.get_view(episode, view)["logits"] for view in views]).to(device)
            return fusion.fuse(logits, prototypes)

        def update(metrics: Dict[str, Any], output, target: int) -> None:
            scores = output.logits if class_tensor is None else output.logits.index_select(-1, class_tensor)
            prediction_index = int(torch.argmax(scores).item())
            prediction = prediction_index if class_tensor is None else int(class_tensor[prediction_index].item())
            top5_index = torch.topk(scores, k=min(5, scores.numel())).indices.tolist()
            top5 = top5_index if class_tensor is None else class_tensor[top5_index].tolist()
            metrics["top1"] += int(prediction == target)
            metrics["top5"] += int(target in top5)
            metrics["entropy_sum"] += float(output.entropy.item())

        names = ("single", "random", "fixed", "policy", "oracle")
        totals = {name: {"top1": 0, "top5": 0, "entropy_sum": 0.0, "movement_sum": 0.0} for name in names}
        selected_hist = {name: [0, 0, 0, 0] for name in names}
        rng = random.Random(20260906)
        predictions = []
        with torch.inference_mode():
            for episode in range(len(cache)):
                target = int(cache.arrays["labels"][episode])
                initial = 0
                single_output = fused(episode, (initial,))
                update(totals["single"], single_output, target)

                state, _ = (builder.build(episode, (initial,)), None)
                state_batch = {key: value.unsqueeze(0) for key, value in state.items()}
                q = DoubleDQNAgent.masked_q(network(state_batch), state["action_mask"].bool().unsqueeze(0))[0]
                valid = torch.nonzero(state["action_mask"].bool(), as_tuple=False).flatten().tolist()
                policy_view = int(torch.argmax(q).item())
                random_view = int(rng.choice(valid))
                fixed_view = 1 if 1 in valid else int(valid[0])

                candidates = []
                for candidate in valid:
                    output = fused(episode, (initial, candidate))
                    candidates.append((float(output.logits[target].item()), candidate))
                oracle_view = max(candidates)[1]
                choices = {"random": random_view, "fixed": fixed_view, "policy": policy_view, "oracle": oracle_view}
                row = {"episode_id": episode, "initial_view": initial, "target": target, "selected": choices}
                for name, view in choices.items():
                    output = fused(episode, (initial, view))
                    update(totals[name], output, target)
                    totals[name]["movement_sum"] += cache.get_cost(episode, initial, view)
                    selected_hist[name][view] += 1
                    row[f"{name}_prediction"] = int(torch.argmax(output.logits).item())
                predictions.append(row)

        result: Dict[str, Any] = {"split": split, "episodes": len(cache), "metrics": {}, "selected_view_histogram": selected_hist, "predictions": predictions}
        for name, values in totals.items():
            total = max(len(cache), 1)
            result["metrics"][name] = {
                "top1": values["top1"] / total,
                "top5": values["top5"] / total,
                "mean_entropy": values["entropy_sum"] / total,
                "mean_movement_cost": values["movement_sum"] / total,
            }
        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate active multi-view VPOCLIP.")
    parser.add_argument("--config", default="rl/config_rl.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("val", "test", "seen_val", "pseudo_unseen_val", "unseen_test"), required=True)
    return parser.parse_args()


def main() -> None:
    """Run evaluation and write JSON predictions, metrics, and selected views."""
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["_config_path"] = str(config_path)
    config["_checkpoint_path"] = str(Path(args.checkpoint).resolve())
    result = Evaluator(config).run(args.split)
    output_dir = (config_path.parent / config["train"]["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"evaluation_{args.split}.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "metrics": result["metrics"]}, ensure_ascii=False), flush=True)


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
