"""Train the masked single-step active second-view policy."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml

from .active_view_env import ActiveViewEnv, RewardConfig
from .contracts import CacheShape, IndexTransition
from .double_dqn_agent import DoubleDQNAgent
from .logit_fusion import LogitFusion
from .multiview_cache import MultiViewCache
from .networks import DuelingQNetwork
from .replay_buffer import ReplayBuffer
from .state_builder import StateBuilder


def _resolve(config: Dict[str, Any], value: str | Path) -> Path:
    path = Path(value)
    base = Path(config["_config_path"]).parent
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _seed(value: int) -> None:
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def build_training_components(config: Dict[str, Any]) -> Dict[str, Any]:
    """Construct the cache, policy state builder, environment and DQN."""
    device = torch.device(str(config["runtime"].get("device", "cuda:0")))
    root = _resolve(config, config["cache"]["root"]) / "dqn_train"
    meta = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    cache = MultiViewCache(
        root,
        CacheShape(episodes=len(meta["episodes"]), views=4, embedding_dim=512, classes=55),
        mmap=bool(config["cache"].get("mmap", True)),
    ).open()
    prototypes = torch.from_numpy(np.load(root / "text_prototypes.npy")).float().to(device)
    state_builder = StateBuilder(cache, LogitFusion(55, 5), prototypes, 4, max_selected_views=2)
    env = ActiveViewEnv(
        cache, state_builder,
        RewardConfig(**{key: float(value) for key, value in config["reward"].items() if key in {"ce_weight", "terminal_weight", "movement_weight", "ce_clip"}}),
        training=True,
    )
    online = DuelingQNetwork(4, config["state"]).to(device)
    target = DuelingQNetwork(4, config["state"]).to(device)
    optimizer = torch.optim.AdamW(
        online.parameters(),
        lr=float(config["agent"]["learning_rate"]),
        weight_decay=float(config["agent"]["weight_decay"]),
    )
    agent = DoubleDQNAgent(online, target, state_builder, optimizer, float(config["agent"]["gamma"]), float(config["agent"]["grad_clip"]))
    agent.sync_target()
    return {"cache": cache, "state_builder": state_builder, "env": env, "online": online, "target": target, "optimizer": optimizer, "agent": agent, "device": device}


def _evaluate(components: Dict[str, Any], config: Dict[str, Any], split: str) -> Dict[str, float]:
    root = _resolve(config, config["cache"]["root"]) / split
    if not (root / "metadata.json").exists():
        return {}
    meta = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    cache = MultiViewCache(root, CacheShape(len(meta["episodes"]), 4), mmap=True).open()
    device = components["device"]
    prototypes = torch.from_numpy(np.load(root / "text_prototypes.npy")).float().to(device)
    builder = StateBuilder(cache, LogitFusion(55, 5), prototypes, 4, max_selected_views=2)
    network = components["online"]
    network.eval()
    single_correct = 0
    policy_correct = 0
    random_correct = 0
    total = len(cache)
    generator = random.Random(20260906)
    with torch.inference_mode():
        for episode in range(total):
            a = 0  # fixed navigation start for reproducible validation
            target = int(cache.arrays["labels"][episode])
            a_logits = torch.from_numpy(np.asarray(cache.arrays["logits"][episode, a])).to(device)
            if int(torch.argmax(a_logits).item()) == target:
                single_correct += 1
            state = builder.build(episode, (a,))
            q = DoubleDQNAgent.masked_q(network({key: value.unsqueeze(0) for key, value in state.items()}), state["action_mask"].unsqueeze(0))
            b = int(torch.argmax(q[0]).item())
            available = torch.nonzero(state["action_mask"], as_tuple=False).flatten().tolist()
            random_b = generator.choice(available)
            for selected, field in ((b, "policy"), (random_b, "random")):
                logits = torch.stack([torch.from_numpy(np.asarray(cache.arrays["logits"][episode, a])), torch.from_numpy(np.asarray(cache.arrays["logits"][episode, selected]))]).to(device)
                pred = int(torch.argmax(builder.fusion.fuse(logits, prototypes).logits).item())
                if pred == target:
                    if field == "policy": policy_correct += 1
                    else: random_correct += 1
    network.train()
    return {"single_top1": single_correct / max(total, 1), "policy_top1": policy_correct / max(total, 1), "random_top1": random_correct / max(total, 1), "episodes": float(total)}


def train(config: Dict[str, Any]) -> None:
    """Collect one A->B transition per episode and optimize the masked Q net."""
    _seed(int(config["runtime"].get("seed", 0)))
    components = build_training_components(config)
    cache = components["cache"]
    env = components["env"]
    agent = components["agent"]
    replay = ReplayBuffer(int(config["agent"]["replay_capacity"]), int(config["runtime"].get("seed", 0)))
    output = _resolve(config, config["train"]["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "config_resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    log_path = output / "train.log"
    updates = 0
    global_step = 0
    eps_cfg = config["agent"]["epsilon"]
    warmup = int(config["agent"]["replay_warmup"])
    batch_size = int(config["agent"]["batch_size"])
    epochs = int(config["train"].get("epochs", 1))
    best_policy_top1 = float("-inf")
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        for epoch in range(1, epochs + 1):
            order = np.random.permutation(len(cache))
            rewards, losses = [], []
            for episode in order.tolist():
                epsilon = max(float(eps_cfg["end"]), float(eps_cfg["start"]) - (float(eps_cfg["start"]) - float(eps_cfg["end"])) * global_step / max(1, int(eps_cfg["decay_steps"])))
                state, info = env.reset(int(episode))
                action = agent.select_action(state, epsilon)
                next_state, reward, done, _truncated, _step_info = env.step(action)
                replay.add(IndexTransition(int(episode), (int(info["selected_views"][0]),), int(action), float(reward), (int(info["selected_views"][0]), int(action)), bool(done)))
                rewards.append(float(reward)); global_step += 1
                if len(replay) >= warmup:
                    sample_size = min(batch_size, len(replay))
                    metrics = agent.update(replay.sample(sample_size))
                    losses.append(metrics["loss"]); updates += 1
                    if updates % int(config["agent"]["target_update_interval"]) == 0:
                        agent.sync_target()
            validation = _evaluate(components, config, "val") if epoch % int(config["train"].get("validation_interval", 1)) == 0 else {}
            record = {"epoch": epoch, "steps": global_step, "updates": updates, "epsilon": epsilon, "reward_mean": float(np.mean(rewards)), "loss_mean": float(np.mean(losses)) if losses else None, "validation": validation}
            print(json.dumps(record), flush=True); log.write(json.dumps(record) + "\n")
            checkpoint = {"online": agent.online.state_dict(), "target": agent.target.state_dict(), "optimizer": agent.optimizer.state_dict(), "epoch": epoch, "updates": updates, "config": config}
            torch.save(checkpoint, output / "last.pt")
            if validation and validation.get("policy_top1", -1.0) >= best_policy_top1:
                best_policy_top1 = float(validation["policy_top1"])
                torch.save(checkpoint, output / "best.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).with_name("config_rl.yaml")))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.config).resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    config["_config_path"] = str(path)
    train(config)


if __name__ == "__main__":
    main()
