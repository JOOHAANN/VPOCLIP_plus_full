"""GPU-resident trainer for the terminal A->B active-view DDQN task."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from .contracts import CacheShape, PolicyState
from .multiview_cache import MultiViewCache
from .networks import DuelingQNetwork


def _resolve(config: Mapping[str, Any], value: str | Path) -> Path:
    path = Path(value)
    base = Path(config["_config_path"]).parent
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class GPUCache:
    """Copy the compact frozen-recognizer cache to GPU once."""

    def __init__(self, root: Path, device: torch.device) -> None:
        metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
        cache = MultiViewCache(root, CacheShape(len(metadata["episodes"]), 4), mmap=True).open()
        self.device = device

        def load(name: str, dtype: torch.dtype | None = None) -> torch.Tensor:
            array = np.array(cache.arrays[name], copy=True)
            return torch.from_numpy(array).to(device=device, dtype=dtype, non_blocking=False)

        self.z = load("z", torch.float32)
        self.logits = load("logits", torch.float32)
        self.pose = load("pose", torch.float32)
        self.object_map = load("object_map", torch.float32).reshape(len(cache), 4, -1)
        self.quality = load("image_quality", torch.float32)
        self.geometry = load("view_geometry", torch.float32)
        self.reachable = load("reachable", torch.bool)
        self.cost = load("move_cost", torch.float32)
        self.labels = load("target_columns", torch.long)
        prototypes = np.load(root / "text_prototypes.npy", allow_pickle=False)
        self.prototypes = torch.from_numpy(np.array(prototypes, copy=True)).to(device=device, dtype=torch.float32)
        self.num_episodes = len(cache)

    def build_states(self, episodes: torch.Tensor, initial_views: torch.Tensor) -> PolicyState:
        """Vectorize the exact StateBuilder s_A contract on the GPU."""
        episodes = episodes.long()
        initial_views = initial_views.long()
        count = episodes.numel()
        device = self.device
        rows = torch.arange(count, device=device)

        view_features = torch.zeros((count, 3, 512), device=device)
        pose_features = torch.zeros((count, 3, self.pose.shape[-1]), device=device)
        object_features = torch.zeros((count, 3, self.object_map.shape[-1]), device=device)
        quality_features = torch.zeros((count, 3, self.quality.shape[-1]), device=device)
        view_features[:, 0] = self.z[episodes, initial_views]
        pose_features[:, 0] = self.pose[episodes, initial_views]
        object_features[:, 0] = self.object_map[episodes, initial_views]
        quality_features[:, 0] = self.quality[episodes, initial_views]

        logits = self.logits[episodes, initial_views]
        probabilities = torch.softmax(logits, dim=-1)
        top_probabilities, top_indices = torch.topk(probabilities, k=5, dim=-1)
        semantic = (
            top_probabilities.unsqueeze(-1) * self.prototypes[top_indices]
        ).sum(dim=1)
        entropy = -(probabilities.clamp_min(1e-8).log() * probabilities).sum(dim=-1)
        entropy = entropy / math.log(float(logits.shape[-1]))
        margin = top_probabilities[:, 0] - top_probabilities[:, 1]
        evidence = torch.cat((semantic, top_probabilities, entropy[:, None], margin[:, None]), dim=-1)

        current = F.one_hot(initial_views, num_classes=4).float()
        reachable = self.reachable[episodes, initial_views]
        costs = self.cost[episodes, initial_views].nan_to_num(1.0, 1.0, 1.0).clamp(0.0, 1.0)
        remaining = torch.ones((count, 1), device=device)
        candidate_geometry = self.geometry[episodes].clone()
        candidate_geometry[:, :, 2] = costs
        candidate_geometry[:, :, 3] = reachable.float()
        # Label-free candidate complementarity features for unseen transfer.
        candidate_logits = 0.5 * (self.logits[episodes, initial_views, None, :] + self.logits[episodes])
        candidate_probs = torch.softmax(candidate_logits, dim=-1)
        cand_entropy = -(candidate_probs * candidate_probs.clamp_min(1e-8).log()).sum(-1)
        cand_entropy = cand_entropy / math.log(float(self.logits.shape[-1]))
        top2 = candidate_probs.topk(2, dim=-1).values
        cand_margin = top2[..., 0] - top2[..., 1]
        current_prob = torch.softmax(self.logits[episodes, initial_views], dim=-1)
        disagreement = 1.0 - F.cosine_similarity(candidate_probs, current_prob[:, None, :], dim=-1)
        proxy = torch.stack((candidate_probs.max(-1).values, cand_entropy, cand_margin, disagreement), dim=-1)
        candidate_geometry = torch.cat((candidate_geometry, proxy), dim=-1)
        current_angle = candidate_geometry[rows, initial_views, :2]
        yaw_confidence = quality_features[:, 0, 3:4]
        robot_context = torch.cat(
            (
                current,
                current,
                reachable.float(),
                costs,
                remaining,
                current_angle,
                yaw_confidence,
            ),
            dim=-1,
        )
        action_mask = reachable.clone()
        action_mask[rows, initial_views] = False
        slot_mask = torch.zeros((count, 3), device=device)
        slot_mask[:, 0] = 1.0
        return {
            "view_features": view_features,
            "evidence": evidence,
            "robot_context": robot_context,
            "slot_mask": slot_mask,
            "action_mask": action_mask,
            "candidate_geometry": candidate_geometry,
            "pose_features": pose_features,
            "object_features": object_features,
            "quality_features": quality_features,
        }

    def all_context_states(self) -> PolicyState:
        episodes = torch.arange(self.num_episodes, device=self.device).repeat_interleave(4)
        initial = torch.arange(4, device=self.device).repeat(self.num_episodes)
        return self.build_states(episodes, initial)

    def reward_table(self, reward_config: Mapping[str, Any]) -> torch.Tensor:
        """Return exact terminal rewards for every [episode,A,B]."""
        labels = self.labels
        count = self.num_episodes
        class_ids = reward_config.get("class_ids")
        score_logits = self.logits if class_ids is None else self.logits[..., list(class_ids)]
        target_a = labels[:, None, None].expand(count, 4, 1)
        target_local = labels if class_ids is None else torch.tensor(
            [list(class_ids).index(int(x)) for x in labels.detach().cpu().tolist()], device=labels.device
        )
        ce_a = -F.log_softmax(score_logits, dim=-1).gather(-1, target_local[:, None, None].expand(count, 4, 1)).squeeze(-1)
        fused = 0.5 * (self.logits[:, :, None, :] + self.logits[:, None, :, :])
        fused_score = fused if class_ids is None else fused[..., list(class_ids)]
        ce_ab = -F.log_softmax(fused_score, dim=-1).gather(-1, target_local[:, None, None, None].expand(count, 4, 4, 1)).squeeze(-1)
        improvement = (ce_a[:, :, None] - ce_ab).clamp(
            -float(reward_config["ce_clip"]), float(reward_config["ce_clip"])
        )
        if class_ids is None:
            correct = fused.argmax(dim=-1).eq(labels[:, None, None])
        else:
            class_tensor = torch.as_tensor(list(class_ids), device=fused.device)
            correct = class_tensor[fused_score.argmax(dim=-1)].eq(labels[:, None, None])
        terminal = torch.where(correct, 1.0, -1.0)
        return (
            float(reward_config["ce_weight"]) * improvement
            + float(reward_config["terminal_weight"]) * terminal
            - float(reward_config["movement_weight"]) * self.cost
        )


class GPUReplay:
    """Fixed-size ring buffer containing only terminal transition indices."""

    def __init__(self, capacity: int, device: torch.device) -> None:
        self.capacity = int(capacity)
        self.context = torch.empty(self.capacity, dtype=torch.long, device=device)
        self.action = torch.empty(self.capacity, dtype=torch.long, device=device)
        self.reward = torch.empty(self.capacity, dtype=torch.float32, device=device)
        self.size = 0
        self.cursor = 0

    def add(self, context: torch.Tensor, action: torch.Tensor, reward: torch.Tensor) -> None:
        if context.numel() >= self.capacity:
            context, action, reward = context[-self.capacity :], action[-self.capacity :], reward[-self.capacity :]
        count = context.numel()
        first = min(count, self.capacity - self.cursor)
        self.context[self.cursor : self.cursor + first] = context[:first]
        self.action[self.cursor : self.cursor + first] = action[:first]
        self.reward[self.cursor : self.cursor + first] = reward[:first]
        rest = count - first
        if rest:
            self.context[:rest] = context[first:]
            self.action[:rest] = action[first:]
            self.reward[:rest] = reward[first:]
        self.cursor = (self.cursor + count) % self.capacity
        self.size = min(self.capacity, self.size + count)

    def sample(self, count: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        indices = torch.randint(self.size, (int(count),), device=self.context.device)
        return self.context[indices], self.action[indices], self.reward[indices]


def _take_state(states: PolicyState, indices: torch.Tensor) -> PolicyState:
    return {key: value[indices] for key, value in states.items()}


def _masked_q(network: DuelingQNetwork, state: PolicyState) -> torch.Tensor:
    q = network(state)
    return q.masked_fill(~state["action_mask"].bool(), torch.finfo(q.dtype).min)


def _best_logged_score(path: Path) -> float:
    best = float("-inf")
    if not path.exists():
        return best
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
            best = max(best, float(record.get("validation", {}).get("policy_top1", best)))
        except (ValueError, TypeError, json.JSONDecodeError):
            continue
    return best


@torch.inference_mode()
def evaluate(
    network: DuelingQNetwork,
    cache: GPUCache,
    states: PolicyState,
    batch_size: int,
    class_ids: list[int] | None = None,
) -> Dict[str, float]:
    network.eval()
    count = cache.num_episodes
    policy_actions = torch.empty(count, dtype=torch.long, device=cache.device)
    for start in range(0, count, batch_size):
        stop = min(count, start + batch_size)
        context = torch.arange(start, stop, device=cache.device) * 4
        state = _take_state(states, context)
        policy_actions[start:stop] = _masked_q(network, state).argmax(dim=-1)
    episodes = torch.arange(count, device=cache.device)
    initial = torch.zeros(count, dtype=torch.long, device=cache.device)
    generator = torch.Generator(device=cache.device).manual_seed(20260906)
    random_scores = torch.rand((count, 4), generator=generator, device=cache.device)
    random_scores[:, 0] = -1.0
    random_actions = random_scores.argmax(dim=-1)
    labels = cache.labels
    class_tensor = None if class_ids is None else torch.as_tensor(class_ids, device=cache.device)
    score = cache.logits if class_tensor is None else cache.logits.index_select(-1, class_tensor)
    single = score[:, 0].argmax(dim=-1)
    if class_tensor is not None:
        single = class_tensor[single]
    policy_logits = 0.5 * (cache.logits[:, 0] + cache.logits[episodes, policy_actions])
    random_logits = 0.5 * (cache.logits[:, 0] + cache.logits[episodes, random_actions])
    policy_score = policy_logits if class_tensor is None else policy_logits.index_select(-1, class_tensor)
    random_score = random_logits if class_tensor is None else random_logits.index_select(-1, class_tensor)
    if class_tensor is not None:
        policy_pred = class_tensor[policy_score.argmax(dim=-1)]
        random_pred = class_tensor[random_score.argmax(dim=-1)]
    else:
        policy_pred = policy_score.argmax(dim=-1)
        random_pred = random_score.argmax(dim=-1)
    network.train()
    return {
        "single_top1": float(single.eq(labels).float().mean().item()),
        "policy_top1": float(policy_pred.eq(labels).float().mean().item()),
        "random_top1": float(random_pred.eq(labels).float().mean().item()),
        "episodes": float(count),
    }


def train(config: Dict[str, Any], resume: Path | None, benchmark_updates: int = 0) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("train_rl_fast requires CUDA")
    device = torch.device(str(config["runtime"].get("device", "cuda:0")))
    seed = int(config["runtime"].get("seed", 0))
    _seed(seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    fast = config["fast"]
    batch_size = int(fast["batch_size"])
    updates_per_epoch = int(fast["updates_per_epoch"])
    selection_batch_size = int(fast.get("selection_batch_size", batch_size))

    cache_root = _resolve(config, config["cache"]["root"])
    train_cache = GPUCache(cache_root / "dqn_train", device)
    val_cache = GPUCache(cache_root / "val", device)
    print(
        json.dumps({
            "stage": "gpu_preload_complete",
            "train_episodes": train_cache.num_episodes,
            "val_episodes": val_cache.num_episodes,
            "gpu_allocated_gib": round(torch.cuda.memory_allocated(device) / 2**30, 3),
        }),
        flush=True,
    )
    train_states = train_cache.all_context_states()
    val_states = val_cache.all_context_states()
    rewards = train_cache.reward_table(config["reward"])
    print(
        json.dumps({
            "stage": "gpu_states_complete",
            "contexts": int(train_cache.num_episodes * 4),
            "gpu_allocated_gib": round(torch.cuda.memory_allocated(device) / 2**30, 3),
        }),
        flush=True,
    )

    online = DuelingQNetwork(4, config["state"]).to(device)
    target = DuelingQNetwork(4, config["state"]).to(device)
    optimizer = torch.optim.AdamW(
        online.parameters(),
        lr=float(config["agent"]["learning_rate"]),
        weight_decay=float(config["agent"]["weight_decay"]),
    )
    if resume is not None:
        checkpoint = torch.load(resume, map_location=device, weights_only=False)
        online.load_state_dict(checkpoint["online"], strict=True)
        target.load_state_dict(checkpoint.get("target", checkpoint["online"]), strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        updates = int(checkpoint.get("updates", 0))
        global_step = int(checkpoint.get("global_step", (start_epoch - 1) * train_cache.num_episodes))
    else:
        target.load_state_dict(online.state_dict())
        start_epoch = 1
        updates = 0
        global_step = 0
    epochs = int(config["train"]["epochs"])
    replay = GPUReplay(int(fast.get("replay_capacity", config["agent"]["replay_capacity"])), device)
    grad_clip = float(config["agent"]["grad_clip"])
    eps_cfg = config["agent"]["epsilon"]

    compiled: Any = online
    if bool(fast.get("compile", True)):
        compiled = torch.compile(online, mode=str(fast.get("compile_mode", "reduce-overhead")))
    amp_dtype = torch.bfloat16 if str(fast.get("amp_dtype", "bfloat16")) == "bfloat16" else torch.float16

    output = _resolve(config, config["train"]["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "config_resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    log_path = output / "train.log"
    best_policy_top1 = _best_logged_score(log_path)
    benchmark_mode = benchmark_updates > 0
    epoch_range = range(start_epoch, epochs + 1) if not benchmark_mode else range(start_epoch, start_epoch + 1)
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        marker = {
            "event": "fast_resume" if resume is not None else "fast_start",
            "checkpoint": str(resume) if resume is not None else None,
            "resume_epoch": start_epoch - 1,
            "batch_size": batch_size,
            "updates_per_epoch": updates_per_epoch,
            "compile": bool(fast.get("compile", True)),
        }
        print(json.dumps(marker), flush=True)
        if not benchmark_mode:
            log.write(json.dumps(marker) + "\n")
        for epoch in epoch_range:
            epoch_start = time.perf_counter()
            episodes = torch.randperm(train_cache.num_episodes, device=device)
            initial = torch.randint(0, 4, (train_cache.num_episodes,), device=device)
            context = episodes * 4 + initial
            greedy = torch.empty_like(initial)
            online.eval()
            with torch.inference_mode(), torch.autocast("cuda", dtype=amp_dtype):
                for start in range(0, context.numel(), selection_batch_size):
                    stop = min(context.numel(), start + selection_batch_size)
                    state = _take_state(train_states, context[start:stop])
                    greedy[start:stop] = _masked_q(online, state).argmax(dim=-1)
            step_numbers = global_step + torch.arange(context.numel(), device=device)
            epsilon = (
                float(eps_cfg["start"])
                - (float(eps_cfg["start"]) - float(eps_cfg["end"]))
                * step_numbers.float()
                / max(1, int(eps_cfg["decay_steps"]))
            ).clamp_min(float(eps_cfg["end"]))
            random_scores = torch.rand((context.numel(), 4), device=device)
            state_masks = train_states["action_mask"][context]
            random_actions = random_scores.masked_fill(~state_masks, -1.0).argmax(dim=-1)
            actions = torch.where(torch.rand(context.numel(), device=device) < epsilon, random_actions, greedy)
            reward = rewards[episodes, initial, actions]
            replay.add(context, actions, reward)
            global_step += context.numel()
            online.train()

            losses: list[torch.Tensor] = []
            run_updates = benchmark_updates if benchmark_mode else updates_per_epoch
            torch.cuda.synchronize(device)
            update_start = time.perf_counter()
            for _ in range(run_updates):
                sampled_context, sampled_action, sampled_reward = replay.sample(batch_size)
                state = _take_state(train_states, sampled_context)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=amp_dtype):
                    q = compiled(state).gather(1, sampled_action[:, None]).squeeze(1)
                    loss = F.smooth_l1_loss(q.float(), sampled_reward)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(online.parameters(), grad_clip)
                optimizer.step()
                # Keep scalar losses on-device; .item() here would force one
                # CPU/GPU synchronization per optimizer update.
                losses.append(loss.detach())
                updates += 1
            torch.cuda.synchronize(device)
            update_seconds = time.perf_counter() - update_start
            throughput = run_updates * batch_size / max(update_seconds, 1e-9)
            if benchmark_mode:
                print(json.dumps({
                    "benchmark_updates": run_updates,
                    "batch_size": batch_size,
                    "seconds": update_seconds,
                    "transitions_per_second": throughput,
                    "gpu_peak_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                }), flush=True)
                return

            target.load_state_dict(online.state_dict())
            validation = evaluate(
                online, val_cache, val_states,
                int(fast.get("validation_batch_size", batch_size)),
                class_ids=config.get("evaluation", {}).get("seen_class_ids"),
            )
            record = {
                "epoch": epoch,
                "trainer": "gpu_resident_terminal_ddqn",
                "steps": global_step,
                "updates": updates,
                "epsilon": float(epsilon[-1].item()),
                "reward_mean": float(reward.mean().item()),
                "loss_mean": float(torch.stack(losses).mean().item()),
                "update_seconds": update_seconds,
                "transitions_per_second": throughput,
                "epoch_seconds": time.perf_counter() - epoch_start,
                "validation": validation,
            }
            print(json.dumps(record), flush=True)
            log.write(json.dumps(record) + "\n")
            saved = {
                "online": online.state_dict(),
                "target": target.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "updates": updates,
                "global_step": global_step,
                "config": config,
                "trainer": "gpu_resident_terminal_ddqn",
                "validation": validation,
            }
            torch.save(saved, output / "last.pt")
            if validation["policy_top1"] >= best_policy_top1:
                best_policy_top1 = validation["policy_top1"]
                torch.save(saved, output / "best.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).with_name("config_rl.yaml")))
    parser.add_argument("--resume", default=None)
    parser.add_argument("--benchmark-updates", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--updates-per-epoch", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["_config_path"] = str(config_path)
    if args.batch_size is not None:
        config["fast"]["batch_size"] = int(args.batch_size)
    if args.updates_per_epoch is not None:
        config["fast"]["updates_per_epoch"] = int(args.updates_per_epoch)
    resume = Path(args.resume).resolve() if args.resume else None
    train(config, resume, args.benchmark_updates)


if __name__ == "__main__":
    main()
