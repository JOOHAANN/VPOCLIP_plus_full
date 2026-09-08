# VPOCLIP Active-View Double Dueling DQN

This directory is an implementation scaffold for selecting three synchronized
camera views while keeping the existing VPOCLIP recognizer frozen.

## Non-Negotiable Boundary

- VPOCLIP maps each view to a 512-D visual-language embedding and 55 class logits.
- The DQN outputs only 8 or 16 destination Q-values.
- One episode starts from A, selects B, selects C, classifies, and terminates.
- A, B, and C stay independent until their logits are fused.
- Labels are allowed in the seen-training reward path only.
- Real unseen evaluation performs no reward calculation and no parameter update.

## Package Map

| File | Responsibility |
|---|---|
| `contracts.py` | Shared cache, state, replay, reward, and diagnostic types. |
| `vpoclip_adapter.py` | Load, freeze, and batch-export VPOCLIP embeddings/logits. |
| `build_cache.py` | Build synchronized episode manifests and cache arrays. |
| `multiview_cache.py` | Memory-map arrays and enforce alignment/shape contracts. |
| `logit_fusion.py` | Fuse independent logits and derive Top-5/H/margin evidence. |
| `state_builder.py` | Build the only train/eval representation of s_A/s_AB/s_ABC. |
| `active_view_env.py` | Enforce the two-decision episode and training reward. |
| `networks.py` | Shared ViewEncoder and Dueling Q value/advantage heads. |
| `replay_buffer.py` | Store compact episode/view transitions. |
| `double_dqn_agent.py` | Masked epsilon-greedy and Double-DQN Bellman updates. |
| `train_rl.py` | Seen-only training orchestration and checkpoint selection. |
| `evaluate_rl.py` | Label-free policy execution and baseline reporting. |
| `config_rl.yaml` | Select M0/M1/M2 and all cache, reward, and agent settings. |

Every Python file ends with an English `Implementation guide` section. Those
sections define the intended code to write without hiding unfinished behavior.

## State Contract

The policy receives a dictionary with these tensors:

| Key | Intended shape | Meaning |
|---|---:|---|
| `view_features` | `[B,3,...]` | Fixed A/B/C slots with zero padding. |
| `evidence` | `[B,519]` | 512-D Top-5 prototype summary, five probabilities, H, margin. |
| `robot_context` | `[B,4V+4]` | Current/selected/reachable/costs, budget, `sin/cos(delta_A)`, yaw confidence. |
| `slot_mask` | `[B,3]` | Which A/B/C slots contain observations. |
| `action_mask` | `[B,V]` | Reachable destinations that have not been selected. |

For eight views, the encoded trunk input is
`3*256 + 128 + 64 + 3 = 963` dimensions. The network then uses a
`963 -> 512 -> 256` trunk, a `256 -> 128 -> 1` value head, and a
`256 -> 128 -> V` advantage head.

## Cache Contract

Required arrays:

```text
z.npy               [N,V,512]
logits.npy          [N,V,55]
reachable.npy       [N,V,V] bool
move_cost.npy       [N,V,V]
labels.npy          [N]
target_columns.npy  [N]
```

Optional M2 arrays:

```text
joint_xy.npy        [N,V,13,17,2]
joint_conf.npy      [N,V,13,17,1]
object_map.npy      [N,V,50,6,6]
image_quality.npy   [N,V,4]
view_geometry.npy   [N,V,d] or [V,d]
```

The manifest must prove that all views in one episode share person, action,
repetition, scene, and temporal window identifiers.

## Implementation Order

1. Build and manually inspect the synchronized episode manifest.
2. Implement the frozen adapter and prove cached logits match direct inference.
3. Implement fusion and StateBuilder; verify all slot states have fixed shapes.
4. Implement ActiveViewEnv and prove every episode creates exactly two transitions.
5. Implement DuelingQNetwork, replay, and masked Double-DQN updates.
6. Overfit a small seen subset and then train the complete seen split.
7. Evaluate single, random, fixed, DQN, and oracle policies on all allowed splits.
8. Enable M1 Top-5 evidence, then M2 pose/object branches as separate ablations.

## Running After Implementation

```bash
python -m rl.build_cache --config rl/config_rl.yaml --split seen_train
python -m rl.train_rl --config rl/config_rl.yaml
python -m rl.evaluate_rl \
  --config rl/config_rl.yaml \
  --checkpoint work_dir/active_view_ddqn/best.pt \
  --split unseen_test
python -m unittest discover -s rl/tests -v
```

At scaffold stage, CLI methods raise `NotImplementedError` and acceptance tests
are explicitly skipped. This is deliberate: incomplete RL behavior must not look
like a valid experiment.
