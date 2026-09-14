# Dual-robot depth-aware DDQN results

Primary test bank: true unseen five-way classes `[14, 15, 25, 39, 46]`.
Each cell is the mean over 30 random-start seed aggregates with a 95% normal CI.

| method | weighted-fusion accuracy | normalized move cost | angular cost (deg-equivalent) | completion |
|---|---:|---:|---:|---:|
| `cyclic_adjacent_pair` | 0.6875 [0.6863, 0.6886] | 0.2359 [0.2313, 0.2405] | 42.46 [41.63, 43.29] | 1.0000 |
| `cyclic_pair` | 0.6900 [0.6887, 0.6914] | 0.2274 [0.2223, 0.2325] | 40.93 [40.01, 41.84] | 1.0000 |
| `full_depth19` | 0.6834 [0.6811, 0.6857] | 0.2266 [0.2226, 0.2306] | 40.79 [40.07, 41.51] | 1.0000 |
| `full_lstm19` | 0.6754 [0.6730, 0.6778] | 0.2358 [0.2320, 0.2395] | 42.44 [41.76, 43.12] | 1.0000 |
| `human_only` | 0.6849 [0.6820, 0.6878] | 0.2663 [0.2618, 0.2707] | 47.93 [47.12, 48.73] | 1.0000 |
| `human_only_lstm` | 0.6856 [0.6824, 0.6888] | 0.2651 [0.2596, 0.2705] | 47.71 [46.72, 48.70] | 1.0000 |
| `object_only_depth19` | 0.6935 [0.6911, 0.6959] | 0.2282 [0.2230, 0.2335] | 41.08 [40.14, 42.02] | 1.0000 |
| `object_only_lstm19` | 0.6801 [0.6771, 0.6830] | 0.3004 [0.2981, 0.3027] | 54.07 [53.66, 54.48] | 1.0000 |
| `random_pair` | 0.6868 [0.6834, 0.6903] | 0.2279 [0.2234, 0.2324] | 41.02 [40.22, 41.83] | 1.0000 |

## Protocol notes

* Robot A starts at a uniformly sampled valid slot and performs one single-view recognition. This view is not fused during selection; its human/object state is the policy input.
* Robot B starts independently at a uniformly sampled valid slot. The selected pair is assigned to the two robots using the cheaper feasible assignment; a selected current slot therefore has zero cost for that robot.
* `cyclic_pair` uses `(phase, (phase+2) mod 4)`; `cyclic_adjacent_pair` is also reported as a sensitivity check.
* Movement values are normalized angular costs from the cache. They are not meters. The degree-equivalent column is `180 * normalized_move_cost`.
* The final output uses the existing lightweight weighted-logit fusion gate over the two selected per-view logits.

## Resolved configuration

```json
{
  "experiment": "dual_robot_depth_ddqn_v1",
  "cache_root": "/home/youhan/ws/VPOCLIP_plus_full/data/frame0_body_angle_rank_g1_pseudo_selected_stage_b/cache",
  "track_root": "/home/youhan/ws/VPOCLIP_plus_full/data/frame0_body_angle_rank_g1_pseudo_selected_stage_b/object_tracks_full",
  "depth_root": "/home/youhan/ws/VPOCLIP_plus_full/data/new50_5_group1_zsl_depth_first_frame_fp32",
  "gate_checkpoint": "/home/youhan/ws/VPOCLIP_plus_full/work_dir/fusion_lightweight_gating_v1/models/seed_20260910/best.pt",
  "device": "cuda:0",
  "training_classes": [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    40,
    41,
    42,
    43,
    44,
    45,
    47,
    48,
    49,
    50,
    51,
    52,
    53,
    54
  ],
  "true_unseen_test_classes": [
    14,
    15,
    25,
    39,
    46
  ],
  "pseudo_validation_classes": [
    0,
    2,
    9,
    10,
    11,
    17,
    26,
    34,
    49,
    50
  ],
  "train_seeds": [
    20260909,
    20260910,
    20260911
  ],
  "eval_seeds": [
    20260920,
    20260921,
    20260922,
    20260923,
    20260924,
    20260925,
    20260926,
    20260927,
    20260928,
    20260929,
    20260930,
    20260931,
    20260932,
    20260933,
    20260934,
    20260935,
    20260936,
    20260937,
    20260938,
    20260939,
    20260940,
    20260941,
    20260942,
    20260943,
    20260944,
    20260945,
    20260946,
    20260947,
    20260948,
    20260949
  ],
  "variants": [
    "full_depth19",
    "human_only",
    "object_only_depth19",
    "full_lstm19",
    "human_only_lstm",
    "object_only_lstm19"
  ],
  "human_channels": [
    "center_x",
    "center_y",
    "velocity_x",
    "velocity_y",
    "sin_3d_body_yaw",
    "cos_3d_body_yaw"
  ],
  "human_yaw_source": "released 3-D skeleton body_yaw_deg.npy; repeated over 13 frames",
  "temporal_input": "13-frame sequence for both human and object branches",
  "temporal_encoder_by_variant": {
    "full_depth19": "Conv1d",
    "human_only": "Conv1d",
    "object_only_depth19": "Conv1d",
    "full_lstm19": "one-layer LSTM, hidden_size=32",
    "human_only_lstm": "one-layer LSTM, hidden_size=32",
    "object_only_lstm19": "one-layer LSTM, hidden_size=32"
  },
  "object_raw_track": [
    "presence",
    "center_x",
    "center_y",
    "confidence"
  ],
  "object_pooled_fields": [
    "center_x",
    "center_y",
    "confidence",
    "relative_x_to_person",
    "relative_y_to_person",
    "relative_depth_to_person"
  ],
  "object_pooled_width_with_depth": 19,
  "object_pooled_width_without_depth": 16,
  "object_category_or_slot_seen_by_policy": false,
  "relative_depth_definition": "object MiDaS relative inverse depth minus person-slot depth; used only as object relative z; normalized first-frame depth; not metric",
  "robot_protocol": "robot A random valid start performs single-view recognition; robot B random independent start; select two distinct final targets",
  "action_space": "four target slots with invalid, already-selected, and infeasible pairs masked",
  "assignment_cost": "min(cost(A->i)+cost(B->j), cost(A->j)+cost(B->i))",
  "movement_metric": "normalized angular movement cost from move_cost.npy; degree equivalent equals cost*180; not meters",
  "policy_algorithm": "offline masked Double Dueling DQN",
  "reward": "terminal seen-bank weighted-fusion margin minus distance_lambda times assignment cost",
  "distance_lambda": 0.0,
  "hyperparameters": {
    "epochs": 30,
    "batch_size": 32768,
    "updates_per_epoch": 96,
    "learning_rate": 0.0003,
    "weight_decay": 0.0001,
    "gamma": 0.99,
    "target_update_interval": 250
  },
  "depth_join_audits": {
    "dqn_train": {
      "split": "dqn_train",
      "episodes": 3328,
      "views_joined": 12541,
      "depth_rows_available": 91406,
      "join_key": "metadata.episodes[*].views[*].sample_name",
      "depth_definition": "first-frame MiDaS normalized relative inverse depth; not metric depth"
    },
    "val": {
      "split": "val",
      "episodes": 1312,
      "views_joined": 4991,
      "depth_rows_available": 91406,
      "join_key": "metadata.episodes[*].views[*].sample_name",
      "depth_definition": "first-frame MiDaS normalized relative inverse depth; not metric depth"
    },
    "test": {
      "split": "test",
      "episodes": 2974,
      "views_joined": 11324,
      "depth_rows_available": 91406,
      "join_key": "metadata.episodes[*].views[*].sample_name",
      "depth_definition": "first-frame MiDaS normalized relative inverse depth; not metric depth"
    }
  },
  "pool": {
    "transitions": 741888,
    "stage1": 185472,
    "stage2": 556416,
    "training_episodes_seen_classes_four_valid_views": 2898
  },
  "selected_checkpoints": {
    "full_depth19": "work_dir/dual_robot_depth_ddqn_v1_lstm_compare_lambda0/models/full_depth19/seed_20260910/best.pt",
    "human_only": "work_dir/dual_robot_depth_ddqn_v1_lstm_compare_lambda0/models/human_only/seed_20260910/best.pt",
    "object_only_depth19": "work_dir/dual_robot_depth_ddqn_v1_lstm_compare_lambda0/models/object_only_depth19/seed_20260910/best.pt",
    "full_lstm19": "work_dir/dual_robot_depth_ddqn_v1_lstm_compare_lambda0/models/full_lstm19/seed_20260909/best.pt",
    "human_only_lstm": "work_dir/dual_robot_depth_ddqn_v1_lstm_compare_lambda0/models/human_only_lstm/seed_20260909/best.pt",
    "object_only_lstm19": "work_dir/dual_robot_depth_ddqn_v1_lstm_compare_lambda0/models/object_only_lstm19/seed_20260910/best.pt"
  },
  "depth_audits": {
    "dqn_train": {
      "split": "dqn_train",
      "episodes": 3328,
      "views_joined": 12541,
      "depth_rows_available": 91406,
      "join_key": "metadata.episodes[*].views[*].sample_name",
      "depth_definition": "first-frame MiDaS normalized relative inverse depth; not metric depth"
    },
    "val": {
      "split": "val",
      "episodes": 1312,
      "views_joined": 4991,
      "depth_rows_available": 91406,
      "join_key": "metadata.episodes[*].views[*].sample_name",
      "depth_definition": "first-frame MiDaS normalized relative inverse depth; not metric depth"
    },
    "test": {
      "split": "test",
      "episodes": 2974,
      "views_joined": 11324,
      "depth_rows_available": 91406,
      "join_key": "metadata.episodes[*].views[*].sample_name",
      "depth_definition": "first-frame MiDaS normalized relative inverse depth; not metric depth"
    }
  }
}
```
