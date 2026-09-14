# Dual-robot depth-aware DDQN experiment

This directory is an isolated experiment for the two-robot active-view
protocol.  It does not modify the historical v1--v6 policies, their caches,
or their checkpoints.

## State variants

Every policy receives the initial observation from robot A and the start
view of robot B.  The policy then selects two distinct target view slots from
the four available slots.

The human branch has six channels per frame:

```
torso center x, torso center y, velocity x, velocity y,
sin(body yaw), cos(body yaw)
```

The yaw is the frame-0 body-forward yaw extracted from the released 3-D
skeleton.  It is repeated over the 13-frame state window because this cache
contains one frame-0 orientation label per view.

The category-free object branch starts from the 50-slot detector track only
as an on-disk representation.  The policy never sees category IDs or slot
IDs.  For every frame, present objects are pooled using count ratio plus
mean/min/max of:

```
center x, center y, confidence, relative x to the person,
relative y to the person, relative depth to the person
```

This gives `1 + 6*3 = 19` pooled channels.  The relative depth is the
first-frame MiDaS normalized relative inverse-depth value, not metric depth.
It is used only as the relative `z` coordinate of a detected object with
respect to the person; it is not a depth image, a robot-distance signal, or a
separate human feature.  It is repeated over the 13-frame window.  The strict legacy no-depth form is
also implemented as a 16-channel diagnostic (`1 + 5*3`).

The default comparison is:

* `full_depth19`: six-channel human branch + nineteen-channel object branch;
* `human_only`: six-channel human branch;
* `object_only_depth19`: nineteen-channel object branch;
* `full_lstm19`: same two streams as `full_depth19`, with LSTM temporal encoders;
* `human_only_lstm`: same human stream as `human_only`, with an LSTM encoder;
* `object_only_lstm19`: same object stream as `object_only_depth19`, with an LSTM encoder;
* `random_pair`: uniform feasible `C(4,2)` pair;
* `cyclic_pair`: deterministic opposite pair `(p,(p+2) mod 4)`, with a
  feasible fallback when the requested pair is unavailable.

The object-only policy still uses person center and person depth as reference
coordinates when preprocessing relative object features.  It does not have
the separate human trajectory branch.

## Policy and reward

The network is a compact dueling Q-network.  Its state encoders are a
6-channel human temporal encoder and an optional 19/16-channel object
temporal encoder.  The default models use temporal convolutions; the three
`*_lstm*` models replace only these temporal convolutions with one-layer
32-hidden-unit LSTMs.  Thus both the human and object branches consume the
full 13-frame sequences, not a single static frame.  The encoders are joined
with the two robot start directions and the current selection/stage.  Each
candidate receives target direction, direction from both starts, A/B movement
costs, pair assignment cost, feasibility, and the selected flag.  Invalid,
visited, and infeasible actions are masked.

Training uses offline masked Double-DQN targets.  The recognizer and the
existing lightweight weighted-logit fusion gate are frozen.  The terminal
reward is the seen-bank fused-logit margin of the selected pair minus a
movement-cost penalty.  No labels, logits, future-view features, or depth
features from unselected views enter the policy state.

The policy is fitted on seen classes only under the existing 50/5 split
(`{14,15,25,39,46}` are held out).  The test report evaluates the true
unseen five-way bank and also writes a seen-bank diagnostic.

## Movement metric

The source cache provides normalized angular movement cost, not robot metric
trajectory length.  Therefore the report calls it `normalized_move_cost` and
also gives the equivalent angular cost in degrees (`cost * 180`).  It must
not be interpreted as meters.

## Run

Use the project environment that contains CUDA PyTorch:

```bash
/home/youhan/.conda/envs/clipgcn/bin/python -u -m \
  rl.dual_robot_depth_ddqn_v1.experiment \
  --output-root work_dir/dual_robot_depth_ddqn_v1
```

The command runs a smoke check, three training seeds per learned variant,
then 30 evaluation seeds and 95% confidence intervals over seed-level means.
The resolved configuration, per-seed metrics, checkpoints, and
`RESULTS.md` are written below `work_dir/dual_robot_depth_ddqn_v1`.
