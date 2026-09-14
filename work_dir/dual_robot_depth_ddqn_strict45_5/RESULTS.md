# Dual-robot active zero-shot HAR: independent random-start results

## Scope of this report

This report contains the final comparison under the **independent random-start protocol**. Each sample initializes the two robots from valid candidate viewpoints before target selection.

The table combines the two previously reported results into one view. It reports unseen accuracy, 95% confidence intervals, accuracy differences relative to Random and the non-adjacent cyclic baseline, movement cost, movement confidence intervals, and movement relative to Random.

## Main result

| Model | Unseen accuracy | 95% CI | Delta vs. Random (percentage points) | Delta vs. Cyclic (percentage points) | Movement cost | 95% CI | Degree-equivalent movement | Movement / Random |
|---|---:|:---:|---:|---:|---:|:---:|---:|---:|
| Random | 64.56% | [64.07, 65.04] | -- | -- | 0.2698 | [0.2647, 0.2749] | 48.56° [47.65°, 49.48°] | 100.0% |
| Cyclic (non-adjacent / 隔位循环) | 64.95% | [64.57, 65.32] | +0.39 | -- | 0.2476 | [0.2431, 0.2520] | 44.56° [43.76°, 45.37°] | 91.8% |
| Cyclic (adjacent / 相邻循环) | 64.98% | [64.68, 65.28] | +0.42 | +0.03 | 0.2798 | [0.2752, 0.2844] | 50.36° [49.54°, 51.19°] | 103.7% |
| **Full DQN (DDQN): human + object + relative depth** | **65.33%** | **[64.95, 65.70]** | **+0.77** | **+0.38** | **0.0737** | **[0.0713, 0.0762]** | **13.27° [12.83°, 13.71°]** | **27.3%** |
| Object-only DQN (with relative depth) | 65.22% | [64.81, 65.64] | +0.66 | +0.27 | 0.0671 | [0.0642, 0.0701] | 12.09° [11.56°, 12.61°] | 24.9% |
| Human trajectory-only DQN | 65.06% | [64.76, 65.36] | +0.50 | +0.11 | 0.1293 | [0.1263, 0.1324] | 23.28° [22.73°, 23.83°] | 47.9% |
| Full LSTM DQN | 64.97% | [64.66, 65.28] | +0.41 | +0.02 | 0.1271 | [0.1236, 0.1307] | 22.89° [22.25°, 23.52°] | 47.1% |
| Object-only LSTM DQN (with relative depth) | 64.66% | [64.36, 64.97] | +0.10 | -0.29 | 0.1412 | [0.1380, 0.1443] | 25.41° [24.85°, 25.97°] | 52.3% |
| Human-only LSTM DQN | 64.62% | [64.35, 64.88] | +0.06 | -0.33 | 0.1524 | [0.1484, 0.1564] | 27.43° [26.71°, 28.15°] | 56.5% |
| Human orientation-only DQN | 64.46% | [64.18, 64.73] | -0.10 | -0.49 | 0.2241 | [0.2194, 0.2288] | 40.34° [39.50°, 41.19°] | 83.1% |
| Human-only DQN | 64.25% | [63.99, 64.51] | -0.31 | -0.70 | 0.1444 | [0.1403, 0.1486] | 26.00° [25.25°, 26.74°] | 53.5% |

`Delta vs. Random` and `Delta vs. Cyclic` are accuracy differences in percentage points. The cyclic reference is the non-adjacent/隔位 cyclic policy. Movement cost is the sum of the two robots' normalized angular costs after the minimum-cost robot-to-target assignment; it is not a distance in metres. The degree-equivalent value is `180 x movement cost`.

The Full DQN is the best recognition model in this run: it reaches 65.33%, which is 0.77 percentage points above Random and 0.38 points above the non-adjacent cyclic policy. Its movement cost is 0.0737, or 27.3% of Random, corresponding to a 72.7% reduction. The object-only DQN has slightly lower accuracy but the smallest movement cost among the learned policies.

The confidence intervals overlap, so these values should be reported as empirical means and not as a claim of statistical significance between every pair of methods.

## VPOCLIP-only no-movement baseline

To separate the recognizer from active view selection, we additionally
evaluated the frozen VPOCLIP recognizer on the same strict `45/5/5` final
unseen bank `{25, 39, 46, 52, 54}`. In this test, one candidate view is
sampled as Robot A's initial observation and is classified directly by
VPOCLIP. No DDQN policy, target selection, robot movement, or multi-view
fusion is applied. The same 30 evaluation seeds and 291 complete final-unseen
episodes per seed are used as in the main table.

| Method | Final unseen accuracy | 95% CI | Movement | Fusion/policy |
|---|---:|:---:|---:|---|
| VPOCLIP-only, random initial view | 63.64% | [63.21, 64.08] | 0 | none |

For reference, the deterministic fixed-rank single-view accuracies are
62.20%, 63.57%, 65.64%, and 63.92% for ranks 0--3, respectively. Averaging
the four single-view correctness indicators per episode gives 63.83%; this is
reported only as a diagnostic because it is not a deployable fixed-view
policy. Under the random-initial-view protocol, Full DDQN improves over the
VPOCLIP-only baseline by 1.69 percentage points, while the two-view Random
baseline improves by 0.92 points.

The no-movement result is a recognizer baseline rather than another active
policy: its purpose is to quantify the accuracy obtained from one locally
acquired observation before any view-selection benefit or multi-view fusion is
introduced. The machine-readable calculation is stored in
`vpoclip_nomove.json`.

## Evaluation protocol

- Dataset protocol: strict `45/5/5`.
- The recognizer and the active policy are trained using only the 45 seen classes.
- Five classes are held out as pseudo-unseen classes for model/gate selection: `{1, 7, 14, 15, 18}`.
- The final true-unseen test classes are `{25, 39, 46, 52, 54}`.
- The pseudo-unseen classes are excluded from policy replay/training. They are used only for the prescribed pseudo-unseen selection procedure.
- Each sample has four available low-view candidates and a two-robot, two-target budget.
- Robot A starts from a randomly sampled valid candidate and performs the initial single-view recognition. Robot B has an independently sampled valid start. The initial recognition is used to obtain the current observation/state; it is not fused as the final two-view result.
- The policy selects two distinct final target views from the four candidates. The current robot position is a legal target: assigning a robot to its current view has zero movement cost.
- Invalid candidates, already selected targets, and infeasible target pairs are masked.
- The two selected targets are assigned to the two robots using
  `min(cost(A -> i) + cost(B -> j), cost(A -> j) + cost(B -> i))`.
- Random, both cyclic baselines, and all learned variants use the same candidate set, independent random starts, feasibility rules, two-target budget, assignment rule, and weighted-logit fusion.

The evaluation uses 30 independent evaluation seeds (`20260920`--`20260949`). Each seed evaluates 291 complete test episodes. The reported 95% intervals are normal-approximation intervals over the 30 seed-level means.

## Experimental procedure

The complete experiment is organized as a fixed-budget perception pipeline. Every sample is processed independently through the same sequence of data preparation, feature extraction, target selection, robot assignment, multi-view recognition, and metric computation. The observation budget is fixed at two final views, one supplied by each robot.

### 1. Dataset organization and episode construction

The experiment uses the four low-view recordings associated with each ETRI-Activity3D sample. Each recording is treated as one candidate observation of the same human action instance. For every candidate, an 8-second temporal window is sampled into 13 frames. The four candidates are grouped by sample identity so that RGB clips, 2-D poses, object tracks, depth maps, 3-D-skeleton orientation, and VPOCLIP logits remain aligned.

The class protocol is fixed before any policy training. The 45 seen classes provide the recognition and policy-training episodes. Five classes are reserved as pseudo-unseen validation classes for selecting the fusion/checkpoint configuration, and five different classes are retained as the final unseen test bank. No final-test label is used to fit the policy or select its checkpoint.

### 2. Per-view visual recognition

For each candidate view, the recognition pipeline processes the modalities in parallel:

1. RGB frames are encoded by the X3D video stream.
2. RTMPose extracts 17 image-plane joints for every sampled frame; the resulting motion sequence is encoded by the CTR-GCN stream.
3. YOLO detections provide object locations and confidence values for the object-context stream.
4. The three streams are spatially and temporally fused to produce one visual representation and one class-logit vector for the candidate view.
5. Text descriptions are encoded once by the frozen CLIP text encoder. The corresponding text prototypes define the action bank used for both seen-class training and unseen-class evaluation.

VPOCLIP is trained before active-view learning and then frozen. Therefore, the active policy learns where to observe using a fixed recognition function; it does not change the recognizer while learning target selection.

### 3. Human, object, and relative-depth feature construction

The active policy does not consume the complete RGB feature tensor. Instead, it receives compact temporal state features that can be computed locally on the robot.

For every frame, the human state is formed from the normalized person center, its frame-to-frame velocity, and the body orientation represented by `sin(theta)` and `cos(theta)`. The orientation is computed from the released 3-D skeleton body yaw and aligned to the corresponding low-view sample. This produces a six-channel human sequence for the Full and Human-only policies. The trajectory-only and orientation-only ablations retain four and two channels, respectively.

For the object state, all detections in a frame are pooled; the policy does not preserve a fixed list of object categories or object slots. The object center, confidence, person-relative image displacement, and person-relative depth are summarized by their mean, minimum, and maximum, together with the normalized number of detected objects. The depth term is obtained from the normalized first-frame MiDaS relative inverse-depth map and is used only as the relative object-depth coordinate. It is not interpreted as metric distance.

Each feature sequence is normalized using the training statistics and retains its temporal order. Human and object sequences are then passed to the selected temporal encoder: Conv1D for the standard policies or a one-layer LSTM with hidden size 32 for the LSTM ablations.

### 4. Initial observation and state formation

At the start of an episode, Robot A and Robot B are assigned valid candidate start positions according to the evaluation seed. Robot A acquires the initial observation and performs a single-view recognition pass. This pass provides the current temporal state and candidate context; the final prediction is produced only after the two target views have been selected and fused.

The policy state contains:

- the 13-frame human trajectory/orientation sequence;
- the 13-frame pooled object sequence;
- the current robot-to-person/candidate geometry;
- the candidate validity and reachability indicators;
- the selected-target mask and the two-robot assignment state.

The state is category-free on the policy side: object category names and fixed detector-slot identities are not used as policy inputs. Candidate identity is represented by its available geometry and feasibility information.

### 5. Two-robot target selection

The four candidate views define the discrete target space. The policy selects two distinct target slots. At each decision, invalid candidates, previously selected targets, and infeasible target pairs are masked before the Q-value maximization. A robot is allowed to keep its current view, which contributes zero movement for that robot.

After a target pair `(i,j)` is selected, the two robots are assigned to the two targets by evaluating both assignments and retaining the smaller total angular cost:

```text
assignment_cost(i,j) = min(
    cost(robot_A -> i) + cost(robot_B -> j),
    cost(robot_A -> j) + cost(robot_B -> i)
)
```

This assignment is used consistently for the learned policies and the Random/Cyclic baselines. It ensures that the reported movement cost reflects the best feasible assignment of the two physical robots rather than an arbitrary robot-to-view ordering.

### 6. Offline DDQN training

The target-selection policy is trained offline from seen-class episodes. For each complete four-view episode, the available transitions are generated for the valid first and second target-selection stages. Each transition contains the current compact state, the masked candidate action set, the selected target, the next state, and the terminal fused-recognition reward.

The reward uses the correct-class margin of the fused seen-class logits as the recognition term and subtracts the movement cost with coefficient 0.25:

```text
terminal reward = fused correct-class margin
                  - 0.25 * assignment movement cost
```

The online and target networks use the same masked action space. Double-DQN target selection reduces the maximization bias, while the dueling decomposition separates the state value from candidate-specific advantages. The recognizer and the weighted-fusion gate remain frozen during this stage. Three policy-training seeds are run, and the validation procedure selects the checkpoint used by each ablation.

The resulting replay pool contains 649,984 transitions, including 162,496 first-stage and 487,488 second-stage transitions. The policy-training episodes contain four valid candidate views and belong to the 45 seen classes.

### 7. Target-view acquisition and final recognition

For each test episode, the selected target pair is converted into the two robot observations. The two candidate-view logit vectors are passed through the same frozen lightweight weighted-logit fusion module used for all methods. The fused logits determine the final action prediction by an argmax over the final unseen class bank.

Random and both cyclic baselines follow exactly the same acquisition and fusion pipeline. They differ only in how the two target slots are selected. Consequently, recognition budget, robot assignment, masking, and fusion are held constant across all rows of the main table.

### 8. Metrics and repeated evaluation

The primary recognition metric is top-1 accuracy on the five final unseen classes. The movement metric is the sum of the two assigned robots' normalized angular movement costs. For readability, the normalized cost is also multiplied by 180 to report a degree-equivalent angle. The report additionally records completion rate, per-seed episode count, accuracy difference from Random and Cyclic, and movement as a percentage of Random.

The complete test procedure is repeated for 30 evaluation seeds. For each seed, the method is evaluated on 291 complete episodes, and the per-seed mean is stored. The final mean is the average of these 30 seed-level means. The reported interval is the normal-approximation 95% CI computed from the 30 seed-level values, applied separately to accuracy, normalized movement cost, and degree-equivalent movement.

## Input features and model variants

All policy inputs are temporal: each view contributes a 13-frame sequence. The policy does not receive action labels, object category IDs, object slot IDs, future-view features, or the recognizer's visual embedding at execution time.

### Human stream

The full human stream has six channels per frame:

1. normalized human center `(center_x, center_y)`;
2. center velocity `(velocity_x, velocity_y)`;
3. `sin(theta)` and `cos(theta)`, where `theta` is the body yaw computed from the released 3-D skeleton.

The orientation-only ablation keeps only the last two channels. The trajectory-only ablation keeps only center and velocity, giving four channels. The 3-D skeleton is used to provide the orientation label/feature; it is not supplied as a full 3-D skeleton sequence to the policy.

### Object stream

The object branch starts from per-object detections containing presence, image-plane center, and confidence. For every frame, all detected objects are pooled rather than assigned to 50 categories or fixed slots. The pooled fields are the mean, minimum, and maximum of:

- object center coordinates;
- detector confidence;
- object position relative to the person in the image plane;
- relative depth to the person.

The depth-enabled branch therefore has 19 pooled channels: the normalized object-count ratio plus three statistics for each of the six scalar fields. The corresponding no-depth object representation has 16 channels. The relative depth is only an object-coordinate feature, not a separate metric-depth sensor: it is formed from normalized first-frame MiDaS relative inverse depth as object depth minus person-slot depth.

### Temporal encoders

The standard DQN variants use temporal 1-D convolution for the human and object sequences. The LSTM variants replace the temporal encoder with a one-layer LSTM with hidden size 32. The variants in the table are:

| Variant | Human input | Object input | Temporal encoder |
|---|---|---|---|
| Full DQN | 6 channels | 19 channels, including relative depth | Conv1D |
| Human-only DQN | 6 channels | none | Conv1D |
| Human trajectory-only DQN | 4 channels | none | Conv1D |
| Human orientation-only DQN | 2 channels | none | Conv1D |
| Object-only DQN | none | 19 channels, including relative depth | Conv1D |
| Full LSTM DQN | 6 channels | 19 channels, including relative depth | 1-layer LSTM, hidden size 32 |
| Human-only LSTM DQN | 6 channels | none | 1-layer LSTM, hidden size 32 |
| Object-only LSTM DQN | none | 19 channels, including relative depth | 1-layer LSTM, hidden size 32 |

## Recognizer, fusion, and policy training

VPOCLIP is trained separately and frozen during active-view learning. It produces one class-logit vector per candidate view. The final prediction uses the same lightweight weighted-logit fusion gate for Random, cyclic, and learned policies; only the selected two views are fused for the reported prediction.

The active policy is an offline masked Double Dueling DQN. Its action space is the four candidate target views, with invalid, already selected, and infeasible pair actions masked. The policy is trained on seen-class episodes and uses the current human/object temporal state plus candidate geometry and robot state.

For this run, the terminal reward is the seen-bank weighted-fusion correct-class margin minus the movement penalty:

```text
reward = fused seen-class margin - 0.25 * assignment movement cost
```

Thus, the current result uses the fused recognition margin as the accuracy-oriented signal. Entropy is evaluated separately only if explicitly added to a later reward configuration; it is not an additional reward term in the present `strict45_5` run.

| Training setting | Value |
|---|---:|
| Policy algorithm | masked Double Dueling DQN |
| Training epochs | 30 |
| Batch size | 32,768 |
| Updates per epoch | 96 |
| Optimizer | AdamW |
| Learning rate | 0.0003 |
| Weight decay | 0.0001 |
| Discount factor | 0.99 |
| Target-network update interval | 250 updates |
| Movement penalty coefficient | 0.25 |
| Policy replay transitions | 649,984 |
| Four-valid-view seen-class training episodes | 2,539 |
| Policy training seeds | 20260909, 20260910, 20260911 |

The checkpoint for each variant was selected from the validation procedure using the prescribed pseudo-unseen setup; the final true-unseen classes were not used for checkpoint selection.

## Data and depth audits

The temporal samples are joined by `metadata.episodes[*].views[*].sample_name`. The depth feature is the normalized first-frame MiDaS relative inverse depth and is not metric depth.

| Split | Episodes | Joined views | Available depth rows |
|---|---:|---:|---:|
| DQN training | 3,657 | 13,847 | 30,698 |
| Validation | 1,446 | 5,527 | 30,698 |
| Final test | 2,974 | 11,324 | 30,698 |

All reported runs completed with the same candidate-view and depth-join protocol. The test completion rate was 100% for every method and every evaluation seed.

## Interpretation

1. The Full DQN is the strongest overall trade-off in this independent-start experiment: it is above both Random and the two cyclic baselines in unseen accuracy while moving substantially less.
2. Object evidence with relative depth is highly useful. Object-only DQN is only 0.11 percentage points below Full DQN and moves 24.9% as much as Random.
3. Human trajectory information is more useful than orientation alone for this policy. The trajectory-only model reaches 65.06%, whereas the orientation-only model reaches 64.46%.
4. Replacing Conv1D with the tested one-layer LSTM does not consistently improve the policy. Full LSTM DQN is close to the cyclic baselines but below Full DQN.
5. The movement result is reported under the same independent-start protocol for every method, so the movement comparison uses a common initial-state distribution.
