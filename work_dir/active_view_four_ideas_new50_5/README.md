# Four candidate-selection experiments

All four experiments were specified before inspecting their test results. Models use only the current observation and known candidate geometry at inference. Unvisited views and true labels are used only for offline supervision/evaluation. VPOCLIP is frozen. This is offline single-step action selection, not multi-step DDQN bootstrapping.

| Version | Hypothesis | Objective / inference |
| --- | --- | --- |
| v8 | Any correctly classifying candidate is acceptable | Maximize total probability of the correct candidate set; no decision gradient for all-correct/all-wrong states |
| v9 | Relative ordering matters more than absolute utility | Gap-weighted pairwise ranking, ignoring utility differences <= 0.05 |
| v10 | Independent models reduce unstable individual preferences | Three independently trained centered-utility regressors; average their action scores |
| v11 | The usefulness of semantics varies by observation | Learned gate between semantic and quality/geometry branches, with relative-utility supervision |

Each version trains seeds 20260909/20260910/20260911 for 40 epochs, 48 updates per epoch, batch size 8192. Dynamic task banks are 75% random five-way and 25% all 40 policy-training classes. Half of sampling probability is class-balanced ordinary sampling and half is class-balanced sampling of states with both correct and incorrect candidates. Utility for v9-v11 is correctness + 0.2*tanh(true-class margin/10). Candidate scoring is permutation-equivariant.

The original recognizer's 50 seen classes are split into 40 policy-training classes and 10 held-out policy-validation classes, matching v6/v7. The recognizer has seen those 10 classes; this proxy is not strict recognizer zero-shot validation. Checkpoint selection uses 0.75 * proxy five-way macro gain + 0.25 * seen 50-way validation macro gain, averaged over fixed0/random starts. For v8/v9/v11, the best validation seed is selected. For v10, each member's checkpoint is selected on validation and all three are averaged. Overall version selection uses validation only. No unseen-test labels influence training or selection.

Evaluation: seen 50-way and unseen five-way, separately for fixed view0 and seeded random first views. Four second-view comparisons plus single-view baseline are retained, along with exact uniform-random expected accuracy and the classification-oracle ceiling. The historical `oracle` is true-class fused-score argmax, which is not necessarily the classification-accuracy ceiling. Fusion is 1:1; movement reward is zero but movement metrics are still reported. Two-view recordings have zero decision loss because only one destination exists. A049/A050 merged-group handling and all cached splits are inherited unchanged.

Each version has an independent output directory, resolved config, source copies, checkpoints, seed logs, selection.json, evaluation reports and summary.json. `selected_policy.pt` stores the deployable model state; reconstruct with `make_model(version)`, or an `Ensemble` of three `make_model('v10')` members. Candidate feature extraction is `causal_state` from v6.

`v6_preserved/` snapshots all RL Python/YAML source plus the v6 config and summary, with SHA-256 checks. Original v6 source, cache and checkpoints remain at their original locations. `verify_v6()` checks preserved original files after each experiment.

Run/resume from the project root:

```bash
/home/youhan/.conda/envs/clipgcn/bin/python -u -m rl.run_four_ideas
```

The same command skips completed versions/seeds and resumes incomplete seeds from the last atomic checkpoint, including optimizer and random states. Preserve the original 40-epoch setting when resuming.

Live log:

```bash
tail -f /home/youhan/ws/VPOCLIP_plus_full/logs/rl_four_ideas_new50_5/pipeline.log
```
