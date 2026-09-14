# Strict 45/5/5 sequential experiment

Entry point: `rl.dual_robot_depth_ddqn_v1.strict_sequential`.

The 45 original seen classes train the policy. Classes 1, 7, 14, 15, 18 select checkpoints. Classes 25, 39, 46, 52, 54 are the final test bank. Each pseudo class has complete four-view validation recordings.

Robot positions persist through shuffled recordings of the same subject. Initial positions are drawn only at subject boundaries. Physical camera identities C001/C003/C005/C007 map previous assigned targets into the current recording's body-relative ranks. Either robot can remain at its current position. All six distinct target pairs are permitted when reachable; adjacency is not hard-masked.

Training enumerates legal start/action transitions on seen recordings. Pair completion bootstraps into the next recording at the assigned camera identities. A subject's final recording terminates the stream. This is offline replay with counterfactual transitions, not online robot data collection. Validation and test execute each policy's own sequence of decisions and carry its positions forward. Random and cyclic baselines use the same subject streams and initialization rule.

Train all eight variants, three independent training seeds, 30 epochs. Evaluate with 30 stream-order/initial-position seeds. Select checkpoints by pseudo-unseen accuracy subject to movement <= 40% of pseudo-unseen random movement; if none are feasible, select the lowest movement ratio and disclose the failure. Reward remains fused margin minus 0.25 times movement, preserving the preceding comparison's reward coefficients. No coefficient sweep is part of this run.

Movement is summed normalized angular cost for the cheaper feasible robot assignment, not metric navigation distance. Sample ordering is synthetic because the cache supplies no chronological cross-action navigation sequence.

The inherited per-model audit's `stage2_terminal_transitions` and `terminal` fields describe the original two-stage pool. In this entry point, `sequential.pool` changes pair completions to nonterminal successor transitions except at stream boundaries; the sequential protocol here takes precedence over those inherited descriptions.

Results will be written to RESULTS.md and test/summary.json after training and evaluation finish. Existing independent-start results remain in the separate dual_robot_depth_ddqn_strict45_5 directory.
