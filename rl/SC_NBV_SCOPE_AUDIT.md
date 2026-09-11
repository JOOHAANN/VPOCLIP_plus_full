# SC-NBV evidence audit

## Checkpoint scope

- `sc_nbv_current_v27_transition_policy.yaml` uses the original recognizer at `work_dir/allviews_lowview50_5_final_aug_entropy_single_h2/allviews_lowview50_5_20260907/last_model.pth` and unseen class IDs `[0,2,26,34,50]`.
- `sc_nbv_new50_5_group1_label_free_v36.yaml` and `sc_nbv_new50_5_group1_anchored_v38.yaml` use `work_dir/new50_5_group1/recipe_stage_b/best_model.pth` and unseen IDs `[14,15,25,39,46]`.
- These are different recognizers and class splits. Their absolute accuracies cannot establish the effect of correcting geometry. The newer experiments do not prove success on the originally requested old-recognizer task. The requested conditional migration to the new `last_model.pth` is not established by these results.

## Camera identity versus geometry

- Preserved `test/backup_v1_v9/code/build_offline_cache.py` constructs geometry from each view's `relative_bearing_deg` and movement costs from angle differences.
- Preserved `test/backup_v1_v9/code/state_builder.py` reads cached geometry when present. Its missing-geometry fallback assigns equally spaced angles to row IDs.
- Preserved `test/backup_v1_v9/code/build_cache.py::_angle` reads sample-specific CSV angles first, with a hardcoded camera-order fallback for missing entries.
- Thus unsafe fallback assumptions exist in historical code, but their existence alone does not prove that a historical run used them. Audit actual cache metadata before claiming an observed accuracy loss was caused by this bug.
- Current `sc_nbv.py::g25_candidate_features` computes angular ranks per recording from relative bearings. Camera rows remain lookup keys. This removes a fixed-ID directional assumption from that encoding, but does not establish the physical accuracy of input bearings.
- Current CSV optical-axis yaw fallback is an estimate, not independently verified anatomical body-relative camera position.

### Actual original-recognizer cache audit

Inspected every view in `data/rl_candidate_rank_v6_new50_5/cache/*/metadata.json` and compared `view_geometry.npy[..., :2]` against sine/cosine of the corresponding `relative_bearing_deg`.

| Split | Episodes | Views | Skeleton camera center | Optical-axis fallback | Unavailable |
|---|---:|---:|---:|---:|---:|
| dqn_train | 3154 | 11861 | 11857 | 4 | 0 |
| seen_test | 2561 | 9689 | 9682 | 5 | 2 |
| test | 414 | 1656 | 1656 | 0 | 0 |
| val | 1251 | 4739 | 4729 | 2 | 8 |

All views report `angle_source=csv`; none report `camera_order_fallback`. All contain relative bearings. Maximum tensor-versus-metadata error is below 3e-8 in all four splits. Thus this retained old cache supplies per-view geometry, not a fixed direction assigned by row ID. The optional optical-axis fallback is distinct from the hardcoded camera-order fallback. This evidence refutes a blanket claim that this old cache trained on fixed-ID directions. It does not prove physical calibration accuracy or the provenance of every earlier experiment/cache.

## Evaluation requirements still open

- The guide's item 24 requires architecture and hyperparameter selection on pseudo-unseen only. Repeated true-unseen inspection makes current results exploratory; multiple positive seeds on this repeatedly inspected split are insufficient evidence of untouched-test generalization.
- A controlled original-recognizer comparison must keep class split, fusion, initial view, data and training budget fixed while varying geometry representation.
- Verify actual training-cache classes are disjoint from pseudo-unseen classes; configuration lists alone are insufficient (the v27 lists overlap).
- Verified `data/sc_nbv_current_v24_current_logits/cache/*/labels.npy`: train has 11,861 contexts and includes all ten pseudo-unseen classes `[5,6,14,22,25,38,42,44,45,51]`; pseudo-unseen has 940 contexts. Real unseen has 1,656 contexts with IDs `[0,2,26,34,50]`, disjoint from training. Consequently v27 validation is not class-held-out validation. The cache builder now rejects overlapping class splits before constructing output.
- Preserve prior runs and finish the already live v38 process. Do not claim goal completion from the newer recognizer or a selected favorable variant.

## v39 validation sensitivity

Using the strict v25 cache reused by v39, fixed view0 selects 245 pseudo-unseen contexts. Recomputed classification directly from raw val logits, the ten pseudo-unseen class columns, legal-action masks and the evaluator's 50:50 logit fusion. Only **2 of 245 states** change classification correctness depending on the second view. Exact random accuracy is 97.95918%, the accuracy-maximizing candidate upper bound is 98.36735%, and the worst legal candidate is 97.55102%. Consequently Top-1 selection has very little resolution and many ties; this is a measured limitation of this proxy validation set, not proof the selector has converged. Continuous held-out margin/regret should be examined before a further capacity experiment. Real-unseen results must not choose the tie-break rule.

The three candidate geometry tests passed: candidate permutation equivariance, common absolute rotation invariance for g23, and recording-specific angular-rank equivariance for g25. These verify feature construction, not real-world calibration accuracy or every input of the complete network.

## v39/v40 complete selector smoke checks

On 16 actual pseudo-unseen cache states, instantiated both configured selectors in eval mode on CPU. Small model: 43,458 parameters; large model: 130,626. Replacing labels, utility and oracle_action with constant 17 changed neither model's scores (maximum difference exactly zero). Permuting candidate rows and corresponding metadata, then restoring score order, produced maximum errors 1.49e-7 and 5.96e-8 respectively. These are random-initialization architecture checks, not trained-checkpoint performance measurements. `future_logits` was absent from this batch, so this check does not claim to audit its full cache provenance. Model capacity differs by approximately 3x.

v40 repeats v39 with the same seeds and training configuration but uses pseudo-unseen margin gain for selection, motivated by the measured Top-1 saturation above. Its selection rule was set before inspecting v39 true-unseen results.

## v41 diagnostic outcome

Both utility variants completed all three seeds. Validation selected centered_margin (mean margin gain over random 0.094865; minimum 0.045648). Its unseen accuracies were 52.899%, 52.415%, 51.932%, versus 52.899% exact random. centered_entropy_js gave 52.415%, 53.140%, 52.657%; it was not validation-selected.

Reloaded all six best checkpoints and counted selected actions on the 245 fixed0 pseudo-unseen states. Counts for slots `[0,1,2,3]`:

- centered_margin: `[0,116,71,58]`, `[0,95,65,85]`, `[0,92,66,87]`.
- centered_entropy_js: `[0,102,77,66]`, `[0,104,69,72]`, `[0,86,97,62]`.

There is no complete collapse to one action on these validation states. Better pseudo-unseen margin has not translated into better true-unseen classification. This does not prove impossibility of prediction from the current state. Actual fixed0 training uses 2,531 contexts (285 with one legal destination), from a cache of 9,485 contexts across starts; previous statements of 9,485 training states describe cache size, not the effective fixed0 sample pool.
