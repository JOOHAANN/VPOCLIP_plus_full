# Published selector adaptations versus preserved v5

## Scope and provenance

These are **ETRI/frozen-VPOCLIP adaptations**, not claims to reproduce the
original paper benchmark numbers or to establish current SOTA.

* MVSelect: official repository https://github.com/hou-yz/MVSelect,
  commit `beece4b1c0b6079d2a8ddd98fc5365bce896823b`.
  `CamSelect` parameters/branches are imported directly. Observed-feature max
  aggregation, camera embeddings, TD smooth-L1, target network and CE improvement
  reward are retained. The frozen-recognizer setting is supported by upstream.
* Wang et al., CVPR 2020, *Active Vision for Early Recognition of Human Actions*:
  https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_Active_Vision_for_Early_Recognition_of_Human_Actions_CVPR_2020_paper.html
  No official implementation was located on the paper/author pages. This is a
  from-paper implementation of independent missing-frame LSTMs with learned
  decay-to-null, elapsed-time belief state, and A2C camera selection. The downloaded
  paper is in `upstream/`. It must not be presented as running official code.

## Adaptations that matter

Both methods output discrete unvisited-camera indices. Only acquired views'
features enter selection. Illegal/unvisited masks are enforced in exploration,
exploitation and bootstrap targets. Slot indices are identities, **not angles**;
neither native camera embedding nor per-camera LSTM encodes a physical direction.
Their camera-identity assumptions may generalize poorly across ETRI recordings;
this limitation must be reported, not interpreted as universal inferiority.

MVSelect uses the frozen 512-D per-view descriptor, not original spatial maps.
Its native downstream classifier/max fusion is replaced by fixed VPOCLIP logits.
The mfLSTM version uses 136 pairwise distances from 17 2-D joints and hidden size
100. The original NTU experiment had 3-D skeletons; these are unavailable here.
Each acquisition reveals one 13-frame cached clip; elapsed ticks are internal
acquisition ticks, **not synchronized wall-clock robot motion**. The original
early-recognition/revisiting protocol is replaced by 1 or 2 non-repeating moves.
The auxiliary seen-class heads initialize the recurrent encoder using random
frame dropout (10 epochs, endpoint CE); they never classify the final result.
Native elapsed-time-weighted classification is intentionally replaced by the
same frozen weighted VPOCLIP fusion as v5. Recurrent state is subsequently trained
with A2C, gamma 0.9, log-probability reward, critic baseline and entropy 0.01.
This is a core-mechanism transfer baseline, not a full early-recognition replica.

Training uses 40 policy epochs and 64 updates per epoch, Adam 1e-4. Batches are
2048 for MVSelect and 512 for mfLSTM. MVSelect gamma is 0.99, target EMA 0.01,
epsilon decays to 0.05. These are documented adaptation budgets, not identical
optimizer/update budgets to v5. No VPOCLIP parameters are trained.

## Fair evaluation and limitations

* Same frozen `g1_pseudo_selected_stage_b/last_model.pth` cache as v5.
* Same 2,900 four-view seen training episodes; no real unseen labels in training.
* Audit finding: the anchor's named pseudo-unseen validation classes overlap its
  50 training classes. We preserve that split for comparability and **do not call
  it class-disjoint validation**. Model selection uses only val fixed0 three-view
  equal-fusion gain, never real unseen performance.
* Train seeds 20260909/10/11. Every trained best checkpoint is evaluated and
  reported, alongside the validation-selected seed; no test-based seed selection.
* Exact saved v5 IDs, starts, random paths and predictions are reused. Fixed
  view0/1/2/3 plus random starts, 30 seeds 20260920–20260949, one/two moves.
* Seen 50-way and real unseen **5-way** are kept separate. All five preserved
  fusion methods are applied identically; fusion is not tuned on test labels.
* 95% paired episode-bootstrap CIs average evaluation seeds first (5,000 draws).
  They do not establish independence of repeated subjects or account for all
  model-selection/multiple-comparison uncertainty. 30 evaluation seeds are not
  30 independent policy training runs. v5 remains the saved selected anchor.
* This is a system comparison with method-native state representations, not a
  controlled same-input optimizer ablation: v5 uses person/object trajectories,
  MVSelect uses frozen appearance descriptors, mfLSTM uses skeleton distances.

Everything is isolated here. Anchors, recognizer, data and existing policies are
read-only. `smoke.py` checks no future-feature leakage and non-repeating actions.

## Run

```bash
cd /home/youhan/ws/VPOCLIP_plus_full
/home/youhan/.conda/envs/clipgcn/bin/python -u work_dir/published_selectors_v1/run.py --model mvselect --batch 2048
/home/youhan/.conda/envs/clipgcn/bin/python -u work_dir/published_selectors_v1/run.py --model mflstm_a2c --batch 512
```

Completed seeds are skipped; incomplete policy training resumes from `last.pt`.
The smoke experiment is separate and excluded from reported comparisons.

## Old 50/5 split

The old split run is isolated at `mflstm_a2c_old_split/`. It uses 2,734 complete
four-view seen training episodes, 414 real unseen 5-way test episodes, the old
pseudo-unseen list, and one move to match the old weighted-fusion protocol. The
v5 reference was regenerated in `v5_old_weighted_anchor/` from the preserved old
v5 checkpoint; the existing v7 anchor and historical reports were not overwritten.
The compact result table is `OldSplit_MF_LSTM_Report.md`.
