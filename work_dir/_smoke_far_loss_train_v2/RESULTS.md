# Far-view loss-aware DDQN results

Subject-wise sequential replay; 30 evaluation seeds; movement is an angular proxy.
The policy is trained on seen classes and tested on the true unseen five-way bank.

|Method|Accuracy|95% CI|Fused CE|Entropy|GT margin|Angular separation|Loss gain|Move cost|RL reward|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|cyclic_adjacent_pair|67.53%|[67.53, 67.53]|0.9817|0.6816|0.3520|31.24|-0.2885|0.2364|-0.0575|
|cyclic_pair|67.53%|[67.53, 67.53]|0.9716|0.6820|0.3646|39.48|-0.2784|0.3105|-0.0034|
|full_depth19|68.63%|[68.63, 68.63]|0.9817|0.6754|0.3512|54.51|-0.2885|0.1565|0.0412|
|random_nonadjacent|68.63%|[68.63, 68.63]|0.9790|0.6772|0.3545|48.35|-0.2858|0.1875|0.0205|
|random_pair|69.37%|[69.37, 69.37]|0.9667|0.6753|0.3819|36.05|-0.2735|0.1865|-0.0054|

## Interpretation

* `Fused CE` and `Entropy` are lower-is-better; both are computed after weighted fusion of the selected two views.
* `GT margin` and `Loss gain` are higher-is-better. `GT margin` uses the fused true class against the strongest fused rival.
* `Angular separation` is the actual body-relative target-view angle in degrees; it is separate from the start-to-target movement cost.
* `RL reward` is the configured training objective. Bellman loss is stored in `config_resolved.json` under `selected_training_loss`; it measures Q-target fitting, not recognition accuracy.
* Every scalar in the table is aggregated over 30 evaluation seeds with a 95% normal CI.
