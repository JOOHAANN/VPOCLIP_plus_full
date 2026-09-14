# Far-view loss-aware DDQN results

Subject-wise sequential replay; 30 evaluation seeds; movement is an angular proxy.
The policy is trained on seen classes and tested on the true unseen five-way bank.

|Method|Accuracy|95% CI|Fused CE|Entropy|GT margin|Angular separation|Loss gain|Move cost|RL reward|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|cyclic_adjacent_pair|68.19%|[67.39, 69.00]|0.9836|0.6805|0.3552|31.64|-0.2904|0.2319|-0.3934|
|cyclic_pair|68.71%|[67.78, 69.63]|0.9622|0.6813|0.3782|40.22|-0.2690|0.3061|-0.3062|
|full_depth19|68.04%|[67.68, 68.41]|0.9708|0.6803|0.3665|57.47|-0.2776|0.0837|-0.2596|
|human_only|66.94%|[65.98, 67.90]|0.9805|0.6784|0.3525|51.48|-0.2873|0.0903|-0.3052|
|object_only_depth19|67.68%|[67.18, 68.17]|0.9719|0.6782|0.3621|55.41|-0.2786|0.0926|-0.2681|
|random_nonadjacent|69.23%|[67.80, 70.65]|0.9711|0.6794|0.3688|48.78|-0.2779|0.1814|-0.2919|
|random_pair|68.93%|[66.58, 71.28]|0.9623|0.6804|0.3785|34.98|-0.2691|0.1993|-0.3278|

## Interpretation

* `Fused CE` and `Entropy` are lower-is-better; both are computed after weighted fusion of the selected two views.
* `GT margin` and `Loss gain` are higher-is-better. `GT margin` uses the fused true class against the strongest fused rival.
* `Angular separation` is the actual body-relative target-view angle in degrees; it is separate from the start-to-target movement cost.
* `RL reward` is the configured training objective. Bellman loss is stored in `config_resolved.json` under `selected_training_loss`; it measures Q-target fitting, not recognition accuracy.
* Every scalar in the table is aggregated over 30 evaluation seeds with a 95% normal CI.
