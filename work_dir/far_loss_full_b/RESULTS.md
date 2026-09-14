# Far-view loss-aware DDQN results

Subject-wise sequential replay; 30 evaluation seeds; movement is an angular proxy.
The policy is trained on seen classes and tested on the true unseen five-way bank.

|Method|Accuracy|95% CI|Fused CE|Entropy|GT margin|Angular separation|Loss gain|Move cost|RL reward|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|cyclic_adjacent_pair|68.50%|[68.10, 68.90]|0.9751|0.6789|0.3632|31.48|-0.2819|0.2323|-0.3654|
|cyclic_pair|68.67%|[68.35, 68.99]|0.9634|0.6817|0.3763|40.20|-0.2702|0.3038|-0.3208|
|full_depth19|68.30%|[68.08, 68.53]|0.9749|0.6777|0.3596|58.41|-0.2816|0.1039|-0.2614|
|human_only|68.04%|[67.88, 68.21]|0.9780|0.6775|0.3511|59.61|-0.2848|0.1098|-0.2616|
|object_only_depth19|68.35%|[68.05, 68.65]|0.9801|0.6778|0.3479|57.76|-0.2868|0.1004|-0.2713|
|random_nonadjacent|69.08%|[68.60, 69.56]|0.9692|0.6795|0.3707|48.69|-0.2759|0.1831|-0.2911|
|random_pair|69.08%|[68.59, 69.56]|0.9702|0.6806|0.3681|34.66|-0.2770|0.1988|-0.3488|

## Interpretation

* `Fused CE` and `Entropy` are lower-is-better; both are computed after weighted fusion of the selected two views.
* The Q-learning terminal reward uses fused CE and fused entropy directly; single-view CE/entropy differences are diagnostics only.
* `GT margin` and `Loss gain` are higher-is-better. `GT margin` uses the fused true class against the strongest fused rival.
* `Angular separation` is the actual body-relative target-view angle in degrees; it is separate from the start-to-target movement cost.
* `RL reward` is the configured training objective. Bellman loss is stored in `config_resolved.json` under `selected_training_loss`; it measures Q-target fitting, not recognition accuracy.
* Every scalar in the table is aggregated over 30 evaluation seeds with a 95% normal CI.
