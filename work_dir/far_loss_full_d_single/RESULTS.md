# Far-view loss-aware DDQN results

Subject-wise sequential replay; 30 evaluation seeds; movement is an angular proxy.
The policy is trained on seen classes and tested on the true unseen five-way bank.

|Method|Accuracy|95% CI|Fused CE|Entropy|GT margin|Angular separation|Loss gain|Move cost|RL reward|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|cyclic_adjacent_pair|68.50%|[68.10, 68.90]|0.9751|0.6789|0.3632|31.48|-0.2819|0.2323|0.3597|
|cyclic_pair|68.67%|[68.35, 68.99]|0.9634|0.6817|0.3763|40.20|-0.2702|0.3038|0.3764|
|full_depth19|69.16%|[68.84, 69.48]|0.9624|0.6799|0.3754|53.43|-0.2691|0.0826|0.3807|
|random_nonadjacent|69.08%|[68.60, 69.56]|0.9692|0.6795|0.3707|48.69|-0.2759|0.1831|0.3741|
|random_pair|69.08%|[68.59, 69.56]|0.9702|0.6806|0.3681|34.66|-0.2770|0.1988|0.3659|

## Interpretation

* `Fused CE` and `Entropy` are lower-is-better; both are computed after weighted fusion of the selected two views.
* The Q-learning terminal reward uses fused CE and fused entropy directly; single-view CE/entropy differences are diagnostics only.
* `GT margin` and `Loss gain` are higher-is-better. `GT margin` uses the fused true class against the strongest fused rival.
* `Angular separation` is the actual body-relative target-view angle in degrees; it is separate from the start-to-target movement cost.
* `RL reward` is the configured training objective. Bellman loss is stored in `config_resolved.json` under `selected_training_loss`; it measures Q-target fitting, not recognition accuracy.
* Every scalar in the table is aggregated over 30 evaluation seeds with a 95% normal CI.
