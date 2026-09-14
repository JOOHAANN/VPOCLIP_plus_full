# Far-view loss-aware DDQN results

Subject-wise sequential replay; 10 evaluation seeds; movement is an angular proxy.
The policy is trained on seen classes and tested on the true unseen five-way bank.

|Method|Accuracy|95% CI|Fused CE|Entropy|GT margin|Angular separation|Loss gain|Move cost|RL reward|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|cyclic_adjacent_pair|68.38%|[67.74, 69.01]|0.9806|0.6792|0.3557|31.62|-0.2874|0.2327|0.3208|
|cyclic_pair|68.38%|[67.75, 69.01]|0.9621|0.6817|0.3790|40.49|-0.2689|0.3025|0.3657|
|full_depth19|68.30%|[67.79, 68.81]|0.9693|0.6796|0.3730|49.51|-0.2761|0.0458|0.3665|
|random_nonadjacent|69.23%|[68.10, 70.35]|0.9675|0.6787|0.3753|48.69|-0.2743|0.1786|0.3787|
|random_pair|69.23%|[68.01, 70.44]|0.9645|0.6803|0.3764|34.66|-0.2713|0.1972|0.3494|

## Interpretation

* `Fused CE` and `Entropy` are lower-is-better; both are computed after weighted fusion of the selected two views.
* The Q-learning terminal reward uses fused CE and fused entropy directly; single-view CE/entropy differences are diagnostics only.
* `GT margin` and `Loss gain` are higher-is-better. `GT margin` uses the fused true class against the strongest fused rival.
* `Angular separation` is the actual body-relative target-view angle in degrees; it is separate from the start-to-target movement cost.
* `RL reward` is the configured training objective. Bellman loss is stored in `config_resolved.json` under `selected_training_loss`; it measures Q-target fitting, not recognition accuracy.
* Every scalar in the table is aggregated over 10 evaluation seeds with a 95% normal CI.
