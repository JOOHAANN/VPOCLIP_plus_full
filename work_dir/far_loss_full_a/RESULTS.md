# Far-view loss-aware DDQN results

Subject-wise sequential replay; 30 evaluation seeds; movement is an angular proxy.
The policy is trained on seen classes and tested on the true unseen five-way bank.

|Method|Accuracy|95% CI|Fused CE|Entropy|GT margin|Angular separation|Loss gain|Move cost|RL reward|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|cyclic_adjacent_pair|68.50%|[68.10, 68.90]|0.9751|0.6789|0.3632|31.48|-0.2819|0.2323|-0.3702|
|cyclic_pair|68.67%|[68.35, 68.99]|0.9634|0.6817|0.3763|40.20|-0.2702|0.3038|-0.3103|
|full_depth19|68.86%|[68.62, 69.10]|0.9765|0.6786|0.3540|55.07|-0.2832|0.0910|-0.2834|
|human_only|67.97%|[67.75, 68.19]|0.9752|0.6796|0.3564|56.66|-0.2820|0.0914|-0.2751|
|object_only_depth19|68.29%|[68.04, 68.54]|0.9686|0.6776|0.3689|55.63|-0.2754|0.1057|-0.2581|
|random_nonadjacent|69.08%|[68.60, 69.56]|0.9692|0.6795|0.3707|48.69|-0.2759|0.1831|-0.2875|
|random_pair|69.08%|[68.59, 69.56]|0.9702|0.6806|0.3681|34.66|-0.2770|0.1988|-0.3477|

## Interpretation

* `Fused CE` and `Entropy` are lower-is-better; both are computed after weighted fusion of the selected two views.
* `GT margin` and `Loss gain` are higher-is-better. `GT margin` uses the fused true class against the strongest fused rival.
* `Angular separation` is the actual body-relative target-view angle in degrees; it is separate from the start-to-target movement cost.
* `RL reward` is the configured training objective. Bellman loss is stored in `config_resolved.json` under `selected_training_loss`; it measures Q-target fitting, not recognition accuracy.
* Every scalar in the table is aggregated over 30 evaluation seeds with a 95% normal CI.
