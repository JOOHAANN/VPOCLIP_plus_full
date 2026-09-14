# Strict 45/5/5 sequential results

Final unseen classes: [25, 39, 46, 52, 54]. Pseudo-unseen selection classes: [1, 7, 14, 15, 18].
30 evaluation seeds; CI describes variation in stream order and initial positions, not independent training runs.
Movement is the summed two-robot normalized angular proxy, not meters. Current positions remain selectable.

|Model|Accuracy (%)|95% CI (%)|Move cost|Move 95% CI|Ratio to random|
|---|---:|---:|---:|---:|---:|
|cyclic_adjacent_pair|64.47|[64.04, 64.90]|0.2744|[0.2712, 0.2776]|1.158|
|cyclic_pair|65.26|[64.89, 65.62]|0.3463|[0.3449, 0.3476]|1.461|
|full_depth19|64.63|[64.16, 65.10]|0.0822|[0.0796, 0.0848]|0.347|
|full_lstm19|63.94|[63.69, 64.19]|0.0975|[0.0947, 0.1004]|0.412|
|human_only|64.98|[64.62, 65.34]|0.0768|[0.0746, 0.0790]|0.324|
|human_only_lstm|63.39|[63.06, 63.72]|0.0701|[0.0671, 0.0731]|0.296|
|human_orientation_only|64.65|[64.38, 64.92]|0.0944|[0.0915, 0.0974]|0.399|
|human_trajectory_only|64.42|[64.04, 64.80]|0.0653|[0.0635, 0.0671]|0.276|
|object_only_depth19|64.12|[63.78, 64.47]|0.0588|[0.0561, 0.0615]|0.248|
|object_only_lstm19|64.39|[64.11, 64.66]|0.1159|[0.1125, 0.1192]|0.489|
|random_pair|64.73|[64.33, 65.14]|0.2369|[0.2331, 0.2408]|1.000|

Only subject boundaries reset positions. Camera identity, rather than body-relative rank, is carried between samples.
Random and both cyclic baselines use the same persistent-start protocol: only the first sample in each subject stream receives a random two-robot start; every later sample starts from the physical cameras assigned by the preceding selected pair. They use the same feasibility mask and robot assignment rule as the learned policy.
Training uses 45 seen classes. Checkpoint selection uses the five pseudo-unseen classes only.
Reward coefficients match the preceding independent-start comparison (margin minus 0.25 times movement).
No reward coefficient sweep is performed in this comparison.
