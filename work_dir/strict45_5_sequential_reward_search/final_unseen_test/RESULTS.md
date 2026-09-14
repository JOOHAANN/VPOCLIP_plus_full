# Pseudo-selected reward: final unseen sequential test

Test bank: [25,39,46,52,54]. 30 stream-order/start seeds; 95% normal CI.
Model and reward coefficients frozen before this evaluation. Movement is a normalized angular proxy, not meters.

|Model|Accuracy %|95% CI|Move|Move/random|Fused CE|Entropy|GT margin|
|---|---:|---:|---:|---:|---:|---:|---:|
|cyclic_adjacent_pair|64.47|[64.04, 64.90]|0.2744|1.158|1.1434|0.8420|0.2269|
|cyclic_pair|65.26|[64.89, 65.62]|0.3463|1.461|1.1405|0.8428|0.2337|
|full_depth19|64.25|[63.89, 64.61]|0.0643|0.272|1.1384|0.8413|0.2361|
|random_pair|64.73|[64.33, 65.14]|0.2369|1.000|1.1436|0.8427|0.2281|
