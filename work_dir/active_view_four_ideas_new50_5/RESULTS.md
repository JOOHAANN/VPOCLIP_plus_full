# Completed comparison

All entries are top-1 percentages. Seen uses 50 candidate classes; unseen uses only five candidate classes. Random is the exact expected accuracy over legal second views, conditional on the same initial view.

| Policy | Seen fixed0 | Seen random start | Unseen fixed0 | Unseen random start |
| --- | ---: | ---: | ---: | ---: |
| Random exact | 94.81 | 94.91 | 52.90 | 52.01 |
| v6 baseline | 94.73 | 95.00 | 51.45 | 50.72 |
| v7 baseline | 95.20 | 95.12 | 52.42 | 51.21 |
| v8 acceptable candidate set | 94.92 | 95.08 | 53.14 | 51.45 |
| v9 pairwise ranking | 94.88 | 95.04 | 53.14 | 52.17 |
| v10 three-member ensemble | 94.85 | 95.12 | 52.66 | 51.21 |
| v11 gated semantic/geometry | 95.04 | 95.00 | 51.93 | 51.69 |

The predeclared validation criterion selects v11 among the four new versions. v9 has the strongest observed unseen result among them, but choosing v9 because of that test result would be post-hoc test selection. Its unseen gains over exact random are only 0.24 and 0.16 percentage points; these results do not establish reliable improvement. No confidence intervals or additional independent test sets were used to establish significance.

Each version completed three 40-epoch runs and both initial-view evaluation protocols. v10 averages the three members, while the other versions select their seed by validation. v6 source preservation checks passed after every version and at completion. Models and test results remain independent of original v6 artifacts.

Full metrics, configurations, selection records and model weights are in the v8/v9/v10/v11 subdirectories. See comparison.json for all metrics and README.md for methods and resume instructions.
