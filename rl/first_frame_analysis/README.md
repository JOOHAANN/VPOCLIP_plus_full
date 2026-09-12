# First-sampled-frame trajectory experiment

This directory contains only the adapter for the first-frame ablation. The
historical `multistep_angle_object_trajectory_policy_v1.py`, v3/v4/v5/v6, and
the object/geometry ablation code are not edited.

For every variant, the original cache is loaded and truncated in memory:

- pose: `[13, 17, 2] -> [1, 17, 2]`
- object track: `[13, 50, C] -> [1, 50, C]`

The original delta-margin target, listwise/pairwise loss, class split, and
two-view fusion evaluation remain unchanged. Outputs go under:
`work_dir/trajectory_first_frame_new50_5/<variant>/`.
