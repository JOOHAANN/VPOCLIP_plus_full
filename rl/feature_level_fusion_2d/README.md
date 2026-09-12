# 512-D feature-level fusion experiment

This directory is isolated from the preserved 2-D anchor at
`work_dir/trajectory_g1_pseudo_selected_stage_b_2d`.

The anchor fuses per-view class logits.  This experiment trains
`FeatureFusion512`, recursively fuses cached normalized visual features, and
only then computes cosine similarity with normalized text prototypes.  The
trajectory policy inputs, class split, recognizer cache, three training seeds,
fixed starts, random starts, and 30 evaluation seeds are unchanged.

The pipeline writes all outputs under
`work_dir/trajectory_g1_pseudo_selected_stage_b_feature_level_fusion_2d` and
logs under the matching `logs/` directory.  It does not overwrite any v1--v9
or anchor output.

