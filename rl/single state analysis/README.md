# Single-state second-view analysis

This directory is an isolated experiment line.  It does not modify the
preserved v1--v24 implementations or their checkpoints.

## Changed target

The old trajectory/SC-NBV experiments ranked a candidate by the recognition
margin after fusing it with the currently observed view.  This experiment
changes only the offline action target:

```text
target(a) = 1[standalone VPOCLIP(view_a) predicts the ground-truth class]
```

The first view is still part of the causal state because it is the view the
robot currently has, but the target for the next action does not use the
current-view logits or a current+candidate fusion.  Thus the selector is
trained to predict a good standalone second view.

## Evaluation

For every fixed start (`view0` ... `view3`) and the random-start protocol,
the test reports exactly these quantities:

* `policy_second_single`: selected second view by itself;
* `policy_fused`: first view + selected second view;
* `random_second_single`: random legal second view by itself;
* `random_fused`: first view + random second view.

The five true unseen classes are used only for the final test.  Model
selection uses held-out pseudo-unseen classes from the corresponding split.
All random-start results use the same 30 evaluation seeds as the previous
reports (`20260920`--`20260949`).

## Scripts

* `train_trajectory_single_state.py`: trains v1/v4/v5/v6/object-only/
  geometry-only with the standalone-view target.
* `evaluate_trajectory_single_state.py`: evaluates the four requested
  single/fused policy and random metrics on the true unseen test split.

The SC-NBV v9--v24 runner is kept separate so its original cache and model
contracts remain untouched.
