# Isolated iterative RL search

This directory is separate from the existing v1-v9 experiments. The original
source/config snapshot is in `backup_v1_v9/`; new generations write only under
`work_dir/active_view_iterative_test/` and `logs/rl_iterative_test/`.

The success rule for a candidate generation is intentionally strict: its
unseen 5-way Top-1 must exceed the random baseline for all three seeds under
both fixed-view0 and random-start protocols. Test results are never used to
select checkpoints; validation remains the selection split.
