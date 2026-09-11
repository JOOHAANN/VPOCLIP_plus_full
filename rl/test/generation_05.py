"""Generation 05: generation-04 low-capacity policies trained on all 50 seen classes."""

from . import generation_04 as runner


runner.GENERATION = "generation_05"
runner.OUT = runner.ROOT / "work_dir/active_view_iterative_test/generation_05"
runner.TRAIN_POOL_MODE = "all50"
runner.VARIANTS = {
    "g05_linear_transition": "all50 seen + 4-view geometry Fourier transition",
    "g05_linear_causal": "all50 seen + 4-view causal evidence interaction",
    "g05_small_transition": "all50 seen + 4-view low-capacity nonlinear transition",
    "g05_linear_pose_transition": "all50 seen + 4-view compact pose transition",
}


if __name__ == "__main__":
    runner.main()
