"""Generation 07: current posterior over the candidate class set."""

from . import generation_06 as runner


runner.GENERATION = "generation_07"
runner.OUT = runner.ROOT / "work_dir/active_view_iterative_test/generation_07"
runner.POOL_MODE = "all50"
runner.VARIANTS = {
    "g07_banksemantic": "candidate-bank semantic posterior + geometry",
    "g07_banksemantic_pose": "candidate-bank semantic posterior + pose + geometry",
    "g07_banksemantic_z": "candidate-bank semantic posterior + z + pose + geometry",
    "g07_banksemantic_evidence": "candidate-bank semantic posterior + evidence + geometry",
}


if __name__ == "__main__":
    runner.main()
