"""Second generation: 5-way task distribution and all-seen data coverage."""

from . import iterative_runner as runner


runner.GENERATION = "generation_02"
runner.TASK_FULL_FRACTION = 0.0
runner.VARIANTS = {
    "g02_5way_pairwise": "policy40 + pure 5-way + v9 pairwise",
    "g02_5way_listwise": "policy40 + pure 5-way + listwise/pairwise",
    "g02_5way_regression": "policy40 + pure 5-way + relative regression",
    "g02_5way_oracle_ce": "policy40 + pure 5-way + acceptable-set CE",
    "g02_all50_listwise": "all 50 seen + pure 5-way + listwise/pairwise",
}


def main():
    runner.TRAIN_POOL_MODE = "policy40"
    runner.main()


if __name__ == "__main__":
    main()
