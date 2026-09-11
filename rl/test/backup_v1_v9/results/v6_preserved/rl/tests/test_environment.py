"""Acceptance tests for action masking, reward direction, and episode length."""

import unittest


class ActiveViewEnvironmentTests(unittest.TestCase):
    @unittest.skip("Scaffold: implement ActiveViewEnv before enabling this test.")
    def test_action_mask(self) -> None:
        """Selected and unreachable views must never be legal destinations."""

    @unittest.skip("Scaffold: implement ActiveViewEnv before enabling this test.")
    def test_episode_length(self) -> None:
        """Each reset must permit exactly B and C before termination."""

    @unittest.skip("Scaffold: implement reward computation before enabling this test.")
    def test_reward_direction(self) -> None:
        """Increasing target probability is positive and confident errors are penalized."""


# Implementation guide
# 1. Use a deterministic three-step cache with one unreachable destination.
# 2. Assert the first step is non-terminal and the second is terminal.
# 3. Verify illegal actions raise before state or reward is mutated.
