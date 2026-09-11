"""Acceptance tests for masked Double-DQN targets."""

import unittest


class DoubleDQNTests(unittest.TestCase):
    @unittest.skip("Scaffold: implement DoubleDQNAgent before enabling this test.")
    def test_next_action_uses_online_and_value_uses_target(self) -> None:
        """The online network selects a_star and the target network evaluates it."""

    @unittest.skip("Scaffold: implement DoubleDQNAgent before enabling this test.")
    def test_terminal_target(self) -> None:
        """A done transition must have Bellman target equal to reward."""


# Implementation guide
# 1. Use tiny deterministic networks whose online and target argmax differ.
# 2. Include a high-valued invalid action to prove next-state masking is applied.
# 3. Assert terminal targets are independent of every next-state Q-value.
