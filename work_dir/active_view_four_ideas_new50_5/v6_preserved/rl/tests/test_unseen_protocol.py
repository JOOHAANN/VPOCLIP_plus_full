"""Acceptance test preventing target leakage during real unseen evaluation."""

import unittest


class UnseenProtocolTests(unittest.TestCase):
    @unittest.skip("Scaffold: implement Evaluator before enabling this test.")
    def test_unseen_protocol(self) -> None:
        """Unseen policy execution must not read labels, reward, or call optimizer.step."""


# Implementation guide
# 1. Replace labels with an object that raises on access during policy execution.
# 2. Spy on reward computation and optimizer.step and assert neither is called.
# 3. Reveal labels only after predictions and selected-view traces are finalized.
