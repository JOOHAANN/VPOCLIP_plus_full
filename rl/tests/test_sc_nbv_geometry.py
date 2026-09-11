import unittest

import torch

from rl.sc_nbv import g23_candidate_features, g25_candidate_features


class RelativeCandidateGeometryTests(unittest.TestCase):
    def _fixture(self):
        angles = torch.tensor(
            [[0.1, 1.0, 2.0, -2.0], [0.2, -1.0, 2.0, 0.0]],
            dtype=torch.float32,
        )
        candidate = torch.zeros(2, 4, 8)
        candidate[..., 2] = angles.sin()
        candidate[..., 3] = angles.cos()
        candidate[..., 6] = torch.tensor(
            [[0.1, 0.4, 0.7, 0.2], [0.3, 0.5, 0.2, 0.6]]
        )
        candidate[..., 7] = torch.tensor(
            [[0.0, 1.0, 1.0, 1.0], [0.0, 1.0, 1.0, 0.0]]
        )
        confidence = torch.ones(2, 4)
        return candidate, confidence

    def test_is_candidate_permutation_equivariant(self):
        candidate, confidence = self._fixture()
        source = torch.zeros(2, 2)
        original = g23_candidate_features(candidate, source, confidence)
        permutation = torch.tensor([[2, 0, 3, 1], [1, 3, 0, 2]])
        shuffled_candidate = torch.gather(
            candidate, 1, permutation[..., None].expand(-1, -1, candidate.shape[-1])
        )
        shuffled_confidence = torch.gather(confidence, 1, permutation)
        shuffled = g23_candidate_features(
            shuffled_candidate, source, shuffled_confidence
        )
        inverse = permutation.argsort(dim=-1)
        restored = torch.gather(
            shuffled, 1, inverse[..., None].expand(-1, -1, shuffled.shape[-1])
        )
        torch.testing.assert_close(original, restored, rtol=0.0, atol=1e-6)

    def test_ignores_common_absolute_rotation(self):
        candidate, confidence = self._fixture()
        source = torch.zeros(2, 2)
        relative = torch.atan2(candidate[..., 2], candidate[..., 3])
        phase = torch.tensor([0.7, -1.2])
        rotated = candidate.clone()
        rotated[..., 0] = torch.sin(relative + phase[:, None])
        rotated[..., 1] = torch.cos(relative + phase[:, None])
        rotated[..., 4] = torch.sin(phase)[:, None]
        rotated[..., 5] = torch.cos(phase)[:, None]
        torch.testing.assert_close(
            g23_candidate_features(candidate, source, confidence),
            g23_candidate_features(rotated, source, confidence),
            rtol=0.0,
            atol=1e-6,
        )

    def test_rank_one_hot_is_recording_specific_and_equivariant(self):
        candidate, confidence = self._fixture()
        source = torch.zeros(2, 2)
        original = g25_candidate_features(candidate, source, confidence)
        self.assertEqual(tuple(original.shape), (2, 4, 22))
        # The first three appended fields encode angular ranks; the invalid
        # current row is all zero and does not receive a fake rank.
        torch.testing.assert_close(
            original[0, 0, -4:], torch.zeros(4), rtol=0.0, atol=0.0
        )
        self.assertEqual(int(original[0, 3, -4:].argmax()), 0)
        self.assertEqual(int(original[0, 1, -4:].argmax()), 1)
        self.assertEqual(int(original[0, 2, -4:].argmax()), 2)
        permutation = torch.tensor([[2, 0, 3, 1], [1, 3, 0, 2]])
        shuffled_candidate = torch.gather(
            candidate, 1, permutation[..., None].expand(-1, -1, candidate.shape[-1])
        )
        shuffled_confidence = torch.gather(confidence, 1, permutation)
        shuffled = g25_candidate_features(
            shuffled_candidate, source, shuffled_confidence
        )
        inverse = permutation.argsort(dim=-1)
        restored = torch.gather(
            shuffled, 1, inverse[..., None].expand(-1, -1, shuffled.shape[-1])
        )
        torch.testing.assert_close(original, restored, rtol=0.0, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
