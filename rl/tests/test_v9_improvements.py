import unittest
import torch

from rl.causal_rank_v6 import causal_state
from rl.tests.test_causal_v6 import fixture
from rl.v9_improvement_models import decision_loss, make_model
from rl.run_v9_improvements import dropout_state


class V9ImprovementTests(unittest.TestCase):
    def test_permutation_and_causal_inputs(self):
        c = fixture(); e = torch.arange(2); starts = torch.zeros(2, dtype=torch.long)
        bank = torch.zeros(2, 55, dtype=torch.bool); bank[:, :5] = True
        before = causal_state(c, e, starts, bank)
        for name in ('logits', 'z', 'pose', 'object_map', 'quality'):
            getattr(c, name)[:, 1:] = torch.randn_like(getattr(c, name)[:, 1:]) * 100
        c.labels[:] = 54
        after = causal_state(c, e, starts, bank)
        perm = torch.tensor([2, 0, 3, 1])
        shuffled = dict(before, candidate=before['candidate'][:, perm], mask=before['mask'][:, perm])
        for version in ('v12', 'v13', 'v14', 'v15'):
            net = make_model(version).eval()
            torch.testing.assert_close(net(before), net(after), rtol=0, atol=0)
            torch.testing.assert_close(net(shuffled), net(before)[:, perm])

    def test_losses_rank_better_candidate_and_mask_invalid(self):
        valid = torch.tensor([[False, True, True, False]])
        utility = torch.tensor([[0., 1.2, -.1, 0.]])
        for version in ('v12', 'v13', 'v14', 'v15'):
            q = torch.zeros(1, 4, requires_grad=True)
            decision_loss(version, q, utility, valid).backward()
            self.assertLess(q.grad[0, 1], 0)
            self.assertGreater(q.grad[0, 2], 0)
            self.assertEqual(q.grad[0, 0], 0)
            self.assertEqual(q.grad[0, 3], 0)

    def test_candidate_dropout_keeps_one_action(self):
        c = fixture(); e = torch.arange(2); starts = torch.zeros(2, dtype=torch.long)
        bank = torch.zeros(2, 55, dtype=torch.bool); bank[:, :5] = True
        s = causal_state(c, e, starts, bank)
        gen = torch.Generator().manual_seed(4)
        for _ in range(100):
            dropped = dropout_state(s, gen)
            self.assertTrue((dropped['mask'].sum(-1) >= 1).all())
            torch.testing.assert_close(dropped['candidate'][..., -1], dropped['mask'].float())


if __name__ == '__main__':
    unittest.main()
