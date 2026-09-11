import unittest
import torch
from rl.causal_rank_v6 import causal_state, class_mask
from rl.tests.test_causal_v6 import fixture
from rl.four_ideas_models import Ensemble, decision_loss, make_model


class FourIdeasTests(unittest.TestCase):
    def test_causal_and_permutation(self):
        c = fixture(); e = torch.arange(2); start = torch.zeros(2, dtype=torch.long)
        bank = class_mask(range(5), 2, 'cpu')
        before = causal_state(c, e, start, bank)
        for name in ('logits', 'z', 'pose', 'object_map', 'quality'):
            getattr(c, name)[:, 1:] = torch.randn_like(getattr(c, name)[:, 1:]) * 100
        c.labels[:] = 54
        after = causal_state(c, e, start, bank)
        perm = torch.tensor([2, 0, 3, 1])
        shuffled = dict(before, candidate=before['candidate'][:, perm], mask=before['mask'][:, perm])
        for version in ('v8', 'v9', 'v10', 'v11'):
            net = make_model(version).eval()
            torch.testing.assert_close(net(before), net(after), rtol=0, atol=0)
            torch.testing.assert_close(net(shuffled), net(before)[:, perm])
        ensemble = Ensemble([make_model('v10').eval(), make_model('v10').eval()]).eval()
        torch.testing.assert_close(ensemble(shuffled), ensemble(before)[:, perm])

    def test_correct_ordering_and_mask(self):
        valid = torch.tensor([[False, True, True, False]])
        u = torch.tensor([[0., 1.1, -.1, 0.]])
        for version in ('v8', 'v9', 'v10', 'v11'):
            q = torch.zeros(1, 4, requires_grad=True)
            decision_loss(version, q, u, valid).backward()
            self.assertLess(q.grad[0, 1], 0)
            self.assertGreater(q.grad[0, 2], 0)
            self.assertEqual(q.grad[0, 0], 0)
            self.assertEqual(q.grad[0, 3], 0)

    def test_two_views_no_choice(self):
        for version in ('v8', 'v9', 'v10', 'v11'):
            q = torch.randn(2, 4, requires_grad=True)
            valid = torch.tensor([[False, True, False, False]] * 2)
            decision_loss(version, q, torch.rand(2, 4), valid).backward()
            self.assertTrue(torch.equal(q.grad, torch.zeros_like(q)))

    def test_v8_tied_corrects_and_v9_near_ties(self):
        valid = torch.tensor([[False, True, True, True]])
        for version, u in [('v8', torch.tensor([[0., 1.1, 1.2, 1.05]])),
                           ('v9', torch.tensor([[0., .1, .11, .12]]))]:
            q = torch.zeros(1, 4, requires_grad=True)
            decision_loss(version, q, u, valid).backward()
            self.assertTrue(torch.equal(q.grad, torch.zeros_like(q)))


if __name__ == '__main__':
    unittest.main()
