import unittest
import torch
from rl.train_candidate_rank import targets
from rl.train_sc_nbv import _utility_from_task_mask


class CandidateTargetsTest(unittest.TestCase):
    def test_padding_ties_and_unique_destination(self):
        logits=torch.zeros(2,4,3)
        labels=torch.tensor([0,0])
        present=torch.tensor([[True,True,False,False],[True,True,True,True]])
        context,mask,reward,pairs,info,u=targets(logits,labels,[0,1,2],present)
        self.assertEqual(context.tolist(),[0,1,4,5,6,7])
        self.assertEqual(mask[:2].sum(-1).tolist(),[1,1])
        self.assertFalse(mask[:2,2:].any())
        self.assertFalse(pairs.any())
        self.assertTrue(torch.isfinite(reward).all())
        self.assertTrue((reward==0).all())

    def test_correct_candidate_outranks_wrong_candidate(self):
        logits=torch.tensor([[[0.,0.,0.],[8.,0.,0.],[0.,8.,0.],[0.,0.,8.]]])
        context,mask,reward,pairs,info,u=targets(logits,torch.tensor([0]),[0,1,2],torch.ones(1,4,dtype=torch.bool))
        self.assertTrue(pairs[0,1,2])
        self.assertTrue(pairs[0,1,3])
        self.assertFalse(pairs[0,2,1])
        self.assertEqual(float(reward[0,1]),0.)
        self.assertLess(float(reward[0,2]),-.5)
        self.assertTrue(info[0])

    def test_label_free_utility_does_not_depend_on_gt_label(self):
        generator = torch.Generator().manual_seed(7)
        current = torch.randn(3, 55, generator=generator)
        future = torch.randn(3, 4, 55, generator=generator)
        mask = torch.zeros(3, 55, dtype=torch.bool)
        mask[:, :5] = True
        valid = torch.ones(3, 4, dtype=torch.bool)
        first = _utility_from_task_mask(
            current, future, torch.tensor([0, 1, 2]), valid, mask,
            "entropy_js", temperature=2.0,
        )
        second = _utility_from_task_mask(
            current, future, torch.tensor([4, 3, 2]), valid, mask,
            "entropy_js", temperature=2.0,
        )
        torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)


if __name__=='__main__':unittest.main()
