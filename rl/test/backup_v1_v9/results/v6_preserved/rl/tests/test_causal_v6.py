import copy
from types import SimpleNamespace
import unittest
import torch
from rl.causal_rank_v6 import causal_state,class_mask,CausalRanker,supervision,selection_loss


def fixture():
    torch.manual_seed(1)
    return SimpleNamespace(device=torch.device('cpu'),logits=torch.randn(2,4,55),z=torch.randn(2,4,512),
        pose=torch.randn(2,4,442),object_map=torch.rand(2,4,1800)*10,quality=torch.rand(2,4,4),
        geometry=torch.randn(2,4,4),reachable=torch.ones(2,4,4,dtype=torch.bool),labels=torch.tensor([0,1]))


class CausalTests(unittest.TestCase):
    def test_future_observations_and_labels_cannot_change_policy(self):
        c=fixture();e=torch.arange(2);a=torch.zeros(2,dtype=torch.long);bank=class_mask([0,1,2,3,4],2,'cpu')
        before=causal_state(c,e,a,bank)
        for name in ('logits','z','pose','object_map','quality'):
            getattr(c,name)[:,1:]=torch.randn_like(getattr(c,name)[:,1:])*1000
        c.labels[:]=54
        after=causal_state(c,e,a,bank)
        for k in before:torch.testing.assert_close(before[k],after[k],rtol=0,atol=0)
        m=CausalRanker().eval();torch.testing.assert_close(m(before),m(after),rtol=0,atol=0)

    def test_candidate_permutation_equivariance(self):
        c=fixture();e=torch.arange(2);a=torch.zeros(2,dtype=torch.long);bank=class_mask(range(5),2,'cpu')
        before=causal_state(c,e,a,bank);perm=torch.tensor([2,0,3,1]);d=copy.deepcopy(c)
        for k in ('logits','z','pose','object_map','quality','geometry'):setattr(d,k,getattr(c,k)[:,perm])
        d.reachable=c.reachable[:,perm][:,:,perm]
        after=causal_state(d,e,torch.ones_like(a),bank);m=CausalRanker().eval()
        torch.testing.assert_close(m(after),m(before)[:,perm])

    def test_tied_correct_actions_and_two_view_mask(self):
        c=fixture();c.logits.zero_();c.logits[0,:,0]=10;c.logits[1,:,1]=10
        e=torch.arange(2);a=torch.zeros(2,dtype=torch.long);bank=class_mask(range(5),2,'cpu')
        valid=torch.tensor([[False,True,False,False],[False,True,True,True]])
        target,correct,selective=supervision(c,e,a,bank,valid)
        self.assertFalse(selective.any());torch.testing.assert_close(target[1],torch.tensor([0.,1/3,1/3,1/3]))
        q=torch.zeros(2,4,requires_grad=True);loss=selection_loss(q,target,correct,selective,valid)
        loss.backward();self.assertTrue(torch.isfinite(loss));self.assertEqual(q.grad[0,2].item(),0.)


if __name__=='__main__':unittest.main()
