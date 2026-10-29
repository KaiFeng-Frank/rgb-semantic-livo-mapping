import sys
from pathlib import Path
import unittest
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from semantic_memory import MemoryConfig
from semantic_repair import JointRepair,RepairConfig,repair_loss
from .repair_supervision import current_token_input,supervision_groups,repair_objective


class SupervisionTrainingTests(unittest.TestCase):
    def test_cached_current_encoding_is_native_current_input(self):
        torch.manual_seed(5);m=JointRepair(MemoryConfig(),RepairConfig());f=torch.randn(7,64);a=torch.randn(7,16)
        saved={};handle=m.net.register_forward_pre_hook(lambda mod,args:saved.update(z=args[0].detach().clone()))
        c=dict(features=torch.randn(7,4,64),probabilities=torch.softmax(torch.randn(7,4,16),-1),
            delta=torch.randn(7,4,3),age=torch.rand(7,4),valid=torch.rand(7,4)>.3)
        m(f,a,c);cached=saved['z']
        cur=dict(features=f[:,None],probabilities=a.softmax(-1)[:,None],delta=torch.zeros(7,1,3),
            age=torch.zeros(7,1),valid=torch.zeros(7,1,dtype=torch.bool))
        m(f,a,cur);torch.testing.assert_close(current_token_input(cached),saved['z'],rtol=0,atol=0)
        handle.remove()

    def test_group_membership_obeys_bound_history_validity_and_ignored_labels(self):
        a=torch.tensor([[0.,2.],[0.,4.],[2.,0.],[2.,0.],[1.,1.]])
        y=torch.tensor([0,0,0,0,-1]);h=torch.tensor([[0],[0],[1],[1],[0]])
        valid=torch.tensor([[1],[1],[1],[0],[1]],dtype=torch.bool)
        torch.testing.assert_close(supervision_groups(a,y,h,valid),torch.tensor([0,2,1,2,-1]))

    def test_standard_reduction_matches_original_loss_and_gradients(self):
        torch.manual_seed(9);z=torch.randn(8,16,requires_grad=True);a=torch.randn(8,16);y=torch.tensor([0,3,-1,6,3,9,5,1]);g=torch.tensor([0,1,-1,2,0,1,2,2]);w=torch.linspace(.25,2,16)
        output=dict(query_indices=torch.arange(8),selected_logits=z,anchor_logits=a)
        old=repair_loss(output,y,w);new,_=repair_objective(z,a,y,g,w,False)
        torch.testing.assert_close(new,old,rtol=0,atol=0)
        go=torch.autograd.grad(old,z,retain_graph=True)[0];gn=torch.autograd.grad(new,z)[0]
        torch.testing.assert_close(gn,go,rtol=0,atol=0)

    def test_balanced_mean_is_invariant_to_replication_within_group(self):
        a=torch.tensor([[0.,2.],[2.,0.],[2.,0.]])
        z=torch.tensor([[.1,1.],[1.,.1],[3.,.1]],requires_grad=True)
        y=torch.tensor([0,0,0]);g=torch.tensor([0,1,2]);w=torch.ones(2)
        b,_=repair_objective(z,a,y,g,w,True)
        draw=torch.tensor([0,1,2,2,2,2,2])
        bb,_=repair_objective(z[draw],a[draw],y[draw],g[draw],w,True)
        torch.testing.assert_close(b,bb,rtol=0,atol=1e-7)
        self.assertTrue(torch.isfinite(torch.autograd.grad(bb,z)[0]).all())
        empty=torch.full((3,),-1)
        zero,_=repair_objective(z,a,empty,empty,w,True)
        self.assertEqual(zero.item(),0.)


if __name__=='__main__':unittest.main()
