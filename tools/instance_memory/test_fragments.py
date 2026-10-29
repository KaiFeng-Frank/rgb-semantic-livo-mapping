"""Causality and non-self-confirmation tests for memory-based evidence admission."""
import copy
import unittest
import numpy as np
import torch
from .model import MemoryHead
from .runtime import Runtime
from .fragments import FragmentRuntime,confirm
from .test_memory import observation


class FragmentTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3);torch.set_num_threads(1)
        self.model=MemoryHead(np.zeros(320),np.ones(320)).eval()
        torch.nn.init.normal_(self.model.delta.weight,0,.1)
        self.identity=copy.deepcopy(self.model)

    def frame(self,t,*,trusted=True,center=(0,0,0),track=1):
        o=observation(t,track,center);o['trusted']=trusted
        return dict(stamp=o['stamp'],observations=[o])

    def test_two_past_observations_required_and_no_class_gate(self):
        rt=FragmentRuntime(self.model,self.identity)
        rt.step(self.frame(1))
        _,ids,_=rt.step(self.frame(1.5,trusted=False));self.assertEqual(ids,[0])
        rt.step(self.frame(2))
        f=self.frame(2.5,trusted=False);f['observations'][0]['visual_label']=9
        _,ids,info=rt.step(f)
        self.assertGreater(ids[0],0);self.assertEqual(info['admitted_fragments'],1)

    def test_candidates_do_not_create_or_refresh_their_own_history(self):
        rt=FragmentRuntime(self.model,self.identity)
        _,a,_=rt.step(self.frame(1));rt.step(self.frame(1.5))
        old=copy.deepcopy(rt.base.memory.tracks)
        for t in [2,2.5,3]:
            _,ids,_=rt.step(self.frame(t,trusted=False));self.assertEqual(ids,a)
            self.assertEqual(old,rt.base.memory.tracks)
        _,ids,_=rt.step(self.frame(3.5,trusted=False));self.assertEqual(ids,[0])
        fresh=FragmentRuntime(self.model,self.identity)
        for t in [1,1.5,2]:self.assertEqual(fresh.step(self.frame(t,trusted=False))[1],[0])
        self.assertEqual(fresh.base.memory.tracks,{})

    def test_other_nearby_identity_vetoes_a_visual_id_hint(self):
        rt=FragmentRuntime(self.model,self.identity)
        for t in [1,1.5]:
            f=self.frame(t,center=(-1,0,0));other=observation(t,2,(1,0,0));other['trusted']=True
            f['observations'].append(other);rt.step(f)
        _,ids,info=rt.step(self.frame(2,trusted=False,center=(0,0,0)))
        self.assertEqual(ids,[0]);self.assertEqual(info['candidate_reasons'].get('ambiguous'),1)

    def test_trusted_stream_state_and_outputs_are_identical(self):
        rt=FragmentRuntime(self.model,self.identity);control=Runtime(self.model,self.identity,'matched_current')
        for t in [1,1.5,2,2.5]:
            f=self.frame(t);original=copy.deepcopy(f)
            candidate=observation(t,2,(10,0,0));candidate['trusted']=False;f['observations'].append(candidate)
            p,ids,_=rt.step(f);q,control_ids,_=control.step(original)
            torch.testing.assert_close(p[0],q[0],rtol=0,atol=0)
            self.assertEqual(ids[0],control_ids[0]);self.assertEqual(rt.base.memory.tracks,control.memory.tracks)
            self.assertEqual(ids[1],0)

    def test_future_camera_and_reused_image_rejected(self):
        rt=FragmentRuntime(self.model,self.identity);rt.step(self.frame(1));rt.step(self.frame(1.5))
        f=self.frame(2,trusted=False);f['observations'][0]['camera_stamp']=3_000_000_000
        with self.assertRaises(ValueError):rt.step(f)
        f['observations'][0]['camera_stamp']=1_500_000_000-1000
        self.assertEqual(rt.step(f)[1],[0])

    def test_eval_targets_cannot_change_admission(self):
        a=FragmentRuntime(self.model,self.identity);b=FragmentRuntime(self.model,self.identity)
        for t in [1,1.5]:a.step(self.frame(t));b.step(self.frame(t))
        f=self.frame(2,trusted=False);g=copy.deepcopy(f)
        g['gt']=[999]*6;g['future_truth']=[1]*100;g['observations'][0]['gt_identity']=42
        pa,ia,_=a.step(f);pb,ib,_=b.step(g)
        self.assertEqual(ia,ib);torch.testing.assert_close(pa[0],pb[0],rtol=0,atol=0)


if __name__=='__main__':unittest.main()
