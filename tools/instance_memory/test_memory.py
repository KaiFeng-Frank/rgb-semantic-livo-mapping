"""Contracts that distinguish identity, semantic evidence and causal state."""
import copy
import unittest
import numpy as np
import torch

from .memory import ObjectMemory,Config
from .model import MemoryHead,batch_inputs
from .metrics import IdentityMetrics
from .runtime import Runtime


def observation(stamp,track=1,center=(0,0,0),cls=3):
    raw=np.full(16,.01,np.float32);raw[cls]=.85;raw/=raw.sum()
    return dict(stamp=int(stamp*1e9),camera_stamp=int(stamp*1e9)-1000,
        x=np.zeros(320,np.float32).tolist(),center=list(center),sensor_center=list(center),extent=[1,1,1],
        visual_track=track,visual_session='a',visual_confidence=.7,raw_mean=raw.tolist(),
        raw_probabilities=[raw.tolist()]*6,indices=list(range(6)),visual_label=cls)


class MemoryTests(unittest.TestCase):
    def step(self,m,obs,emb):
        stamp=obs[0]['stamp'];ids=m.candidates(stamp)
        old=np.stack([self.descriptors[k] for k in ids]) if ids else np.empty((0,32))
        tracks,hist,info=m.match(obs,emb,ids,old,stamp)
        m.write(obs,tracks,stamp)
        for k,v in zip(tracks,emb):self.descriptors[k]=v
        return tracks,hist

    def setUp(self):
        self.descriptors={};self.e=np.eye(32)[:2]

    def test_visual_id_changes_but_object_persists(self):
        m=ObjectMemory()
        a,_=self.step(m,[observation(1)],self.e[:1])
        b,h=self.step(m,[observation(1.5,track=27,cls=9)],self.e[:1])
        self.assertEqual(a,b);self.assertEqual(len(h[0]),1)
        self.assertEqual(h[0][0]['stamp'],1_000_000_000)

    def test_no_semantic_gate_and_no_identity_reuse_for_other_object(self):
        m=ObjectMemory()
        a,_=self.step(m,[observation(1,center=(0,0,0)),observation(1,2,(3,0,0))],self.e)
        # Deliberately swapped 2D IDs; actual position/appearance remain distinct.
        b,_=self.step(m,[observation(1.5,2,(0,0,0),9),observation(1.5,1,(3,0,0),3)],self.e)
        self.assertEqual(a,b)

    def test_bounded_raw_history_and_expiry(self):
        m=ObjectMemory()
        for t in np.arange(1,10,.5):
            o=observation(t);o['enhanced_probabilities']=[99]*16
            ids,h=self.step(m,[o],self.e[:1])
            self.assertLessEqual(len(h[0]),6)
            for r in h[0]:
                self.assertNotIn('enhanced_probabilities',r)
                self.assertLess(r['stamp'],o['stamp'])
        old=ids[0]
        new,_=self.step(m,[observation(20)],self.e[:1])
        self.assertNotEqual(old,new[0])

    def test_no_read_until_write_and_no_time_reversal(self):
        m=ObjectMemory();o=observation(1)
        ids=m.candidates(o['stamp'])
        out,_,_=m.match([o],self.e[:1],ids,np.empty((0,32)),o['stamp'])
        with self.assertRaises(ValueError):m.candidates(o['stamp']+1)
        m.write([o],out,o['stamp'])
        with self.assertRaises(ValueError):m.candidates(o['stamp'])

    def test_pose_revision_keeps_each_objects_sensor_center(self):
        m=ObjectMemory()
        ids,_=self.step(m,[observation(1,center=(0,0,0)),observation(1,2,(3,0,0))],self.e)
        pose=np.eye(4);pose[0,3]=100
        m.revise({1_000_000_000:pose})
        self.assertEqual(m.tracks[ids[0]]['last']['center'],[100.,0.,0.])
        self.assertEqual(m.tracks[ids[1]]['last']['center'],[103.,0.,0.])

    def test_zero_long_memory_really_zero(self):
        m=ObjectMemory(Config(short_records=1,long_records=0))
        for t in [1,1.5,2]:self.step(m,[observation(t)],self.e[:1])
        self.assertEqual(next(iter(m.tracks.values()))['long'],[])

    def test_unchanged_growing_trajectory_preserves_motion_state(self):
        m=ObjectMemory()
        for t,x in [(1,0),(1.5,.2),(2,.8)]:
            self.step(m,[observation(t,center=(x,0,0))],self.e[:1])
        before=copy.deepcopy(m.tracks)
        m.revise({int(t*1e9):np.eye(4) for t in [1,1.5,2,2.5]})
        self.assertEqual(before,m.tracks)


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(0)
        self.model=MemoryHead(np.zeros(320),np.ones(320))

    def test_current_only_does_not_read_past(self):
        self.model.eval();torch.nn.init.normal_(self.model.delta.weight,.0,.1)
        o=observation(2);past=observation(1)
        a=self.model(*batch_inputs(self.model,[o],[[]],o['stamp'],False))
        b=self.model(*batch_inputs(self.model,[o],[[past]],o['stamp'],False))
        torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_future_history_is_rejected(self):
        with self.assertRaises(ValueError):
            batch_inputs(self.model,[observation(1)],[[observation(2)]],1_000_000_000)

    def test_gradients_reach_reader(self):
        torch.nn.init.normal_(self.model.delta.weight,.0,.1)
        o=observation(2);p=observation(1);p['x']=np.ones(320).tolist()
        out=self.model(*batch_inputs(self.model,[o],[[p]],o['stamp']))
        out.square().sum().backward()
        self.assertGreater(float(self.model.attention.in_proj_weight.grad.abs().sum()),0)

    def test_runtime_empty_frames_and_gt_cannot_change_prediction(self):
        self.model.eval();identity=copy.deepcopy(self.model)
        a=Runtime(self.model,identity);b=Runtime(self.model,identity)
        f=dict(stamp=1_000_000_000,observations=[observation(1)])
        tampered=copy.deepcopy(f);tampered['gt']=[99]*6;tampered['future']=[1]*320
        pa,ia,_=a.step(f);pb,ib,_=b.step(tampered)
        torch.testing.assert_close(pa[0],pb[0]);self.assertEqual(ia,ib)
        p,ids,_=a.step(dict(stamp=2_000_000_000,observations=[]))
        self.assertEqual(p,[]);self.assertEqual(ids,[])


class MetricTests(unittest.TestCase):
    def test_perfect_and_missing_coverage(self):
        m=IdentityMetrics();m.add(np.array([1,1,2,2]),np.array([9,9,8,8]),0,np.ones(4,bool))
        self.assertEqual(m.result()['point_idf1'],1)
        self.assertEqual(m.result()['association_quality'],1)
        empty=IdentityMetrics();empty.add(np.array([1,1]),np.array([0,0]),0,np.ones(2,bool))
        self.assertEqual(empty.result()['point_idf1'],0)
        self.assertEqual(empty.result()['association_quality'],0)

    def test_stable_semantics_does_not_hide_identity_switch(self):
        m=IdentityMetrics();g=np.array([1]*10+[2]*10)
        m.add(g,np.array([7]*10+[8]*10),0,np.ones(20,bool))
        m.add(g,np.array([8]*10+[7]*10),1,np.ones(20,bool))
        self.assertEqual(m.result()['id_switches'],2)
        self.assertEqual(m.result()['point_idf1'],.5)


if __name__=='__main__':unittest.main()
