import unittest

import numpy as np

from .semantics import InstanceSemantics


class InstanceSemanticTests(unittest.TestCase):
    def observation(self, car=.95, x=0., visual=3, score=.9, tid=7, session='a'*32):
        xyz = np.array([[x+.02*i, 0., 5.] for i in range(6)])
        p = np.zeros((6,16),np.float32); p[:,3]=car;p[:,9]=1-car
        a = dict(ids=np.full(6,tid),labels=np.full(6,visual),scores=np.full(6,score),
                 info=dict(session=session),appearance={})
        return xyz,p,a

    def step(self, core, i, **kwargs):
        return core.step(*self.observation(**kwargs), stamp_ns=int((i+1)*500_000_000),pose=np.eye(4))

    def test_same_object_weak_class_flip_uses_shared_history(self):
        c=InstanceSemantics(); self.step(c,0)
        r=self.step(c,1,car=.45,visual=9,score=.4)
        self.assertEqual(r['labels'].tolist(),[3]*6)
        self.assertEqual(r['raw_labels'].tolist(),[9]*6)
        self.assertEqual(r['info']['history_instances'],1)
        self.assertEqual(len(set(r['instance_confidence'])),1)

    def test_strong_new_evidence_corrects_wrong_initial_label(self):
        c=InstanceSemantics();a=self.step(c,0,car=.05,visual=9)
        b=self.step(c,1,car=.95)
        self.assertEqual(a['labels'].tolist(),[9]*6)
        self.assertEqual(b['labels'].tolist(),[3]*6)
        self.assertEqual(b['info']['semantic_reset'],1)
        np.testing.assert_array_equal(a['epochs'],b['epochs'])

    def test_same_id_other_object_does_not_inherit_semantics(self):
        c=InstanceSemantics();a=self.step(c,0)
        b=self.step(c,1,x=4.,car=.05,visual=9)
        self.assertEqual(b['info']['motion'],1)
        self.assertEqual(b['labels'].tolist(),[9]*6)
        self.assertNotEqual(a['epochs'][0],b['epochs'][0])

    def test_appearance_veto_is_independent_of_same_semantic_class(self):
        c=InstanceSemantics()
        for i in range(2):
            xyz,p,a=self.observation(); h=np.zeros(24);h[i]=1.;a['appearance']={7:h}
            r=c.step(xyz,p,a,stamp_ns=int((i+1)*5e8),pose=np.eye(4))
        self.assertEqual(r['info']['appearance'],1)

    def test_confident_current_counterevidence_abstains_and_clears_old_semantics(self):
        c=InstanceSemantics(); self.step(c,0)
        xyz,p,a=self.observation(car=.56)
        p[4:,3]=.3;p[4:,9]=.7
        r=c.step(xyz,p,a,stamp_ns=1_000_000_000,pose=np.eye(4))
        np.testing.assert_array_equal(r['labels'],p.argmax(1))
        self.assertEqual(r['info']['semantic_conflict'],1)
        self.assertFalse(r['valid'].any())
        self.assertEqual(len(c.states[7]['history']),1)

    def test_competing_object_prevents_history_transfer_even_without_large_motion(self):
        c=InstanceSemantics();self.step(c,0)
        xyz,p,a=self.observation(x=.2,tid=8)
        c.step(xyz,p,a,stamp_ns=750_000_000,pose=np.eye(4))
        r=self.step(c,1,x=.2)
        self.assertEqual(r['info']['competing_history'],1)

    def test_current_instance_can_correct_outlier_without_history_manufacturing_confidence(self):
        c=InstanceSemantics();self.step(c,0)
        xyz,p,a=self.observation();p[-1,3]=.3;p[-1,9]=.7
        r=c.step(xyz,p,a,stamp_ns=1_000_000_000,pose=np.eye(4))
        self.assertTrue(r['valid'].all())
        self.assertEqual(r['labels'].tolist(),[3]*6)
        self.assertNotIn('semantic_conflict',r['info'])

    def test_pose_revision_reanchors_history_without_false_identity_break(self):
        c=InstanceSemantics();a=self.step(c,0)
        pose=np.eye(4);pose[0,3]=100.
        c.revise_poses([500_000_000],[pose])
        b=c.step(*self.observation(),stamp_ns=1_000_000_000,pose=pose)
        self.assertEqual(b['info']['accepted'],1)
        np.testing.assert_array_equal(a['epochs'],b['epochs'])

    def test_disconnected_background_is_not_unified(self):
        xyz,p,a=self.observation();xyz=np.concatenate([xyz,[[0,0,15],[.02,0,15]]])
        p=np.concatenate([p,np.eye(16,dtype=np.float32)[[10,10]]])
        a['ids']=np.full(8,7);a['labels']=np.full(8,3);a['scores']=np.full(8,.9)
        r=InstanceSemantics().step(xyz,p,a,stamp_ns=1,pose=np.eye(4))
        self.assertEqual(r['labels'][-2:].tolist(),[10,10])
        self.assertEqual(r['valid'].tolist(),[1]*6+[0]*2)

    def test_enhanced_output_never_reenters_evidence(self):
        c=InstanceSemantics();self.step(c,0)
        r=self.step(c,1,car=.45,visual=9,score=.4)
        saved=c.states[7]['history'][-1]['posterior'].copy()
        self.assertEqual(int(saved.argmax()),9)
        r['labels'][:]=10
        np.testing.assert_array_equal(c.states[7]['history'][-1]['posterior'],saved)

    def test_session_ttl_and_absent_vision_do_not_propagate_old_label(self):
        c=InstanceSemantics();self.step(c,0)
        r=self.step(c,1,car=.05,visual=9,session='b'*32)
        self.assertEqual(r['labels'].tolist(),[9]*6)
        self.assertNotIn('history_instances',r['info'])
        xyz,p,a=self.observation();a['ids'][:]=0;a['info']['session']=None
        r=c.step(xyz,p,a,stamp_ns=5_000_000_000,pose=np.eye(4))
        self.assertEqual(r['info']['tracks'],0)
        np.testing.assert_array_equal(r['labels'],p.argmax(1))


if __name__=='__main__': unittest.main()
