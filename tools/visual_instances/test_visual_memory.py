import json
import unittest
import numpy as np
from .geometry import InstanceFrame
from .visual_memory import VisualMemory


class VisualMemoryTests(unittest.TestCase):
    def frame(self,t,value=.8,tid=1,session='a'*32):
        scores=np.zeros((1,80),np.float32);scores[0,0]=value;scores[0,2]=1-value
        return InstanceFrame(t,session,np.ones((20,20),np.uint16),np.array([tid]),np.array([6]),np.array([.9]),class_scores=scores)

    def test_read_before_write_raw_only_and_no_confidence_from_repetition(self):
        m=VisualMemory();f=self.frame(0);p,n,_=m.step(f)
        np.testing.assert_array_equal(p,f.class_scores);self.assertEqual(n[0],0)
        f=self.frame(200_000_000,.1);m.step(f);f.class_scores[:]=1  # caller mutation cannot corrupt stored raw evidence
        f=self.frame(400_000_000,.1);p,n,_=m.step(f)
        w=np.exp(-np.array([.4,.2])/.75);w/=max(1,w.sum())
        self.assertAlmostEqual(float(p[0,0]),(.1+w[0]*.8+w[1]*.1)/(1+w.sum()),places=6)
        self.assertEqual(n[0],2)
        m=VisualMemory()
        for i in range(40):
            f=self.frame(i*50_000_000);p,_,stats=m.step(f);np.testing.assert_allclose(p,f.class_scores,atol=1e-7)
        self.assertLessEqual(stats['raw_records'],12)

    def test_causality_expiry_identity_and_session_isolation(self):
        m=VisualMemory();m.step(self.frame(0))
        with self.assertRaises(ValueError):m.step(self.frame(0))
        f=self.frame(200_000_000,.1,tid=2);p,n,_=m.step(f);self.assertEqual(n[0],0)
        f=self.frame(2_300_000_000,.2);p,n,_=m.step(f);self.assertEqual(n[0],0)
        f=self.frame(2_400_000_000,.3,session='b'*32);p,n,_=m.step(f);self.assertEqual(n[0],0)
        np.testing.assert_array_equal(p,f.class_scores)

    def test_track_bound_and_wire_validation(self):
        m=VisualMemory()
        for i in range(150):p,n,stats=m.step(self.frame(i*1_000_000,tid=i+1))
        self.assertEqual(stats['tracks'],128)
        f=self.frame(200_000_000,tid=150);f.temporal_class_scores,f.temporal_observations,_=m.step(f)
        restored=InstanceFrame.decode(f.encode())
        np.testing.assert_array_equal(restored.temporal_class_scores,f.temporal_class_scores)
        np.testing.assert_array_equal(restored.class_scores,f.class_scores)
        d=json.loads(f.encode());d['temporal_observations']=[1.5]
        with self.assertRaises(ValueError):InstanceFrame.decode(json.dumps(d))
        d=json.loads(f.encode());d['temporal_class_scores'][0][0]=float('nan')
        with self.assertRaises(ValueError):InstanceFrame.decode(json.dumps(d))


if __name__=='__main__':unittest.main()
