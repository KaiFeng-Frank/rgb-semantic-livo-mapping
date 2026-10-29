import unittest
import numpy as np
from .raw_context import RawContext,Config,keys


class RawContextTests(unittest.TestCase):
    def setUp(self):
        self.pose=np.eye(4)
        self.current=np.array([[0,0,0,.1],[.39,.39,.39,.2],[1,1,1,.3],[-1,-1,-1,.4]],np.float32)
        self.old=np.array([[.11,.11,.11,.5],[.24,.24,.24,.6],[.36,.36,.36,.7],[8,8,8,.8]],np.float32)

    def test_original_rows_native_voxels_bounds_and_budget_preserved(self):
        m=RawContext();m.store(self.old,1_000_000_000,self.pose,[])
        out,info=m.prepare(self.current,1_500_000_000,self.pose,[],'ego')
        self.assertEqual(out[:4].tobytes(),self.current.tobytes());self.assertGreater(info['history_points'],0)
        self.assertLessEqual(len(out),6)
        self.assertFalse(np.isin(keys(out[4:,:3],.05),keys(self.current[:,:3],.05)).any())
        np.testing.assert_array_equal(out[:,:3].min(0),self.current[:,:3].min(0))
        np.testing.assert_array_equal(out[:,:3].max(0),self.current[:,:3].max(0))

    def test_memory_contains_only_a_copy_of_raw_evidence_and_expires(self):
        m=RawContext();old=self.old.copy();m.store(old,1_000_000_000,self.pose,[]);old[:]=999
        np.testing.assert_array_equal(m.previous['points'],self.old)
        out,info=m.prepare(self.current,2_100_000_000,self.pose,[],'ego')
        np.testing.assert_array_equal(out,self.current);self.assertEqual(info['history_points'],0)
        with self.assertRaises(ValueError):m.prepare(self.current,1_000_000_000,self.pose,[],'ego')

    def test_current_object_geometry_cannot_come_from_future(self):
        m=RawContext();m.store(self.old,1_000_000_000,self.pose,[])
        o=dict(stamp=1_500_000_000,camera_stamp=2_000_000_000,center=[0,0,0])
        with self.assertRaises(ValueError):m.prepare(self.current,1_500_000_000,self.pose,[o],'motion')

    def test_modes_read_same_raw_state_and_gt_is_not_retained(self):
        o=dict(stamp=1_000_000_000,camera_stamp=999_999_999,indices=[0],center=[.1,0,0],visual_session='a',visual_track=1,gt=42,fused_probabilities=[1,0])
        m=RawContext();m.store(self.old,1_000_000_000,self.pose,[o])
        self.assertNotIn('gt',m.previous['observations'][0]);self.assertNotIn('fused_probabilities',m.previous['observations'][0])
        a,ia=m.prepare(self.current,1_500_000_000,self.pose,[],'ego')
        m.prepare(self.current,1_500_000_000,self.pose,[],'unwarped')
        b,ib=m.prepare(self.current,1_500_000_000,self.pose,[],'ego')
        np.testing.assert_array_equal(a,b);self.assertEqual(ia['unmatched_instance_points'],1)


if __name__=='__main__':unittest.main()
