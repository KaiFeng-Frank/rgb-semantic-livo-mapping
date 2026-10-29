import unittest
import numpy as np
from .point_history_audit import RawPointHistory,neighbors,vote


class PointHistoryTests(unittest.TestCase):
    def test_raw_only_read_before_write_expiry_epoch_and_causality(self):
        m=RawPointHistory();xyz=np.array([[0,0,0],[.01,0,0],[.02,0,0],[.03,0,0]],float)
        first=m.step(xyz,np.ones(4,dtype=int),0,'a')[0]
        np.testing.assert_array_equal(first['previous_raw'],1)
        second=m.step(xyz,np.full(4,2),500000000,'a')[0]
        np.testing.assert_array_equal(second['previous_raw'],1)
        # The corrected ones must not become the next frame's memory.
        third=m.step(xyz,np.full(4,3),1000000000,'a')[0]
        np.testing.assert_array_equal(third['previous_raw'],2)
        with self.assertRaises(ValueError):m.step(xyz,np.ones(4,int),1000000000,'a')
        np.testing.assert_array_equal(m.step(xyz,np.full(4,4),2000000000,'a')[0]['previous_raw'],4)
        np.testing.assert_array_equal(m.step(xyz,np.full(4,5),0,'b')[0]['previous_raw'],5)

    def test_self_exclusion_duplicates_missing_and_tie(self):
        xyz=np.zeros((3,3));j,v=neighbors(xyz,xyz,same_scan=True)
        self.assertTrue(np.all((j!=np.arange(3)[:,None])|~v));self.assertTrue(np.all(v.sum(1)==2))
        np.testing.assert_array_equal(vote(np.array([3]),np.array([2]),np.zeros((1,4),int),np.array([[1,0,0,0]],bool)),3)
        j,v=neighbors(np.array([[10.,0,0]]),xyz);self.assertFalse(v.any())
        np.testing.assert_array_equal(vote(np.array([3]),np.array([2,2,2]),j,v),3)


if __name__=='__main__':unittest.main()
