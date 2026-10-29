import unittest
import numpy as np
from .point_history_causes import truth_masks


class CauseTests(unittest.TestCase):
    def test_class_identity_unknown_and_invalid_are_distinct(self):
        gt=np.array([3,3,3,10,-1]);ids=np.array([4010,4011,0,0,0])
        old_gt=np.array([3,10,-1]);old_ids=np.array([4010,0,0])
        j=np.array([[0,1],[0,1],[0,1],[1,0],[2,0]]);valid=np.ones_like(j,dtype=bool)
        cls,obj=truth_masks(gt,ids,old_gt,old_ids,j,valid)
        np.testing.assert_array_equal(cls[:,0],[1,1,1,1,0])
        np.testing.assert_array_equal(obj[:,0],[1,0,0,1,0])
        self.assertFalse(cls[:,1].any());self.assertFalse(obj[:,1].any())


if __name__=='__main__':unittest.main()
