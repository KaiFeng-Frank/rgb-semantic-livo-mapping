import unittest
import numpy as np
from .feature_supervision_audit import voxel_truth,object_rows


class SupervisionObserverTests(unittest.TestCase):
    def test_voxel_identity_rejects_mixed_or_unknown_labeled_points(self):
        inverse=np.array([0,0,1,1,2,2,3,3,4])
        gt=np.array([3,3,3,3,3,-1,3,3,-1])
        ids=np.array([17,17,17,18,17,0,17,0,0])
        labels,vid=voxel_truth(inverse,gt,ids,5)
        np.testing.assert_array_equal(labels,[3,3,3,3,-1])
        np.testing.assert_array_equal(vid,[17,0,17,0,0])

    def test_object_coverage_half_correct_and_ignored_points(self):
        gt=np.array([3]*6+[6]*5+[-1]*3)
        identity=np.array([17]*6+[18]*5+[18]*3)
        a=np.array([3]*3+[1]*3+[6]*2+[1]*3+[6]*3)
        b=np.array([3]*2+[1]*4+[6]*3+[1]*2+[6]*3)
        rows=object_rows(gt,identity,dict(a=a,b=b))
        self.assertEqual(rows[17]['success'],dict(a=True,b=False))
        self.assertEqual(rows[18]['success'],dict(a=False,b=True))
        self.assertEqual(rows[18]['points'],5)
        self.assertEqual(object_rows(np.array([3]*4),np.array([17]*4),dict(a=np.array([3]*4))),{})


if __name__=='__main__':unittest.main()
