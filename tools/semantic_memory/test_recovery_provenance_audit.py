import unittest
import numpy as np
from .recovery_provenance_audit import provenance_masks


class ProvenanceTests(unittest.TestCase):
    def test_identity_class_and_prediction_are_distinct_requirements(self):
        qgt=np.array([3,3,3,10,-1]);qid=np.array([17001,17001,0,0,0])
        oldgt=np.array([3,3,10]);oldid=np.array([17001,17002,0])
        src=np.array([[0,0],[0,1],[0,1],[2,2],[0,0]])
        valid=np.ones((5,2),bool);valid[3,1]=False
        hp=np.array([[3,6],[3,3],[3,3],[10,10],[3,3]])
        m=provenance_masks(qgt,qid,oldgt,oldid,src,valid,hp)
        np.testing.assert_array_equal(m['same'],[[1,1],[1,0],[0,0],[1,0],[0,0]])
        np.testing.assert_array_equal(m['correct'],[[1,0],[1,0],[0,0],[1,0],[0,0]])
        np.testing.assert_array_equal(m['unknown'],[[0,0],[0,0],[1,1],[0,0],[1,1]])

    def test_no_source_does_not_invent_correspondence(self):
        m=provenance_masks(np.array([3]),np.array([17001]),None,None,None,np.zeros((1,4),bool),np.zeros((1,4),int))
        self.assertTrue(all(not x.any() for x in m.values()))


if __name__=='__main__':unittest.main()
