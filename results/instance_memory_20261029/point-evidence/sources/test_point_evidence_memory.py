import unittest
import numpy as np
from .point_evidence_memory import EvidencePointMemory, owned_readout, evidence_veto, visual_arrays


class EvidenceTests(unittest.TestCase):
    def test_ownership_distinguishes_unknown_from_matching_identity(self):
        p=np.zeros((4,16),np.float32);p[:,0]=.6;p[:,1]=.4
        old=np.eye(16,dtype=np.float32)[[1,1,1,1]]
        j=np.repeat(np.arange(4)[:,None],4,axis=1);valid=np.ones((4,4),bool)
        y,keep=owned_readout(p,old,np.array([0,2,3,4]),np.array([0,2,5,0]),j,valid)
        np.testing.assert_array_equal(y,[1,1,0,0])
        np.testing.assert_array_equal(keep.any(1),[1,1,0,0])

    def test_veto_can_save_correct_current_but_also_block_a_correct_history(self):
        raw=np.array([0,0,0,0]);proposal=np.array([1,1,1,0]);visual=np.array([0,0,-1,0])
        y,mask=evidence_veto(raw,proposal,visual)
        np.testing.assert_array_equal(y,[0,0,1,0])
        np.testing.assert_array_equal(mask,[1,1,0,0])
        gt=np.array([0,1,1,0])
        self.assertEqual(int((mask&(y==gt)&(proposal!=gt)).sum()),1)
        self.assertEqual(int((mask&(y!=gt)&(proposal==gt)).sum()),1)

    def test_causal_expiry_epoch_and_raw_only_writeback(self):
        m=EvidencePointMemory();xyz=np.zeros((1,3));ids=np.array([7]);visual=np.array([-1])
        old=np.eye(16,dtype=np.float32)[[1]]
        p=old*.4;p[:,0]=.6
        m.step(xyz,old,ids,visual,0,'a')
        y,_=m.step(xyz,p,ids,visual,500000000,'a')
        np.testing.assert_array_equal(y,[1]);np.testing.assert_array_equal(m.previous['p'],p)
        with self.assertRaises(ValueError):m.step(xyz,p,ids,visual,500000000,'a')
        np.testing.assert_array_equal(m.step(xyz,p,ids,visual,1500000001,'a')[0],[0])
        np.testing.assert_array_equal(m.step(xyz,p,ids,visual,0,'b')[0],[0])

    def test_visual_projection_overlap_and_class_metadata_are_checked(self):
        score=np.zeros(80);score[7]=.8
        obs=dict(indices=[1,2],visual_id=4,current_scores=score,memory_scores=score,
                 current_class=9,memory_class=9,history_records=1)
        ids,cur,mem=visual_arrays(4,[obs])
        np.testing.assert_array_equal(ids,[0,4,4,0]);np.testing.assert_array_equal(cur,[-1,9,9,-1])
        with self.assertRaises(ValueError):visual_arrays(4,[obs,obs])
        with self.assertRaises(ValueError):visual_arrays(4,[dict(obs,memory_class=3)])


if __name__=='__main__':unittest.main()
