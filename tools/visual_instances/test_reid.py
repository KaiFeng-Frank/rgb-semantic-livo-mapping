import unittest
import numpy as np
import torch
from .reid import InitializedBOTrack
from .evidence_predictor import EvidenceResults


class ReIDTests(unittest.TestCase):
    def test_visual_precision_isolation_restores_ptv3_caller(self):
        from .fullrate import native_visual_precision
        old=(torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
        try:
            torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=False
            with self.assertRaisesRegex(RuntimeError,'test'):
                with native_visual_precision():
                    self.assertFalse(torch.backends.cuda.matmul.allow_tf32)
                    self.assertTrue(torch.backends.cudnn.allow_tf32)
                    raise RuntimeError('test')
            self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
            self.assertFalse(torch.backends.cudnn.allow_tf32)
        finally:torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32=old

    def test_initial_feature_and_ema_have_initialized_storage(self):
        f=np.array([3.,4.],np.float32);before=f.copy()
        track=InitializedBOTrack(np.array([10,10,4,4,0]),.9,2,f)
        np.testing.assert_array_equal(f,before)
        self.assertEqual(len(track.features),1);self.assertTrue(np.isfinite(track.smooth_feat).all())
        track.update_features(np.array([4.,3.],np.float32))
        self.assertEqual(len(track.features),2);self.assertAlmostEqual(np.linalg.norm(track.smooth_feat),1.,places=6)

    def test_tracker_indexing_keeps_scores_features_and_masks_aligned(self):
        r=EvidenceResults(orig_img=np.zeros((20,20,3),np.uint8),path='test',names={0:'a'},
            boxes=torch.tensor([[1,1,4,4,.9,0],[5,5,8,8,.8,0]]),
            masks=torch.stack([torch.zeros(20,20),torch.ones(20,20)]),
            class_scores=torch.tensor([[1.,2.],[3.,4.]]),feats=torch.tensor([[5.,6.],[7.,8.]]))
        s=r[np.array([1,0])].cpu()
        torch.testing.assert_close(s.feats,torch.tensor([[7.,8.],[5.,6.]]))
        torch.testing.assert_close(s.class_scores,torch.tensor([[3.,4.],[1.,2.]]))
        self.assertEqual(float(s.masks.data[0].mean()),1.)


if __name__=='__main__':unittest.main()
