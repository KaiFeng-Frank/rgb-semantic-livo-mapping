"""Integration contracts run in the pinned Open3D/GTSAM CPU environment."""
import unittest
from types import SimpleNamespace

import numpy as np

from .run import aggregate, raw_scan, features
from tools.loop_closure.common import S, transform
from tools.loop_closure.backend import cloud
from .test_core import CFG, GEO


class InputContracts(unittest.TestCase):
    def test_trace_preserves_geometric_centroids_and_mixed_label_evidence(self):
        rng = np.random.default_rng(91)
        xyz = rng.uniform(-5, 5, (10000, 3))
        evidence = np.zeros((len(xyz), 16))
        evidence[:, 14] = .8
        sampled = aggregate(xyz, evidence, .4)
        np.testing.assert_array_equal(sampled['xyz'], np.asarray(cloud(xyz, .4).points))
        np.testing.assert_array_equal(sampled['label'], np.full(len(sampled['xyz']), 14))
        np.testing.assert_allclose(sampled['confidence'], .8)
        mixed = aggregate(np.array([[0.,0,0], [.01,0,0]]), np.eye(16)[[3,14]], .4)
        np.testing.assert_allclose(mixed['evidence'][0,[3,14]], [.5,.5])
        self.assertEqual(mixed['confidence'][0], .5)

    def test_time_sort_validity_and_range_masks_keep_original_prediction_indices(self):
        points = np.array([[1,0,0,0], [100,0,0,0], [4,0,0,0], [np.nan,0,0,0], [2,0,0,0]],float)
        order = np.array([4,0,2,1,3])
        valid = np.array([True,False,True,True,False])
        traj = SimpleNamespace(valid=lambda t: valid, query_one=lambda t:(np.eye(4),True))
        def world(xyz,t,keep):
            np.testing.assert_array_equal(keep,valid)
            idx=np.flatnonzero(valid)
            return transform(xyz[idx],S.T_I_L),idx
        fake=SimpleNamespace(read=lambda frame:(points,order,np.arange(5)),traj=traj,images=[0.],world=world)
        xyz,idx,n,_=raw_scan(fake,0)
        np.testing.assert_array_equal(idx,[4,2])
        np.testing.assert_allclose(xyz[:,0],[2,4],atol=1e-5)
        self.assertEqual(n,5)

    def test_future_and_cross_visit_history_rejected_before_reading_files(self):
        invalid=dict(CFG, source_history_frames=[1040,1070])
        with self.assertRaisesRegex(ValueError,'causal'):
            features(None,None,'ZS',invalid,GEO)
        invalid=dict(CFG, source_history_frames=[20,1050])
        with self.assertRaisesRegex(ValueError,'disjoint'):
            features(None,None,'ZS',invalid,GEO)


if __name__ == '__main__':
    unittest.main()
