import json
from pathlib import Path
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from .core import weights, temporal_evidence, jacobian, register, guard, transform

CFG = json.loads(Path(__file__).with_name('protocol.json').read_text())
GEO = json.loads((Path(__file__).parents[1]/'loop_closure/protocol.json').read_text())


def features(points, label=14, confidence=1., frame=0):
    n = len(points)
    return dict(xyz=np.asarray(points), label=np.full(n, label), confidence=np.full(n, confidence),
                temporal_count=np.zeros(n, int), temporal_stability=np.ones(n), frame=frame)


class Contracts(unittest.TestCase):
    def test_unknown_or_low_confidence_labels_are_neutral_in_all_arms(self):
        a, b = features(np.zeros((4, 3))), features(np.zeros((4, 3)))
        a['label'][:] = [3, -1, 14, 6]
        b['label'][:] = [3, -1, 10, 6]
        a['confidence'][:] = b['confidence'][:] = [.3, 1., .49, 0.]
        for arm in CFG['arms']:
            np.testing.assert_array_equal(weights(a,b,np.arange(4),np.arange(4),arm,CFG),np.ones(4))

    def test_soft_dynamic_and_compatibility_weights_preserve_points(self):
        a, b = features(np.zeros((3,3))), features(np.zeros((3,3)))
        a['label'][:] = [3, 14, 10]
        b['label'][:] = [2, 10, 12]
        i = np.arange(3)
        np.testing.assert_allclose(weights(a,b,i,i,'D',CFG), [.25, 1, 1])
        np.testing.assert_allclose(weights(a,b,i,i,'S',CFG), [.25, .35, 1])
        self.assertEqual(len(a['xyz']), 3)

    def test_temporal_unknown_is_neutral_and_duplicate_frame_is_rejected(self):
        a = features([[0.,0,0],[5.,0,0]])
        h1, h2 = features([[.1,0,0]],frame=1), features([[.4,0,0]],frame=2)
        n, s = temporal_evidence(a,[h1,h2],CFG)
        np.testing.assert_array_equal(n,[2,0])
        np.testing.assert_allclose(s,[.5,1.])
        with self.assertRaises(ValueError):
            temporal_evidence(a,[h1,h1],CFG)

    def test_jacobian_matches_left_rotation_and_translation(self):
        rng = np.random.default_rng(51)
        p, n = rng.normal(size=(20,3)), rng.normal(size=(20,3))
        analytic = jacobian(p,n)
        for i in range(6):
            delta = np.eye(4)
            if i<3:
                v=np.zeros(3); v[i]=1e-7
                delta[:3,:3]=Rotation.from_rotvec(v).as_matrix()
            else:
                delta[i-3,3]=1e-7
            numeric = np.einsum('ij,ij->i',n,transform(p,delta)-p)/1e-7
            np.testing.assert_allclose(analytic[:,i],numeric,atol=1e-6)

    def test_weighted_solver_recovers_transform_and_unknown_semantics_match_G(self):
        rng = np.random.default_rng(42)
        p = rng.uniform(-5,5,(6000,3)); n = np.zeros_like(p)
        p[:2000,0]=-4; n[:2000,0]=1
        p[2000:4000,1]=3; n[2000:4000,1]=1
        p[4000:,2]=-2; n[4000:,2]=1
        truth=np.eye(4); truth[:3,:3]=Rotation.from_euler('xyz',[.01,-.02,.03]).as_matrix()
        truth[:3,3]=[.2,-.15,.1]
        target=features(p,label=-1); target['normal']=n
        source=features(transform(p,np.linalg.inv(truth)),label=-1)
        results=[]
        for arm in CFG['arms']:
            pose, stats=register(source,target,np.eye(4),arm,CFG,GEO)
            self.assertTrue(stats['accepted'],stats)
            np.testing.assert_allclose(pose,truth,atol=.003)
            results.append(pose)
        for p in results[1:]:
            np.testing.assert_array_equal(p,results[0])

    def test_planar_degeneracy_is_rejected_without_artificial_constraints(self):
        rng=np.random.default_rng(11)
        p=rng.uniform(-4,4,(1000,3)); p[:,2]=0
        source,target=features(p),features(p)
        target['normal']=np.tile([0.,0,1],(len(p),1))
        _,stats=register(source,target,np.eye(4),'G',CFG,GEO)
        self.assertFalse(stats['accepted'])
        self.assertEqual(stats['failure'],'geometric_degeneracy')

    def test_guard_does_not_use_gt_and_preserves_rejected_candidate(self):
        candidate=np.eye(4); candidate[0,3]=2
        before=candidate.copy()
        result,stats=guard(candidate,{'accepted':True},np.eye(4),CFG)
        self.assertTrue(stats['fallback'])
        np.testing.assert_array_equal(result,np.eye(4))
        np.testing.assert_array_equal(candidate,before)


if __name__ == '__main__':
    unittest.main()
