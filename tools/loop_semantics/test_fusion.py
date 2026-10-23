"""Evidence independence, moving-object vetoes and held-out frame separation."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from .fusion import Visit, propose, apply_proposal
from .run import build_maps, configuration, S


def evidence(label, confidence, distance, groups, xyz=(0., 0., 0.)):
    score = np.zeros((1, 16))
    score[0, label] = confidence / (1 + (distance/20.)**2)
    return dict(keys=np.array([123]), score=score, count=np.array([groups]), xyz=np.array([xyz]))


class Contracts(unittest.TestCase):
    def setUp(self):
        self.cfg = configuration()

    def test_point_density_does_not_multiply_frame_evidence(self):
        def run(repeats):
            v = Visit(16, 5, 20.)
            for f in [0, 5]:
                v.add(f, np.tile([1, 1, 2], repeats), np.tile([10, 12, 14], repeats),
                      np.tile([.6, .8, .9], repeats), np.tile([[0,0,0],[0,0,0],[1,1,1]], (repeats,1)),
                      np.tile([10.,10.,20.], repeats))
            return v.finish()
        a, b = run(1), run(100)
        np.testing.assert_array_equal(a['keys'], b['keys'])
        np.testing.assert_allclose(a['score'], b['score'], atol=1e-14)
        np.testing.assert_array_equal(a['count'], [2, 2])

    def test_repeated_visit_does_not_overwhelm_clearer_revisit(self):
        a = evidence(14, .95, 40, 100)
        b = evidence(15, .9, 5, 2)
        result = propose(a, b, self.cfg)
        self.assertEqual(result['group_balanced']['label'].item(), 14)
        self.assertEqual(result['revisit_balanced']['label'].item(), 15)
        self.assertTrue(result['revisit_balanced']['accept'].item())
        a['count'] *= 10
        repeated = propose(a, b, self.cfg)
        np.testing.assert_array_equal(result['revisit_balanced']['label'], repeated['revisit_balanced']['label'])
        np.testing.assert_array_equal(result['revisit_balanced']['share'], repeated['revisit_balanced']['share'])

    def test_geometry_and_weak_support_reject_revision(self):
        a, b = evidence(14, .8, 20, 4), evidence(15, .9, 5, 2, (.3,0,0))
        self.assertFalse(propose(a, b, self.cfg)['revisit_balanced']['accept'].item())
        b['xyz'][:] = 0
        b['count'][:] = 1
        self.assertFalse(propose(a, b, self.cfg)['revisit_balanced']['accept'].item())

    def test_moving_class_evidence_and_protected_voxels_never_relabelled(self):
        a, b = evidence(14, .8, 40, 4), evidence(15, .9, 5, 2)
        a['score'][0, 6] = .001
        self.assertFalse(propose(a, b, self.cfg)['eligible'].item())
        a['score'][0, 6] = 0
        p = propose(a, b, self.cfg)
        result = apply_proposal(np.array([123, 456]), np.array([14, 10]), np.array([True, False]), p, 'revisit_balanced')
        np.testing.assert_array_equal(result, [14, 10])

    def test_unknown_keys_and_unmatched_visits_cannot_create_voxels(self):
        a, b = evidence(14, .8, 40, 4), evidence(15, .9, 5, 2)
        p = propose(a, b, self.cfg)
        result = apply_proposal(np.array([10]), np.array([10]), np.array([False]), p, 'revisit_balanced')
        np.testing.assert_array_equal(result, [10])
        b['keys'][:] = 456
        p = propose(a, b, self.cfg)
        self.assertEqual(len(p['keys']), 0)

    def test_actual_build_excludes_heldout_predictions(self):
        cfg = dict(self.cfg, frames=6)
        mapper = S.SemanticVoxelMap
        def small_mapper(**kwargs):
            return mapper(voxel=kwargs['voxel'], cap0=8, hash_cap=32)
        def observation(scans, frame, config):
            n = 100 if frame == 5 else 1
            return dict(idx=np.arange(n), keep=np.ones(n, bool), world=np.zeros((n,3)),
                        label=np.full(n, 15 if frame == 5 else 10), confidence=np.ones(n), ranges=np.ones(n))
        with tempfile.TemporaryDirectory() as tmp, patch('tools.loop_semantics.run.configuration', return_value=cfg), \
             patch('tools.loop_semantics.run.setup', return_value=(None, np.zeros(3), [(0,1),(4,5)])), \
             patch('tools.loop_semantics.run.read_observation', side_effect=observation), \
             patch('tools.loop_semantics.run.S.SemanticVoxelMap', side_effect=small_mapper):
            result = build_maps(Path(tmp)/'no-gt-directory-exists', Path(tmp)/'out')
            self.assertFalse(result['ground_truth_read'])
            self.assertNotIn(5, result['modes']['heldout']['input_frames'])
            with np.load(Path(tmp)/'out/all-maps.npz') as full, np.load(Path(tmp)/'out/heldout-maps.npz') as hold:
                self.assertEqual(full['ordinary'].item(), 15)
                self.assertEqual(hold['ordinary'].item(), 10)


if __name__ == '__main__':
    unittest.main()
