"""CPU-only contracts and an independent raw-point reference for semantic fusion.

Run: python3 -m unittest tools.loop_closure.test_semantic_fusion -v
Opt-in dense benchmark/parity: python3 -m tools.loop_closure.test_semantic_fusion --benchmark
"""
import json
import sys
import unittest
from time import perf_counter

import numpy as np

from .semantic_fusion import DEFAULT_EXCLUDED_CLASSES, ReversibleSemanticMap


def rigid(x=0., y=0., z=0., yaw=0., pitch=0.):
    c, s = np.cos(yaw), np.sin(yaw)
    a, b = np.cos(pitch), np.sin(pitch)
    pose = np.eye(4)
    pose[:3, :3] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ np.array(
        [[a, 0, b], [0, 1, 0], [-b, 0, a]])
    pose[:3, 3] = [x, y, z]
    return pose


def reference(packets, poses, voxel=.2, classes=16, excluded=(), minimum=0.):
    """Independent Python-dict reducer: no tiles, encoded IDs or production helpers."""
    votes, point_counts, frame_counts = {}, {}, {}
    for stamp in sorted(packets):
        points, labels, confidence = packets[stamp]
        seen = set()
        for point, label, weight in zip(points, labels, np.asarray(confidence, np.float32)):
            if int(label) in excluded or weight <= 0 or weight < np.float32(minimum):
                continue
            world = poses[stamp][:3, :3] @ np.asarray(point, np.float64) + poses[stamp][:3, 3]
            key = tuple(int(v) for v in np.floor(world/voxel))
            if key not in votes:
                votes[key] = np.zeros(classes, np.float64)
                point_counts[key] = frame_counts[key] = 0
            votes[key][int(label)] += float(weight)
            point_counts[key] += 1
            seen.add(key)
        for key in seen:
            frame_counts[key] += 1
    keys = sorted(votes)
    scores = np.array([votes[k] for k in keys], np.float64).reshape(-1, classes)
    labels = scores.argmax(axis=1).astype(np.uint16)
    confidence = (scores[np.arange(len(keys)), labels]/scores.sum(axis=1)).astype(np.float32)
    keys_array = np.array(keys, np.int64).reshape(-1, 3)
    return dict(keys=keys_array, scores=scores, labels=labels, confidence=confidence,
                points=(keys_array.astype(np.float64)+.5)*voxel,
                point_counts=np.array([point_counts[k] for k in keys], np.int64),
                keyframe_counts=np.array([frame_counts[k] for k in keys], np.int64))


class SemanticFusionTests(unittest.TestCase):
    def assertMap(self, actual, expected):
        for field in ('keys', 'labels', 'point_counts', 'keyframe_counts'):
            np.testing.assert_array_equal(actual[field], expected[field], err_msg=field)
        for field in ('scores', 'confidence', 'points'):
            np.testing.assert_allclose(actual[field], expected[field], atol=1e-12, rtol=1e-12,
                                       err_msg=field)

    def add(self, m, stamp, points, labels=None, confidence=None, pose=None):
        points = np.asarray(points, np.float64).reshape(-1, 3)
        labels = np.full(len(points), 10, np.uint16) if labels is None else labels
        confidence = np.ones(len(points), np.float32) if confidence is None else confidence
        return m.add(stamp, points, labels, confidence, np.eye(4) if pose is None else pose)

    def assertUnchanged(self, m, callback, exception=ValueError):
        before, counts = m.snapshot(), m.summary()
        with self.assertRaises(exception):
            callback()
        self.assertMap(m.snapshot(), before)
        self.assertEqual(m.summary(), counts)

    def test_weighted_votes_ties_centers_and_evidence_counts(self):
        m = ReversibleSemanticMap(excluded_classes=())
        p = np.array([[.01, .02, .03], [.08, .04, .02], [.1, .1, .1]])
        labels, confidence = np.array([10, 11, 10], np.uint16), np.array([.25, .5, .25], np.float32)
        self.assertEqual(m.add(0, p, labels, confidence, rigid())['status'], 'updated')
        self.add(m, 1, [[.02, .02, .02]], [11], [.25])
        expected = reference({0: (p, labels, confidence), 1: (np.array([[.02]*3]), [11], [.25])},
                             {0: rigid(), 1: rigid()})
        self.assertMap(m.snapshot(), expected)
        self.assertEqual(m.snapshot()['labels'][0], 11)
        self.assertAlmostEqual(float(m.snapshot()['confidence'][0]), .6, places=7)
        self.assertEqual(m.snapshot()['point_counts'][0], 4)
        self.assertEqual(m.snapshot()['keyframe_counts'][0], 2)
        # Equal raw support chooses the smallest class, not insertion order.
        tie = ReversibleSemanticMap(excluded_classes=())
        self.add(tie, 0, [[0, 0, 0], [0, 0, 0]], [12, 11], [.5, .5])
        self.assertEqual(tie.snapshot()['labels'][0], 11)

    def test_negative_floor_tile_boundaries_rotation_and_sorted_keys(self):
        m = ReversibleSemanticMap(voxel=1., tile_voxels=2, excluded_classes=())
        p = np.array([[-2.01, -.01, 0], [-2, 0, 0], [-.01, 0, 0], [0, 0, 0],
                      [1.99, 0, 0], [2, 0, 0], [.3, -2.1, .9]])
        labels = np.arange(10, 10+len(p)) % 16
        weights = np.ones(len(p), np.float32)
        m.add(0, p, labels, weights, rigid())
        self.assertMap(m.snapshot(), reference({0: (p, labels, weights)}, {0: rigid()}, voxel=1.))
        pose = rigid(.13, -.07, .51, yaw=.73, pitch=-.21)
        m.update_poses([0], [pose])
        self.assertMap(m.snapshot(), reference({0: (p, labels, weights)}, {0: pose}, voxel=1.))

    def test_default_protected_policy_keeps_barrier_cone_and_static_classes(self):
        m = ReversibleSemanticMap()
        p = np.column_stack((np.arange(16), np.zeros((16, 2))))
        self.add(m, 1, p, np.arange(16, dtype=np.uint16))
        self.assertEqual(DEFAULT_EXCLUDED_CLASSES, (1, 2, 3, 4, 5, 6, 8, 9))
        self.assertEqual(set(m.snapshot()['labels']), {0, 7, 10, 11, 12, 13, 14, 15})
        self.assertEqual(m.summary()['excluded_points'], 8)
        self.assertEqual(m.summary()['stored_points'], 8)
        m.update_poses([1], [rigid(x=10)])
        self.assertEqual(m.snapshot()['scores'][:, list(DEFAULT_EXCLUDED_CLASSES)].sum(), 0.)

    def test_caller_policy_threshold_zero_weight_and_filtered_count(self):
        m = ReversibleSemanticMap(excluded_classes=(0, 7), minimum_confidence=.35,
                                  max_total_points=2)
        p = np.zeros((7, 3))
        labels = np.array([0, 7, 10, 11, 12, 13, 14], np.uint16)
        weights = np.array([1, 1, .349, .35, .8, 0, .1], np.float32)
        result = m.add(1, p, labels, weights, rigid())
        self.assertEqual(result['status'], 'updated')
        self.assertEqual(result['accepted_points'], 2)
        counts = m.summary()
        self.assertEqual((counts['input_points'], counts['stored_points'], counts['filtered_points']), (7, 2, 5))
        self.assertEqual((counts['excluded_points'], counts['low_confidence_points']), (2, 3))
        self.assertMap(m.snapshot(), reference({1: (p, labels, weights)}, {1: rigid()},
                                              excluded=(0, 7), minimum=.35))

    def test_all_empty_and_all_filtered_packets_still_own_timestamps(self):
        m = ReversibleSemanticMap(max_total_points=1)
        self.assertEqual(m.add(0, np.empty((0, 3)), [], [], rigid())['status'], 'updated')
        self.add(m, 1, [[1, 0, 0]], [3])
        self.add(m, 2, [[2, 0, 0]], [10], [0.])
        self.assertEqual((m.summary()['keyframes'], m.summary()['stored_points']), (3, 0))
        report = m.update_poses([0, 1, 2], [rigid(x=1), rigid(x=2), rigid(x=3)])
        self.assertEqual((report['changed_keyframes'], report['dirty_tiles']), (3, 0))
        snap = m.snapshot(0)
        for field in ('keys', 'points'):
            self.assertEqual(snap[field].shape, (0, 3))
        self.assertEqual(snap['scores'].shape, (0, 16))
        self.assertEqual(snap['labels'].dtype, np.uint16)
        self.assertEqual(snap['confidence'].dtype, np.float32)
        self.assertTrue(m.verify_rebuild()['ok'])

    def test_dense_frame_preserves_all_200k_observations_without_truncation(self):
        m = ReversibleSemanticMap()
        points = np.zeros((200000, 3), np.float64)
        points[:, 0] = np.arange(200000) % 200 * .2 + .013
        result = self.add(m, 1, points, confidence=np.full(200000, .5, np.float32))
        self.assertEqual((result['status'], result['accepted_points']), ('updated', 200000))
        snapshot = m.snapshot(17, include_scores=False)
        self.assertEqual((snapshot['stored_points'], snapshot['map_voxels'], snapshot['display_points']),
                         (200000, 200, 17))
        full = m.snapshot()
        self.assertEqual(full['point_counts'].sum(), 200000)
        self.assertEqual(full['scores'].sum(), 100000.)
        np.testing.assert_array_equal(full['point_counts'], np.full(200, 1000))
        self.assertEqual(m.summary()['max_points'], 200000)

    def test_overlap_removal_retains_other_keyframes_and_erases_empty_tiles(self):
        m = ReversibleSemanticMap()
        self.add(m, 1, [[.01, 0, 0], [1.01, 0, 0]], [10, 12], [.75, 1.])
        self.add(m, 2, [[.02, 0, 0]], [11], [.5])
        report = m.update_poses([1], [rigid(x=8)])
        snap = m.snapshot()
        self.assertEqual(report['dirty_tiles'], 2)
        self.assertEqual(snap['labels'].tolist(), [11, 10, 12])
        self.assertEqual(snap['point_counts'].tolist(), [1, 1, 1])
        np.testing.assert_array_equal(snap['scores'][0, [10, 11]], [0., .5])
        m.update_poses([2], [rigid(x=8)])
        self.assertEqual(m.summary()['map_tiles'], 1)
        self.assertNotIn([0, 0, 0], m.snapshot()['keys'].tolist())
        self.assertTrue(m.verify_rebuild()['ok'])

    def test_repeated_duplicate_and_subvoxel_updates_do_not_accumulate_votes(self):
        m = ReversibleSemanticMap()
        points = [[.01, 0, 0], [.03, 0, 0]]
        self.add(m, 1, points)
        before, summary = m.snapshot(), m.summary()
        self.assertEqual(self.add(m, 1, [[99, 0, 0]])['status'], 'duplicate_stamp')
        self.assertEqual(m.update_poses([1], [rigid()])['status'], 'unchanged')
        self.assertEqual(m.summary(), summary)
        pose = rigid(x=1e-11)
        report = m.update_poses([1], [pose])
        self.assertEqual(report['changed_keyframes'], 1)
        self.assertMap(m.snapshot(), before)
        revision = m.summary()['revision']
        for _ in range(20):
            result = m.update_poses([1], [pose])
            self.assertEqual((result['changed_keyframes'], result['dirty_tiles']), (0, 0))
        self.assertEqual(m.summary()['revision'], revision)
        self.assertMap(m.snapshot(), before)

    def test_immutable_inputs_outputs_and_pose_revision_round_trip(self):
        m = ReversibleSemanticMap(excluded_classes=())
        p = np.array([[.013, .027, -.113], [.223, -.57, 1.99]])
        labels, confidence, pose = np.array([11, 12], np.uint16), np.array([.7, .9], np.float32), rigid()
        m.add(1, p, labels, confidence, pose)
        original = m.snapshot()
        p[:] = 999
        labels[:] = 0
        confidence[:] = 0
        pose[0, 3] = 99
        for n in range(30):
            revised = rigid(x=.37*n, y=-.13*n, yaw=.09*n, pitch=.01*n)
            m.update_poses([1], [revised])
            revised[:] = 0
        m.update_poses([1], [rigid()])
        self.assertMap(m.snapshot(), original)
        exposed = m.snapshot()
        for key in ('keys', 'points', 'scores', 'labels', 'confidence', 'point_counts', 'keyframe_counts'):
            exposed[key][:] = 0
        self.assertMap(m.snapshot(), original)
        self.assertTrue(m.verify_rebuild()['ok'])

    def test_keyframe_input_and_total_capacities_reject_without_mutation(self):
        m = ReversibleSemanticMap(max_keyframes=2, max_points=3, max_total_points=3)
        self.add(m, 1, [[0, 0, 0], [1, 0, 0]])
        counts, before = m.summary(), m.snapshot()
        for points, labels, reason in (
                (np.zeros((4, 3)), [3]*4, 'max_points'),
                (np.zeros((2, 3)), [10]*2, 'max_total_points')):
            result = self.add(m, 2, points, labels)
            self.assertEqual((result['status'], result['reason']), ('capacity_reached', reason))
            self.assertEqual(m.summary(), counts)
            self.assertMap(m.snapshot(), before)
        self.assertEqual(self.add(m, 2, [[2, 0, 0]])['status'], 'updated')
        counts = m.summary()
        self.assertEqual(self.add(m, 3, [])['reason'], 'max_keyframes')
        self.assertEqual(m.summary(), counts)
        self.assertEqual(m.summary()['peak_stored_points'], 3)
        self.assertTrue(m.verify_rebuild()['ok'])

    def test_stamps_strictly_increase_with_nanosecond_precision(self):
        m = ReversibleSemanticMap()
        t = 1700000000000000000
        self.add(m, np.int64(t), [[0, 0, 0]])
        self.add(m, t+1, [[1, 0, 0]])
        counts = m.summary()
        self.assertEqual(self.add(m, t-1, [])['status'], 'nonmonotonic_stamp')
        self.assertEqual(self.add(m, t, [])['status'], 'duplicate_stamp')
        self.assertEqual(m.summary(), counts)
        for bad in (-1, True, np.bool_(False), 1., np.nan, np.inf, '12', 1 << 63):
            with self.subTest(stamp=bad):
                self.assertUnchanged(m, lambda: self.add(m, bad, []))

    def test_full_graph_unknown_stamps_are_validated_then_ignored(self):
        m = ReversibleSemanticMap(max_keyframes=5)
        self.add(m, 2, [[0, 0, 0]])
        result = m.update_poses([1, 2, 3], [rigid(x=99), rigid(x=2), rigid(x=-99)])
        self.assertEqual((result['changed_keyframes'], result['unknown_keyframes']), (1, 2))
        self.assertEqual(m.snapshot()['keys'].tolist(), [[10, 0, 0]])
        counts = m.summary()
        result = m.update_poses([4, 3], [rigid(), rigid()])
        self.assertEqual((result['status'], result['unknown_keyframes']), ('unchanged', 2))
        self.assertEqual(m.summary(), counts)
        invalid = rigid(); invalid[0, 0] = 2
        self.assertUnchanged(m, lambda: m.update_poses([2, 3], [rigid(x=3), invalid]))
        self.assertEqual(m.update_poses([], [])['changed_keyframes'], 0)

    def test_pose_batch_bounds_shapes_stamps_and_duplicates_are_atomic(self):
        m = ReversibleSemanticMap(max_keyframes=2)
        self.add(m, 1, [[0, 0, 0]])
        self.add(m, 2, [[1, 0, 0]])
        invalid_batches = [([1, 2, 3], [rigid()]*3), ([1, 1], [rigid(x=3)]*2),
                           ([1., 2.], [rigid()]*2), ([True, 2], [rigid()]*2),
                           ([-1], [rigid()]), ([1 << 63], [rigid()]),
                           ([[1, 2]], [rigid()]), ([1], rigid()), ([1, 2], [rigid()]),
                           ([], [rigid()])]
        for stamps, poses in invalid_batches:
            with self.subTest(stamps=stamps):
                self.assertUnchanged(m, lambda: m.update_poses(stamps, poses))

    def test_invalid_geometry_labels_confidence_are_rejected_even_if_filtered(self):
        m = ReversibleSemanticMap()
        self.add(m, 0, [[0, 0, 0]])
        p, labels, weights = np.zeros((2, 3)), np.array([3, 10], np.uint16), np.ones(2, np.float32)
        invalid = [(np.zeros((2, 4)), labels, weights), (np.zeros(3), labels, weights),
                   (np.array([[np.nan, 0, 0], [0, 0, 0]]), labels, weights),
                   (np.array([[np.inf, 0, 0], [0, 0, 0]]), labels, weights),
                   (p.astype(complex), labels, weights), (p, [-1, 10], weights),
                   (p, [16, 10], weights), (p, [65535, 10], weights),
                   (p, [3., 10.], weights), (p, [True, False], weights),
                   (p, [[3], [10]], weights), (p, labels, [np.nan, 1]),
                   (p, labels, [np.inf, 1]), (p, labels, [-.1, 1]),
                   (p, labels, [1.01, 1]), (p, labels, [1]),
                   (p, labels, ['1', '1']), (p, labels, np.ones((2, 1)))]
        for points, lab, conf in invalid:
            with self.subTest(points_shape=np.shape(points), labels=lab, confidence=conf):
                self.assertUnchanged(m, lambda: m.add(1, points, lab, conf, rigid()))

    def test_invalid_transforms_in_add_and_update_are_atomic(self):
        m = ReversibleSemanticMap()
        self.add(m, 0, [[0, 0, 0]])
        self.add(m, 1, [[1, 0, 0]])
        bad = [np.zeros((3, 4)), np.zeros((4, 4)), rigid().astype(complex)]
        for i, j, value in ((0, 0, 2), (0, 0, -1), (0, 1, .1), (3, 0, 1e-5),
                            (3, 3, 1.000001), (0, 3, np.nan), (2, 3, np.inf)):
            pose = rigid(); pose[i, j] = value
            bad.append(pose)
        for pose in bad:
            with self.subTest(pose=pose):
                self.assertUnchanged(m, lambda: self.add(m, 2, [[0, 0, 0]], pose=pose))
                # Ensure an earlier valid changed pose never leaks through.
                if pose.shape == (4, 4):
                    self.assertUnchanged(m, lambda: m.update_poses([0, 1], [rigid(x=5), pose]))

    def test_geometry_overflow_after_valid_rigid_transforms_is_atomic(self):
        m = ReversibleSemanticMap()
        self.add(m, 0, [[0, 0, 0]])
        self.add(m, 1, [[1, 0, 0]])
        self.assertUnchanged(m, lambda: self.add(m, 2, [[1e100, 0, 0]]))
        self.assertUnchanged(m, lambda: m.update_poses([0, 1], [rigid(x=3), rigid(x=1e100)]))
        # Exercise the non-packed tile-index fallback without overflowing int64.
        wide = ReversibleSemanticMap(voxel=1., tile_voxels=1)
        points = [[-1e12, 2e12, -3e12], [1e12, -2e12, 3e12]]
        self.add(wide, 0, points)
        self.assertEqual(len(wide.snapshot()['keys']), 2)
        self.assertTrue(wide.verify_rebuild()['ok'])

    def test_configuration_rejects_invalid_values(self):
        invalid = [dict(voxel=0), dict(voxel=-.1), dict(voxel=np.nan), dict(voxel=np.inf),
                   dict(voxel=True), dict(voxel='0.2'), dict(classes=0), dict(classes=65537),
                   dict(classes=16.), dict(tile_voxels=0), dict(tile_voxels=1 << 30),
                   dict(max_keyframes=0), dict(max_keyframes=1001), dict(max_points=0),
                   dict(max_points=True), dict(max_total_points=0),
                   dict(minimum_confidence=-.1), dict(minimum_confidence=1.1),
                   dict(minimum_confidence=np.nan), dict(excluded_classes=(16,)),
                   dict(excluded_classes=(-1,)), dict(excluded_classes=(1.,)),
                   dict(excluded_classes=None)]
        for config in invalid:
            with self.subTest(config=config):
                with self.assertRaises(ValueError):
                    ReversibleSemanticMap(**config)
        self.assertEqual(ReversibleSemanticMap(max_total_points=8000000).summary()['max_total_points'], 8000000)

    def test_snapshot_cap_is_deterministic_and_does_not_limit_full_map(self):
        m = ReversibleSemanticMap(voxel=1., tile_voxels=2)
        points = np.column_stack((np.arange(47), np.zeros((47, 2))))
        self.add(m, 0, points)
        full, before = m.snapshot(), m.summary()
        small = m.snapshot(7, include_scores=False)
        self.assertEqual((small['map_voxels'], small['display_points']), (47, 7))
        self.assertNotIn('scores', small)
        np.testing.assert_array_equal(small['keys'], full['keys'][np.arange(7)*47//7])
        np.testing.assert_array_equal(small['keys'], m.snapshot(7)['keys'])
        self.assertEqual(m.summary(), before)
        self.assertMap(m.snapshot(100), full)
        for value in (-1, 1., True, np.nan):
            with self.subTest(cap=value):
                self.assertUnchanged(m, lambda: m.snapshot(value))

    def test_partial_revision_work_is_bounded_to_relevant_tiles(self):
        m = ReversibleSemanticMap()
        for stamp in range(20):
            self.add(m, stamp, [[.01, 0, 0]]*10, pose=rigid(x=stamp*20))
        report = m.update_poses(np.arange(20), [rigid(x=k*20+(8 if k == 4 else 0)) for k in range(20)])
        self.assertEqual(report['changed_keyframes'], 1)
        self.assertEqual(report['reprojected_points'], 10)
        self.assertEqual(report['dirty_tiles'], 2)
        self.assertEqual(report['recomputed_keyframe_tiles'], 1)
        self.assertEqual(report['map_voxels'], 20)
        self.assertGreaterEqual(report['dirty_snapshot_ms'], 0)
        self.assertGreaterEqual(report['tile_recompute_ms'], report['dirty_snapshot_ms'])
        self.assertTrue(m.verify_rebuild()['ok'])

    def test_random_revisions_match_independent_raw_history_reference(self):
        rng = np.random.default_rng(8147)
        packets, poses = {}, {}
        config = dict(voxel=.2, tile_voxels=5, excluded_classes=(1, 2, 3, 4, 5, 6, 8, 9),
                      minimum_confidence=.35)
        m = ReversibleSemanticMap(**config)
        for stamp in range(12):
            # Quantized anchors create substantial repeated/overlapping evidence.
            anchors = rng.integers(-7, 8, (90, 3))*.21
            p = anchors[rng.integers(0, len(anchors), 350)] + rng.uniform(-.02, .02, (350, 3))
            labels = rng.integers(0, 16, len(p), dtype=np.uint16)
            weights = rng.uniform(0, 1, len(p)).astype(np.float32)
            packets[stamp] = p, labels, weights
            poses[stamp] = rigid(x=.031*stamp, y=-.017*stamp, yaw=.011*stamp)
            m.add(stamp, p, labels, weights, poses[stamp])
        for round_index in range(8):
            stamps = rng.choice(12, size=12 if round_index == 7 else 5, replace=False)
            for stamp in stamps:
                poses[int(stamp)] = rigid(*rng.uniform(-.6, .6, 3), yaw=rng.uniform(-.3, .3),
                                         pitch=rng.uniform(-.1, .1))
            report = m.update_poses(stamps, [poses[int(k)] for k in stamps])
            self.assertEqual(report['changed_keyframes'], len(stamps))
            self.assertMap(m.snapshot(), reference(packets, poses, excluded=config['excluded_classes'], minimum=.35))
            self.assertTrue(m.verify_rebuild()['ok'])
            repeated = m.update_poses(np.arange(12), [poses[k] for k in range(12)])
            self.assertEqual((repeated['changed_keyframes'], repeated['dirty_tiles']), (0, 0))

    def test_update_batch_order_and_grouping_do_not_change_final_evidence(self):
        rng = np.random.default_rng(173)
        first, second = ReversibleSemanticMap(), ReversibleSemanticMap()
        final = {}
        for stamp in range(8):
            p = rng.normal(size=(170, 3))*.15
            labels = rng.integers(10, 16, len(p), dtype=np.uint16)
            weights = rng.uniform(.01, 1, len(p)).astype(np.float32)
            for m in (first, second):
                m.add(stamp, p, labels, weights, rigid())
            final[stamp] = rigid(x=.13*stamp, yaw=.03*stamp)
        first.update_poses(list(reversed(range(8))), [final[k] for k in reversed(range(8))])
        for k in (4, 2, 7, 1, 6, 0, 3, 5):
            second.update_poses([k], [final[k]])
        self.assertMap(first.snapshot(), second.snapshot())
        np.testing.assert_array_equal(first.snapshot()['scores'], second.snapshot()['scores'])

    def test_offline_rebuild_preserves_configuration_state_and_reports_errors(self):
        m = ReversibleSemanticMap(max_keyframes=3, max_points=200001, max_total_points=8000000,
                                  excluded_classes=(), voxel=.31, tile_voxels=3, minimum_confidence=.4)
        self.add(m, 1, [[0, 0, 0], [.5, .1, .2]], [0, 7], [.3, .7])
        m.update_poses([1], [rigid(x=.5, yaw=.2)])
        before, counts = m.snapshot(), m.summary()
        check = m.verify_rebuild()
        self.assertTrue(check['ok'], check)
        self.assertEqual((check['stored_points'], check['max_score_error'], check['errors']), (1, 0., []))
        self.assertMap(m.snapshot(), before)
        self.assertEqual(m.summary(), counts)
        # Deliberately corrupt a cached tile to prove this diagnostic detects it.
        tile = next(iter(m._tiles.values()))
        tile.scores.setflags(write=True)
        tile.scores[0, 7] += .1
        tile.scores.setflags(write=False)
        check = m.verify_rebuild()
        self.assertFalse(check['ok'])
        self.assertIn('scores_equal', check['errors'])
        self.assertGreater(check['max_score_error'], .09)


def benchmark():
    """Representative synthetic dense CPU workload; no wall-clock pass assertion."""
    rng = np.random.default_rng(20261026)
    frames, count = 120, 24000
    # Shared surface anchors give roughly 100k-200k voxels, with revisit overlap.
    anchors = np.column_stack((rng.uniform(-36, 36, 60000), rng.uniform(-18, 18, 60000),
                               rng.choice([-.8, 0., .8, 1.6], 60000)))
    m = ReversibleSemanticMap(minimum_confidence=.35)
    poses = np.repeat(np.eye(4)[None], frames, axis=0)
    add_ms = []
    start = perf_counter()
    for stamp in range(frames):
        p = anchors[rng.integers(0, len(anchors), count)] + rng.uniform(-.015, .015, (count, 3))
        labels = rng.integers(10, 16, count, dtype=np.uint16)
        confidence = rng.uniform(.35, 1., count).astype(np.float32)
        result = m.add(stamp, p, labels, confidence, poses[stamp])
        if result['status'] != 'updated':
            raise AssertionError(result)
        add_ms.append(result['total_ms'])
    ingest_ms = (perf_counter()-start)*1000
    partial = poses.copy()
    for k in range(115, 120):
        partial[k] = rigid(x=.13, y=-.07, yaw=.002)
    partial_report = m.update_poses(np.arange(frames), partial)
    partial_check = m.verify_rebuild()
    moved = np.array([rigid(x=.2+.001*k, y=-.1, yaw=.003) for k in range(frames)])
    whole_report = m.update_poses(np.arange(frames), moved)
    display = m.snapshot(200000, include_scores=False)
    final_check = m.verify_rebuild()
    result = dict(workload='synthetic CPU: 120 x 24k retained body observations',
                  ingest_ms=ingest_ms, add_median_ms=float(np.median(add_ms)),
                  add_p95_ms=float(np.percentile(add_ms, 95)), summary=m.summary(),
                  partial_revision=partial_report, partial_parity=partial_check,
                  whole_history_revision=whole_report, final_parity=final_check,
                  display={k: display[k] for k in ('map_voxels', 'display_points', 'sort_ms', 'snapshot_ms')})
    print(json.dumps(result, indent=2))
    if not partial_check['ok'] or not final_check['ok']:
        raise AssertionError('dense benchmark parity failed')


if __name__ == '__main__':
    if '--benchmark' in sys.argv:
        benchmark()
    else:
        unittest.main()
