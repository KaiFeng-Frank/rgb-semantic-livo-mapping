"""Small semantic adapter contracts; the trajectory tests need only NumPy.

Run with ``python -m unittest tools.loop_closure.test_semantic_online -v``.
Source ROS externally to also exercise the actual PointCloud2 output schema.
The separate validate_semantic_ros fixture exercises native graph/ROS processes.
"""
from collections import OrderedDict
from copy import deepcopy
import importlib.util
import threading
import unittest

import numpy as np

from .semantic_node import checked_trajectory, claim_packet
from .semantic_trajectory import compact_snapshot


def revision(times=(1_000_000_000, 11_000_000_000), version=0):
    poses = np.repeat(np.eye(4)[None], len(times), axis=0)
    poses[:, 0, 3] = np.arange(len(times)) * 2.1
    return dict(frame_id='map', stamp_ns=times[-1], version=version,
                times_ns=list(times), poses=poses.tolist())


class TrajectoryContracts(unittest.TestCase):
    def test_valid_revision_is_independent_of_ros_and_native_backend(self):
        packet = revision()
        before = deepcopy(packet)
        result = checked_trajectory(packet)
        self.assertEqual(packet, before)
        self.assertEqual(result['stamp_ns'], packet['stamp_ns'])
        self.assertEqual(result['version'], 0)
        self.assertEqual(result['times_ns'].dtype, np.dtype('int64'))
        np.testing.assert_array_equal(result['poses'], packet['poses'])

    def test_unknown_frame_is_rejected(self):
        for frame in ('', 'odom', 'backend_lidar', '/map'):
            with self.subTest(frame=frame), self.assertRaises(ValueError):
                checked_trajectory(dict(revision(), frame_id=frame))

    def test_timestamp_and_version_require_actual_integers(self):
        for key in ('stamp_ns', 'version'):
            for bad in (1.0, True, '1', np.int64(1)):
                with self.subTest(key=key, bad=bad), self.assertRaises(ValueError):
                    checked_trajectory(dict(revision(), **{key: bad}))
        for bad in (1.0, True, '1', -1):
            packet = revision()
            packet['times_ns'][0] = bad
            with self.subTest(time=bad), self.assertRaises(ValueError):
                checked_trajectory(packet)

    def test_timestamp_order_watermark_capacity_and_version(self):
        for times in ((1, 1), (2, 1), (-1, 1)):
            with self.subTest(times=times), self.assertRaises(ValueError):
                checked_trajectory(revision(times))
        for patch in (dict(stamp_ns=12), dict(version=-1), dict(times_ns=[])):
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                checked_trajectory(dict(revision(), **patch))
        with self.assertRaises(ValueError):
            checked_trajectory(revision(), capacity=1)

    def test_nonrigid_or_nonfinite_poses_are_rejected(self):
        for row, column, value in ((0, 0, 2.), (0, 1, .2), (0, 0, -1.),
                                   (3, 0, .01), (3, 3, 0.), (0, 3, float('nan')),
                                   (1, 3, float('inf'))):
            packet = revision()
            packet['poses'][0][row][column] = value
            with self.subTest(row=row, column=column, value=value), self.assertRaises(ValueError):
                checked_trajectory(packet)
        with self.assertRaises(ValueError):
            checked_trajectory(dict(revision(), poses=[np.eye(3).tolist()]))

    def test_reset_reordering_and_version_rollback_are_rejected(self):
        previous = checked_trajectory(revision(version=2))
        for packet in (revision((1_000_000_000,), 2), revision(version=1),
                       revision((2_000_000_000, 11_000_000_000), 3)):
            with self.subTest(packet=packet), self.assertRaises(ValueError):
                checked_trajectory(packet, previous)

    def test_append_without_revising_history_keeps_revision(self):
        previous = checked_trajectory(revision())
        packet = revision((1_000_000_000, 11_000_000_000, 21_000_000_000))
        result = checked_trajectory(packet, previous)
        self.assertEqual(result['version'], 0)
        np.testing.assert_array_equal(result['poses'][:2], previous['poses'])

    def test_exact_duplicate_and_new_revision_are_accepted(self):
        packet = revision()
        previous = checked_trajectory(packet)
        np.testing.assert_array_equal(checked_trajectory(packet, previous)['poses'], previous['poses'])
        packet['version'] += 1
        packet['poses'][1][0][3] -= .2
        result = checked_trajectory(packet, previous)
        self.assertEqual(result['version'], 1)
        self.assertNotEqual(result['poses'][1, 0, 3], previous['poses'][1, 0, 3])

    def test_same_revision_different_history_is_rejected(self):
        # A changed pose under the same revision otherwise bypasses on_graph's
        # duplicate check, or silently changes history when a keyframe is added.
        previous = checked_trajectory(revision())
        for times in ((1_000_000_000, 11_000_000_000),
                      (1_000_000_000, 11_000_000_000, 21_000_000_000)):
            packet = revision(times)
            packet['poses'][1][0][3] += .1
            with self.subTest(times=times), self.assertRaises(ValueError):
                checked_trajectory(packet, previous)

    def test_compact_graph_payload_excludes_dense_geometry_and_labels(self):
        graph = checked_trajectory(revision())
        payload = dict(graph, map_points=np.ones((20, 3)), labels=np.ones(20),
                       correction=np.eye(4), attempts=[{'accepted': True}])
        compact = compact_snapshot(lambda: payload)
        self.assertEqual(set(compact), {'stamp_ns', 'version', 'times_ns', 'poses'})
        np.testing.assert_array_equal(compact['poses'], graph['poses'])
        self.assertIsNone(compact_snapshot(lambda: None))


class PacketOwnershipContracts(unittest.TestCase):
    def test_claimed_packet_cannot_also_be_evicted_or_counted_again(self):
        # Hold the worker after its atomic claim, as if update_poses were slow.
        # New arrivals must evict only pending packets, never this owned frame.
        payloads = {stamp: object() for stamp in range(1, 5)}
        pending = OrderedDict((stamp, (payloads[stamp], 99.)) for stamp in (1, 2))
        lock, claimed, release = threading.Lock(), threading.Event(), threading.Event()
        owned = []

        def worker():
            with lock:
                owned.append(claim_packet(pending, {'stamp_ns': 2}, 100., 3.))
            claimed.set()
            release.wait(2.)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        evicted = []
        try:
            self.assertTrue(claimed.wait(1.), 'worker did not claim a packet')
            with lock:
                self.assertNotIn(1, pending)
                for stamp in (3, 4):
                    pending[stamp] = (payloads[stamp], 100.)
                    while len(pending) > 2:
                        evicted.append(pending.popitem(last=False)[0])
                self.assertEqual(list(pending), [3, 4])
            self.assertEqual(evicted, [2])
            self.assertEqual(owned[0][0], 1)
            self.assertIs(owned[0][1][0], payloads[1])
        finally:
            release.set()
            thread.join(timeout=2.)
        self.assertFalse(thread.is_alive())
        processed = [owned[0][0]]
        with lock:
            while pending:
                processed.append(claim_packet(pending, {'stamp_ns': 4}, 100., 3.)[0])
            self.assertIsNone(claim_packet(pending, {'stamp_ns': 4}, 100., 3.))
        self.assertEqual(processed, [1, 3, 4])
        self.assertEqual(sorted(processed + evicted), [1, 2, 3, 4])
        self.assertTrue(set(processed).isdisjoint(evicted))

    def test_future_unanchored_packet_waits_for_its_graph(self):
        packet = (object(), 10.)
        pending = OrderedDict([(2, packet)])
        self.assertIsNone(claim_packet(pending, {'stamp_ns': 1}, 11., 3.))
        self.assertIs(pending[2], packet)
        claimed = claim_packet(pending, {'stamp_ns': 2}, 11., 3.)
        self.assertEqual(claimed[0], 2)
        self.assertIs(claimed[1], packet)
        self.assertFalse(pending)

    def test_stale_future_packet_is_claimed_once_for_rejection(self):
        pending = OrderedDict([(2, (object(), 10.))])
        self.assertIsNone(claim_packet(pending, {'stamp_ns': 1}, 13., 3.))
        claimed = claim_packet(pending, {'stamp_ns': 1}, 13.001, 3.)
        self.assertEqual(claimed[0], 2)
        self.assertFalse(pending)
        self.assertIsNone(claim_packet(pending, {'stamp_ns': 1}, 14., 3.))

    def test_past_unanchored_packet_is_claimed_for_rejection(self):
        graph = checked_trajectory(revision((1, 3)))
        pending = OrderedDict([(2, (object(), 10.)), (4, (object(), 10.))])
        claimed = claim_packet(pending, graph, 10.5, 3.)
        self.assertEqual(claimed[0], 2)
        self.assertNotIn(claimed[0], graph['times_ns'])
        self.assertEqual(list(pending), [4])
        self.assertIsNone(claim_packet(pending, graph, 10.5, 3.))

    def test_no_graph_or_empty_queue_never_transfers_ownership(self):
        pending = OrderedDict([(1, (object(), 10.))])
        original = list(pending.items())
        self.assertIsNone(claim_packet(pending, None, 100., 3.))
        self.assertEqual(list(pending.items()), original)
        self.assertIsNone(claim_packet(OrderedDict(), {'stamp_ns': 1}, 100., 3.))


@unittest.skipUnless(importlib.util.find_spec('sensor_msgs') is not None,
                     'source ROS externally for PointCloud2 schema tests')
class SemanticOutputSchema(unittest.TestCase):
    def test_actual_label_confidence_schema_round_trip(self):
        from .semantic_outputs import cloud
        from .validate_semantic_ros import semantic_rows
        points = np.array([[1., 2., 3.], [-.5, 0., 9.]], np.float32)
        labels = np.array([7, 15], np.uint16)
        confidence = np.array([.5, .875], np.float32)
        stamp = 32_123_456_789
        message = cloud(points, labels, confidence, 'map', stamp)
        self.assertEqual(message.header.frame_id, 'map')
        self.assertEqual(message.header.stamp.sec * 10**9 + message.header.stamp.nanosec, stamp)
        self.assertEqual([(f.name, f.offset, f.datatype, f.count) for f in message.fields],
                         [('x', 0, 7, 1), ('y', 4, 7, 1), ('z', 8, 7, 1),
                          ('label', 12, 4, 1), ('confidence', 16, 7, 1)])
        self.assertEqual(message.point_step, 20)
        self.assertEqual(message.row_step, 40)
        self.assertEqual(len(message.data), 40)
        xyz, actual_labels, actual_confidence = semantic_rows(message)
        np.testing.assert_array_equal(xyz, points)
        np.testing.assert_array_equal(actual_labels, labels)
        np.testing.assert_array_equal(actual_confidence, confidence)

    def test_empty_output_retains_the_schema(self):
        from .semantic_outputs import cloud
        from .validate_semantic_ros import semantic_rows
        message = cloud(np.empty((0, 3)), np.empty(0, np.uint16), np.empty(0), 'map', 1)
        xyz, labels, confidence = semantic_rows(message)
        self.assertEqual(xyz.shape, (0, 3))
        self.assertEqual(labels.dtype, np.dtype('<u2'))
        self.assertEqual(confidence.dtype, np.dtype('<f4'))


if __name__ == '__main__':
    unittest.main()
