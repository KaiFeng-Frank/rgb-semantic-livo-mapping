"""Numerical and lifecycle contracts for causal online graph/IPC, no datasets."""
import sys
import time
import unittest
import os
import signal
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from .common import transform
from .online import IncrementalGraph, OnlineBackend, configuration
from .online_transport import BackendProcess, ExactPairs


def scene():
    rng = np.random.default_rng(42)
    p = rng.uniform(-8, 8, (12000, 3)).astype(np.float32)
    p[:4000, 0], p[4000:8000, 1], p[8000:, 2] = -6, 7, -2
    return p


def pose(x=0., yaw=0.):
    out = np.eye(4)
    out[:3, :3] = Rotation.from_euler('z', yaw).as_matrix()
    out[0, 3] = x
    return out


class OnlineContracts(unittest.TestCase):
    def test_no_loop_is_exact_identity_and_input_immutable(self):
        graph = IncrementalGraph(configuration())
        raw = np.array([pose(k*.3, k*.02) for k in range(60)])
        before = raw.copy()
        for p in raw:
            graph.append(p)
        np.testing.assert_array_equal(graph.poses(), raw)
        np.testing.assert_allclose(np.array([graph.isam.calculateEstimatePose3(i).matrix()
                                            for i in range(len(raw))]), raw, atol=1e-9)
        np.testing.assert_array_equal(graph.correction, np.eye(4))
        np.testing.assert_array_equal(raw, before)

    def test_loop_direction_and_future_odometry_not_double_corrected(self):
        graph = IncrementalGraph(configuration())
        for x in (0, 1, 2, .3):
            graph.append(pose(x))
        graph.add_loop(0, 3, np.eye(4))
        self.assertLess(abs(graph.poses()[3, 0, 3]), .3)
        graph.append(pose(1.3))
        poses = graph.poses()
        np.testing.assert_allclose(np.linalg.inv(poses[3]) @ poses[4], pose(1.), atol=1e-7)
        np.testing.assert_allclose(graph.correction @ pose(1.3), poses[4], atol=1e-9)

    def test_rotation_and_left_correction(self):
        graph = IncrementalGraph(configuration())
        raw = [pose(0), pose(2, .2), pose(.2, .04)]
        for p in raw:
            graph.append(p)
        graph.add_loop(0, 2, np.eye(4))
        extrinsic = pose(.8, .12)
        np.testing.assert_allclose((graph.correction @ raw[-1]) @ extrinsic,
                                   graph.poses()[-1] @ extrinsic, atol=1e-9)

    def test_actual_retrieval_icp_incremental_loop_and_map_rebuild(self):
        cfg = configuration()
        cfg.update(submap_half_window=0, max_map_points=1000, map_every_keyframes=1)
        backend = OnlineBackend(cfg)
        world = scene()
        for stamp, actual_x, odom_x in ((0, 0, 0), (10, 2, 2.1), (20, 4, 4.2), (31, 0, .3)):
            local = transform(world, np.linalg.inv(pose(actual_x)))
            result = backend.ingest(stamp*10**9, pose(odom_x), local)
            self.assertEqual(result['status'], 'updated')
        self.assertEqual(result['loop_count'], 1, result['attempts'])
        self.assertLess(abs(result['poses'][-1, 0, 3]), .3)
        self.assertLessEqual(len(result['map_points']), cfg['max_map_points'])
        baseline_map = backend._display_map(np.array([k.odom for k in backend.keyframes]))
        self.assertGreater(np.max(np.abs(result['map_points']-baseline_map)), .01)
        for item in result['attempts']:
            self.assertLess(item['i'], item['j'])

    def test_invalid_cloud_and_out_of_order_never_mutate_graph(self):
        backend = OnlineBackend()
        self.assertEqual(backend.ingest(10**9, pose(), scene())['status'], 'updated')
        for stamp in (0, 10**9):
            self.assertEqual(backend.ingest(stamp, pose(), scene())['status'], 'nonmonotonic_stamp')
        bad = np.full((100, 3), np.nan)
        self.assertEqual(backend.ingest(2*10**9, pose(), bad)['status'], 'insufficient_points')
        with self.assertRaises(ValueError):
            backend.ingest(2*10**9, np.zeros((4, 4)), scene())
        self.assertEqual(len(backend.keyframes), 1)

    def test_session_capacity_is_explicit(self):
        cfg = configuration()
        cfg['max_keyframes'] = 2
        backend = OnlineBackend(cfg)
        for stamp in (0, 10**9):
            backend.ingest(stamp, pose(), scene())
        self.assertEqual(backend.ingest(2*10**9, pose(), scene())['status'], 'capacity_reached')
        self.assertEqual(len(backend.keyframes), 2)

    def test_skipped_keyframe_gap_increases_edge_uncertainty(self):
        cfg = configuration()
        backend = OnlineBackend(cfg)
        backend.ingest(0, pose(), scene())
        result = backend.ingest(4*10**9, pose(1.), scene())
        self.assertEqual(result['odometry_variance_scale'], 4.)
        factor = backend.graph.isam.getFactorsUnsafe().at(1)
        # Four nominal intervals must not receive the confidence of one interval.
        np.testing.assert_allclose(factor.noiseModel().sigmas(),
                                   backend.graph.odom_noise.sigmas()*2.)

    def test_pairs_are_exact_order_independent_and_bounded(self):
        pairs = ExactPairs(2)
        self.assertIsNone(pairs.add('pose', 100, 'p100'))
        self.assertIsNone(pairs.add('cloud', 101, 'c101'))
        self.assertEqual(pairs.add('cloud', 100, 'c100'), ('p100', 'c100'))
        self.assertEqual(pairs.add('pose', 101, 'p101'), ('p101', 'c101'))
        for stamp in range(10):
            pairs.add('cloud', stamp, str(stamp))
        self.assertEqual(len(pairs.clouds), 2)
        self.assertEqual(pairs.evicted, 8)

    def test_worker_process_round_trip_and_shutdown(self):
        client = BackendProcess(sys.executable, configuration())
        try:
            deadline = time.monotonic()+30
            while not client.ready.wait(.05):
                self.assertIsNone(client.error)
                self.assertLess(time.monotonic(), deadline)
            self.assertTrue(client.submit(10**9, pose(), scene()))
            while client.snapshot() is None:
                self.assertIsNone(client.error)
                self.assertLess(time.monotonic(), deadline)
                time.sleep(.01)
            self.assertEqual(client.snapshot()['keyframe'], 0)
            np.testing.assert_array_equal(client.snapshot()['correction'], np.eye(4))
        finally:
            client.close()
        self.assertIsNotNone(client.process.poll())
        self.assertFalse(client.thread.is_alive())

    def test_idle_worker_failure_is_reported_without_new_inputs(self):
        client = BackendProcess(sys.executable, configuration())
        try:
            self.assertTrue(client.ready.wait(30), client.error)
            client.process.terminate()
            client.process.wait(timeout=3)
            deadline = time.monotonic()+2
            while client.error is None and time.monotonic() < deadline:
                time.sleep(.01)
            self.assertIsNotNone(client.error)
            self.assertFalse(client.ready.is_set())
            self.assertFalse(client.submit(10**9, pose(), scene()))
        finally:
            client.close()

    def test_overload_keeps_latest_pending_frames(self):
        client = BackendProcess(sys.executable, configuration())
        try:
            self.assertTrue(client.ready.wait(30), client.error)
            os.kill(client.process.pid, signal.SIGSTOP)
            client.submit(10**9, pose(), scene())
            deadline = time.monotonic()+2
            while not client.pending.empty() and time.monotonic() < deadline:
                time.sleep(.01)
            for stamp in (2, 3, 4):
                self.assertTrue(client.submit(stamp*10**9, pose(), scene()))
            self.assertEqual(client.stats['queue_full'], 1)
            self.assertEqual([client.pending.get_nowait()['stamp_ns'] for _ in range(2)],
                             [3*10**9, 4*10**9])
        finally:
            if client.process.poll() is None:
                os.kill(client.process.pid, signal.SIGCONT)
            client.close()

    def test_watchdog_releases_partial_payload_recv(self):
        cfg = configuration()
        cfg['worker_timeout_s'] = 1.
        real_popen = __import__('subprocess').Popen
        def partial_worker(command, **kwargs):
            code = ('import os,struct,sys,time; from multiprocessing.connection import Connection; '
                    'fd=int(sys.argv[1]); c=Connection(fd); c.recv(); '
                    'c.send(dict(status="ready")); c.recv(); '
                    'os.write(fd,struct.pack(">i",128)+b"partial"); time.sleep(10)')
            return real_popen([sys.executable, '-c', code, command[-1]], **kwargs)
        with patch('tools.loop_closure.online_transport.subprocess.Popen', side_effect=partial_worker):
            client = BackendProcess(sys.executable, cfg)
        try:
            self.assertTrue(client.ready.wait(3), client.error)
            client.submit(10**9, pose(), scene())
            deadline = time.monotonic()+3
            while client.error is None and time.monotonic() < deadline:
                time.sleep(.01)
            self.assertIn('complete backend exchange', client.error)
            client.process.wait(timeout=2)
            self.assertFalse(client.ready.is_set())
        finally:
            client.close()

    def test_watchdog_cancellation_and_next_request_are_atomic(self):
        client = BackendProcess(sys.executable, configuration())
        try:
            self.assertTrue(client.ready.wait(30), client.error)
            for next_deadline in (None, time.monotonic()+20):
                with client.io_lock:
                    client.io_deadline = time.monotonic()+.05
                    time.sleep(.15)
                    client.io_deadline = next_deadline
                time.sleep(.1)
                self.assertIsNone(client.error)
                self.assertIsNone(client.process.poll())
        finally:
            client.close()


if __name__ == '__main__':
    unittest.main()
