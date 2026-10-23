"""Contract tests: graph direction, frame conversion, no-loop control, yaw retrieval."""
import unittest
import numpy as np
from scipy.spatial.transform import Rotation

from .common import S, corrected_tum, matrices, protocol, transform
from .backend import descriptor, descriptor_distance, optimize, cloud, register


class Contracts(unittest.TestCase):
    def test_world_correction_preserves_imu_lidar_extrinsic(self):
        tum = np.array([[1, 1, 2, 3, 0, 0, 0, 1], [2, 4, 5, 6, 0, 0, 0, 1]], float)
        original = matrices(tum) @ S.T_I_L
        c = np.eye(4)
        c[:3, :3] = Rotation.from_euler('z', .7).as_matrix()
        c[:3, 3] = [3, -2, 1]
        corrected = corrected_tum(tum, tum[:, 0], original, c@original)
        np.testing.assert_allclose(matrices(corrected)@S.T_I_L, c@original, atol=1e-8)
        np.testing.assert_array_equal(corrected_tum(tum, tum[:, 0], original, original), tum)

    def test_odometry_only_graph_and_loop_direction(self):
        poses = np.tile(np.eye(4), (4, 1, 1))
        poses[:, 0, 3] = [0, 1, 2, .3]
        no_loop, _ = optimize(poses, [], protocol())
        np.testing.assert_allclose(no_loop, poses, atol=1e-8)
        # The last pose returns to the origin, measurement T_0_3 = identity.
        closed, _ = optimize(poses, [(0, 3, np.eye(4))], protocol())
        self.assertLess(abs(closed[3, 0, 3]), .3)
        np.testing.assert_allclose(closed[0], poses[0], atol=1e-6)

    def test_scan_context_rotation(self):
        rng = np.random.default_rng(71)
        points = rng.normal(size=(10000, 3))*[20, 20, 4]
        a = descriptor(points, protocol())
        b = np.roll(a, 13, axis=1)
        distance, shift = descriptor_distance(a, b)
        self.assertAlmostEqual(distance, 0., places=10)
        self.assertEqual(shift, 47)
        self.assertEqual(descriptor_distance(np.zeros_like(a), a)[0], 1.)

    def test_icp_transform_direction_on_three_planes(self):
        rng = np.random.default_rng(42)
        p = rng.uniform(-5, 5, (12000, 3))
        p[:4000, 0] = -4
        p[4000:8000, 1] = 3
        p[8000:, 2] = -2
        true = np.eye(4)
        true[:3, :3] = Rotation.from_euler('xyz', [.01, -.02, .03]).as_matrix()
        true[:3, 3] = [.2, -.15, .1]
        target = cloud(p, .15)
        source = cloud(transform(p, np.linalg.inv(true)), .15)
        estimated, stats = register(source, target, np.eye(4), protocol())
        self.assertTrue(stats['accepted'], stats)
        np.testing.assert_allclose(estimated, true, atol=.01)


if __name__ == '__main__':
    unittest.main()
