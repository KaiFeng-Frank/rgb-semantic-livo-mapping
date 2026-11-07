"""Causal, bounded-session Pose3 backend. Only the worker imports GTSAM/Open3D.

Input: a sensor-time pose T_odom_body and deskewed points in that SAME body frame.
Output: T_map_odom, optimized keyframes and a bounded geometric display map.
No GT, semantic labels, frontend state mutation or historical interpolation.
"""
from dataclasses import dataclass
import time

import gtsam
import numpy as np

from .backend import cloud, descriptor, descriptor_distance, register
from .common import transform
from .online_transport import load_settings


def configuration(path=None):
    return validate_configuration(load_settings(path))


def validate_configuration(cfg):
    for key in ('max_keyframes', 'max_points_per_keyframe', 'max_map_points',
                'map_every_keyframes', 'max_icp_attempts', 'pending_keyframes',
                'pair_cache_size', 'minimum_points', 'loop_update_passes'):
        if not isinstance(cfg[key], int) or cfg[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    for key in ('keyframe_interval_s', 'max_queue_age_s', 'worker_timeout_s',
                'relinearize_threshold'):
        if not np.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f'{key} must be positive and finite')
    if cfg['minimum_points'] > cfg['max_points_per_keyframe']:
        raise ValueError('minimum_points exceeds point cap')
    return cfg


def checked_pose(pose):
    pose = np.asarray(pose, dtype=np.float64)
    if (pose.shape != (4, 4) or not np.isfinite(pose).all()
            or not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-9)
            or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-6)
            or abs(np.linalg.det(pose[:3, :3])-1) > 1e-6):
        raise ValueError('expected a finite rigid T_odom_body')
    return pose.copy()


@dataclass
class Keyframe:
    stamp_ns: int
    odom: np.ndarray
    points: np.ndarray
    description: np.ndarray


class IncrementalGraph:
    """Raw odometry defines edges; corrected poses only initialize new variables."""
    def __init__(self, cfg):
        params = gtsam.ISAM2Params()
        params.setRelinearizeThreshold(cfg['relinearize_threshold'])
        params.relinearizeSkip = 1
        self.isam = gtsam.ISAM2(params)
        diag = gtsam.noiseModel.Diagonal.Sigmas
        self.prior_noise = diag(np.full(6, 1e-6))
        self.odom_noise = diag(np.r_[np.full(3, np.deg2rad(cfg['odometry_rotation_sigma_deg'])),
                                     np.full(3, cfg['odometry_translation_sigma_m'])])
        self.loop_noise = gtsam.noiseModel.Robust.Create(
            gtsam.noiseModel.mEstimator.Huber.Create(cfg['loop_huber_sigma']),
            diag(np.r_[np.full(3, np.deg2rad(cfg['loop_rotation_sigma_deg'])),
                       np.full(3, cfg['loop_translation_sigma_m'])]))
        self.cfg, self.odom = cfg, []
        self.correction = np.eye(4)
        self.loop_count = 0

    def append(self, odom, variance_scale=1.):
        odom = checked_pose(odom)
        if not np.isfinite(variance_scale) or variance_scale < 1:
            raise ValueError('odometry variance scale must be finite and >= 1')
        j = len(self.odom)
        if j >= self.cfg['max_keyframes']:
            raise ValueError('session keyframe capacity reached')
        graph, values = gtsam.NonlinearFactorGraph(), gtsam.Values()
        values.insert(j, gtsam.Pose3(self.correction @ odom))
        if j:
            noise = gtsam.noiseModel.Diagonal.Sigmas(self.odom_noise.sigmas()*np.sqrt(variance_scale))
            graph.add(gtsam.BetweenFactorPose3(j-1, j,
                gtsam.Pose3(np.linalg.inv(self.odom[-1]) @ odom), noise))
        else:
            graph.add(gtsam.PriorFactorPose3(0, gtsam.Pose3(odom), self.prior_noise))
        self.isam.update(graph, values)
        self.odom.append(odom)
        self._correction()
        return j

    def add_loop(self, historical, current, measurement):
        if not 0 <= historical < current < len(self.odom):
            raise ValueError('loop must connect existing ordered keys')
        graph = gtsam.NonlinearFactorGraph()
        graph.add(gtsam.BetweenFactorPose3(historical, current,
                  gtsam.Pose3(checked_pose(measurement)), self.loop_noise))
        self.isam.update(graph, gtsam.Values())
        for _ in range(self.cfg['loop_update_passes']):
            self.isam.update()
        self.loop_count += 1
        self._correction()

    def _correction(self):
        # Preserve exact identity before the first loop, avoiding numerical drift.
        if self.loop_count:
            j = len(self.odom)-1
            self.correction = self.isam.calculateEstimatePose3(j).matrix() @ np.linalg.inv(self.odom[j])

    def poses(self):
        if not self.loop_count:
            return np.asarray(self.odom).copy()
        values = self.isam.calculateEstimate()
        return np.stack([values.atPose3(i).matrix() for i in range(len(self.odom))])


class OnlineBackend:
    def __init__(self, cfg=None):
        self.cfg = configuration() if cfg is None else validate_configuration(cfg)
        self.graph = IncrementalGraph(self.cfg)
        self.keyframes = []
        self.last_loop = -self.cfg['loop_cooldown_keyframes']
        self.last_stamp = -1
        self.version = 0
        self.previous_optimized = np.empty((0, 4, 4))

    def ingest(self, stamp_ns, pose, points):
        start = time.monotonic()
        if not isinstance(stamp_ns, (int, np.integer)) or stamp_ns < 0:
            raise ValueError('stamp must be a nonnegative integer in nanoseconds')
        if stamp_ns <= self.last_stamp:
            return dict(status='nonmonotonic_stamp', stamp_ns=int(stamp_ns))
        if len(self.keyframes) >= self.cfg['max_keyframes']:
            return dict(status='capacity_reached', stamp_ns=int(stamp_ns))
        if self.keyframes and stamp_ns-self.keyframes[-1].stamp_ns < self.cfg['keyframe_interval_s']*1e9:
            return dict(status='not_keyframe', stamp_ns=int(stamp_ns))
        pose = checked_pose(pose)
        points = np.asarray(points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) > self.cfg['max_points_per_keyframe']:
            raise ValueError('expected capped Nx3 deskewed body points')
        points = points[np.isfinite(points).all(axis=1)]
        if len(points) < self.cfg['minimum_points']:
            return dict(status='insufficient_points', stamp_ns=int(stamp_ns))
        description = descriptor(points, self.cfg)
        points = np.asarray(cloud(points, self.cfg['icp_voxel_m']).points, dtype=np.float32)
        if len(points) < self.cfg['minimum_points']:
            return dict(status='insufficient_voxels', stamp_ns=int(stamp_ns))
        variance_scale = max(1., (stamp_ns-self.last_stamp)*1e-9/self.cfg['keyframe_interval_s']) if self.keyframes else 1.
        j = self.graph.append(pose, variance_scale)
        self.keyframes.append(Keyframe(int(stamp_ns), pose, points, description))
        self.last_stamp = stamp_ns
        graph_ms = (time.monotonic()-start)*1000
        attempts = self._try_loop(j)
        loop = any(a['accepted'] for a in attempts)
        optimized = self.graph.poses()
        # iSAM2 may refine old poses on subsequent odometry insertions as well.
        # Revision/map invalidation must follow changed history, not only a new
        # loop edge; otherwise path/map consumers can retain inconsistent poses.
        history_changed = bool(len(self.previous_optimized) and not np.allclose(
            optimized[:-1], self.previous_optimized, atol=1e-10, rtol=0))
        if loop or history_changed:
            self.version += 1
        self.previous_optimized = optimized.copy()
        display = None
        if loop or history_changed or j % self.cfg['map_every_keyframes'] == 0:
            display = self._display_map(optimized)
        return dict(status='updated', stamp_ns=int(stamp_ns), keyframe=j,
                    version=self.version, loop_count=self.graph.loop_count,
                    correction=self.graph.correction.copy(),
                    times_ns=np.array([k.stamp_ns for k in self.keyframes], dtype=np.int64),
                    poses=optimized, map_points=display, attempts=attempts,
                    odometry_variance_scale=variance_scale,
                    graph_and_cloud_ms=graph_ms, total_ms=(time.monotonic()-start)*1000)

    def _try_loop(self, j):
        cfg, current = self.cfg, self.keyframes[j]
        if j-self.last_loop < cfg['loop_cooldown_keyframes']:
            return []
        # At most max_keyframes entries; all are already observed. Candidate
        # radius deliberately retains the existing odometry-limited policy.
        eligible = [i for i, k in enumerate(self.keyframes[:j])
                    if (current.stamp_ns-k.stamp_ns)*1e-9 > cfg['minimum_loop_seconds']
                    and np.linalg.norm(k.odom[:3, 3]-current.odom[:3, 3]) < cfg['maximum_candidate_distance_m']]
        if not eligible:
            return []
        ring = current.description.mean(axis=1)
        candidates = sorted(eligible, key=lambda i: np.linalg.norm(
            self.keyframes[i].description.mean(axis=1)-ring))[:cfg['ring_candidates']]
        scored = sorted((*descriptor_distance(self.keyframes[i].description, current.description), i)
                        for i in candidates)
        attempts, registrations = [], 0
        # Use the current graph estimate consistently for historical submaps,
        # while odometry edges remain based on the uncorrected frontend poses.
        poses = self.graph.poses()
        for distance, shift, i in scored:
            record = dict(i=i, j=j, descriptor_distance=distance, yaw_shift=shift, accepted=False)
            if distance > cfg['scan_context_threshold']:
                record['rejection'] = 'descriptor'
                attempts.append(record)
                continue
            if registrations >= cfg['max_icp_attempts']:
                break
            registrations += 1
            inv = np.linalg.inv(poses[i])
            half = cfg['submap_half_window']
            pieces = [transform(self.keyframes[k].points, inv @ poses[k])
                      for k in range(max(0, i-half), min(j, i+half+1))
                      if (current.stamp_ns-self.keyframes[k].stamp_ns)*1e-9 > cfg['minimum_loop_seconds']]
            if not pieces:
                continue
            measurement, stats = register(cloud(current.points, cfg['icp_voxel_m']),
                cloud(np.concatenate(pieces), cfg['icp_voxel_m']), inv @ poses[j], cfg)
            record.update(stats)
            record['T_historical_current'] = measurement.tolist()
            attempts.append(record)
            if stats['accepted']:
                self.graph.add_loop(i, j, measurement)
                self.last_loop = j
                break
        return attempts

    def _display_map(self, poses):
        # Cap BEFORE concatenate. Each keyframe stays in its own body frame so
        # loop deformation moves old geometry, not just the newest cloud/TF.
        total = sum(len(k.points) for k in self.keyframes)
        stride = max(1, int(np.ceil(total/self.cfg['max_map_points'])))
        pieces, offset = [], 0
        for k, pose in zip(self.keyframes, poses):
            start = (-offset) % stride
            pieces.append(transform(k.points[start::stride], pose))
            offset += len(k.points)
        return np.asarray(np.concatenate(pieces), dtype=np.float32)
