"""Frame/time adapters shared by the offline backend and its separate evaluators."""
import calendar
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'src'))
import sem_core as S
from kitti_scan import synth

DRIVE = '2011_09_30_drive_0027_sync'


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def protocol():
    return json.loads(Path(__file__).with_name('protocol.json').read_text())


def timestamps(path):
    values = []
    for line in Path(path).read_text().splitlines():
        date, time = line.split()
        hms, frac = time.split('.')
        y, mo, d = map(int, date.split('-'))
        h, mi, sec = map(int, hms.split(':'))
        values.append(calendar.timegm((y, mo, d, h, mi, sec, 0, 0, 0)) * 10**9
                      + int((frac + '000000000')[:9]))
    # Match the archived replay's integer-nanosecond conversion exactly.
    return np.asarray(values, np.int64)


def transform(points, pose):
    return np.asarray(points) @ pose[:3, :3].T + pose[:3, 3]


def rigid_pose(pose):
    """Project rounded calibration rotations onto SO(3) for rigid graph math.

    Keep sem_core's frozen decimal extrinsic untouched for historical replay.
    GTSAM's Rot3 inverse assumes orthonormality, unlike numpy.linalg.inv.
    """
    result = np.array(pose, dtype=np.float64, copy=True)
    result[:3, :3] = Rotation.from_matrix(result[:3, :3]).as_matrix()
    return result


def matrices(tum):
    out = np.tile(np.eye(4), (len(tum), 1, 1))
    out[:, :3, :3] = Rotation.from_quat(tum[:, 4:8]).as_matrix()
    out[:, :3, 3] = tum[:, 1:4]
    return out


def check_time_tables(relative, image_times):
    """Audit the auxiliary odometry clock without replacing the raw camera clock.

    KITTI seq07 has a real 4.788 ms discrepancy in times.txt. The frozen ATE
    evaluator logs it and uses raw image timestamps. Reject an ambiguity of
    half a frame, not a sub-frame clock fluctuation.
    """
    assert len(relative) == len(image_times)
    assert np.all(np.diff(relative)>0) and np.all(np.diff(image_times)>0)
    difference = (relative-relative[0])-(image_times-image_times[0])
    limit = .5*min(float(np.min(np.diff(relative))), float(np.min(np.diff(image_times))))
    maximum = float(np.max(np.abs(difference)))
    assert maximum < limit, (maximum, limit)
    return dict(max_difference_s=maximum, half_frame_bound_s=limit,
                association_clock='raw image_02 timestamps', samples=len(relative))


def interpolate(times, poses, query):
    # Offset absolute UNIX times to avoid losing precision in Slerp internals.
    query = np.clip(query, times[0], times[-1])
    out = np.tile(np.eye(4), (len(query), 1, 1))
    out[:, :3, :3] = Slerp(times-times[0], Rotation.from_matrix(poses[:, :3, :3]))(
        query-times[0]).as_matrix()
    for k in range(3):
        out[:, k, 3] = np.interp(query, times, poses[:, k, 3])
    return out


def corrected_tum(tum, key_times, original_lidar, optimized_lidar):
    """Interpolate WORLD-frame corrections and left-multiply IMU poses.

    (C @ T_W_I) @ T_I_L = C @ T_W_L. No right-side frame conversion
    is lost. Outside keyframe coverage use the nearest correction, not motion
    extrapolation. Original timestamps and sample count are retained.
    """
    if np.array_equal(original_lidar, optimized_lidar):
        return tum.copy()
    corrections = optimized_lidar @ np.linalg.inv(original_lidar)
    corrected = interpolate(key_times, corrections, tum[:, 0]) @ matrices(tum)
    return np.column_stack([tum[:, 0], corrected[:, :3, 3],
                            Rotation.from_matrix(corrected[:, :3, :3]).as_quat()])


class Scans:
    def __init__(self, root, trajectory):
        self.root = Path(root)
        self.raw = self.root / 'data/raw/2011_09_30' / DRIVE
        self.starts = timestamps(self.raw / 'velodyne_points/timestamps_start.txt')
        self.ends = timestamps(self.raw / 'velodyne_points/timestamps_end.txt')
        self.images = timestamps(self.raw / 'image_02/timestamps.txt') * 1e-9
        self.traj = S.TrajInterp(trajectory)

    def read(self, frame):
        pts = np.fromfile(self.raw / f'velodyne_points/data/{frame:010d}.bin', np.float32).reshape(-1, 4)
        _, relative, order = synth(pts[:, :3], sweep_duration=(self.ends[frame]-self.starts[frame])/1e9)
        return pts, order, self.starts[frame] * 1e-9 + relative

    def world(self, sorted_xyz, times, keep):
        """Archived map_eval's 128-bin deskew, including its confidence-gate bin endpoints."""
        if not np.any(keep):
            return np.empty((0, 3)), np.empty(0, np.int64)
        valid = self.traj.valid(times)
        idx = np.flatnonzero(valid)
        lo, hi = times[keep][[0, -1]]
        bins = protocol()['deskew_bins']
        centers = lo + (np.arange(bins) + .5) * (hi-lo) / bins
        rot, pos, _ = self.traj.query(centers)
        bi = np.clip(((times[idx]-lo) / max(hi-lo, 1e-9) * bins).astype(np.int64), 0, bins-1)
        edges = np.searchsorted(bi, np.arange(bins+1))
        pi = transform(sorted_xyz[idx].astype(np.float64), S.T_I_L)
        pw = np.empty_like(pi)
        for k, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
            if b > a:
                np.matmul(pi[a:b], rot[k].T, out=pw[a:b])
                pw[a:b] += pos[k]
        return pw, idx

    def local(self, frame):
        pts, order, times = self.read(frame)
        valid = self.traj.valid(times)
        pose, ok = self.traj.query_one(self.images[frame])
        if not ok or valid.sum() < 100:
            raise ValueError(f'No trajectory support for keyframe {frame}')
        world, _ = self.world(pts[order, :3], times, valid)
        lidar = rigid_pose(pose @ S.T_I_L)
        return transform(world, np.linalg.inv(lidar)), lidar
