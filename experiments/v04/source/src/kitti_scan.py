#!/usr/bin/env python3
"""
kitti_scan.py -- synthesise the `ring` and `time` fields that KITTI .bin files
do not carry, and put the points in the order a Velodyne driver would emit.

WHY
---
KITTI velodyne_points/data/*.bin holds only (x, y, z, intensity) float32.
FAST-LIVO2's velodyne_handler does pcl::fromROSMsg into

    velodyne_ros::Point { float x, y, z, intensity, time; uint16 ring; }

so both `ring` and `time` have to be reconstructed.

RING
----
KITTI stores a scan LASER-MAJOR: all points of laser 0 for one full
revolution, then all points of laser 1, and so on.  Measured on
2011_09_30_drive_0016 frame 0: the azimuth atan2(y, x) is monotonically
INCREASING in file order and crosses +-pi exactly 64 times, and the total
unwrapped azimuth span is 63.90 revolutions.  So:

    d      = unwrapped diff of atan2(y, x)          (>= 0 everywhere)
    cum    = cumulative sum, starting at 0
    ring   = floor(cum / 2pi)                        -> 0 .. 63
    phase  = cum - ring * 2pi                        -> [0, 2pi)

This recovers exactly 64 rings whose mean elevation angles decrease
monotonically from +2.35 deg to -23.66 deg, i.e. the HDL-64E vertical FOV
(+2 .. -24.8 deg).  Binning by elevation would NOT work: HDL-64E spacing is
non-uniform (~1/3 deg upper block, ~1/2 deg lower block).

Note the direction: azimuth INCREASES in KITTI file order, so the per-point
time is (az - az_start), not (az_start - az).

TIME
----
KITTI ships velodyne_points/timestamps_start.txt and timestamps_end.txt.
Measured: end_i == start_{i+1} to 1.4 us, and end_i - start_i = 0.104105 s
(mean over drive 0016) -- the HDL-64E is spinning at 9.606 Hz, NOT 10 Hz.
timestamps.txt is exactly the arithmetic midpoint of start and end and carries
no azimuth information.

Therefore the i-th .bin holds exactly the points acquired in [start_i, end_i]
and the per-point offset from the start of the sweep is

    t = phase / (2pi) * (end_i - start_i)            -> [0, 0.1041)

ORDER
-----
The points are re-sorted by t (azimuth-major, all 64 lasers interleaved),
which is what a real Velodyne driver emits, because FAST-LIVO2's
ImuProcess::UndistortPcl walks the cloud BACKWARD
(IMU_Processing.cpp:514, `for (; it_pcl->curvature/1000 > head->offset_time; it_pcl--)`)
and its sort() at IMU_Processing.cpp:286 is COMMENTED OUT.  A laser-major
cloud would break undistortion silently.

`synth()` also returns the permutation, so SemanticKITTI .label files (which
are indexed in the ORIGINAL .bin order) can still be matched:

    ring, t, order = synth(pts)
    labels_in_bag_order = labels_from_file[order]
"""

import numpy as np

__all__ = ["synth", "N_RINGS", "NOMINAL_SWEEP"]

N_RINGS = 64
NOMINAL_SWEEP = 0.104105          # s, measured mean of end-start on 2011_09_30


def synth(pts_xyz, sweep_duration=NOMINAL_SWEEP, n_rings=N_RINGS, reorder=True):
    """
    Parameters
    ----------
    pts_xyz        (N,>=3) float array, velodyne frame, in ORIGINAL .bin order
    sweep_duration seconds for one full revolution (use end_i - start_i)
    reorder        if True, return everything sorted by time (driver order)

    Returns
    -------
    ring   (N,) uint16   0 .. n_rings-1
    t      (N,) float64  seconds since the start of the sweep, [0, sweep)
    order  (N,) int64    permutation applied (identity if reorder=False);
                         bag_order_array = original_array[order]
    """
    p = np.asarray(pts_xyz, dtype=np.float64)
    x, y = p[:, 0], p[:, 1]
    az = np.arctan2(y, x)

    d = np.diff(az)
    d = np.where(d < -np.pi, d + 2.0 * np.pi, d)
    d = np.where(d > np.pi, d - 2.0 * np.pi, d)
    # a handful of samples per scan step backwards by <0.001 deg (encoder
    # quantisation); clamp so the cumulative azimuth is strictly monotonic and
    # the ring index can never flicker at a 2pi boundary.
    np.clip(d, 0.0, None, out=d)

    cum = np.empty(p.shape[0], dtype=np.float64)
    cum[0] = 0.0
    np.cumsum(d, out=cum[1:])

    two_pi = 2.0 * np.pi
    ring = np.floor(cum / two_pi).astype(np.int64)
    np.clip(ring, 0, n_rings - 1, out=ring)
    phase = cum - ring * two_pi
    np.clip(phase, 0.0, two_pi * (1.0 - 1e-12), out=phase)

    t = phase / two_pi * sweep_duration

    if reorder:
        order = np.argsort(t, kind="stable")
        return ring[order].astype(np.uint16), t[order], order

    order = np.arange(p.shape[0], dtype=np.int64)
    return ring.astype(np.uint16), t, order
