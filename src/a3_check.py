#!/usr/bin/env python3
"""
a3_check.py -- EMPIRICAL verification of the T_I_L contract.

    T_W_L = T_W_I @ T_I_L      vs      T_W_L = T_W_I        (control)

Accumulates the same raw sweeps both ways (identical points, identical per-point
de-skew, identical everything except the right-multiplication) and measures map
sharpness.  Purely geometric -- PTv3 plays no part, so nothing semantic can
confound the result.

Metrics (higher sharpness = smaller number, except where noted)
  * local surface thickness : per 1 m cube with >= 30 points, sqrt of the
    smallest eigenvalue of the point covariance.  This is the standard
    "how thick is a surface that should be a sheet" measure and is orientation
    free, so it scores road, facade and pole alike.  Reported as the median and
    the mean over all qualifying cells.
  * ground-plane thickness  : robust std of z inside 2 x 2 m columns of the
    lowest-lying points, median over columns.
  * occupied voxels         : the same points rasterised at 10 cm.  A blurred
    map spreads the same returns over more voxels.

Why this discriminates: the omitted factor is a CONSTANT right-hand transform,
so while the vehicle drives straight it is a constant world offset and nothing
blurs.  It only shows up where R_W_I(t) changes, because the error becomes
R_W_I(t) @ t_IL.  The window is therefore centred on the sharpest turn of
seq07 (107 deg over 100 sweeps).
"""
import argparse
import json
import sys
import numpy as np

sys.path.insert(0, "/data/livo_sem/src")
import sem_core as S


def read_scans(bag, topic, i0, i1):
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import PointCloud2
    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=bag, storage_id="mcap"),
           rosbag2_py.ConverterOptions("", ""))
    r.set_filter(rosbag2_py.StorageFilter(topics=[topic]))
    out = []
    k = 0
    while r.has_next():
        _, d, _ = r.read_next()
        if k >= i1:
            break
        if k >= i0:
            m = deserialize_message(d, PointCloud2)
            raw = np.frombuffer(m.data, dtype=np.uint8).reshape(-1, m.point_step)
            xyz = raw[:, 0:12].copy().view(np.float32).reshape(-1, 3).astype(np.float64)
            toff = raw[:, 16:20].copy().view(np.float32).ravel().astype(np.float64)
            t0 = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
            out.append((xyz, t0 + toff * 1e-6))
        k += 1
    return out


def accumulate(scans, traj, T_IL, deskew=True):
    P = []
    for xyz, tpt in scans:
        ok = traj.valid(tpt)
        if ok.sum() < 100:
            continue
        x, t = xyz[ok], tpt[ok]
        if deskew:
            R, p, _ = traj.query(t)
        else:
            R1, p1, _ = traj.query(np.array([t.mean()]))
            R = np.broadcast_to(R1[0], (len(x), 3, 3))
            p = np.broadcast_to(p1[0], (len(x), 3))
        h = np.hstack([x, np.ones((len(x), 1))])
        b = (h @ T_IL.T)[:, :3]
        P.append(np.einsum("nij,nj->ni", R, b) + p)
    return np.concatenate(P)


def cell_moments(P, cell):
    g = np.floor(P / cell).astype(np.int64)
    g -= g.min(0)
    e = g.max(0) + 1
    key = (g[:, 0] * e[1] + g[:, 1]) * e[2] + g[:, 2]
    uk, inv = np.unique(key, return_inverse=True)
    m = len(uk)
    n = np.bincount(inv, minlength=m).astype(np.float64)
    s = np.stack([np.bincount(inv, weights=P[:, i], minlength=m) for i in range(3)], 1)
    idx = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
    q = {}
    for a, b in idx:
        q[(a, b)] = np.bincount(inv, weights=P[:, a] * P[:, b], minlength=m)
    return n, s, q, m


def surface_thickness(P, cell=1.0, min_pts=30):
    n, s, q, m = cell_moments(P, cell)
    keep = n >= min_pts
    n, s = n[keep], s[keep]
    mu = s / n[:, None]
    C = np.empty((keep.sum(), 3, 3))
    for a in range(3):
        for b in range(a, 3):
            v = q[(a, b)][keep] / n - mu[:, a] * mu[:, b]
            C[:, a, b] = v
            C[:, b, a] = v
    C += np.eye(3) * 1e-12
    w = np.linalg.eigvalsh(C)
    th = np.sqrt(np.maximum(w[:, 0], 0.0))
    return th, int(keep.sum())


def ground_thickness(P, cell=2.0, min_pts=40, band=0.6):
    """Lowest-lying returns in each 2x2 m column: robust std of z."""
    g = np.floor(P[:, :2] / cell).astype(np.int64)
    g -= g.min(0)
    key = g[:, 0] * (g[:, 1].max() + 1) + g[:, 1]
    order = np.argsort(key, kind="stable")
    ks, zs = key[order], P[order, 2]
    first = np.ones(len(ks), bool)
    first[1:] = ks[1:] != ks[:-1]
    starts = np.nonzero(first)[0]
    ends = np.append(starts[1:], len(ks))
    out = []
    for a, b in zip(starts, ends):
        if b - a < min_pts:
            continue
        z = zs[a:b]
        zl = np.percentile(z, 5)
        sel = z[z < zl + band]
        if len(sel) < min_pts // 2:
            continue
        med = np.median(sel)
        out.append(1.4826 * np.median(np.abs(sel - med)))
    return np.array(out)


def occupied(P, v):
    return len(np.unique(S.voxel_key(P, v)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", default="/data/livo_sem/bags/kitti_seq07_us")
    ap.add_argument("--traj", default="/data/livo_sem/out/kitti_seq07_fastlivo2_tum.txt")
    ap.add_argument("--i0", type=int, default=243)
    ap.add_argument("--n", type=int, default=110)
    ap.add_argument("--out", default="/data/livo_sem/out/a3_T_I_L.json")
    a = ap.parse_args()

    traj = S.TrajInterp(a.traj)
    scans = read_scans(a.bag, "/velodyne_points", a.i0, a.i0 + a.n)
    print("scans loaded: %d  (%d points total)" % (len(scans), sum(len(s[0]) for s in scans)))
    t_lo = min(s[1].min() for s in scans)
    t_hi = max(s[1].max() for s in scans)
    R, p, _ = traj.query(np.array([t_lo, t_hi]))
    yaw = np.degrees(np.arctan2(R[:, 1, 0], R[:, 0, 0]))
    dy = (yaw[1] - yaw[0] + 180) % 360 - 180
    print("window t [%.3f, %.3f]  |yaw change| = %.1f deg" % (t_lo, t_hi, abs(dy)))

    res = {"window": dict(i0=a.i0, n=len(scans), t_lo=t_lo, t_hi=t_hi,
                          yaw_change_deg=float(abs(dy)))}
    for name, T in (("with_T_I_L", S.T_I_L), ("without_T_I_L", np.eye(4))):
        P = accumulate(scans, traj, T, deskew=True)
        th, ncell = surface_thickness(P, cell=1.0, min_pts=30)
        gt = ground_thickness(P)
        r = dict(
            n_points=int(len(P)),
            surf_thickness_median_m=float(np.median(th)),
            surf_thickness_mean_m=float(th.mean()),
            surf_thickness_p75_m=float(np.percentile(th, 75)),
            n_cells=ncell,
            ground_thickness_median_m=float(np.median(gt)),
            ground_thickness_mean_m=float(gt.mean()),
            n_ground_columns=int(len(gt)),
            occupied_voxels_10cm=int(occupied(P, 0.10)),
            occupied_voxels_20cm=int(occupied(P, 0.20)),
        )
        res[name] = r
        print("\n[%s]" % name)
        for k, v in r.items():
            print("   %-28s %s" % (k, ("%.4f" % v) if isinstance(v, float) else v))

    w, o = res["with_T_I_L"], res["without_T_I_L"]
    res["verdict"] = dict(
        surf_thickness_ratio_without_over_with=o["surf_thickness_median_m"] / w["surf_thickness_median_m"],
        ground_thickness_ratio=o["ground_thickness_median_m"] / w["ground_thickness_median_m"],
        occupied_10cm_ratio=o["occupied_voxels_10cm"] / w["occupied_voxels_10cm"],
        with_T_I_L_is_sharper=bool(
            w["surf_thickness_median_m"] < o["surf_thickness_median_m"] and
            w["occupied_voxels_10cm"] < o["occupied_voxels_10cm"]),
    )
    print("\nVERDICT", json.dumps(res["verdict"], indent=2))
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2)
    print("-> %s" % a.out)


if __name__ == "__main__":
    main()
