#!/usr/bin/env python3
"""Plumbing probe: an unambiguous synthetic street scene in the nuScenes sensor frame.
If the model labels the ground driveable_surface and the car-sized box car, the wiring
is correct and the KITTI numbers are a domain gap, not a preprocessing bug."""
import sys, numpy as np
sys.path.insert(0, "/data/livo_sem/src")
from ptv3_infer import PTv3Segmenter, NUSCENES16, read_bin

H = 1.84          # nuScenes LIDAR_TOP height -> ground at z = -H in sensor frame
rng = np.random.default_rng(0)

def plane(xr, yr, z, step=0.10):
    x = np.arange(*xr, step); y = np.arange(*yr, step)
    X, Y = np.meshgrid(x, y)
    return np.stack([X.ravel(), Y.ravel(), np.full(X.size, z)], 1)

def box(cx, cy, L, W, Ht, step=0.05, zb=-H):
    pts = []
    xs = np.arange(-L/2, L/2, step); ys = np.arange(-W/2, W/2, step); zs = np.arange(0, Ht, step)
    for s, e in ((ys, zs), (ys, zs)):
        Y, Z = np.meshgrid(s, e); n = Y.size
        pts.append(np.stack([np.full(n, -L/2), Y.ravel(), Z.ravel()], 1))
        pts.append(np.stack([np.full(n, +L/2), Y.ravel(), Z.ravel()], 1))
        break
    X, Z = np.meshgrid(xs, zs); n = X.size
    pts.append(np.stack([X.ravel(), np.full(n, -W/2), Z.ravel()], 1))
    pts.append(np.stack([X.ravel(), np.full(n, +W/2), Z.ravel()], 1))
    X, Y = np.meshgrid(xs, ys); n = X.size
    pts.append(np.stack([X.ravel(), Y.ravel(), np.full(n, Ht)], 1))
    p = np.concatenate(pts, 0)
    p[:, 0] += cx; p[:, 1] += cy; p[:, 2] += zb
    return p

parts = {}
parts["ground"] = plane((-35, 35), (-9, 9), -H, 0.12)
parts["sidewalk_L"] = plane((-35, 35), (9, 12), -H + 0.12, 0.12)
parts["sidewalk_R"] = plane((-35, 35), (-12, -9), -H + 0.12, 0.12)
parts["building_L"] = plane((-35, 35), (-H, 8), 0, 0.12)[:, [0, 2, 1]] * np.array([1, 1, 1])
parts["building_L"] = np.stack([parts["building_L"][:, 0], np.full(len(parts["building_L"]), 12.0),
                                parts["building_L"][:, 1] - 0], 1)
bl = plane((-35, 35), (0, 9), 0, 0.12)
parts["building_L"] = np.stack([bl[:, 0], np.full(len(bl), 12.0), bl[:, 1] - H], 1)
parts["building_R"] = np.stack([bl[:, 0], np.full(len(bl), -12.0), bl[:, 1] - H], 1)
parts["car1"] = box(9, 3.0, 4.5, 1.9, 1.5)
parts["car2"] = box(-11, -3.0, 4.5, 1.9, 1.5)

names, pts = [], []
for k, v in parts.items():
    v = v + rng.normal(0, 0.01, v.shape)
    pts.append(v); names += [k] * len(v)
pts = np.concatenate(pts, 0).astype(np.float32)
names = np.array(names)
r = np.linalg.norm(pts[:, :2], axis=1)
keep = r < 40
pts, names = pts[keep], names[keep]
inten = np.full((len(pts), 1), 0.30, np.float32)
scan = np.concatenate([pts, inten], 1).astype(np.float32)
print("synthetic scene: %d points" % len(scan))

seg = PTv3Segmenter(amp=False)
lb, cf = seg.segment(scan)
for k in ["ground", "sidewalk_L", "building_L", "car1", "car2"]:
    m = names == k
    c = np.bincount(lb[m].astype(np.int64), minlength=16); o = np.argsort(-c)[:3]
    print("  %-12s n=%6d -> %s" % (k, m.sum(),
          "  ".join("%s %.0f%%" % (NUSCENES16[i], 100*c[i]/m.sum()) for i in o)))
