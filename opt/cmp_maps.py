#!/usr/bin/env python3
"""Compare two published maps voxel by voxel.  The snapshot does not store the
hash key, but the centroid always lies inside its own voxel, so floor(xyz/voxel)
recovers the voxel index exactly and the two maps can be joined on it."""
import sys, numpy as np
V = 0.20
a = np.load(sys.argv[1]); b = np.load(sys.argv[2])

def key(d):
    g = np.floor(d["xyz"].astype(np.float64) / V).astype(np.int64)
    return (g[:, 0] + (1 << 20)) * (1 << 42) + (g[:, 1] + (1 << 20)) * (1 << 21) + (g[:, 2] + (1 << 20))

ka, kb = key(a), key(b)
print("voxels: A %d   B %d" % (len(ka), len(kb)))
ua, ia = np.unique(ka, return_index=True)
ub, ib = np.unique(kb, return_index=True)
common, ja, jb = np.intersect1d(ua, ub, return_indices=True)
print("shared voxels: %d  (A-only %d, B-only %d)" % (len(common), len(ua) - len(common), len(ub) - len(common)))
A = {k: a[k][ia][ja] for k in ("xyz", "rgb", "cls", "conf", "has_rgb")}
B = {k: b[k][ib][jb] for k in ("xyz", "rgb", "cls", "conf", "has_rgb")}
same_cls = (A["cls"] == B["cls"])
print("class      identical : %.6f%%  (%d differ)" % (100 * same_cls.mean(), (~same_cls).sum()))
print("has_rgb    identical : %.6f%%" % (100 * (A["has_rgb"] == B["has_rgb"]).mean()))
print("rgb        identical : %.6f%%" % (100 * (A["rgb"] == B["rgb"]).all(axis=1).mean()))
d = np.abs(A["xyz"] - B["xyz"]).max(axis=1)
print("centroid   max |delta| : %.3e m   (identical %.6f%%)" % (d.max(), 100 * (d == 0).mean()))
dc = np.abs(A["conf"] - B["conf"])
print("confidence max |delta| : %.3e     (identical %.6f%%)" % (dc.max(), 100 * (dc == 0).mean()))
for nm, x in (("A", a), ("B", b)):
    h = np.bincount(x["cls"].astype(np.int64), minlength=16)
    print("  %s class hist: %s" % (nm, h.tolist()))
