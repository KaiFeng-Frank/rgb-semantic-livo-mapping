#!/usr/bin/env python3
"""Domain-gap sweep: which preprocessing makes the nuScenes PTv3 checkpoint behave on KITTI."""
import sys, numpy as np, torch
sys.path.insert(0, "/data/livo_sem/src")
from ptv3_infer import (PTv3Segmenter, read_bin, sk_labels_to_coarse,
                        NUSC16_TO_COARSE, COARSE, IGNORE, NUSCENES16)

SCAN = "/data/livo_sem/data/raw/2011_09_30/2011_09_30_drive_0027_sync/velodyne_points/data/0000000000.bin"
GT   = "/data/livo_sem/data/odometry/dataset/sequences/07/labels/000000.label"

pts_all = read_bin(SCAN)
gt_all  = sk_labels_to_coarse(GT)
print("N =", pts_all.shape[0])
z = pts_all[:, 2]
h, e = np.histogram(z, bins=60, range=(-3, 3))
peak = e[h.argmax()]
print("z histogram peak (ground level) = %.2f m ; z range [%.2f, %.2f]" % (peak, z.min(), z.max()))

seg = PTv3Segmenter(amp=False)

def score(pts, idx, tag):
    lb, cf = seg.segment(pts)
    pr = NUSC16_TO_COARSE[lb.astype(np.int64)]
    gt = gt_all[idx]
    m = gt != IGNORE
    acc = (pr[m] == gt[m]).mean() * 100
    ious = []
    for k in range(len(COARSE)):
        pk = (pr == k) & m; gk = (gt == k) & m
        if gk.sum() == 0: continue
        u = (pk | gk).sum()
        ious.append(((pk & gk).sum() / u) if u else 0.0)
    miou = 100 * np.mean(ious)
    top = np.bincount(lb.astype(np.int64), minlength=16)
    order = np.argsort(-top)[:4]
    ts = " ".join("%s=%.0f%%" % (NUSCENES16[i], 100 * top[i] / len(lb)) for i in order)
    print("%-28s N=%6d vox=%6d  acc=%5.2f%%  mIoU=%5.2f%%  conf=%.3f | %s"
          % (tag, len(pts), seg.last_num_voxels, acc, miou, cf.mean(), ts))
    return acc

rng = np.random.default_rng(0)
allidx = np.arange(len(pts_all))

score(pts_all, allidx, "base")

for zs in (-0.11, +1.73, -1.0):
    seg.z_shift = zs; score(pts_all, allidx, "z_shift=%+.2f" % zs)
seg.z_shift = 0.0

for s in (0.0, 0.5, 2.0, 1/255.0, 255.0):
    seg.intensity_scale = s; score(pts_all, allidx, "intensity_scale=%g" % s)
seg.intensity_scale = 1.0

for frac in (0.5, 0.33, 0.25):
    idx = np.sort(rng.choice(len(pts_all), int(len(pts_all) * frac), replace=False))
    score(pts_all[idx], idx, "subsample %.2f" % frac)

for r in (50.0, 40.0):
    d = np.linalg.norm(pts_all[:, :2], axis=1)
    idx = np.where(d <= r)[0]
    score(pts_all[idx], idx, "range_crop %.0fm" % r)

for g in (0.10, 0.15, 0.025):
    seg.grid_sample.grid_size = g
    score(pts_all, allidx, "grid_size=%.3f" % g)
seg.grid_sample.grid_size = 0.05

th = np.deg2rad(90.0)
R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]], np.float32)
p2 = pts_all.copy(); p2[:, :3] = pts_all[:, :3] @ R.T
score(p2, allidx, "yaw +90deg")

seg2 = PTv3Segmenter(amp=False, enable_flash=False)
_s, seg = seg, seg2
score(pts_all, allidx, "no_flash (fp32 attn)")
seg = _s
