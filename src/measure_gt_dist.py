#!/usr/bin/env python3
"""Raw SemanticKITTI id histogram over seq07, all points and camera-frustum points.
Pure numpy, CPU only."""
import os, sys, glob, json
import numpy as np
sys.path.insert(0, "/data/livo_sem/src")
from kitti_calib import KittiCalib

RAW = "/data/livo_sem/data/raw/2011_09_30/2011_09_30_drive_0027_sync"
SCANS = sorted(glob.glob(RAW + "/velodyne_points/data/*.bin"))
LABELS = sorted(glob.glob("/data/livo_sem/data/odometry/dataset/sequences/07/labels/*.label"))
calib = KittiCalib("/data/livo_sem/data/raw/2011_09_30")
W, H = int(calib.img_size[0]), int(calib.img_size[1])
print("image size", W, H, "frames", len(LABELS))

NID = 300
h_all = np.zeros(NID, np.int64)
h_fov = np.zeros(NID, np.int64)
n_pts = 0; n_fov = 0
mismatch = []
T = calib.T_velo_rect0; P = calib.P_rect_02

for i, lp in enumerate(LABELS):
    sp = SCANS[i]
    assert int(os.path.basename(sp)[:-4]) == int(os.path.basename(lp)[:-6]) == i, (sp, lp)
    p = np.fromfile(sp, dtype=np.float32).reshape(-1, 4)
    g = (np.fromfile(lp, dtype=np.uint32) & 0xFFFF).astype(np.int64)
    if len(p) != len(g):
        mismatch.append((i, len(p), len(g))); continue
    n_pts += len(p)
    h_all += np.bincount(g, minlength=NID)
    hom = np.hstack([p[:, :3].astype(np.float64), np.ones((len(p), 1))])
    uvw = (P @ (T @ hom.T)).T
    d = uvw[:, 2]
    ok = d > 0.5
    u = np.where(ok, uvw[:, 0] / np.where(ok, d, 1.0), -1e9)
    v = np.where(ok, uvw[:, 1] / np.where(ok, d, 1.0), -1e9)
    ok &= (u >= 0) & (u <= W - 1) & (v >= 0) & (v <= H - 1)
    n_fov += int(ok.sum())
    h_fov += np.bincount(g[ok], minlength=NID)
    if i % 200 == 0:
        print("  frame %d/%d" % (i, len(LABELS)), flush=True)

print("mismatches:", mismatch)
print("total points %d, in-frustum %d (%.2f%%)" % (n_pts, n_fov, 100.0*n_fov/n_pts))
json.dump(dict(h_all=h_all.tolist(), h_fov=h_fov.tolist(), n_pts=n_pts, n_fov=n_fov,
               frames=len(LABELS), mismatch=mismatch),
          open("/data/livo_sem/out/labelspace/gt_raw_hist.json", "w"))
NAMES = {0:"unlabeled",1:"outlier",10:"car",11:"bicycle",13:"bus",15:"motorcycle",
 16:"on-rails",18:"truck",20:"other-vehicle",30:"person",31:"bicyclist",32:"motorcyclist",
 40:"road",44:"parking",48:"sidewalk",49:"other-ground",50:"building",51:"fence",
 52:"other-structure",60:"lane-marking",70:"vegetation",71:"trunk",72:"terrain",80:"pole",
 81:"traffic-sign",99:"other-object",252:"moving-car",253:"moving-bicyclist",
 254:"moving-person",255:"moving-motorcyclist",256:"moving-on-rails",257:"moving-bus",
 258:"moving-truck",259:"moving-other-vehicle"}
print("\n%-5s %-22s %14s %8s %14s %8s" % ("id","name","all_pts","all_%","fov_pts","fov_%"))
for k in sorted(NAMES):
    if h_all[k] or h_fov[k]:
        print("%-5d %-22s %14d %8.4f %14d %8.4f" % (k, NAMES[k], h_all[k],
              100.0*h_all[k]/n_pts, h_fov[k], 100.0*h_fov[k]/max(n_fov,1)))
extra = [k for k in range(NID) if h_all[k] and k not in NAMES]
print("ids present but not in official list:", extra)
