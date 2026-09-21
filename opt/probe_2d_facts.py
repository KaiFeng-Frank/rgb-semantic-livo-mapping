#!/usr/bin/env python3
"""CPU-only facts needed to pin the 2D-vs-3D protocol. No GPU, no model."""
import sys, glob, json
import numpy as np
sys.path.insert(0, "/data/livo_sem/src")
import sem_core as S
from kitti_calib import load_calib

RAW = "/data/livo_sem/data/raw/2011_09_30/2011_09_30_drive_0027_sync"
LAB = "/data/livo_sem/data/odometry/dataset/sequences/07/labels"
CAL = "/data/livo_sem/data/raw/2011_09_30"
W, H = 1226, 370

cal = load_calib(CAL)
print("img_size from calib:", cal.img_size)
print("K_rect:\n", cal.K_rect)
print("T_velo_cam2:\n", np.round(cal.T_velo_cam2, 9))

frames = list(range(0, 1101, 11))
lut = S.sk_lut()
rawhist = np.zeros(300, np.int64)
rawhist_fr = np.zeros(300, np.int64)
tot = infr = 0
depths = []
cov_per_frame = []
for f in frames:
    p = np.fromfile(RAW + "/velodyne_points/data/%010d.bin" % f, np.float32).reshape(-1, 4)
    g = np.fromfile(LAB + "/%06d.label" % f, np.uint32) & 0xFFFF
    assert len(p) == len(g), (f, len(p), len(g))
    uv, d, m = cal.project_velo_to_cam2(p[:, :3], img_shape=(H, W), min_depth=0.5,
                                        return_mask=True)
    tot += len(p); infr += int(m.sum()); cov_per_frame.append(m.mean())
    rawhist += np.bincount(g, minlength=300)
    rawhist_fr += np.bincount(g[m], minlength=300)
    depths.append(d)

d = np.concatenate(depths)
print("\n=== FRUSTUM COVERAGE over %d frames ===" % len(frames))
print("points total %d  in-frustum %d  = %.3f%%" % (tot, infr, 100.0*infr/tot))
print("per-frame coverage mean %.4f min %.4f max %.4f" %
      (np.mean(cov_per_frame), np.min(cov_per_frame), np.max(cov_per_frame)))
print("in-frustum depth: p50 %.1f p90 %.1f p99 %.1f max %.1f m" %
      tuple(np.percentile(d, [50, 90, 99, 100])))
for lo, hi in [(0,10),(10,20),(20,30),(30,50),(50,1e9)]:
    print("   depth %5s-%-5s : %8d pts (%.2f%% of in-frustum)" %
          (lo, hi, ((d>=lo)&(d<hi)).sum(), 100.0*((d>=lo)&(d<hi)).mean()))

print("\n=== GT raw-id histogram (global / in-frustum) ===")
NAMES = {0:"unlabeled",1:"outlier",10:"car",11:"bicycle",13:"bus",15:"motorcycle",
 16:"on-rails",18:"truck",20:"other-vehicle",30:"person",31:"bicyclist",
 32:"motorcyclist",40:"road",44:"parking",48:"sidewalk",49:"other-ground",
 50:"building",51:"fence",52:"other-structure",60:"lane-marking",70:"vegetation",
 71:"trunk",72:"terrain",80:"pole",81:"traffic-sign",99:"other-object",
 252:"moving-car",253:"moving-bicyclist",254:"moving-person",255:"moving-motorcyclist",
 256:"moving-on-rails",257:"moving-bus",258:"moving-truck",259:"moving-other-veh"}
for i in np.nonzero(rawhist)[0]:
    print("  %3d %-20s %10d  %10d" % (i, NAMES.get(i,"?"), rawhist[i], rawhist_fr[i]))

print("\n=== RIDER DISAMBIGUATION (Cityscapes 'rider') ===")
bic = rawhist[31] + rawhist[253]; mot = rawhist[32] + rawhist[255]
bicf = rawhist_fr[31] + rawhist_fr[253]; motf = rawhist_fr[32] + rawhist_fr[255]
print("  bicyclist(31,253) global %d  in-frustum %d" % (bic, bicf))
print("  motorcyclist(32,255) global %d  in-frustum %d" % (mot, motf))

print("\n=== coarse GT histogram ===")
cg = lut[np.arange(300)]
for k, name in enumerate(S.COARSE):
    ids = np.nonzero(cg == k)[0]
    print("  %-14s global %10d  in-frustum %10d" %
          (name, rawhist[ids].sum(), rawhist_fr[ids].sum()))
ign = np.nonzero(cg == S.IGNORE)[0]
print("  %-14s global %10d  in-frustum %10d" % ("IGNORE", rawhist[ign].sum(), rawhist_fr[ign].sum()))
json.dump(dict(frames=len(frames), tot=int(tot), infr=int(infr),
               cov=float(infr)/tot), open("/data/livo_sem/opt/out2d/facts.json","w"), indent=2)
