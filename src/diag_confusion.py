#!/usr/bin/env python3
import sys, numpy as np
sys.path.insert(0, "/data/livo_sem/src")
from ptv3_infer import PTv3Segmenter, read_bin, NUSCENES16

SCAN = "/data/livo_sem/data/raw/2011_09_30/2011_09_30_drive_0027_sync/velodyne_points/data/0000000000.bin"
GT   = "/data/livo_sem/data/odometry/dataset/sequences/07/labels/000000.label"
SKNAME = {0:"unlabeled",1:"outlier",10:"car",11:"bicycle",13:"bus",15:"motorcycle",16:"on-rails",
          18:"truck",20:"other-vehicle",30:"person",31:"bicyclist",32:"motorcyclist",40:"road",
          44:"parking",48:"sidewalk",49:"other-ground",50:"building",51:"fence",52:"other-struct",
          60:"lane-marking",70:"vegetation",71:"trunk",72:"terrain",80:"pole",81:"traffic-sign",
          99:"other-object",252:"moving-car",253:"moving-bicyclist",254:"moving-person",
          255:"moving-motorcyclist",256:"moving-on-rails",257:"moving-bus",258:"moving-truck",
          259:"moving-other-veh"}

pts = read_bin(SCAN)
sem = (np.fromfile(GT, dtype=np.uint32) & 0xFFFF).astype(np.int32)
seg = PTv3Segmenter(amp=False)
lb, cf = seg.segment(pts)

print("=== confusion: GT SemanticKITTI class -> top-3 predicted nuScenes class ===")
for g in sorted(set(sem.tolist())):
    m = sem == g
    if m.sum() < 300: continue
    c = np.bincount(lb[m].astype(np.int64), minlength=16)
    o = np.argsort(-c)[:3]
    print("  %-16s n=%7d | %s" % (SKNAME.get(g, str(g)), m.sum(),
          "  ".join("%s %.0f%%" % (NUSCENES16[i], 100*c[i]/m.sum()) for i in o)))

print("\n=== geometry of predicted classes (meters) ===")
print("  %-22s %7s %8s %8s %8s %8s" % ("class","n","z_med","z_p05","z_p95","range_med"))
r = np.linalg.norm(pts[:, :2], axis=1)
for i in range(16):
    m = lb == i
    if m.sum() < 200: continue
    z = pts[m, 2]
    print("  %-22s %7d %8.2f %8.2f %8.2f %8.1f" % (NUSCENES16[i], m.sum(),
          np.median(z), np.percentile(z,5), np.percentile(z,95), np.median(r[m])))

print("\n=== where are the 'trailer' points? ===")
m = lb == NUSCENES16.index("trailer")
print("  n=%d  x[%.1f,%.1f] y[%.1f,%.1f] z[%.1f,%.1f]" % (m.sum(),
      pts[m,0].min(), pts[m,0].max(), pts[m,1].min(), pts[m,1].max(),
      pts[m,2].min(), pts[m,2].max()))
c = np.bincount(sem[m], minlength=260); o = np.argsort(-c)[:5]
print("  their GT:", ", ".join("%s %.0f%%" % (SKNAME.get(int(k), str(k)), 100*c[k]/m.sum()) for k in o))
