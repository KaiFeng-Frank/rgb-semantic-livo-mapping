#!/usr/bin/env python3
"""Final quantitative report: 20 frames of SemanticKITTI seq 07 (== raw drive 0027)."""
import sys, glob, numpy as np, torch
sys.path.insert(0, "/data/livo_sem/src")
from ptv3_infer import (PTv3Segmenter, read_bin, sk_labels_to_coarse,
                        NUSC16_TO_COARSE, COARSE, IGNORE, NUSCENES16)
D = "/data/livo_sem/data"
sc = sorted(glob.glob(D + "/raw/2011_09_30/2011_09_30_drive_0027_sync/velodyne_points/data/*.bin"))
lb = sorted(glob.glob(D + "/odometry/dataset/sequences/07/labels/*.label"))
fl = [(sc[i], lb[i]) for i in range(0, 1100, 55)]
print("frames:", len(fl), " (seq 07 / drive 0027, every 55th)")
seg = PTv3Segmenter()
K = len(COARSE)
inter = np.zeros(K, np.int64); pc = np.zeros(K, np.int64); gc = np.zeros(K, np.int64)
ok = tot = 0; hist = np.zeros(16, np.int64); confs = []
for sp, gp in fl:
    p = read_bin(sp); g = sk_labels_to_coarse(gp)
    assert len(p) == len(g)
    l, c = seg.segment(p); hist += np.bincount(l.astype(np.int64), minlength=16); confs.append(c)
    pr = NUSC16_TO_COARSE[l.astype(np.int64)]; m = g != IGNORE
    ok += int((pr[m] == g[m]).sum()); tot += int(m.sum())
    for k in range(K):
        pk = (pr == k) & m; gk = (g == k) & m
        inter[k] += int((pk & gk).sum()); pc[k] += int(pk.sum()); gc[k] += int(gk.sum())
print("\n=== predicted nuScenes-16 histogram over %d frames (%d points) ===" % (len(fl), hist.sum()))
for i in np.argsort(-hist):
    print("   %-22s %9d  %6.2f%%" % (NUSCENES16[i], hist[i], 100*hist[i]/hist.sum()))
print("\n=== agreement with SemanticKITTI GT in the coarse common label space ===")
print("overall point accuracy = %.2f%%  (%d / %d labelled points)" % (100*ok/tot, ok, tot))
print("  %-16s%>9s" % ("", "") if False else "  coarse class        GT pts     pred pts     correct    recall      IoU")
ious = []
for k in range(K):
    if gc[k] == 0 and pc[k] == 0: continue
    u = pc[k] + gc[k] - inter[k]
    iou = inter[k]/u if u else 0.0
    rec = inter[k]/gc[k] if gc[k] else float('nan')
    if gc[k] > 0: ious.append(iou)
    print("  %-16s%10d%13d%12d%9.2f%%%9.2f%%" % (COARSE[k], gc[k], pc[k], inter[k], 100*rec, 100*iou))
print("  mIoU over %d GT-present classes = %.2f%%" % (len(ious), 100*np.mean(ious)))

print("\n=== is the prediction trustworthy where it is confident? ===")
p = read_bin(fl[0][0]); g = sk_labels_to_coarse(fl[0][1])
l1, c1 = seg.segment(p); l2, c2 = seg.segment(p)
pr = NUSC16_TO_COARSE[l1.astype(np.int64)]; m = g != IGNORE
print("  conf bin      n      accuracy   run-to-run repeatability")
for lo, hi in [(0.0,0.3),(0.3,0.5),(0.5,0.7),(0.7,0.9),(0.9,1.01)]:
    b = (c1 >= lo) & (c1 < hi)
    if b.sum() == 0: continue
    a = (pr[b & m] == g[b & m]).mean()*100 if (b & m).sum() else float('nan')
    print("  [%.1f,%.1f) %8d    %6.2f%%          %6.2f%%" % (lo, hi, b.sum(), a, 100*(l1[b]==l2[b]).mean()))
