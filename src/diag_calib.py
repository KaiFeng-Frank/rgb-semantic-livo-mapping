#!/usr/bin/env python3
"""Calibrate on seq 04 (drive_0016), evaluate on seq 07 (drive_0027). Elevation-based beam decimation."""
import sys, glob, numpy as np
sys.path.insert(0, "/data/livo_sem/src")
from ptv3_infer import (PTv3Segmenter, read_bin, sk_labels_to_coarse,
                        NUSC16_TO_COARSE, COARSE, IGNORE, NUSCENES16)

D = "/data/livo_sem/data"
SEQ = {"04": (f"{D}/raw/2011_09_30/2011_09_30_drive_0016_sync/velodyne_points/data",
              f"{D}/odometry/dataset/sequences/04/labels"),
       "07": (f"{D}/raw/2011_09_30/2011_09_30_drive_0027_sync/velodyne_points/data",
              f"{D}/odometry/dataset/sequences/07/labels")}

def frames(seq, n, stride):
    sd, ld = SEQ[seq]
    s = sorted(glob.glob(sd + "/*.bin")); l = sorted(glob.glob(ld + "/*.label"))
    idx = list(range(0, min(len(s), len(l)), stride))[:n]
    return [(s[i], l[i]) for i in idx]

def beam_decimate(pts, n_out=32, n_in=64, fov=(-24.8, 2.0)):
    r = np.linalg.norm(pts[:, :3], axis=1) + 1e-9
    el = np.degrees(np.arcsin(pts[:, 2] / r))
    b = np.clip(((el - fov[0]) / (fov[1] - fov[0]) * n_in).astype(int), 0, n_in - 1)
    step = n_in // n_out
    return np.where(b % step == 0)[0]

seg = PTv3Segmenter(amp=False)

def run(fl, tag, iscale=1.0, zsh=0.0, decim=0):
    seg.intensity_scale = iscale; seg.z_shift = zsh
    K = len(COARSE); inter = np.zeros(K, np.int64); pc = np.zeros(K, np.int64); gc = np.zeros(K, np.int64)
    ok = tot = 0; hist = np.zeros(16, np.int64)
    for sp, gp in fl:
        p = read_bin(sp); g = sk_labels_to_coarse(gp)
        assert len(p) == len(g), (sp, len(p), gp, len(g))
        if decim:
            keep = beam_decimate(p, decim); p = p[keep]; g = g[keep]
        lb, _ = seg.segment(p); hist += np.bincount(lb.astype(np.int64), minlength=16)
        pr = NUSC16_TO_COARSE[lb.astype(np.int64)]; m = g != IGNORE
        ok += int((pr[m] == g[m]).sum()); tot += int(m.sum())
        for k in range(K):
            pk = (pr == k) & m; gk = (g == k) & m
            inter[k] += int((pk & gk).sum()); pc[k] += int(pk.sum()); gc[k] += int(gk.sum())
    ious = [inter[k] / max(pc[k] + gc[k] - inter[k], 1) for k in range(K) if gc[k] > 0]
    o = np.argsort(-hist)[:4]
    print("%-38s acc=%5.2f%% mIoU=%5.2f%% | %s" % (tag, 100*ok/max(tot,1), 100*np.mean(ious),
          " ".join("%s=%.0f%%" % (NUSCENES16[i], 100*hist[i]/hist.sum()) for i in o)))
    return 100*ok/max(tot,1), 100*np.mean(ious)

cal = frames("04", 8, 30)
print("calibration frames (seq 04):", len(cal))
p0 = read_bin(cal[0][0]); k32 = beam_decimate(p0, 32)
print("beam decimation check: %d -> %d pts (32/64 beams)" % (len(p0), len(k32)))
print()
for d in (0, 32, 16):
    run(cal, "seq04 decim=%d intens=1.0" % d, 1.0, 0.0, d)
print()
best = None
for s in (1.0, 0.7, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1):
    a, m = run(cal, "seq04 intens x%.2f" % s, s, 0.0, 0)
    if best is None or m > best[1]: best = (s, m)
print("-> best intensity_scale on seq04 by mIoU:", best)
print()
for zs in (0.0, -0.3, -0.5, -1.0):
    run(cal, "seq04 intens x%.2f z%+.1f" % (best[0], zs), best[0], zs, 0)
