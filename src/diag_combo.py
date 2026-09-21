#!/usr/bin/env python3
"""Beam-pattern emulation (64->32), intensity recalibration, TTA."""
import sys, numpy as np, torch
sys.path.insert(0, "/data/livo_sem/src")
from ptv3_infer import (PTv3Segmenter, read_bin, sk_labels_to_coarse,
                        NUSC16_TO_COARSE, COARSE, IGNORE, NUSCENES16)

SCAN = "/data/livo_sem/data/raw/2011_09_30/2011_09_30_drive_0027_sync/velodyne_points/data/0000000000.bin"
GT   = "/data/livo_sem/data/odometry/dataset/sequences/07/labels/000000.label"
pts_all = read_bin(SCAN); gt_all = sk_labels_to_coarse(GT)

def rings(pts):
    yaw = -np.arctan2(pts[:, 1], pts[:, 0])
    px = 0.5 * (yaw / np.pi + 1)
    br = np.zeros(len(pts), np.int32)
    br[1:] = ((px[1:] < 0.2) & (px[:-1] > 0.8)).astype(np.int32)
    return np.cumsum(br)

R = rings(pts_all)
print("recovered rings: max=%d  counts(first 5)=%s" % (R.max(), np.bincount(R)[:5]))

seg = PTv3Segmenter(amp=False)

def score(pts, idx, tag):
    lb, cf = seg.segment(pts)
    pr = NUSC16_TO_COARSE[lb.astype(np.int64)]; gt = gt_all[idx]; m = gt != IGNORE
    acc = (pr[m] == gt[m]).mean() * 100
    ious = []
    for k in range(len(COARSE)):
        pk = (pr == k) & m; gk = (gt == k) & m
        if gk.sum() == 0: continue
        u = (pk | gk).sum(); ious.append(((pk & gk).sum()/u) if u else 0.0)
    t = np.bincount(lb.astype(np.int64), minlength=16); o = np.argsort(-t)[:4]
    print("%-34s N=%6d vox=%6d acc=%5.2f%% mIoU=%5.2f%% | %s" % (tag, len(pts), seg.last_num_voxels,
          acc, 100*np.mean(ious), " ".join("%s=%.0f%%" % (NUSCENES16[i], 100*t[i]/len(lb)) for i in o)))
    return acc

keep = np.where(R % 2 == 0)[0]
print()
score(pts_all, np.arange(len(pts_all)), "base")
score(pts_all[keep], keep, "32-beam (every 2nd ring)")
for s in (0.6, 0.4, 0.3):
    seg.intensity_scale = s
    score(pts_all[keep], keep, "32-beam + intens x%.1f" % s)
seg.intensity_scale = 1.0
keep4 = np.where(R % 4 == 0)[0]
score(pts_all[keep4], keep4, "16-beam (every 4th ring)")

# combos on full cloud
for s in (0.6, 0.5, 0.4, 0.3, 0.2):
    seg.intensity_scale = s
    score(pts_all, np.arange(len(pts_all)), "full + intens x%.1f" % s)
seg.intensity_scale = 0.4
for zs in (0.0, -0.5, -1.0):
    seg.z_shift = zs
    score(pts_all, np.arange(len(pts_all)), "full + i0.4 + z%+.1f" % zs)
seg.z_shift = 0.0

# ---- TTA (official test_cfg: scales 0.9..1.1 x {no flip, flip y})
print()
def tta(pts, idx, tag, scales=(0.9, 0.95, 1.0, 1.05, 1.1), flips=(False, True)):
    prob = None
    for sc in scales:
        for fl in flips:
            p = pts.copy(); p[:, :3] *= sc
            if fl: p[:, 1] *= -1
            lb, cf, logits, inv = seg.segment(p, return_logits=True)
            pr = torch.softmax(torch.from_numpy(logits), -1).numpy()[inv]
            prob = pr if prob is None else prob + pr
    lb = prob.argmax(1).astype(np.uint16)
    pr = NUSC16_TO_COARSE[lb.astype(np.int64)]; gt = gt_all[idx]; m = gt != IGNORE
    acc = (pr[m] == gt[m]).mean()*100
    ious = []
    for k in range(len(COARSE)):
        pk = (pr == k) & m; gk = (gt == k) & m
        if gk.sum() == 0: continue
        u = (pk | gk).sum(); ious.append(((pk & gk).sum()/u) if u else 0.0)
    t = np.bincount(lb.astype(np.int64), minlength=16); o = np.argsort(-t)[:4]
    print("%-34s acc=%5.2f%% mIoU=%5.2f%% | %s" % (tag, acc, 100*np.mean(ious),
          " ".join("%s=%.0f%%" % (NUSCENES16[i], 100*t[i]/len(lb)) for i in o)))

seg.intensity_scale = 1.0
tta(pts_all, np.arange(len(pts_all)), "TTA10 full")
seg.intensity_scale = 0.4
tta(pts_all, np.arange(len(pts_all)), "TTA10 full + intens x0.4")
