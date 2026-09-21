#!/usr/bin/env python3
"""
eval_report.py -- accuracy of the SHIPPED inference path against SemanticKITTI GT.

MUST be run with the ptv3 conda python:
    /opt/miniconda3/envs/ptv3/bin/python src/eval_report.py [flags]

WHY THIS FILE WAS REWRITTEN (2026-09-21)
----------------------------------------
The previous version did `from ptv3_infer import PTv3Segmenter` and instantiated it
with defaults.  ptv3_infer.py is the DO-NOT-USE module: POINTCEPT_ROOT defaults to
src/Pointcept (HEAD) and INTENSITY_SCALE_KITTI defaults to 1.0.  Run as-is it scored
a *different model* from the one the node deploys and reported ~25.7 % point accuracy
with car IoU 0.00 %.  So the stated acceptance instrument did not measure the thing
under test, and any re-score after an optimisation would have looked like a
catastrophic regression caused by that optimisation.

This version imports ptv3_worker.Segmenter -- literally the class the node's
co-process runs -- so what is scored is what is deployed.

20 frames of SemanticKITTI seq 07 (== raw drive 0027), every 55th, coarse label space.
"""
import os
import sys
import glob
import json
import time
import argparse

import numpy as np

sys.path.insert(0, "/data/livo_sem/src")
import sem_core as S

D = "/data/livo_sem/data"
SCANS = D + "/raw/2011_09_30/2011_09_30_drive_0027_sync/velodyne_points/data/*.bin"
LABELS = D + "/odometry/dataset/sequences/07/labels/*.label"


def read_bin(p):
    return np.fromfile(p, dtype=np.float32).reshape(-1, 4)


def sk_labels_to_coarse(p, lut):
    raw = np.fromfile(p, dtype=np.uint32) & 0xFFFF
    return lut[raw]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid-size", type=float, default=0.05)
    ap.add_argument("--intensity-scale", type=float, default=0.2)
    ap.add_argument("--half", type=int, default=0)
    ap.add_argument("--shuffle", type=int, default=1,
                    help="shuffle_orders; the frozen config ships 1")
    ap.add_argument("--fast-voxel", type=int, default=0)
    ap.add_argument("--stem-fast", type=int, default=0)
    ap.add_argument("--tf32", type=int, default=0)
    ap.add_argument("--gpu-voxel", type=int, default=0)
    ap.add_argument("--fast-hilbert", type=int, default=0)
    ap.add_argument("--nframes", type=int, default=20)
    ap.add_argument("--stride", type=int, default=55)
    ap.add_argument("--timing", type=int, default=0,
                    help="also time the segmenter over the eval frames (>=N reps)")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    os.environ["PTV3_GRID_SIZE"] = "%.6f" % a.grid_size
    os.environ["PTV3_INTENSITY_SCALE"] = "%.6f" % a.intensity_scale
    os.environ["PTV3_HALF"] = str(int(a.half))
    os.environ["PTV3_SHUFFLE"] = str(int(a.shuffle))
    os.environ["PTV3_FAST_VOXEL"] = str(int(a.fast_voxel))
    os.environ["PTV3_STEM_FAST"] = str(int(a.stem_fast))
    os.environ["PTV3_TF32"] = str(int(a.tf32))
    os.environ["PTV3_GPU_VOXEL"] = str(int(a.gpu_voxel))
    os.environ["PTV3_FAST_HILBERT"] = str(int(a.fast_hilbert))

    import ptv3_worker as W
    seg = W.Segmenter()

    sc = sorted(glob.glob(SCANS))
    lb = sorted(glob.glob(LABELS))
    fl = [(sc[i], lb[i]) for i in range(0, a.stride * a.nframes, a.stride)]
    lut = S.sk_lut()
    COARSE, IGNORE = S.COARSE, S.IGNORE
    NUSC16_TO_COARSE = S.NUSC16_TO_COARSE
    K = len(COARSE)

    inter = np.zeros(K, np.int64)
    pc = np.zeros(K, np.int64)
    gc = np.zeros(K, np.int64)
    ok = tot = 0
    hist = np.zeros(16, np.int64)
    tms = []
    vox = []

    # warm
    p0 = read_bin(fl[0][0])
    for _ in range(3):
        seg.segment(p0)

    for sp, gp in fl:
        p = read_bin(sp)
        g = sk_labels_to_coarse(gp, lut)
        assert len(p) == len(g), (len(p), len(g), sp)
        t0 = time.perf_counter()
        l, c = seg.segment(p)
        tms.append((time.perf_counter() - t0) * 1000.0)
        vox.append(seg.last_voxels)
        hist += np.bincount(l.astype(np.int64), minlength=16)
        pr = NUSC16_TO_COARSE[l.astype(np.int64)]
        m = g != IGNORE
        ok += int((pr[m] == g[m]).sum())
        tot += int(m.sum())
        for k in range(K):
            pk = (pr == k) & m
            gk = (g == k) & m
            inter[k] += int((pk & gk).sum())
            pc[k] += int(pk.sum())
            gc[k] += int(gk.sum())

    if a.timing:
        tms = []
        for _ in range(max(1, a.timing)):
            for sp, _gp in fl:
                p = read_bin(sp)
                t0 = time.perf_counter()
                seg.segment(p)
                tms.append((time.perf_counter() - t0) * 1000.0)

    acc = 100.0 * ok / tot
    ious = {}
    for k in range(K):
        if gc[k] == 0 and pc[k] == 0:
            continue
        u = pc[k] + gc[k] - inter[k]
        iou = 100.0 * inter[k] / u if u else 0.0
        if gc[k] > 0:
            ious[COARSE[k]] = iou
    miou = float(np.mean(list(ious.values())))
    # The 9 classes CRITICAL_CONSTRAINTS lists for the cited 61.26 % figure.  GT for
    # these 20 frames actually contains 11 coarse classes (bicycle 1579 pts and
    # motorcycle 8077 pts are present), so the cited number is a 9-class mean and is
    # NOT comparable to an 11-class mean.  Both are reported; compare like with like.
    NINE = ["car", "truck", "other_vehicle", "person", "road", "sidewalk",
            "terrain", "vegetation", "manmade"]
    miou9 = float(np.mean([ious[k] for k in NINE if k in ious]))

    res = dict(tag=a.tag, grid_size=a.grid_size, intensity_scale=a.intensity_scale,
               half=a.half, shuffle=a.shuffle, fast_voxel=a.fast_voxel,
               stem_fast=a.stem_fast, tf32=a.tf32,
               gpu_voxel=a.gpu_voxel, fast_hilbert=a.fast_hilbert,
               frames=len(fl), point_acc=acc, coarse_miou=miou, coarse_miou9=miou9,
               per_class_iou={k: round(v, 2) for k, v in ious.items()},
               seg_ms=dict(mean=float(np.mean(tms)), p50=float(np.percentile(tms, 50)),
                           p95=float(np.percentile(tms, 95)), max=float(np.max(tms))),
               voxels=dict(mean=float(np.mean(vox)), max=int(np.max(vox))),
               hist={S.NUSCENES_CLASSES[i]: int(hist[i]) for i in range(16)})

    if not a.quiet:
        print("\n=== %s  grid=%.3f int=%.2f half=%d shuffle=%d fastvox=%d stem=%d tf32=%d ==="
              % (a.tag or "eval", a.grid_size, a.intensity_scale, a.half, a.shuffle,
                 a.fast_voxel, a.stem_fast, a.tf32))
        print("point accuracy = %.2f%%   coarse mIoU = %.2f%% (11 cls) / %.2f%% (9 cls, "
              "the subset CRITICAL_CONSTRAINTS quotes)   (%d frames, %d labelled pts)"
              % (acc, miou, miou9, len(fl), tot))
        for k in sorted(ious, key=lambda x: -ious[x]):
            print("    %-16s IoU %6.2f%%" % (k, ious[k]))
        print("seg_ms mean %.2f p50 %.2f p95 %.2f max %.2f   voxels mean %.0f"
              % (res["seg_ms"]["mean"], res["seg_ms"]["p50"], res["seg_ms"]["p95"],
                 res["seg_ms"]["max"], res["voxels"]["mean"]))
    print("JSONLINE " + json.dumps(res))
    if a.json_out:
        with open(a.json_out, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
