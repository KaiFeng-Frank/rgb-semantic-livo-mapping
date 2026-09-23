#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
align_seq.py -- give score_2d_vs_3d.py an offset-free view of a KITTI raw drive.

WHY.  src/score_2d_vs_3d.py is FROZEN (code_sha1 e07ed80e...); every arm number in
this project was produced by it and editing it invalidates the comparison.  It
addresses a sequence as
    data/raw/2011_09_30/<drive>/velodyne_points/data/%010d.bin
    data/raw/2011_09_30/<drive>/image_02/data/%010d.png
with the ODOMETRY frame index substituted directly into the %010d.  That is correct
for every sequence whose SemanticKITTI frame 0 is raw frame 0 -- which is all of them
EXCEPT seq 08, whose odometry frame 0 is raw frame 1100.

Rather than teach the frozen scorer about offsets, this builds a symlink view
    data/raw/<date>/<drive>_aligned/velodyne_points/data/%010d.bin -> raw %010d+start
    data/raw/<date>/<drive>_aligned/image_02/data/%010d.png       -> raw %010d+start
so the frozen scorer's own path arithmetic lands on the right file and
SEQUENCES["08"] can simply name the _aligned drive.  Zero bytes copied, zero lines
of the scorer changed.

THE OFFSET IS VERIFIED, NOT ASSUMED: for probe frames the raw .bin point count must
equal the .label point count.  A wrong offset on the VALIDATION sequence would
silently corrupt checkpoint selection, so the check is loud and fatal.
"""
import os, sys, argparse

ROOT = "/data/wuyou/livo_sem/data"
ODO = os.path.join(ROOT, "odometry", "dataset", "sequences")

# seq -> (date, drive, n_labels, raw_start)   -- identical table to make_sk_layout.py
MAP = {
    "00": ("2011_10_03", "0027", 4541, 0),
    "01": ("2011_10_03", "0042", 1101, 0),
    "02": ("2011_10_03", "0034", 4661, 0),
    "03": ("2011_09_26", "0067",  801, 0),
    "04": ("2011_09_30", "0016",  271, 0),
    "05": ("2011_09_30", "0018", 2761, 0),
    "06": ("2011_09_30", "0020", 1101, 0),
    "07": ("2011_09_30", "0027", 1101, 0),
    "08": ("2011_09_30", "0028", 4071, 1100),   # the ONLY non-zero start
    "09": ("2011_09_30", "0033", 1591, 0),
    "10": ("2011_09_30", "0034", 1201, 0),
}


def build(seq, force=False):
    date, drive, nlab, start = MAP[seq]
    src = os.path.join(ROOT, "raw", date, f"{date}_drive_{drive}_sync")
    dst = os.path.join(ROOT, "raw", date, f"{date}_drive_{drive}_sync_aligned")
    lab = os.path.join(ODO, seq, "labels")
    if not os.path.isdir(src):
        return f"seq {seq}: SKIP (raw not extracted: {src})"
    if not os.path.isdir(lab):
        return f"seq {seq}: SKIP (no labels)"

    sv = os.path.join(src, "velodyne_points", "data")
    si = os.path.join(src, "image_02", "data")
    dv = os.path.join(dst, "velodyne_points", "data")
    di = os.path.join(dst, "image_02", "data")
    os.makedirs(dv, exist_ok=True)
    os.makedirs(di, exist_ok=True)

    # ---- VERIFY THE OFFSET BEFORE LAYING DOWN ANY LINK ---------------------- #
    probes = [0, nlab // 4, nlab // 2, (3 * nlab) // 4, nlab - 1]
    bad = []
    for i in probes:
        b = os.path.join(sv, f"{start + i:010d}.bin")
        l = os.path.join(lab, f"{i:06d}.label")
        if not os.path.isfile(b):
            bad.append((i, "raw bin missing")); continue
        nb = os.path.getsize(b) // 16          # 4 x float32
        nl = os.path.getsize(l) // 4           # 1 x uint32
        if nb != nl:
            bad.append((i, f"{nb} pts vs {nl} labels"))
    if bad:
        return (f"seq {seq}: *** OFFSET VERIFY FAILED *** start={start} -> {bad}")

    nv = ni = 0
    miss_img = 0
    for i in range(nlab):
        sb = os.path.join(sv, f"{start + i:010d}.bin")
        lb = os.path.join(dv, f"{i:010d}.bin")
        if os.path.islink(lb) or os.path.exists(lb):
            if force: os.remove(lb)
            else: nv += 1; continue
        os.symlink(sb, lb); nv += 1
        sp = os.path.join(si, f"{start + i:010d}.png")
        lp = os.path.join(di, f"{i:010d}.png")
        if not os.path.isfile(sp):
            miss_img += 1; continue
        if os.path.islink(lp) or os.path.exists(lp):
            if force: os.remove(lp)
            else: ni += 1; continue
        os.symlink(sp, lp); ni += 1
    if not force:
        ni = len(os.listdir(di))
    return (f"seq {seq}: OK  start={start}  frames={nlab}  velo_links={nv}  "
            f"img_links={ni}  missing_img={miss_img}  probes_ok={len(probes)}  -> {dst}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("seqs", nargs="*", default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    seqs = a.seqs or ["08"]
    rc = 0
    for s in seqs:
        r = build(s, a.force)
        print(r, flush=True)
        if "FAILED" in r: rc = 1
    sys.exit(rc)
