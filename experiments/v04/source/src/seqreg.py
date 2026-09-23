#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
seqreg.py -- register extra SemanticKITTI sequences with the FROZEN scorer at RUNTIME.

src/score_2d_vs_3d.py ships SEQUENCES = {07: eval, 04: 2D-tuning} and nothing else,
and it is frozen (code_sha1 e07ed80e...).  This module mutates that dict in memory so
the val sequence (08) and the train sequences can be addressed by the same audited
Projector / Arm2D / Arm3D / frame_context code, without editing one byte of the file.

Two facts it encodes, both verified rather than assumed:
  * seq 08's odometry frame 0 is raw frame 1100.  The frozen scorer substitutes the
    odometry index straight into %010d, so 08 is pointed at the `_aligned` symlink
    view built by tools/align_seq.py, where that offset is already absorbed.
  * seqs 00/01/02 were recorded on 2011_10_03, a DIFFERENT calibration day with a
    DIFFERENT rectified image size (1242x375, not 1226x370).  The scorer's IMG_W /
    IMG_H are module-level constants baked into Projector, Arm2D._sample,
    visible_mask and depth_edge_mask defaults, so a sequence from another day needs
    those globals repointed too.  `use(seq)` does that and returns the Projector
    built from that day's own calib.  seq 03's raw drive (2011_09_26_drive_0067) is
    404 on the KITTI mirror and is therefore NOT registered.
"""
import os
import numpy as np

ROOT = "/data/wuyou/livo_sem"

# seq -> (date, drive, n_frames, raw_start, aligned?)
TABLE = {
    "00": ("2011_10_03", "0027", 4541, 0, False),
    "01": ("2011_10_03", "0042", 1101, 0, False),
    "02": ("2011_10_03", "0034", 4661, 0, False),
    # "03" 2011_09_26_drive_0067 -- raw drive returns HTTP 404, unavailable
    "04": ("2011_09_30", "0016",  271, 0, False),
    "05": ("2011_09_30", "0018", 2761, 0, False),
    "06": ("2011_09_30", "0020", 1101, 0, False),
    "07": ("2011_09_30", "0027", 1101, 0, False),
    "08": ("2011_09_30", "0028", 4071, 1100, True),
    "09": ("2011_09_30", "0033", 1591, 0, False),
    "10": ("2011_09_30", "0034", 1201, 0, False),
}

TRAIN_SEQS = ["00", "01", "02", "04", "05", "06", "09", "10"]   # 03 unavailable
VAL_SEQS = ["08"]
TEST_SEQS = ["07"]


def drive_dir(seq):
    date, drive, n, start, aligned = TABLE[seq]
    name = f"{date}_drive_{drive}_sync" + ("_aligned" if aligned else "")
    return date, name, n


def img_size(date):
    """Rectified image_02 size for a calibration day, read from S_rect_02 itself."""
    import sys
    sys.path.insert(0, ROOT + "/src")
    from kitti_calib import load_calib
    cal = load_calib(os.path.join(ROOT, "data", "raw", date))
    return int(cal.img_size[0]), int(cal.img_size[1])


def use(seq, SC=None):
    """Point the frozen scorer module at `seq`.  Returns (SC, Projector, W, H)."""
    import sys
    sys.path.insert(0, ROOT + "/src")
    if SC is None:
        import score_2d_vs_3d as SC          # noqa: N806
    date, name, n = drive_dir(seq)
    W, H = img_size(date)

    # repoint the constants Projector / Arm2D._sample / visible_mask read
    SC.IMG_W, SC.IMG_H = W, H
    SC.SEQUENCES[seq] = (name, n)

    # set_sequence hardcodes .../raw/2011_09_30/<drive>; rebuild the globals by hand
    SC.SEQ_ID = seq
    SC.RAW = os.path.join(ROOT, "data", "raw", date, name)
    SC.SEQ = os.path.join(ROOT, "data", "odometry", "dataset", "sequences", seq)
    SC.LABELS = SC.SEQ + "/labels"
    SC.SCANS = SC.RAW + "/velodyne_points/data"
    SC.IMAGES = SC.RAW + "/image_02/data"
    SC.N_FRAMES_TOTAL = n

    proj = SC.Projector(calib_dir=os.path.join(ROOT, "data", "raw", date), w=W, h=H)
    return SC, proj, W, H


def selftest(seq):
    """Point counts must match between the .bin the scorer will read and the .label."""
    SC, proj, W, H = use(seq)
    n = SC.N_FRAMES_TOTAL
    probes = [0, n // 4, n // 2, (3 * n) // 4, n - 1]
    out = []
    for i in probes:
        b = "%s/%010d.bin" % (SC.SCANS, i)
        l = "%s/%06d.label" % (SC.LABELS, i)
        p = "%s/%010d.png" % (SC.IMAGES, i)
        nb = os.path.getsize(b) // 16 if os.path.isfile(b) else -1
        nl = os.path.getsize(l) // 4 if os.path.isfile(l) else -1
        out.append((i, nb, nl, nb == nl and nb > 0, os.path.isfile(p)))
    ok = all(x[3] for x in out)
    return ok, dict(seq=seq, img_wh=(W, H), n_frames=n, probes=out,
                    scans=SC.SCANS, images=SC.IMAGES)


if __name__ == "__main__":
    import sys, json
    seqs = sys.argv[1:] or sorted(TABLE)
    rc = 0
    for s in seqs:
        try:
            ok, d = selftest(s)
        except Exception as e:
            print("seq %s: ERROR %s" % (s, e)); rc = 1; continue
        print("seq %s  %s  img=%s  n=%d  probes(idx,nbin,nlabel,match,img)=%s"
              % (s, "OK " if ok else "FAIL", d["img_wh"], d["n_frames"], d["probes"]))
        if not ok: rc = 1
    sys.exit(rc)
