#!/usr/bin/env python3
"""
mask_selftest.py -- reference emitter + self-test for the score_dynamic.py interface.

It fabricates exactly the dump the node is expected to write (bag order, a
contiguous pose-gate slice, a confidence gate, `idx`/`xyz`/`eligible`/`pred`),
with a KNOWN dynamic recall and a KNOWN static false-kill, so that the scorer can
be checked against numbers that were injected on purpose.

  python3 opt/mask_selftest.py --out /tmp/mask_demo --f0 745 --f1 800 \
          --recall 0.90 --false-kill 0.01
  python3 opt/score_dynamic.py --mask-dir /tmp/mask_demo --f0 745 --f1 800

THE EMITTER, as the node would do it in stage_b (copy this shape):

    # xyzi, lo, hi are what stage A produced; `drop` is the v0.3 decision, a
    # boolean over the points the gates admitted.
    n = len(xyzi)
    idx = np.arange(lo, hi, dtype=np.int32)          # bag-order positions
    eligible = np.zeros(n, bool); eligible[lo:hi] = True
    if conf_gate > 0: eligible[lo:hi] &= conf[lo:hi] >= conf_gate
    keep = eligible.copy(); keep[idx[drop]] = False
    np.savez(os.path.join(mask_dir, "frame_%06d.npz" % frame_index),
             keep=keep, eligible=eligible, xyz=xyzi[:, :3].astype(np.float32),
             pred=lab.astype(np.uint8), order="bag")
    # (with a full-length keep/eligible, `idx` is not needed at all)
"""
import argparse, calendar, os, sys
import numpy as np
sys.path.insert(0, "/data/livo_sem/src")
import sem_core as S
from kitti_scan import synth
RAW = "/data/livo_sem/data/raw/2011_09_30/2011_09_30_drive_0027_sync"
LB = "/data/livo_sem/data/odometry/dataset/sequences/07/labels"
MOVING = list(range(252, 260))


def parse_ts(p):
    o = []
    for line in open(p):
        line = line.strip()
        if not line:
            continue
        dd, cc = line.split(" ")
        hms, frac = (cc.split(".") + ["0"])[:2]
        y, mo, d = (int(v) for v in dd.split("-"))
        h, mi, s = (int(v) for v in hms.split(":"))
        o.append(calendar.timegm((y, mo, d, h, mi, s, 0, 0, 0)) * 10**9 + int((frac + "000000000")[:9]))
    return np.array(o, dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--f0", type=int, default=745)
    ap.add_argument("--f1", type=int, default=800)
    ap.add_argument("--recall", type=float, default=0.90)
    ap.add_argument("--false-kill", type=float, default=0.01)
    ap.add_argument("--subset", type=int, default=1,
                    help="1: emit only the gated slice with idx; 0: full-length arrays")
    ap.add_argument("--with-pred", type=int, default=0,
                    help="also write a placeholder `pred` array (zeros); off by default")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    ts, te = parse_ts(RAW + "/velodyne_points/timestamps_start.txt"), parse_ts(RAW + "/velodyne_points/timestamps_end.txt")
    traj = S.TrajInterp("/data/livo_sem/out/kitti_seq07_fastlivo2_tum.txt")
    inj_mv = inj_mvd = inj_st = inj_std = 0
    for f in range(a.f0, a.f1):
        p = np.fromfile(RAW + "/velodyne_points/data/%010d.bin" % f, dtype=np.float32).reshape(-1, 4)
        lab = np.fromfile(LB + "/%06d.label" % f, dtype=np.uint32) & 0xFFFF
        _, t, order = synth(p[:, :3], sweep_duration=(te[f] - ts[f]) / 1e9)
        t_abs = t + ts[f] * 1e-9
        sem_bag = lab[order].astype(np.int32)
        xyz_bag = p[order, :3]
        n = len(xyz_bag)
        lo = int(np.searchsorted(t_abs, traj.t0, side="left"))
        hi = int(np.searchsorted(t_abs, traj.t1, side="right"))
        elig = np.zeros(n, bool)
        elig[lo:hi] = True
        mv = np.isin(sem_bag, MOVING)
        ign = np.isin(sem_bag, [0, 1, 99])
        drop = np.zeros(n, bool)
        drop |= mv & (rng.random(n) < a.recall)
        drop |= (~mv & ~ign) & (rng.random(n) < a.false_kill)
        keep = elig & ~drop
        inj_mv += int((mv & elig).sum()); inj_mvd += int((mv & elig & drop).sum())
        inj_st += int((~mv & ~ign & elig).sum()); inj_std += int((~mv & ~ign & elig & drop).sum())
        if a.subset:
            idx = np.arange(lo, hi, dtype=np.int32)
            kw = dict(keep=keep[lo:hi], eligible=elig[lo:hi], idx=idx,
                      xyz=xyz_bag[lo:hi].astype(np.float32), order="bag")
            if a.with_pred:
                kw["pred"] = np.zeros(hi - lo, np.uint8)
            np.savez(os.path.join(a.out, "frame_%06d.npz" % f), **kw)
        else:
            np.savez(os.path.join(a.out, "frame_%06d.npz" % f),
                     keep=keep, eligible=elig, xyz=xyz_bag.astype(np.float32),
                     order="bag")
    print("INJECTED over frames %d..%d:  dynamic recall %.3f %%   static false-kill %.3f %%"
          % (a.f0, a.f1 - 1, 100.0 * inj_mvd / max(1, inj_mv), 100.0 * inj_std / max(1, inj_st)))
    print("-> %s" % a.out)


if __name__ == "__main__":
    main()
