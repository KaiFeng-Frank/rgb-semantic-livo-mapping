#!/usr/bin/env python3
"""cache_pred.py -- one GPU pass caching the DEPLOYED PTv3 argmax+conf per frame.

Runs ptv3_worker.Segmenter (literally the class the node's co-process runs) over
every raw seq07 sweep, in the SAME point order the node sees (kitti_scan.synth,
azimuth-sorted), and writes per frame

    <out>/f%06d.npz   lab uint8 (N,)   conf float16 (N,)

K4: the model is non-deterministic run to run, so this cache is ONE draw.  Use
--tag to keep several and report the spread.
"""
import os, sys, time, argparse
import numpy as np
sys.path.insert(0, "/data/livo_sem/src")

RAW = "/data/livo_sem/data/raw/2011_09_30/2011_09_30_drive_0027_sync"


def parse_ts(path):
    import calendar
    out = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        date, clock = line.split(" ")
        hms, frac = (clock.split(".") + ["0"])[:2]
        y, mo, d = (int(v) for v in date.split("-"))
        h, mi, s = (int(v) for v in hms.split(":"))
        out.append(calendar.timegm((y, mo, d, h, mi, s, 0, 0, 0)) * 10**9
                   + int((frac + "000000000")[:9]))
    return np.array(out, dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--f0", type=int, default=0)
    ap.add_argument("--f1", type=int, default=1101)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    from kitti_scan import synth
    from ptv3_worker import Segmenter
    seg = Segmenter()
    seg.warmup(2)
    ts = parse_ts(RAW + "/velodyne_points/timestamps_start.txt")
    te = parse_ts(RAW + "/velodyne_points/timestamps_end.txt")
    t0 = time.time()
    n = 0
    for f in range(a.f0, a.f1):
        p = RAW + "/velodyne_points/data/%010d.bin" % f
        if not os.path.exists(p):
            continue
        pts = np.fromfile(p, dtype=np.float32).reshape(-1, 4)
        _, _, order = synth(pts[:, :3], sweep_duration=(te[f] - ts[f]) / 1e9)
        bag = np.ascontiguousarray(pts[order])          # the node's point order
        lab, conf = seg.segment(bag)
        np.savez(os.path.join(a.out, "f%06d.npz" % f),
                 lab=lab.astype(np.uint8), conf=conf.astype(np.float16),
                 order=order.astype(np.int32))
        n += 1
        if n % 100 == 0:
            print("%d frames, %.1f s" % (n, time.time() - t0), flush=True)
    print("DONE %d frames in %.1f s -> %s" % (n, time.time() - t0, a.out))


if __name__ == "__main__":
    main()
