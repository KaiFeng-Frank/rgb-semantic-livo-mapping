#!/usr/bin/env python3
"""
measure_ribbon.py -- how much of the accumulated map is a lie, and how long is
the ribbon the known tracked vehicle leaves.  CPU only.

  * transports every GT-moving point (252-259) of seq 07 into the world with the
    node's own chain (synth per-point time -> T_I_L -> interpolated T_W_I);
  * voxelises them at the deployed 0.20 m and compares the occupied set against
    the shipped map out/map_seq07.npz (2.70 M voxels);
  * measures per-instance ribbons: the principal-axis extent of all world points
    an instance deposits over a frame window, versus its extent in ONE sweep.
"""
import argparse, json, os, sys, calendar
from collections import defaultdict
import numpy as np

sys.path.insert(0, "/data/livo_sem/src")
import sem_core as S
from kitti_scan import synth

RAW = "/data/livo_sem/data/raw/2011_09_30/2011_09_30_drive_0027_sync"
LB = "/data/livo_sem/data/odometry/dataset/sequences/07/labels"
MOVING = {252: "moving-car", 253: "moving-bicyclist", 254: "moving-person",
          255: "moving-motorcyclist", 256: "moving-on-rails", 257: "moving-bus",
          258: "moving-truck", 259: "moving-other-vehicle"}


def parse_ts(path):
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


def pca_extent(P):
    if len(P) < 3:
        return (0.0, 0.0, 0.0)
    c = P - P.mean(0)
    _, V = np.linalg.eigh(c.T @ c / len(c))
    pr = c @ V[:, ::-1]
    return tuple(float(pr[:, k].max() - pr[:, k].min()) for k in range(3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-frames", type=int, default=1101)
    ap.add_argument("--traj", default="/data/livo_sem/out/kitti_seq07_fastlivo2_tum.txt")
    ap.add_argument("--map", default="/data/livo_sem/out/map_seq07.npz")
    ap.add_argument("--voxel", type=float, default=0.20)
    ap.add_argument("--window", default="755:788")
    ap.add_argument("--out", default="/data/livo_sem/out/dyn_ribbon.json")
    a = ap.parse_args()
    w0, w1 = (int(v) for v in a.window.split(":"))

    ts = parse_ts(RAW + "/velodyne_points/timestamps_start.txt")
    te = parse_ts(RAW + "/velodyne_points/timestamps_end.txt")
    traj = S.TrajInterp(a.traj)
    R_IL, t_IL = S.T_I_L[:3, :3], S.T_I_L[:3, 3]

    pts_by_inst = defaultdict(list)      # (cls,inst) -> list of (frame, world pts)
    all_mv = []
    for f in range(a.n_frames):
        sp = RAW + "/velodyne_points/data/%010d.bin" % f
        lp = LB + "/%06d.label" % f
        if not (os.path.exists(sp) and os.path.exists(lp)):
            continue
        p = np.fromfile(sp, dtype=np.float32).reshape(-1, 4)
        lab = np.fromfile(lp, dtype=np.uint32)
        if len(p) != len(lab):
            continue
        sem = (lab & 0xFFFF).astype(np.int32)
        inst = (lab >> 16).astype(np.int32)
        mv = (sem >= 252) & (sem <= 259)
        if not mv.any():
            continue
        _, tsyn, order = synth(p[:, :3], sweep_duration=(te[f] - ts[f]) / 1e9)
        t_pt = np.empty(len(p))
        t_pt[order] = tsyn
        t_pt += ts[f] * 1e-9
        xyz = p[mv, :3].astype(np.float64)
        R, pp, ok = traj.query(t_pt[mv])
        pw = np.einsum("nij,nj->ni", R, xyz @ R_IL.T + t_IL) + pp
        pw, semv, instv = pw[ok], sem[mv][ok], inst[mv][ok]
        all_mv.append(pw)
        for c in np.unique(semv):
            for i in np.unique(instv[semv == c]):
                s = (semv == c) & (instv == i)
                if s.sum() >= 5:
                    pts_by_inst[(int(c), int(i))].append((f, pw[s]))

    P = np.concatenate(all_mv)
    key = S.voxel_key(P, a.voxel)
    uk = np.unique(key)
    res = dict(voxel=a.voxel, moving_points_world=int(len(P)),
               moving_voxels=int(len(uk)))

    if os.path.exists(a.map):
        d = np.load(a.map)
        mk = np.unique(S.voxel_key(d["xyz"].astype(np.float64), a.voxel))
        inmap = np.isin(uk, mk)
        res["map_voxels_total"] = int(len(mk))
        res["moving_voxels_present_in_map"] = int(inmap.sum())
        res["frac_of_map_voxels_touched_by_moving_points"] = float(inmap.sum() / len(mk))

    # per-instance ribbons
    ribs = []
    for (c, i), rows in sorted(pts_by_inst.items()):
        allp = np.concatenate([r[1] for r in rows])
        one = [pca_extent(r[1])[0] for r in rows if len(r[1]) >= 30]
        win = [r[1] for r in rows if w0 <= r[0] < w1]
        e = pca_extent(allp)
        ent = dict(cls=c, name=MOVING[c], inst=i, frames=len(rows),
                   first_frame=rows[0][0], last_frame=rows[-1][0],
                   points=int(len(allp)),
                   lifetime_ribbon_m=e[0], lifetime_extent_wh_m=[e[1], e[2]],
                   one_sweep_extent_median_m=float(np.median(one)) if one else None,
                   voxels=int(len(np.unique(S.voxel_key(allp, a.voxel)))))
        if win:
            wp = np.concatenate(win)
            ew = pca_extent(wp)
            ent["window"] = dict(f0=w0, f1=w1, frames=len(win), points=int(len(wp)),
                                 ribbon_m=ew[0], width_m=ew[1],
                                 voxels=int(len(np.unique(S.voxel_key(wp, a.voxel)))))
        ribs.append(ent)
    ribs.sort(key=lambda z: -z["lifetime_ribbon_m"])
    res["instances"] = ribs
    res["total_moving_ribbon_voxels"] = int(len(uk))

    json.dump(res, open(a.out, "w"), indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != "instances"}, indent=2))
    print("\n%-18s %4s %5s %8s %9s %10s %9s %8s" % ("name", "inst", "frms", "points",
          "ribbon_m", "1sweep_m", "voxels", "win_ribbon"))
    for r in ribs[:14]:
        print("%-18s %4d %5d %8d %9.2f %10s %9d %8s"
              % (r["name"], r["inst"], r["frames"], r["points"], r["lifetime_ribbon_m"],
                 ("%.2f" % r["one_sweep_extent_median_m"]) if r["one_sweep_extent_median_m"] else "-",
                 r["voxels"],
                 ("%.2f" % r["window"]["ribbon_m"]) if "window" in r else "-"))
    print("-> %s" % a.out)


if __name__ == "__main__":
    main()
