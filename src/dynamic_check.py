#!/usr/bin/env python3
"""
dynamic_check.py -- acceptance criterion 5: do dynamic vehicles / people smear?

SemanticKITTI labels seq 07 with explicit MOVING classes (252 moving-car,
253 moving-bicyclist, 254 moving-person, 255/256/257/258/259 moving-other), so
the moving objects do not have to be guessed.  For the frame with the most
moving-car returns this script

  1. maps that frame's moving-car points into the world with exactly the node's
     transform chain  (T_W_I(t) interpolated, right-multiplied by T_I_L, per-point
     de-skewed), giving the car's true instantaneous footprint;
  2. measures the footprint's extent along its own principal axis -- the length
     of a real car, ~4.5 m, is the yardstick;
  3. measures the extent of everything the accumulated map holds at that place,
     i.e. how far the car's returns have been dragged along the road by having
     been observed over many sweeps;
  4. prints an RViz focal point so the same place can be looked at by eye.

A static object accumulated over K sweeps stays its own size.  A moving object
accumulated over K sweeps is smeared by (speed x observation span), which is a
property of the SCENE, not of the pipeline -- a mapper with no dynamic-object
handling must smear it.  The test that actually probes SYNC error is the
instantaneous footprint in (1): if the timestamps were mismatched, one sweep of
a moving car would already be elongated, because different parts of the sweep
would be placed with the wrong pose.
"""
import argparse
import json
import sys
import numpy as np

sys.path.insert(0, "/data/livo_sem/src")
import sem_core as S
from a3_check import read_scans

MOVING = {252: "car", 253: "bicyclist", 254: "person", 255: "motorcyclist",
          256: "on-rails", 257: "bus", 258: "truck", 259: "other-vehicle"}


def principal_extent(P):
    if len(P) < 3:
        return 0.0, 0.0, 0.0
    c = P - P.mean(0)
    w, V = np.linalg.eigh(c.T @ c / len(c))
    proj = c @ V[:, ::-1]
    return tuple(float(proj[:, k].max() - proj[:, k].min()) for k in range(3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="/data/livo_sem/data/odometry/dataset/sequences/07/labels")
    ap.add_argument("--bag", default="/data/livo_sem/bags/kitti_seq07_us")
    ap.add_argument("--traj", default="/data/livo_sem/out/kitti_seq07_fastlivo2_tum.txt")
    ap.add_argument("--map", default="/data/livo_sem/out/map_seq07.npz")
    ap.add_argument("--out", default="/data/livo_sem/out/dynamic_check.json")
    ap.add_argument("--stride", type=int, default=1)
    a = ap.parse_args()

    import os
    counts = []
    for f in range(0, 1101, a.stride):
        p = os.path.join(a.labels, "%06d.label" % f)
        if not os.path.exists(p):
            continue
        sem = np.fromfile(p, dtype=np.uint32) & 0xFFFF
        c = {k: int((sem == k).sum()) for k in MOVING}
        counts.append((f, c, sum(c.values())))
    best = max(counts, key=lambda x: x[1][252])
    best_any = max(counts, key=lambda x: x[2])
    print("frames with moving-car returns: %d / %d"
          % (sum(1 for _, c, _ in counts if c[252] > 0), len(counts)))
    print("richest moving-car frame: %06d with %d points" % (best[0], best[1][252]))
    print("richest any-moving frame : %06d  %s"
          % (best_any[0], {MOVING[k]: v for k, v in best_any[1].items() if v}))

    traj = S.TrajInterp(a.traj)
    res = {"richest_moving_car_frame": best[0], "moving_car_points": best[1][252],
           "objects": []}

    for frame, cls_id, name in ((best[0], 252, "moving-car"),
                                (best_any[0], 252, "moving-car(alt frame)")):
        sem = np.fromfile(os.path.join(a.labels, "%06d.label" % frame),
                          dtype=np.uint32) & 0xFFFF
        sc = read_scans(a.bag, "/velodyne_points", frame, frame + 1)
        if not sc:
            continue
        xyz, tpt = sc[0]
        if len(sem) != len(xyz):
            print("WARNING frame %d: %d labels vs %d points" % (frame, len(sem), len(xyz)))
            continue
        m = sem == cls_id
        if m.sum() < 20:
            continue
        pl = np.hstack([xyz[m], np.ones((m.sum(), 1))])
        pl_i = (pl @ S.T_I_L.T)[:, :3]
        R, p, ok = traj.query(tpt[m])
        pw = np.einsum("nij,nj->ni", R, pl_i) + p
        pw = pw[ok]
        # separate the individual vehicles: simple grid-connected clustering
        gb = np.floor(pw / 1.0).astype(np.int64)
        from collections import defaultdict
        occ = set(map(tuple, gb))
        seen, clusters = set(), []
        for v in occ:
            if v in seen:
                continue
            stack, comp = [v], []
            seen.add(v)
            while stack:
                u = stack.pop()
                comp.append(u)
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        for dz in (-1, 0, 1):
                            w = (u[0] + dx, u[1] + dy, u[2] + dz)
                            if w in occ and w not in seen:
                                seen.add(w)
                                stack.append(w)
            clusters.append(set(comp))
        gbt = list(map(tuple, gb))
        for ci, comp in enumerate(sorted(clusters, key=len, reverse=True)[:4]):
            sel = np.array([t in comp for t in gbt])
            P = pw[sel]
            if len(P) < 30:
                continue
            L, W, H = principal_extent(P)
            ent = dict(frame=int(frame), object=name, cluster=ci, n_points=int(len(P)),
                       centroid=[float(x) for x in P.mean(0)],
                       instantaneous_extent_m=[L, W, H])
            if a.map:
                d = np.load(a.map)
                mx = d["xyz"]
                c = P.mean(0)
                near = (np.abs(mx - c) < np.array([8.0, 8.0, 4.0])).all(1)
                if near.sum() > 30:
                    mcls = d["cls"][near]
                    carm = mcls == 3
                    ent["map_points_within_8m"] = int(near.sum())
                    ent["map_car_points_within_8m"] = int(carm.sum())
                    if carm.sum() > 20:
                        ent["map_car_extent_m"] = list(principal_extent(mx[near][carm]))
            res["objects"].append(ent)
            print("  %s cluster %d: %d pts, centroid (%.1f, %.1f, %.1f), "
                  "instantaneous L x W x H = %.2f x %.2f x %.2f m"
                  % (name, ci, len(P), *ent["centroid"], L, W, H))
            if "map_car_extent_m" in ent:
                print("      map `car` voxels within 8 m: %d, extent %.2f x %.2f x %.2f m"
                      % (ent["map_car_points_within_8m"], *ent["map_car_extent_m"]))
            print("      RViz focal point: %.2f; %.2f; %.2f" % tuple(ent["centroid"]))
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2)
    print("-> %s" % a.out)


if __name__ == "__main__":
    main()
