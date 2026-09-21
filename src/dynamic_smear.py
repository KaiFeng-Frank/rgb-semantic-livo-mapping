#!/usr/bin/env python3
"""
dynamic_smear.py -- criterion 5, as a CONTROLLED test.

SemanticKITTI seq 07 labels static cars 10 and MOVING cars 252.  Both are put
through exactly the node's transform chain (interpolated T_W_I, right-multiplied
by T_I_L, per-point de-skew) and, for each isolated single-vehicle cluster, the
length of its ONE-SWEEP world footprint is measured.

Why this is the right test for SYNC error and not for accumulation:
  * a parked car's footprint is its own length whatever the pose error is,
    because every return in the sweep gets essentially the same pose;
  * a MOVING car at 10-14 m/s traverses 1.0-1.5 m during one 104 ms sweep, so if
    the sweep were placed with the wrong pose, or de-skewed with the wrong time,
    the moving car's footprint would be stretched along its travel direction
    while the parked ones stayed sharp.
  => moving-car length ~= parked-car length ~= a real car  <=>  the sweep is
     being placed at the right time.

Accumulated smear across many sweeps is a separate, physical matter: a mapper
with no dynamic-object handling necessarily paints a moving car along its whole
path.  That is measured separately below as the trail length.
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, "/data/livo_sem/src")
import sem_core as S
from a3_check import read_scans


def clusters(P, cell=1.0, min_pts=35, max_pts=3000, max_extent=12.0):
    g = np.floor(P / cell).astype(np.int64)
    keys = list(map(tuple, g))
    occ = {}
    for i, k in enumerate(keys):
        occ.setdefault(k, []).append(i)
    seen, out = set(), []
    for k in occ:
        if k in seen:
            continue
        stack, comp = [k], []
        seen.add(k)
        while stack:
            u = stack.pop(); comp.append(u)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        w = (u[0]+dx, u[1]+dy, u[2]+dz)
                        if w in occ and w not in seen:
                            seen.add(w); stack.append(w)
        idx = np.concatenate([occ[c] for c in comp])
        if not (min_pts <= len(idx) <= max_pts):
            continue
        Q = P[idx]
        c = Q - Q.mean(0)
        w, V = np.linalg.eigh(c.T @ c / len(c))
        pr = c @ V[:, ::-1]
        L = float(pr[:, 0].max() - pr[:, 0].min())
        W = float(pr[:, 1].max() - pr[:, 1].min())
        H = float(pr[:, 2].max() - pr[:, 2].min())
        if L > max_extent:
            continue
        out.append((len(idx), L, W, H, Q.mean(0)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="/data/livo_sem/data/odometry/dataset/sequences/07/labels")
    ap.add_argument("--bag", default="/data/livo_sem/bags/kitti_seq07_us")
    ap.add_argument("--traj", default="/data/livo_sem/out/kitti_seq07_fastlivo2_tum.txt")
    ap.add_argument("--stride", type=int, default=12)
    ap.add_argument("--out", default="/data/livo_sem/out/dynamic_smear.json")
    a = ap.parse_args()
    traj = S.TrajInterp(a.traj)

    res = {"moving": [], "parked": []}
    trails = []
    frames = list(range(0, 1101, a.stride))
    scans = read_scans(a.bag, "/velodyne_points", 0, 1101)
    print("loaded %d scans" % len(scans))
    for f in frames:
        p = os.path.join(a.labels, "%06d.label" % f)
        if not os.path.exists(p) or f >= len(scans):
            continue
        sem = np.fromfile(p, dtype=np.uint32) & 0xFFFF
        xyz, tpt = scans[f]
        if len(sem) != len(xyz):
            continue
        for key, cid in (("moving", 252), ("parked", 10)):
            m = sem == cid
            if m.sum() < 35:
                continue
            pl = np.hstack([xyz[m], np.ones((m.sum(), 1))])
            pli = (pl @ S.T_I_L.T)[:, :3]
            R, pp, ok = traj.query(tpt[m])
            if ok.sum() < 35:
                continue
            pw = (np.einsum("nij,nj->ni", R, pli) + pp)[ok]
            for n, L, W, H, c in clusters(pw):
                res[key].append(dict(frame=int(f), n=int(n), L=L, W=W, H=H,
                                     centroid=[float(x) for x in c]))
    out = {"n_frames_scanned": len(frames)}
    for k in ("moving", "parked"):
        L = np.array([r["L"] for r in res[k]])
        W = np.array([r["W"] for r in res[k]])
        out[k] = dict(n_clusters=int(len(L)),
                      L_median=float(np.median(L)) if len(L) else None,
                      L_mean=float(L.mean()) if len(L) else None,
                      L_p90=float(np.percentile(L, 90)) if len(L) else None,
                      W_median=float(np.median(W)) if len(W) else None)
        print("%-7s single-vehicle clusters: %4d   one-sweep footprint length "
              "median %.2f m  mean %.2f m  p90 %.2f m   width median %.2f m"
              % (k, len(L), np.median(L) if len(L) else -1,
                 L.mean() if len(L) else -1,
                 np.percentile(L, 90) if len(L) else -1,
                 np.median(W) if len(W) else -1))

    # --- accumulated trail of ONE moving car, tracked frame to frame
    best = None
    for f0 in range(0, 1060, 5):
        track, last = [], None
        for f in range(f0, min(f0 + 60, 1101)):
            p = os.path.join(a.labels, "%06d.label" % f)
            if not os.path.exists(p) or f >= len(scans):
                break
            sem = np.fromfile(p, dtype=np.uint32) & 0xFFFF
            xyz, tpt = scans[f]
            if len(sem) != len(xyz):
                break
            m = sem == 252
            if m.sum() < 35:
                break
            pl = np.hstack([xyz[m], np.ones((m.sum(), 1))])
            pli = (pl @ S.T_I_L.T)[:, :3]
            R, pp, ok = traj.query(tpt[m])
            if ok.sum() < 35:
                break
            pw = (np.einsum("nij,nj->ni", R, pli) + pp)[ok]
            cl = clusters(pw)
            if not cl:
                break
            if last is None:
                cl.sort(key=lambda z: -z[0]); pick = cl[0]
            else:
                cl.sort(key=lambda z: np.linalg.norm(z[4] - last))
                pick = cl[0]
                if np.linalg.norm(pick[4] - last) > 3.0:
                    break
            last = pick[4]; track.append((f, pick[4], pick[1]))
        if len(track) >= 8 and (best is None or len(track) > len(best)):
            best = track
    if best:
        C = np.array([t[1] for t in best])
        span = float(np.linalg.norm(C[-1] - C[0]))
        dur = (best[-1][0] - best[0][0]) * 0.1039
        out["tracked_moving_car"] = dict(
            first_frame=best[0][0], last_frame=best[-1][0], n_frames=len(best),
            path_length_m=span, duration_s=dur, speed_mps=span / max(dur, 1e-6),
            one_sweep_length_median_m=float(np.median([t[2] for t in best])),
            expected_accumulated_trail_m=span + float(np.median([t[2] for t in best])),
            centroid_first=[float(x) for x in C[0]],
            centroid_last=[float(x) for x in C[-1]])
        print("\ntracked ONE moving car over frames %d..%d (%d sweeps, %.2f s):"
              % (best[0][0], best[-1][0], len(best), dur))
        print("  it travelled %.1f m (%.1f m/s); its one-sweep footprint stayed %.2f m long"
              % (span, span / max(dur, 1e-6), np.median([t[2] for t in best])))
        print("  => the map must hold a ~%.0f m trail there; that is physics, not sync error"
              % (span + np.median([t[2] for t in best])))
        print("  RViz focal point: %.2f; %.2f; %.2f" % tuple(C[len(C)//2]))
    json.dump(out, open(a.out, "w"), indent=2)
    print("-> %s" % a.out)


if __name__ == "__main__":
    main()
