#!/usr/bin/env python3
"""
measure_dynamic.py -- quantify the dynamic-object problem on SemanticKITTI seq 07.

CPU ONLY, numpy only.  No GPU, no torch, no ROS.  Reads
  * raw scans  .../raw/2011_09_30/2011_09_30_drive_0027_sync/velodyne_points/data/*.bin
  * GT labels  .../odometry/dataset/sequences/07/labels/*.label   (index-aligned, offset 0)
  * sweep start/end timestamps from the raw velodyne_points/timestamps_{start,end}.txt
  * the deployed trajectory out/kitti_seq07_fastlivo2_tum.txt  (T_{W<-IMU})

Everything the node does to a point before it lands in the map is reproduced here
analytically: synthesised per-point time (kitti_scan.synth), T_I_L right-multiply,
per-point de-skew, cam2 projection.  So distances, frustum membership and world
positions are the ones the pipeline actually sees.
"""
import argparse, json, os, sys
from collections import defaultdict
import numpy as np

sys.path.insert(0, "/data/livo_sem/src")
import sem_core as S
from kitti_scan import synth

D = "/data/livo_sem/data"
RAW = D + "/raw/2011_09_30/2011_09_30_drive_0027_sync"
SCANS = RAW + "/velodyne_points/data"
LABELS = D + "/odometry/dataset/sequences/07/labels"

MOVING = {252: "moving-car", 253: "moving-bicyclist", 254: "moving-person",
          255: "moving-motorcyclist", 256: "moving-on-rails", 257: "moving-bus",
          258: "moving-truck", 259: "moving-other-vehicle"}
STATIC = {10: "car", 11: "bicycle", 13: "bus", 15: "motorcycle", 18: "truck",
          20: "other-vehicle", 30: "person", 31: "bicyclist", 32: "motorcyclist",
          16: "on-rails"}
IMG_W, IMG_H = 1226, 370


def parse_ts_file(path):
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
        secs = calendar.timegm((y, mo, d, h, mi, s, 0, 0, 0))
        out.append(secs * 1_000_000_000 + int((frac + "000000000")[:9]))
    return np.array(out, dtype=np.int64)


def q(a, p):
    return float(np.percentile(a, p)) if len(a) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-frames", type=int, default=1101)
    ap.add_argument("--traj", default="/data/livo_sem/out/kitti_seq07_fastlivo2_tum.txt")
    ap.add_argument("--out", default="/data/livo_sem/out/dyn_scale.json")
    a = ap.parse_args()

    t_start = parse_ts_file(RAW + "/velodyne_points/timestamps_start.txt")
    t_end = parse_ts_file(RAW + "/velodyne_points/timestamps_end.txt")
    traj = S.TrajInterp(a.traj)
    print("traj window [%.6f, %.6f]  %d poses" % (traj.t0, traj.t1, len(traj.t)))
    print("sweep 0 start %.6f  end %.6f" % (t_start[0] * 1e-9, t_end[0] * 1e-9))

    R_IL, t_IL = S.T_I_L[:3, :3], S.T_I_L[:3, 3]
    M_CL = S.T_C_L

    ALLC = sorted(set(MOVING) | set(STATIC))
    per_frame = {c: np.zeros(a.n_frames, np.int64) for c in ALLC}
    n_pts = np.zeros(a.n_frames, np.int64)

    # geometry accumulators for moving points
    g_rng, g_hrng, g_hgt, g_front, g_infr = [], [], [], [], []
    g_rng_cls = defaultdict(list)
    ground_z = []
    # per (class, instance) tracks
    tracks = defaultdict(list)          # (cls,inst) -> [(frame, n, cx,cy,cz)]
    inst_pts = defaultdict(int)         # (cls,inst) -> total points
    inst_frames = defaultdict(set)
    moving_inst0 = 0
    n_moving_total = 0
    frustum_num = frustum_den = 0
    front_num = 0

    for f in range(a.n_frames):
        sp = os.path.join(SCANS, "%010d.bin" % f)
        lp = os.path.join(LABELS, "%06d.label" % f)
        if not (os.path.exists(sp) and os.path.exists(lp)):
            continue
        p = np.fromfile(sp, dtype=np.float32).reshape(-1, 4)
        lab = np.fromfile(lp, dtype=np.uint32)
        if len(p) != len(lab):
            print("SKIP frame %d: %d pts vs %d labels" % (f, len(p), len(lab)))
            continue
        sem = (lab & 0xFFFF).astype(np.int32)
        inst = (lab >> 16).astype(np.int32)
        n_pts[f] = len(p)
        for c in ALLC:
            per_frame[c][f] = int((sem == c).sum())

        # ground plane reference: median z of GT road points within 25 m
        road = (sem == 40)
        if road.sum() > 500:
            xy = np.linalg.norm(p[road, :2], axis=1)
            near = road.copy()
            near[road] = xy < 25.0
            if near.sum() > 200:
                ground_z.append(float(np.median(p[near, 2])))
        gz = ground_z[-1] if ground_z else -1.73

        mv = (sem >= 252) & (sem <= 259)
        if not mv.any():
            continue
        n_moving_total += int(mv.sum())
        xyz = p[mv, :3].astype(np.float64)
        semv, instv = sem[mv], inst[mv]
        moving_inst0 += int((instv == 0).sum())

        r = np.linalg.norm(xyz, axis=1)
        hr = np.linalg.norm(xyz[:, :2], axis=1)
        g_rng.append(r); g_hrng.append(hr); g_hgt.append(xyz[:, 2] - gz)
        for c in np.unique(semv):
            g_rng_cls[int(c)].append(r[semv == c])

        # camera frustum, exactly the node's test (z > 0.5, u,v inside image)
        pc = xyz @ M_CL[:3, :3].T + M_CL[:3, 3]
        zc = pc[:, 2]
        front = zc > 0.5
        u = np.full(len(xyz), -1e9); v = np.full(len(xyz), -1e9)
        u[front] = pc[front, 0] / zc[front] * S.FX + S.CX
        v[front] = pc[front, 1] / zc[front] * S.FY + S.CY
        inb = front & (u >= 0) & (u <= IMG_W - 1) & (v >= 0) & (v <= IMG_H - 1)
        frustum_num += int(inb.sum()); frustum_den += len(xyz); front_num += int(front.sum())

        # world centroid per (class, instance), through the node's chain
        _, tsyn, order = synth(p[:, :3])
        t_pt = np.empty(len(p))
        t_pt[order] = tsyn                        # back to .bin order
        t_pt += t_start[f] * 1e-9
        tm = t_pt[mv]
        R, pp, ok = traj.query(tm)
        pw = np.einsum("nij,nj->ni", R, xyz @ R_IL.T + t_IL) + pp
        for c in np.unique(semv):
            for i in np.unique(instv[semv == c]):
                sel = (semv == c) & (instv == i) & ok
                if sel.sum() < 10:
                    continue
                cw = pw[sel].mean(0)
                tracks[(int(c), int(i))].append((f, int(sel.sum()),
                                                 float(cw[0]), float(cw[1]), float(cw[2])))
                inst_pts[(int(c), int(i))] += int(sel.sum())
                inst_frames[(int(c), int(i))].add(f)
        if f % 200 == 0:
            print("  frame %4d  moving %6d" % (f, int(mv.sum())), flush=True)

    # --- also count STATIC instances (cheap second pass over labels only)
    sinst_pts = defaultdict(int)
    sinst_frames = defaultdict(set)
    for f in range(a.n_frames):
        lp = os.path.join(LABELS, "%06d.label" % f)
        if not os.path.exists(lp):
            continue
        lab = np.fromfile(lp, dtype=np.uint32)
        sem = (lab & 0xFFFF).astype(np.int32)
        inst = (lab >> 16).astype(np.int32)
        m = np.isin(sem, list(STATIC))
        if not m.any():
            continue
        k = sem[m].astype(np.int64) * 100000 + inst[m].astype(np.int64)
        uk, cnt = np.unique(k, return_counts=True)
        for kk, cc in zip(uk, cnt):
            key = (int(kk // 100000), int(kk % 100000))
            sinst_pts[key] += int(cc)
            sinst_frames[key].add(f)

    res = {}
    mv_tot = sum(per_frame[c] for c in MOVING)
    st_tot = sum(per_frame[c] for c in (10, 11, 13, 15, 18, 20, 30))
    res["frames"] = int(a.n_frames)
    res["points_total"] = int(n_pts.sum())
    res["points_per_frame_mean"] = float(n_pts.mean())
    res["ground_z_median_velo"] = float(np.median(ground_z))

    def dist(x):
        return dict(mean=float(x.mean()), median=float(np.median(x)),
                    p95=q(x, 95), p99=q(x, 99), max=int(x.max()),
                    zero_frames=int((x == 0).sum()),
                    frac_of_points_mean=float(x.sum() / max(1, n_pts.sum())))
    res["moving_all"] = dist(mv_tot)
    res["static_vehicle_person"] = dist(st_tot)
    res["per_class"] = {}
    for c in ALLC:
        x = per_frame[c]
        if x.sum() == 0:
            continue
        res["per_class"][str(c)] = dict(
            name=MOVING.get(c, STATIC.get(c, "?")), total=int(x.sum()),
            frames_present=int((x > 0).sum()), mean=float(x.mean()),
            median=float(np.median(x)), p95=q(x, 95), max=int(x.max()))

    rng = np.concatenate(g_rng); hrng = np.concatenate(g_hrng); hgt = np.concatenate(g_hgt)
    res["moving_geometry"] = dict(
        n_points=int(len(rng)),
        range_m=dict(mean=float(rng.mean()), median=float(np.median(rng)),
                     p05=q(rng, 5), p95=q(rng, 95), max=float(rng.max())),
        horiz_range_m=dict(median=float(np.median(hrng)), p95=q(hrng, 95)),
        height_above_ground_m=dict(median=float(np.median(hgt)), p05=q(hgt, 5),
                                   p95=q(hgt, 95), max=float(hgt.max())),
        frac_in_front_of_cam=float(front_num / max(1, frustum_den)),
        frac_in_cam_frustum=float(frustum_num / max(1, frustum_den)),
        moving_points_with_instance0=int(moving_inst0))
    res["moving_range_by_class"] = {
        str(c): dict(name=MOVING[c], median=float(np.median(np.concatenate(v))),
                     p95=q(np.concatenate(v), 95))
        for c, v in sorted(g_rng_cls.items())}

    # --- tracks
    sweep_dt = float(np.median(np.diff(t_start)) * 1e-9)
    res["sweep_period_s"] = sweep_dt
    tr = []
    for key, rows in sorted(tracks.items()):
        rows.sort()
        if len(rows) < 3:
            speed = None
        C = np.array([[r[2], r[3], r[4]] for r in rows])
        F = np.array([r[0] for r in rows])
        seg = np.linalg.norm(np.diff(C, axis=0), axis=1)
        dt = np.diff(F) * sweep_dt
        good = dt > 0
        sp = seg[good] / dt[good]
        tr.append(dict(cls=key[0], name=MOVING.get(key[0], STATIC.get(key[0], "?")),
                       inst=key[1], n_frames=len(rows),
                       first_frame=int(F[0]), last_frame=int(F[-1]),
                       points=int(inst_pts[key]),
                       path_length_m=float(seg.sum()),
                       net_displacement_m=float(np.linalg.norm(C[-1] - C[0])),
                       duration_s=float((F[-1] - F[0]) * sweep_dt),
                       speed_median_mps=float(np.median(sp)) if len(sp) else None,
                       speed_p95_mps=q(sp, 95) if len(sp) else None,
                       disp_per_sweep_m=float(np.median(sp) * sweep_dt) if len(sp) else None))
    tr.sort(key=lambda z: -z["path_length_m"])
    res["moving_tracks"] = tr
    res["n_moving_instances"] = len(tracks)
    res["n_static_instances"] = {}
    for c in STATIC:
        ks = [k for k in sinst_pts if k[0] == c]
        if ks:
            res["n_static_instances"][STATIC[c]] = dict(
                n_instances=len(ks), points=int(sum(sinst_pts[k] for k in ks)))
    fast = [t for t in tr if t["speed_median_mps"] and t["speed_median_mps"] > 1.0]
    res["moving_tracks_over_1mps"] = len(fast)
    if fast:
        sp = np.array([t["speed_median_mps"] for t in fast])
        res["moving_speed_mps"] = dict(median=float(np.median(sp)), p95=q(sp, 95),
                                       max=float(sp.max()))
        d = np.array([t["disp_per_sweep_m"] for t in fast])
        res["moving_disp_per_sweep_m"] = dict(median=float(np.median(d)), p95=q(d, 95),
                                              max=float(d.max()))

    json.dump(res, open(a.out, "w"), indent=2)
    print("\n-> %s" % a.out)
    print(json.dumps({k: v for k, v in res.items()
                      if k not in ("moving_tracks", "per_class", "moving_range_by_class")},
                     indent=2))


if __name__ == "__main__":
    main()
