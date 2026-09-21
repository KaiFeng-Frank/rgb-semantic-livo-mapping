#!/usr/bin/env python3
"""
replay_v03.py -- CPU-only full-sequence replay of the v0.3 map, for the
threshold CURVES (D4).  No GPU, no ROS, no DDS.

It reproduces stage B exactly -- cached PTv3 argmax/conf (one real GPU pass,
opt/cache_pred.py), the same confidence gate, the same 128-bin de-skew, the same
insert -- and additionally runs FreeSpaceCarver.  At the end it scores the FINAL
map the way opt/score_dynamic.py does, but at O(1) memory: every point's GT
category is accumulated into per-ROW counters while the map is built, so
    dynamic_recall  = sum(mv_cnt[withheld]) / sum(mv_cnt)
    static_false_kill = sum(st_cnt[withheld]) / sum(st_cnt)
is exact for every k_free at once -- the mask is n_free >= k, so ONE replay gives
the whole k curve.

The point-level agreement between this and opt/score_dynamic.py run on the node's
own mask dump is asserted by opt/verify_v03.py (V7).
"""
import argparse, calendar, json, os, sys, time
import numpy as np
sys.path.insert(0, "/data/livo_sem/src")
import sem_core as S
from kitti_scan import synth

D = "/data/livo_sem/data"
RAW = D + "/raw/2011_09_30/2011_09_30_drive_0027_sync"
LB = D + "/odometry/dataset/sequences/07/labels"
MOVING_IDS = np.arange(252, 260)
STATIC_VP_IDS = np.array([10, 11, 13, 15, 16, 18, 20, 30, 31, 32])
IGNORE_IDS = np.array([0, 1, 99])


def parse_ts(path):
    out = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        d, c = line.split(" ")
        hms, frac = (c.split(".") + ["0"])[:2]
        y, mo, dd = (int(v) for v in d.split("-"))
        h, mi, s = (int(v) for v in hms.split(":"))
        out.append(calendar.timegm((y, mo, dd, h, mi, s, 0, 0, 0)) * 10**9
                   + int((frac + "000000000")[:9]))
    return np.array(out, dtype=np.int64)


def pca_len(P):
    if len(P) < 3:
        return 0.0
    c = P - P.mean(0)
    _, V = np.linalg.eigh(c.T @ c / len(c))
    pr = c @ V[:, ::-1]
    return float(pr[:, 0].max() - pr[:, 0].min())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default="/data/livo_sem/out/predcache/a")
    ap.add_argument("--traj", default="/data/livo_sem/out/kitti_seq07_fastlivo2_tum.txt")
    ap.add_argument("--f0", type=int, default=0)
    ap.add_argument("--f1", type=int, default=1101)
    ap.add_argument("--conf-gate", type=float, default=0.5)
    ap.add_argument("--voxel", type=float, default=0.20)
    ap.add_argument("--deskew-bins", type=int, default=128)
    ap.add_argument("--n-az", type=int, default=450)
    ap.add_argument("--n-el", type=int, default=64)
    ap.add_argument("--margin", type=float, default=0.6)
    ap.add_argument("--r-max", type=float, default=25.0)
    ap.add_argument("--r-min", type=float, default=3.0)
    ap.add_argument("--dil-el", type=int, default=0)
    ap.add_argument("--dil-az", type=int, default=0)
    ap.add_argument("--reset-on-seen", type=int, default=1)
    ap.add_argument("--no-carve", action="store_true")
    ap.add_argument("--window", default="755:788")
    ap.add_argument("--json-out", required=True)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    ts = parse_ts(RAW + "/velodyne_points/timestamps_start.txt")
    te = parse_ts(RAW + "/velodyne_points/timestamps_end.txt")
    traj = S.TrajInterp(a.traj)
    R_IL, t_IL = np.ascontiguousarray(S.T_I_L[:3, :3]), np.ascontiguousarray(S.T_I_L[:3, 3])
    vm = S.SemanticVoxelMap(voxel=a.voxel, cap0=1 << 22, hash_cap=1 << 24, dyn=True)
    fc = S.FreeSpaceCarver(n_az=a.n_az, n_el=a.n_el, margin=a.margin, r_max=a.r_max,
                           r_min=a.r_min, k_free=5, reset_on_seen=a.reset_on_seen,
                           dil_el=a.dil_el, dil_az=a.dil_az)
    CAP = 1 << 22
    mv_cnt = np.zeros(CAP, np.uint32)      # GT-moving points landing in each row
    st_cnt = np.zeros(CAP, np.uint32)      # GT-static (non-ignore) points
    vp_cnt = np.zeros(CAP, np.uint32)      # GT static vehicle/person points
    NC = len(S.COARSE)
    ghist = np.zeros(CAP * NC, np.int32)   # per-row GT coarse-label histogram
    sk = S.sk_lut()
    nv_mv = nv_st = nv_vp = 0              # naive class-delete arm, same frames
    n_mv = n_st = n_vp = 0
    w0, w1 = (int(v) for v in a.window.split(":"))
    rib = {}                               # inst -> [(rows, world pts)]
    t_carve = []
    t_ins = []
    t0 = time.time()
    nf = 0
    for f in range(a.f0, a.f1):
        sp = RAW + "/velodyne_points/data/%010d.bin" % f
        pp = os.path.join(a.pred, "f%06d.npz" % f)
        lp = LB + "/%06d.label" % f
        if not (os.path.exists(sp) and os.path.exists(pp) and os.path.exists(lp)):
            continue
        pts = np.fromfile(sp, dtype=np.float32).reshape(-1, 4)
        lab32 = np.fromfile(lp, dtype=np.uint32)
        sem = (lab32 & 0xFFFF).astype(np.int32)
        inst = (lab32 >> 16).astype(np.int32)
        _, tsyn, order = synth(pts[:, :3], sweep_duration=(te[f] - ts[f]) / 1e9)
        bag = np.ascontiguousarray(pts[order])
        sem_b, inst_b = sem[order], inst[order]
        t_pt = ts[f] * 1e-9 + tsyn
        with np.load(pp) as z:
            cls_all = z["lab"].astype(np.uint8)
            cf_all = z["conf"].astype(np.float32)
        ok = traj.valid(t_pt)
        if ok.sum() < 100:
            continue
        keep = ok & (cf_all >= a.conf_gate)
        idx = np.flatnonzero(keep)
        p3 = bag[idx, :3].astype(np.float64)
        cls, cf, tk = cls_all[idx], cf_all[idx], t_pt[idx]
        n = len(idx)
        # ---- de-skew, identical to semantic_map_node.stage_b
        pl_i = p3 @ R_IL.T
        pl_i += t_IL
        t_lo, t_hi = tk[0], tk[-1]
        NB = a.deskew_bins
        ctr = t_lo + (np.arange(NB) + 0.5) * (t_hi - t_lo) / NB
        Rb, pb, _ = traj.query(ctr)
        bi = np.clip(((tk - t_lo) / (t_hi - t_lo) * NB).astype(np.int64), 0, NB - 1)
        edge = np.searchsorted(bi, np.arange(NB + 1))
        pw = np.empty((n, 3))
        for k in range(NB):
            aa, bb = edge[k], edge[k + 1]
            if bb > aa:
                np.matmul(pl_i[aa:bb], Rb[k].T, out=pw[aa:bb])
                pw[aa:bb] += pb[k]
        # ---- free space BEFORE insert (this sweep's own returns must not vote on
        #      the voxels they are about to create)
        if not a.no_carve:
            R_WL = Rb @ R_IL
            o_W = np.einsum("kij,j->ki", Rb, t_IL) + pb
            az = np.arctan2(bag[idx[0], 1], bag[idx[0], 0])
            azl = np.arctan2(bag[idx[-1], 1], bag[idx[-1], 0])
            span = float(np.mod(azl - az, 2 * np.pi))
            if span < 0.2:
                span = 2 * np.pi
            tc = time.perf_counter()
            fc.carve(vm, bag[idx, :3], R_WL, o_W, float(az), span)
            t_carve.append((time.perf_counter() - tc) * 1e3)
        ti = time.perf_counter()
        vm.insert(pw, cls, cf, None, None, want_point_rows=True)
        t_ins.append((time.perf_counter() - ti) * 1e3)
        rows = vm._last_point_rows
        s = sem_b[idx]
        mv = np.isin(s, MOVING_IDS)
        ig = np.isin(s, IGNORE_IDS)
        st = ~mv & ~ig
        vp = np.isin(s, STATIC_VP_IDS)
        np.add.at(mv_cnt, rows[mv], 1)
        np.add.at(st_cnt, rows[st], 1)
        np.add.at(vp_cnt, rows[vp], 1)
        gc = sk[np.clip(s, 0, 299)]
        gm = gc >= 0
        np.add.at(ghist, rows[gm] * NC + gc[gm], 1)
        n_mv += int(mv.sum()); n_st += int(st.sum()); n_vp += int(vp.sum())
        nd = S.MOVABLE_NUSC[cls]
        nv_mv += int((mv & nd).sum()); nv_st += int((st & nd).sum()); nv_vp += int((vp & nd).sum())
        if w0 <= f < w1 and mv.any():
            j = np.flatnonzero(mv)
            for i2 in np.unique(inst_b[idx][j]):
                m2 = inst_b[idx][j] == i2
                rib.setdefault(int(i2), []).append((rows[j][m2], pw[j][m2]))
        nf += 1
    el = time.time() - t0

    out = dict(tag=a.tag, frames=nf, wall_s=el,
               cfg=dict(n_az=a.n_az, n_el=a.n_el, margin=a.margin, r_max=a.r_max,
                        r_min=a.r_min, dil_el=a.dil_el, dil_az=a.dil_az,
                        reset_on_seen=a.reset_on_seen, conf_gate=a.conf_gate,
                        no_carve=bool(a.no_carve), pred=a.pred),
               map_voxels=int(vm.n), candidates=int(vm.n_cand),
               carve_ms=dict(mean=float(np.mean(t_carve)) if t_carve else 0.0,
                             p95=float(np.percentile(t_carve, 95)) if t_carve else 0.0,
                             max=float(np.max(t_carve)) if t_carve else 0.0),
               insert_ms=dict(mean=float(np.mean(t_ins)), p95=float(np.percentile(t_ins, 95))),
               carver_stats=dict(fc.stats),
               points=dict(moving=n_mv, static=n_st, static_vp=n_vp),
               naive_arm=dict(dynamic_recall=100.0 * nv_mv / max(1, n_mv),
                              static_false_kill=100.0 * nv_st / max(1, n_st),
                              vp_static_false_kill=100.0 * nv_vp / max(1, n_vp)))
    n = vm.n
    nfree = vm.n_free[:n]
    curve = []
    for k in (1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 24):
        dyn = nfree >= k
        md = int(mv_cnt[:n][dyn].sum()); sd = int(st_cnt[:n][dyn].sum())
        vd = int(vp_cnt[:n][dyn].sum())
        e = dict(k_free=k, voxels_withheld=int(dyn.sum()),
                 dynamic_recall=100.0 * md / max(1, n_mv),
                 static_false_kill=100.0 * sd / max(1, n_st),
                 vp_static_false_kill=100.0 * vd / max(1, n_vp),
                 damage_ratio=(sd / md) if md else None)
        for inst, lst in sorted(rib.items()):
            R_ = np.concatenate([x[0] for x in lst]); P_ = np.concatenate([x[1] for x in lst])
            alive = ~dyn[R_]
            kv = S.voxel_key(P_[alive], a.voxel) if alive.any() else np.zeros(0, np.int64)
            bv = S.voxel_key(P_, a.voxel)
            e["ribbon_%d" % inst] = dict(
                points_gt=int(len(P_)), points_retained=int(alive.sum()),
                ribbon_m=pca_len(P_[alive]), baseline_ribbon_m=pca_len(P_),
                voxels_retained=int(len(np.unique(kv))),
                voxels_baseline=int(len(np.unique(bv))),
                voxel_reduction_pct=100.0 * (1.0 - len(np.unique(kv)) / max(1, len(np.unique(bv)))))
        curve.append(e)
    out["k_curve"] = curve
    # ---- diagnostics: WHERE does each ribbon voxel's evidence stop?
    diag = {}
    for inst, lst in sorted(rib.items()):
        R_ = np.unique(np.concatenate([x[0] for x in lst]))
        if len(R_) < 20:
            continue
        cand = vm.in_cand[R_]
        nf_ = vm.n_free[R_].astype(np.int64)
        ns_ = vm.n_seen[R_].astype(np.int64)
        dm = vm.dom[R_]
        diag[str(inst)] = dict(
            voxels=int(len(R_)),
            is_candidate=int(cand.sum()),
            dom_hist={S.NUSCENES_CLASSES[c]: int((dm == c).sum())
                      for c in np.unique(dm)},
            never_touched=int(((nf_ == 0) & (ns_ == 0)).sum()),
            seen_only=int(((nf_ == 0) & (ns_ > 0)).sum()),
            nfree_ge1=int((nf_ >= 1).sum()), nfree_ge3=int((nf_ >= 3).sum()),
            nfree_ge5=int((nf_ >= 5).sum()), nfree_ge8=int((nf_ >= 8).sum()),
            nfree_mean=float(nf_.mean()), nseen_mean=float(ns_.mean()))
    out["ribbon_diag"] = diag
    # ---- MAP-LEVEL accuracy, the thing src/eval_report.py structurally cannot see.
    # eval_report scores per-point INFERENCE and never touches the voxel map, and
    # SK_TO_COARSE collapses 252 onto car, so a map-level change registers as
    # exactly zero there.  Here each map voxel gets a GT coarse class (the majority
    # of the labelled points that fell in it) and a predicted one (argmax of the
    # confidence-weighted vote), weighted by the voxel's labelled point count, and
    # the same is recomputed over the voxels that SURVIVE the publish filter.
    nn = vm.n
    gh = ghist[:nn * NC].reshape(nn, NC)
    wt = gh.sum(1)
    have = wt > 0
    gcls = np.argmax(gh, axis=1)
    pcls = S.NUSC16_TO_COARSE[vm.dom[:nn].astype(np.int64)]
    def mapscore(sel):
        m = have & sel
        w = wt[m].astype(np.int64)
        g, pr = gcls[m], pcls[m]
        acc = 100.0 * w[g == pr].sum() / max(1, w.sum())
        ious = {}
        for k in range(NC):
            gk, pk = (g == k), (pr == k)
            if not gk.any() and not pk.any():
                continue
            inter = w[gk & pk].sum(); union = w[gk].sum() + w[pk].sum() - inter
            if gk.any():
                ious[S.COARSE[k]] = 100.0 * inter / max(1, union)
        NINE = ["car", "truck", "other_vehicle", "person", "road", "sidewalk",
                "terrain", "vegetation", "manmade"]
        return dict(voxels=int(m.sum()), labelled_points=int(w.sum()),
                    point_acc=float(acc),
                    coarse_miou=float(np.mean(list(ious.values()))),
                    coarse_miou9=float(np.mean([ious[k] for k in NINE if k in ious])),
                    per_class_iou={k: round(v, 2) for k, v in ious.items()})
    ma = dict(all_voxels=mapscore(np.ones(nn, bool)))
    for k in (5, 12, 24):
        ma["kept_k%d" % k] = mapscore(~(nfree >= k))
    out["map_accuracy"] = ma
    # ---- global: GT-moving-ONLY voxels (mv_cnt>0, st_cnt==0) over the whole run
    nn = vm.n
    mo = (mv_cnt[:nn] > 0) & (st_cnt[:nn] == 0)
    sv = (vp_cnt[:nn] > 0) & (mv_cnt[:nn] == 0)
    out["voxel_level"] = dict(
        moving_only_voxels=int(mo.sum()), static_vp_voxels=int(sv.sum()),
        moving_only_is_candidate=int(vm.in_cand[:nn][mo].sum()),
        static_vp_is_candidate=int(vm.in_cand[:nn][sv].sum()),
        moving_only_withheld_k5=int((nfree[mo] >= 5).sum()),
        static_vp_withheld_k5=int((nfree[sv] >= 5).sum()),
        moving_only_nfree_ge1=int((nfree[mo] >= 1).sum()),
        moving_only_never_touched=int(((nfree[mo] == 0) & (vm.n_seen[:nn][mo] == 0)).sum()))
    json.dump(out, open(a.json_out, "w"), indent=2)
    print("%s  frames %d  %.0fs  voxels %d  cand %d  carve %.2f ms" %
          (a.tag, nf, el, vm.n, vm.n_cand, out["carve_ms"]["mean"]))
    for e in curve:
        if e["k_free"] in (1, 3, 5, 8, 12):
            print("   k=%-3d recall %6.2f %%  static-FK %6.3f %%  vp-FK %6.3f %%  "
                  "rib7 %5.1f m (%d vox)" %
                  (e["k_free"], e["dynamic_recall"], e["static_false_kill"],
                   e["vp_static_false_kill"],
                   e.get("ribbon_7", {}).get("ribbon_m", -1),
                   e.get("ribbon_7", {}).get("voxels_retained", -1)))


if __name__ == "__main__":
    main()
