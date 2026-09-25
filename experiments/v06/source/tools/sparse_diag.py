#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
opt/replay_v05.py -- MAP-LEVEL semantic quality of the deployed pipeline (v0.5, M2).

Builds on opt/replay_v03.py.  Reused from it: the CPU replay of stage B (cached per-point
PTv3 argmax/conf -> confidence gate -> 128-bin de-skew through TrajInterp -> T_I_L ->
SemanticVoxelMap.insert with want_point_rows), the timestamp parser, and the accumulation
idea of its `map_accuracy`: a per-ROW ground-truth histogram filled while the map is built,
scored against the row's fused class at the end (its voxel-majority `mapscore` is
reproduced here verbatim as `majority`).  What is added:

  * the COMMON-9 label space of the v0.4 offline scorer (label_spaces.sk_lut / nusc_lut:
    GT EXCLUDED dropped, predicted other_flat UNMAPPED = always wrong), so every number is
    directly comparable to out/v04/arms/score_*.json;
  * the in-frustum / out-of-frustum split of that scorer (score_2d_vs_3d.Projector, P3
    rule on the RAW sensor-frame point, via seqreg.use("07")), carried per point through
    the voxelisation as TWO histograms per row;
  * the per-point-GT reading of the map (each labelled point scored against the fused
    class of the voxel it fell in) next to replay_v03's voxel-majority reading;
  * a lookup-based reading over ALL evaluated points -- gated-out points get the class of
    the voxel they would have landed in, no voxel = ABSTAIN -- so the offline->map gap can
    be decomposed into gate selection / fusion / coverage;
  * optional scoring of a LIVE map .npz written by the node (--live-npz), by voxel-key
    lookup from the replay's world positions.

Five readings per subset (out-of-frustum / in-frustum / global):
  offline_all         per-point prediction vs GT, every evaluated point   (== the v0.4 scorer)
  offline_inserted    the same, restricted to the points the pipeline inserted (pose ok, conf >= gate)
  map_pointgt_inserted  fused voxel class vs each inserted point's own GT
  map_majority_inserted fused voxel class vs the voxel's GT majority, point-weighted (replay_v03 definition)
  map_all_lookup      fused voxel class for EVERY evaluated point (gated-out looked up; missing = abstain)

CAVEAT carried over from v0.3 (D5): a coarse label space scores a correctly-classified ghost
(a moving car's trail labelled `car`) as a true positive.  Nothing here withholds anything:
the operating point of this qualification is --dyn OFF, exactly the v0.2 fusion.
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, "/data/livo_sem/src")
sys.path.insert(0, "/data/livo_sem/opt")
import sem_core as S
import label_spaces as LS
from kitti_scan import synth
from replay_v03 import parse_ts, RAW, LB
import seqreg
from scipy.spatial import cKDTree as KDT_     # sparse_diag: module level, so main() never shadows it

K = len(LS.COARSE)
NAMES = list(LS.COARSE)
UNM = int(LS.UNMAPPED)          # -2 prediction side, always wrong
ABST = -1                       # prediction side, no answer
SK = np.asarray(LS.sk_lut(), np.int32)      # raw SemanticKITTI id -> 0..8 | -1 EXCLUDED
NU = np.asarray(LS.nusc_lut(), np.int32)    # nuScenes-16 -> 0..8 | -2 UNMAPPED
SUBS = ("outside", "frustum", "global")


class Conf(object):
    """score_2d_vs_3d.Acc conventions: C[gt, pred] with an extra UNMAPPED column
    (answered, always wrong, never dropped); A[gt] for ABSTAINED (no voxel)."""

    def __init__(self):
        self.C = np.zeros((K, K + 1), np.int64)
        self.A = np.zeros(K, np.int64)

    def add(self, gt, pred):
        if gt.size == 0:
            return
        gt = gt.astype(np.int64); pred = pred.astype(np.int64)
        ab = pred == ABST
        if ab.any():
            self.A += np.bincount(gt[ab], minlength=K)
        g = gt[~ab]; p = pred[~ab]
        if g.size:
            p = np.where(p == UNM, K, p)
            self.C += np.bincount(g * (K + 1) + p, minlength=K * (K + 1)).reshape(K, K + 1)

    def add_hist(self, hist, pred_row):
        """hist (R,K) GT point counts per row; pred_row (R,) fused class per row."""
        pr = np.where(pred_row == UNM, K, pred_row)
        for k in range(K + 1):
            sel = pr == k
            if sel.any():
                self.C[:, k] += hist[sel].sum(0, dtype=np.int64)
        ab = pred_row == ABST
        if ab.any():
            self.A += hist[ab].sum(0, dtype=np.int64)

    def merge(self, o):
        r = Conf(); r.C = self.C + o.C; r.A = self.A + o.A
        return r

    def metrics(self):
        C, A = self.C, self.A
        row = C.sum(1); col = C[:, :K].sum(0); diag = np.diag(C[:, :K])
        n_ans = int(C.sum()); n_ab = int(A.sum()); n = n_ans + n_ab
        present = (row + A) > 0
        iou_ab, iou_st = {}, {}
        for k in range(K):
            if not present[k]:
                continue
            u = row[k] + col[k] - diag[k]
            iou_ab[NAMES[k]] = 100.0 * diag[k] / u if u else 0.0
            u2 = u + A[k]
            iou_st[NAMES[k]] = 100.0 * diag[k] / u2 if u2 else 0.0
        return dict(n_eval=n, n_labelled=n_ans, n_abstained=n_ab,
                    n_unmapped_pred=int(C[:, K].sum()),
                    coverage=(n_ans / n if n else float("nan")),
                    acc_abstain_excluded=(100.0 * diag.sum() / n_ans if n_ans else float("nan")),
                    acc_abstain_wrong=(100.0 * diag.sum() / n if n else float("nan")),
                    miou9_abstain_excluded=float(np.mean(list(iou_ab.values()))) if iou_ab else float("nan"),
                    miou9_abstain_wrong=float(np.mean(list(iou_st.values()))) if iou_st else float("nan"),
                    n_classes=len(iou_ab),
                    per_class_iou_abstain_excluded={k: round(v, 3) for k, v in iou_ab.items()},
                    per_class_iou_abstain_wrong={k: round(v, 3) for k, v in iou_st.items()})


def majority(hist, pred_row):
    """replay_v03.map_accuracy's `mapscore`, in common-9: each row gets the GT class that
    is the MAJORITY of its labelled points, weighted by that count."""
    c = Conf()
    w = hist.sum(1, dtype=np.int64)
    m = w > 0
    g = hist[m].argmax(1).astype(np.int64)
    p = pred_row[m].astype(np.int64)
    p = np.where(p == UNM, K, p)
    ab = p == ABST
    if ab.any():
        c.A += np.bincount(g[ab], weights=w[m][ab], minlength=K).astype(np.int64)
    ok = ~ab
    c.C += np.bincount(g[ok] * (K + 1) + p[ok], weights=w[m][ok],
                       minlength=K * (K + 1)).astype(np.int64).reshape(K, K + 1)
    return c


def deskew(p3_f32, tk, Rb, pb, t_lo, t_hi, NB, R_IL, t_IL):
    """Identical to semantic_map_node.stage_b: T_I_L, then the bin pose.  tk sorted."""
    pl_i = p3_f32.astype(np.float64) @ R_IL.T
    pl_i += t_IL
    n = len(pl_i)
    bi = np.clip(((tk - t_lo) / (t_hi - t_lo) * NB).astype(np.int64), 0, NB - 1)
    edge = np.searchsorted(bi, np.arange(NB + 1))
    pw = np.empty((n, 3))
    for k in range(NB):
        a, b = edge[k], edge[k + 1]
        if b > a:
            np.matmul(pl_i[a:b], Rb[k].T, out=pw[a:b])
            pw[a:b] += pb[k]
    return pw


class KeyIndex(object):
    """read-only voxel-key -> row lookup (sorted keys + searchsorted)."""

    def __init__(self, keys, values):
        o = np.argsort(keys, kind="stable")
        self.k = keys[o]; self.v = values[o]

    def lookup(self, q):
        i = np.searchsorted(self.k, q)
        i = np.minimum(i, len(self.k) - 1)
        hit = self.k[i] == q
        out = np.full(len(q), ABST, np.int64)
        out[hit] = self.v[i[hit]]
        return out, hit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="dir of f%06d.npz with lab (nusc16) / conf")
    ap.add_argument("--pred-order", default="raw", choices=["raw", "bag"],
                    help="raw = tools/cache_trained.py & cache_ptv3.py; bag = opt/cache_pred.py")
    ap.add_argument("--live-npz", default=None, help="node-written map .npz to score by lookup")
    ap.add_argument("--traj", default="/data/livo_sem/out/kitti_seq07_fastlivo2_tum.txt")
    ap.add_argument("--f0", type=int, default=0)
    ap.add_argument("--f1", type=int, default=None, help="default: the sequence length from seqreg")
    ap.add_argument("--seq", default="07", choices=["07", "09"],
                    help="both have raw-to-odometry offset 0 in seqreg, so frame f is raw f and label f")
    ap.add_argument("--conf-gate", type=float, default=0.5)
    ap.add_argument("--voxel", type=float, default=0.20)
    ap.add_argument("--deskew-bins", type=int, default=128)
    ap.add_argument("--json-out", required=True)
    ap.add_argument("--save-map", default=None)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    # map_eval: the sequence comes from seqreg, not replay_v03 (whose RAW / LB are seq 07 only)
    SC, proj, W, H = seqreg.use(a.seq)
    RAW_S, LB_S = SC.RAW, SC.LABELS
    if a.f1 is None:
        a.f1 = SC.N_FRAMES_TOTAL
    ts = parse_ts(RAW_S + "/velodyne_points/timestamps_start.txt")
    te = parse_ts(RAW_S + "/velodyne_points/timestamps_end.txt")
    traj = S.TrajInterp(a.traj)
    R_IL = np.ascontiguousarray(S.T_I_L[:3, :3]); t_IL = np.ascontiguousarray(S.T_I_L[:3, 3])
    vm = S.SemanticVoxelMap(voxel=a.voxel, cap0=1 << 22, hash_cap=1 << 24, dyn=False)
    CAP = 1 << 22
    hin = np.zeros(CAP * K, np.int32)        # GT hist per row, in-frustum points
    hout = np.zeros(CAP * K, np.int32)       # GT hist per row, out-of-frustum points

    off_all = {s: Conf() for s in ("outside", "frustum")}
    off_ins = {s: Conf() for s in ("outside", "frustum")}
    live = {s: Conf() for s in ("outside", "frustum")}
    g_keys, g_gt, g_in = [], [], []
    pp_rows, pp_gt, pp_pred, g_pred = [], [], [], []
    pp_d8, pp_vis, pp_in, g_d8, g_vis = [], [], [], [], []
    iso_d = {"frustum": [], "outside": []}
    iso_c = {"frustum": [], "outside": []}
    nop = {"outside": np.zeros(K, np.int64), "frustum": np.zeros(K, np.int64)}
    n_nopose = n_frames = 0
    n_pts_eval = n_pts_ins = n_pts_gated = n_pts_nopose = 0
    live_idx = None
    live_hits = live_q = 0
    if a.live_npz:
        z = np.load(a.live_npz)
        lk = S.voxel_key(z["xyz"].astype(np.float64), a.voxel)
        lc = NU[z["cls"].astype(np.int64)]
        uniq = len(np.unique(lk))
        live_idx = KeyIndex(lk, lc)
        live_meta = dict(path=a.live_npz, voxels=int(len(lk)), unique_keys_from_centroids=int(uniq))
    t_ins = []
    t0 = time.time()
    for f in range(a.f0, a.f1):
        sp = RAW_S + "/velodyne_points/data/%010d.bin" % f
        pp = os.path.join(a.pred, "f%06d.npz" % f)
        lp = LB_S + "/%06d.label" % f
        if not (os.path.exists(sp) and os.path.exists(pp) and os.path.exists(lp)):
            continue
        pts = np.fromfile(sp, dtype=np.float32).reshape(-1, 4)
        gt9 = SK[(np.fromfile(lp, dtype=np.uint32) & 0xFFFF).astype(np.int64)]
        _, _, _, inm = proj.project(pts[:, :3])
        uc_, vc_, zc_, inc_ = proj.project(pts[:, :3])
        vis_raw = np.zeros(len(pts), bool)
        if inc_.any():
            vis_raw[np.flatnonzero(inc_)] = SC.visible_mask(
                np.rint(uc_[inc_]).astype(np.int64), np.rint(vc_[inc_]).astype(np.int64),
                zc_[inc_], w=W, h=H, half_win=SC.OCC_WIN)
        _, tsyn, order = synth(pts[:, :3], sweep_duration=(te[f] - ts[f]) / 1e9)
        bag = np.ascontiguousarray(pts[order])
        gt_b = gt9[order]; in_b = inm[order]
        vis_b = vis_raw[order]
        ev_ = np.flatnonzero(gt_b >= 0)
        d8_b = np.full(len(bag), np.nan, np.float32)
        if len(ev_) > 9:
            dd_, _ = KDT_(bag[:, :3]).query(bag[ev_, :3], k=9, workers=4)
            d8_b[ev_] = dd_[:, 8].astype(np.float32)
        t_pt = ts[f] * 1e-9 + tsyn
        with np.load(pp) as z:
            lab = z["lab"].astype(np.int32); conf = z["conf"].astype(np.float32)
        if a.pred_order == "raw":
            lab = lab[order]; conf = conf[order]
        assert len(lab) == len(bag), (f, len(lab), len(bag))
        p9 = NU[lab]
        v = gt_b >= 0
        n_frames += 1
        n_pts_eval += int(v.sum())
        off_all["frustum"].add(gt_b[v & in_b], p9[v & in_b])
        off_all["outside"].add(gt_b[v & ~in_b], p9[v & ~in_b])
        dv_ = d8_b[v]; okv_ = (p9[v] == gt_b[v]); iv_ = in_b[v]
        iso_d["frustum"].append(dv_[iv_]); iso_c["frustum"].append(okv_[iv_])
        iso_d["outside"].append(dv_[~iv_]); iso_c["outside"].append(okv_[~iv_])

        ok = traj.valid(t_pt)
        if ok.sum() < 100:                       # the node's pose gate: whole sweep dropped
            n_nopose += 1
            n_pts_nopose += int(v.sum())
            nop["frustum"] += np.bincount(gt_b[v & in_b], minlength=K)
            nop["outside"] += np.bincount(gt_b[v & ~in_b], minlength=K)
            continue
        keep = ok & (conf >= a.conf_gate)
        idx = np.flatnonzero(keep)
        tk = t_pt[idx]
        t_lo, t_hi = tk[0], tk[-1]
        NB = a.deskew_bins
        ctr = t_lo + (np.arange(NB) + 0.5) * (t_hi - t_lo) / NB
        Rb, pb, _ = traj.query(ctr)
        oki = np.flatnonzero(ok)
        pw_all = deskew(bag[oki, :3], t_pt[oki], Rb, pb, t_lo, t_hi, NB, R_IL, t_IL)
        pos = np.searchsorted(oki, idx)
        pw = np.ascontiguousarray(pw_all[pos])
        ti = time.perf_counter()
        vm.insert(pw, lab[idx], conf[idx], None, None, want_point_rows=True)
        t_ins.append((time.perf_counter() - ti) * 1e3)
        rows = vm._last_point_rows.astype(np.int64)
        if vm.n > CAP:
            raise RuntimeError("map exceeded the histogram capacity %d" % CAP)
        gi = gt_b[idx]; ii = in_b[idx]; vv = gi >= 0
        pp_rows.append(rows[vv].astype(np.int32)); pp_gt.append(gi[vv].astype(np.int8))
        pp_pred.append(p9[idx][vv].astype(np.int8))
        pp_d8.append(d8_b[idx][vv]); pp_vis.append(vis_b[idx][vv]); pp_in.append(in_b[idx][vv])
        n_pts_ins += int(vv.sum())
        off_ins["frustum"].add(gi[vv & ii], p9[idx][vv & ii])
        off_ins["outside"].add(gi[vv & ~ii], p9[idx][vv & ~ii])
        for hist, m in ((hin, vv & ii), (hout, vv & ~ii)):
            if m.any():
                u_, c_ = np.unique(rows[m] * K + gi[m], return_counts=True)
                hist[u_] += c_.astype(np.int32)
        gm = ok & ~keep & v
        if gm.any():
            gpos = np.searchsorted(oki, np.flatnonzero(gm))
            g_keys.append(S.voxel_key(pw_all[gpos], a.voxel))
            g_gt.append(gt_b[gm].astype(np.int8)); g_in.append(in_b[gm])
            g_pred.append(p9[gm].astype(np.int8))
            g_d8.append(d8_b[gm]); g_vis.append(vis_b[gm])
            n_pts_gated += int(gm.sum())
        if live_idx is not None:
            m = ok & v
            lpos = np.searchsorted(oki, np.flatnonzero(m))
            pred_l, hit = live_idx.lookup(S.voxel_key(pw_all[lpos], a.voxel))
            live_hits += int(hit.sum()); live_q += len(hit)
            im = in_b[m]
            live["frustum"].add(gt_b[m][im], pred_l[im])
            live["outside"].add(gt_b[m][~im], pred_l[~im])
        if n_frames % 100 == 0:
            print("  %s frame %d  voxels %d  %.0f s" % (a.tag, f, vm.n, time.time() - t0), flush=True)

    # ------------------------------------------------------------- final map readings
    n = vm.n
    dom = np.argmax(vm.score[:n], axis=1)
    pred_row = NU[dom]
    h_in = hin[:n * K].reshape(n, K); h_out = hout[:n * K].reshape(n, K)
    map_pp = {"frustum": Conf(), "outside": Conf()}
    map_pp["frustum"].add_hist(h_in, pred_row); map_pp["outside"].add_hist(h_out, pred_row)
    map_maj = {"frustum": majority(h_in, pred_row), "outside": majority(h_out, pred_row)}
    map_maj_global_v03 = majority(h_in + h_out, pred_row)     # replay_v03's exact global reading
    # all-points reading: inserted (per-point GT) + gated-out by lookup + no-pose abstain
    map_all = {s: Conf() for s in ("frustum", "outside")}
    for s in map_all:
        map_all[s].C += map_pp[s].C; map_all[s].A += map_pp[s].A + nop[s]
    gated_found = 0
    if g_keys:
        gk = np.concatenate(g_keys); gg = np.concatenate(g_gt).astype(np.int64); gin = np.concatenate(g_in)
        kidx = KeyIndex(vm.key[:n].copy(), pred_row.astype(np.int64))
        pg, hit = kidx.lookup(gk)
        gated_found = int(hit.sum())
        map_all["frustum"].add(gg[gin], pg[gin]); map_all["outside"].add(gg[~gin], pg[~gin])
    # centroid -> key self-check (the live-map lookup relies on it)
    cen32 = (vm.xyz[:n] / np.maximum(vm.n_obs[:n], 1)[:, None]).astype(np.float32)
    cen_ok = float(np.mean(S.voxel_key(cen32.astype(np.float64), a.voxel) == vm.key[:n]))

    def three(d):
        out = {s: d[s].metrics() for s in ("outside", "frustum")}
        out["global"] = d["outside"].merge(d["frustum"]).metrics()
        return out

    res = dict(tag=a.tag, pred=a.pred, pred_order=a.pred_order, traj=a.traj,
               cfg=dict(conf_gate=a.conf_gate, voxel=a.voxel, deskew_bins=a.deskew_bins,
                        f0=a.f0, f1=a.f1, dyn=False, label_space="common-9 (label_spaces.COARSE)",
                        frustum="score_2d_vs_3d.Projector P3 rule on the raw sensor-frame point"),
               frames=n_frames, frames_no_pose=n_nopose, wall_s=time.time() - t0,
               map_voxels=int(n), insert_ms=dict(mean=float(np.mean(t_ins)), p95=float(np.percentile(t_ins, 95))),
               points=dict(evaluated=n_pts_eval, inserted=n_pts_ins, gated_out=n_pts_gated,
                           no_pose=n_pts_nopose, gated_out_found_in_map=gated_found,
                           inserted_frac=n_pts_ins / max(1, n_pts_eval)),
               centroid_key_selfcheck=cen_ok,
               offline_all=three(off_all), offline_inserted=three(off_ins),
               map_pointgt_inserted=three(map_pp), map_majority_inserted=three(map_maj),
               map_majority_global_v03=map_maj_global_v03.metrics(),
               map_all_lookup=three(map_all))
    if live_idx is not None:
        res["live_lookup"] = three(live)
        res["live_meta"] = dict(live_meta, query_points=int(live_q), hit_frac=live_hits / max(1, live_q))
    # ============================================================ v0.6 residual anatomy
    # tools/residual_anatomy.py = opt/replay_v05.py verbatim + this block.  Nothing above
    # this line differs from replay_v05.py, so map_all_lookup must reproduce the v0.5 json
    # bit for bit (checked by tools/anatomy_report.py).  Definitions and the verdict rule
    # are in out/v06_anatomy/PREREG.md, written before any number was looked at.
    from scipy.spatial import cKDTree
    kx = KeyIndex(vm.key[:n].copy(), np.arange(n, dtype=np.int64))
    Hs = {"frustum": h_in.astype(np.int64), "outside": h_out.astype(np.int64)}
    novox = {s: nop[s].astype(np.int64).copy() for s in ("frustum", "outside")}
    if g_keys:
        rj, hitj = kx.lookup(gk)
        for s, msk in (("frustum", gin), ("outside", ~gin)):
            sel = msk & hitj
            Hs[s] = Hs[s] + np.bincount(rj[sel] * K + gg[sel], minlength=n * K).reshape(n, K)
            novox[s] += np.bincount(gg[msk & ~hitj], minlength=K)
    Hall = Hs["frustum"] + Hs["outside"]
    hm_ok = Hall.sum(1) > 0
    m_row = np.where(hm_ok, Hall.argmax(1), -1).astype(np.int64)
    f_row = pred_row.astype(np.int64)
    hf_ok = f_row >= 0
    ar = np.arange(n)
    nobs = vm.n_obs[:n].astype(np.int64)
    Rt, Pt, _ = traj.query(np.asarray(traj.t, np.float64))
    lidar_org = Pt + np.einsum("nij,j->ni", Rt, t_IL)
    rng = cKDTree(lidar_org).query(cen32.astype(np.float64), k=1)[0]
    notm = np.ones((n, K), bool)
    notm[ar[hm_ok], m_row[hm_ok]] = False                 # GT class != voxel majority
    notmf = notm.copy()
    notmf[ar[hf_ok], f_row[hf_ok]] = False                # ... and != fused class
    NB_EDGES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 1 << 62]
    RG_EDGES = [0.0, 10.0, 20.0, 30.0, 50.0, 1e9]
    anat = dict(prereg="out/v06_anatomy/PREREG.md", n_voxels=int(n))
    for s in ("frustum", "outside"):
        H = Hs[s]
        tot = H.sum(1)
        corr = np.where(hf_ok, H[ar, np.clip(f_row, 0, K - 1)], 0)
        hm = np.where(hm_ok, H[ar, np.clip(m_row, 0, K - 1)], 0)
        cls = np.where(hm_ok & (f_row != m_row), hm, 0)
        err = tot - corr
        mix = err - cls
        floor = tot - hm
        cm = Conf(); cm.add_hist(H, pred_row); cm.A += novox[s]
        co = Conf(); co.add_hist(H, np.where(hm_ok, m_row, ABST)); co.A += novox[s]
        C = map_all[s].C
        d = dict(
            scored=int(tot.sum()), errors=int(err.sum()), mix=int(mix.sum()),
            cls=int(cls.sum()), floor=int(floor.sum()), novox=int(novox[s].sum()),
            accounting=dict(
                answered_equal=bool(int(tot.sum()) == int(C.sum())),
                errors_equal=bool(int(err.sum()) == int(C.sum() - np.trace(C[:, :K]))),
                novox_equal=bool(int(novox[s].sum()) == int(map_all[s].A.sum())),
                headline_rebuilt=bool(np.array_equal(cm.C, C) and np.array_equal(cm.A, map_all[s].A))),
            model=cm.metrics(), ceiling=co.metrics(),
            mix_by_class={NAMES[k]: int(v) for k, v in enumerate((H * notmf).sum(0))},
            cls_by_class={NAMES[k]: int(v) for k, v in enumerate(
                np.bincount(m_row[cls > 0], weights=cls[cls > 0], minlength=K).astype(np.int64))},
            floor_by_class={NAMES[k]: int(v) for k, v in enumerate((H * notm).sum(0))},
            scored_by_class={NAMES[k]: int(v) for k, v in enumerate(H.sum(0))},
            split={}, nobs_curve=[], range_curve=[])
        for T in (2, 4, 8):
            sp = nobs <= T
            so = Conf(); so.add_hist(H, np.where(sp & hm_ok, m_row, f_row)); so.A += novox[s]
            d["split"][str(T)] = dict(
                sparse=int(cls[sp].sum()), dense=int(cls[~sp].sum()),
                scored_sparse=int(tot[sp].sum()), scored_dense=int(tot[~sp].sum()),
                cls_rate_sparse=float(cls[sp].sum() / max(1, tot[sp].sum())),
                cls_rate_dense=float(cls[~sp].sum() / max(1, tot[~sp].sum())),
                sparse_oracle_miou=so.metrics()["miou9_abstain_excluded"])
        for key_, edges, val in (("nobs_curve", NB_EDGES, nobs), ("range_curve", RG_EDGES, rng)):
            for lo, hi in zip(edges[:-1], edges[1:]):
                b = (val >= lo) & (val < hi)
                d[key_].append(dict(lo=float(lo), hi=float(hi), voxels=int(b.sum()),
                                    scored=int(tot[b].sum()), err=int(err[b].sum()),
                                    mix=int(mix[b].sum()), cls=int(cls[b].sum()),
                                    floor=int(floor[b].sum())))
        anat[s] = d
    res["anatomy"] = anat
    # ============================================================ voxel-weighted (post hoc)
    # tools/residual_anatomy_vox.py = tools/residual_anatomy.py + this block.  Added AFTER
    # the point-weighted verdict was read; rules in out/v06_anatomy/PREREG_voxel.md, written
    # before any voxel-weighted number existed.  Each voxel counts once (SemanticKITTI SSC
    # style): its GT is its own majority m_r, its prediction the fused class.  MIX cannot
    # exist here by construction -- a voxel is scored against its own majority.
    nin = Hs["frustum"].sum(1); nout = Hs["outside"].sum(1)
    vsets = {"global": hm_ok, "outside": hm_ok & (nout > nin), "frustum": hm_ok & (nin >= nout)}
    wrong = hm_ok & (f_row != m_row)

    def vconf(sel, pred):
        c = Conf()
        g = m_row[sel]
        p = pred[sel]
        p = np.where(p == UNM, K, p)
        c.C += np.bincount(g * (K + 1) + p, minlength=K * (K + 1)).reshape(K, K + 1)
        return c

    vox = dict(prereg="out/v06_anatomy/PREREG_voxel.md")
    sp4 = nobs <= 4
    for s, VS in vsets.items():
        d = dict(voxels=int(VS.sum()), wrong=int((VS & wrong).sum()),
                 model=vconf(VS, f_row).metrics(), ceiling=vconf(VS, m_row).metrics(),
                 split={}, nobs_curve=[], range_curve=[], wrong_by_class={})
        for T in (2, 4, 8):
            sp = nobs <= T
            d["split"][str(T)] = dict(
                sparse_voxels=int((VS & sp).sum()), dense_voxels=int((VS & ~sp).sum()),
                sparse_wrong=int((VS & sp & wrong).sum()), dense_wrong=int((VS & ~sp & wrong).sum()),
                sparse_oracle_miou=vconf(VS, np.where(sp, m_row, f_row)).metrics()["miou9_abstain_excluded"])
        for key_, edges, val in (("nobs_curve", NB_EDGES, nobs), ("range_curve", RG_EDGES, rng)):
            for lo, hi in zip(edges[:-1], edges[1:]):
                b = VS & (val >= lo) & (val < hi)
                d[key_].append(dict(lo=float(lo), hi=float(hi), voxels=int(b.sum()),
                                    wrong=int((b & wrong).sum())))
        for k in range(K):
            gk_ = VS & (m_row == k)
            d["wrong_by_class"][NAMES[k]] = dict(
                voxels=int(gk_.sum()), sparse_wrong=int((gk_ & sp4 & wrong).sum()),
                dense_wrong=int((gk_ & ~sp4 & wrong).sum()))
        vox[s] = d
    res["anatomy_voxel"] = vox
    # ============================================================ voxel-reweighted map vs per-scan
    # tools/map_eval.py only.  Each voxel contributes weight 1, spread evenly over all its
    # scored points; GT = each point's own label.  Same points, same weights, same GT on both
    # sides -- only the prediction differs (the point's per-scan prediction vs its voxel's
    # fused class).  This is the voxel-weighted form of the v0.5 comparison "the map beats the
    # per-scan predictions it is built from".  Exactness gate: with uniform weights the map side
    # must equal the headline's answered confusion matrix entry for entry.
    res["seq"] = a.seq
    prow = np.concatenate(pp_rows) if pp_rows else np.zeros(0, np.int32)
    pgt = np.concatenate(pp_gt) if pp_gt else np.zeros(0, np.int8)
    ppr = np.concatenate(pp_pred) if pp_pred else np.zeros(0, np.int8)
    if g_keys:
        gpred = np.concatenate(g_pred)
        prow = np.concatenate([prow, rj[hitj].astype(np.int32)])
        pgt = np.concatenate([pgt, gg[hitj].astype(np.int8)])
        ppr = np.concatenate([ppr, gpred[hitj]])
    tot_all = Hall.sum(1)

    def fconf(sel, use_map, weighted):
        g = pgt[sel].astype(np.int64)
        p = f_row[prow[sel]] if use_map else ppr[sel].astype(np.int64)
        p = np.where(p == UNM, K, p)
        w = (1.0 / np.maximum(tot_all[prow[sel]], 1)) if weighted else None
        return np.bincount(g * (K + 1) + p, weights=w, minlength=K * (K + 1)).reshape(K, K + 1)

    def fmetrics(Cf):
        rowf = Cf.sum(1); colf = Cf[:, :K].sum(0); dg = np.diag(Cf[:, :K])
        iou = {}
        for k in range(K):
            if rowf[k] <= 0:
                continue
            u = rowf[k] + colf[k] - dg[k]
            iou[NAMES[k]] = float(100.0 * dg[k] / u) if u > 0 else 0.0
        return dict(miou9=float(np.mean(list(iou.values()))) if iou else float("nan"),
                    acc=float(100.0 * dg.sum() / Cf.sum()) if Cf.sum() > 0 else float("nan"),
                    per_class={k_: round(v_, 3) for k_, v_ in iou.items()})

    every = np.ones(len(prow), bool)
    rwv = dict(points=int(len(prow)),
               uniform_map_equals_headline=bool(np.array_equal(
                   fconf(every, True, False).astype(np.int64),
                   map_all["outside"].C + map_all["frustum"].C)))
    for s, VS in vsets.items():
        msk = VS[prow]
        rwv[s] = dict(points=int(msk.sum()), voxels=int(VS.sum()),
                      map=fmetrics(fconf(msk, True, True)),
                      per_scan=fmetrics(fconf(msk, False, True)))
    res["reweighted_map_vs_scan"] = rwv
    # ============================================================ sparse-cell failure diagnosis
    # tools/sparse_diag.py only.  Rules: out/v06_sparsediag/PREREG.md (+ amendment A1), written
    # before any number.  CONTEXT / SUPERVISION / FUSION, see there.
    pd8 = np.concatenate(pp_d8) if pp_d8 else np.zeros(0, np.float32)
    pvis = np.concatenate(pp_vis) if pp_vis else np.zeros(0, bool)
    pin = np.concatenate(pp_in) if pp_in else np.zeros(0, bool)
    if g_keys:
        pd8 = np.concatenate([pd8, np.concatenate(g_d8)[hitj]])
        pvis = np.concatenate([pvis, np.concatenate(g_vis)[hitj]])
        pin = np.concatenate([pin, gin[hitj]])
    assert len(pd8) == len(pvis) == len(pin) == len(prow) == len(ppr), "per-point arrays misaligned"
    rr = prow.astype(np.int64)
    vis_any = np.zeros(n, bool)
    vis_any[rr[pvis]] = True
    d8_min = np.full(n, np.inf)
    np.minimum.at(d8_min, rr, pd8.astype(np.float64))
    pp9 = np.where(ppr == UNM, K, ppr).astype(np.int64)
    pc = np.bincount(rr * (K + 1) + pp9, minlength=n * (K + 1)).reshape(n, K + 1)
    ptot = pc.sum(1)
    share_top = pc.max(1) / np.maximum(ptot, 1)
    pattern = np.where(ptot <= 1, 0, np.where(share_top < 2.0 / 3.0, 2, 1))  # 0 SINGLE 1 CONSISTENT 2 SPLIT
    wrong_c = hm_ok & (f_row != m_row)
    sparse4 = nobs <= 4

    def pop(sel):
        d = dict(cells=int(sel.sum()))
        if not sel.any():
            return d
        pat = pattern[sel]
        ns = sel & (pattern != 2)
        d.update(single=int((pat == 0).sum()), consistent=int((pat == 1).sum()),
                 split=int((pat == 2).sum()), camera_labelable=int((sel & vis_any).sum()),
                 nonsplit=int(ns.sum()), nonsplit_camera_labelable=int((ns & vis_any).sum()),
                 d8_min_q=[float(x) for x in np.percentile(d8_min[sel], [25, 50, 75])],
                 nonsplit_d8_min_median=float(np.median(d8_min[ns])) if ns.any() else float("nan"))
        return d

    diag = dict(prereg="out/v06_sparsediag/PREREG.md", populations={})
    for s, VS in (("global", hm_ok), ("outside", vsets["outside"])):
        diag["populations"][s] = dict(
            sparse_wrong=pop(VS & sparse4 & wrong_c), sparse_correct=pop(VS & sparse4 & ~wrong_c),
            dense_wrong=pop(VS & ~sparse4 & wrong_c), dense_correct=pop(VS & ~sparse4 & ~wrong_c))
    iso = {}
    D8_EDGES = [0.0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 1e9]
    for s in ("frustum", "outside"):
        d_ = np.concatenate(iso_d[s]); c_ = np.concatenate(iso_c[s])
        q1, q3 = (float(x) for x in np.percentile(d_, [25, 75]))
        least = d_ <= q1; most = d_ >= q3
        om = off_all[s]
        iso[s] = dict(
            n=int(len(d_)), q1_d8=q1, q3_d8=q3,
            least_isolated=dict(share=float(least.mean()), acc=float(100.0 * c_[least].mean())),
            most_isolated=dict(share=float(most.mean()), acc=float(100.0 * c_[most].mean())),
            curve=[dict(lo=a_, hi=b_, n=int(((d_ >= a_) & (d_ < b_)).sum()),
                        acc=float(100.0 * c_[(d_ >= a_) & (d_ < b_)].mean()) if ((d_ >= a_) & (d_ < b_)).any() else None)
                   for a_, b_ in zip(D8_EDGES[:-1], D8_EDGES[1:])],
            accounting_n_equal=bool(int(len(d_)) == int(om.C.sum())),
            accounting_correct_equal=bool(int(c_.sum()) == int(np.trace(om.C[:, :K]))))
        del d_, c_
    diag["isolation"] = iso
    outp = ~pin
    diag["supervision_reach"] = dict(
        points=int(len(pin)),
        per_scan_camera_labelable_frac=float(pvis.mean()),
        out_of_frustum_points=int(outp.sum()),
        out_of_frustum_points_in_cells_labelable_sometime=float(vis_any[rr[outp]].mean()),
        labelled_cells_labelable_sometime=float(vis_any[hm_ok].mean()))
    res["sparse_diag"] = diag
    json.dump(res, open(a.json_out, "w"), indent=2)
    if a.save_map:
        h_all = h_in + h_out
        np.savez_compressed(a.save_map, xyz=cen32, cls=dom.astype(np.uint16),
                            n_obs=vm.n_obs[:n], key=vm.key[:n],
                            gt_cls=np.where(h_all.sum(1) > 0, h_all.argmax(1), -1).astype(np.int8),
                            gt_cnt=h_all.sum(1).astype(np.int32),
                            gt_in_cnt=h_in.sum(1).astype(np.int32))

    def line(name, m):
        return "  %-22s out %6.2f / %6.2f   in %6.2f / %6.2f   global %6.2f / %6.2f   (acc / mIoU-9, abstain-wrong)" % (
            name, m["outside"]["acc_abstain_wrong"], m["outside"]["miou9_abstain_wrong"],
            m["frustum"]["acc_abstain_wrong"], m["frustum"]["miou9_abstain_wrong"],
            m["global"]["acc_abstain_wrong"], m["global"]["miou9_abstain_wrong"])
    print("%s  frames %d (no-pose %d)  voxels %d  inserted %.3f of evaluated points  %.0f s"
          % (a.tag, n_frames, n_nopose, n, res["points"]["inserted_frac"], res["wall_s"]))
    for k in ("offline_all", "offline_inserted", "map_pointgt_inserted", "map_majority_inserted", "map_all_lookup"):
        print(line(k, res[k]))
    if live_idx is not None:
        print(line("live_lookup", res["live_lookup"]) + "  hit %.4f" % res["live_meta"]["hit_frac"])


if __name__ == "__main__":
    main()
