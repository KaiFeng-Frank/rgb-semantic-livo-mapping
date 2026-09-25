#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
opt/replay_v06.py -- map-level scorer for the v0.6 online arms.  opt/replay_v05.py's main()
with three additions; every reading it shares with v0.5 is computed by the same code
(Conf, majority, deskew, KeyIndex are imported from replay_v05, not copied).

  1. --live-npz scoring of a node-written map by voxel lookup, with the query positions
     computed from the SAME poses the node applied:
       * --pose-used  (npz written by the node with --pose-used-out): per processed sweep the
         128 bin poses the CAUSAL path used -> exact reproduction of where the node put each
         point (`live_lookup_causal`);
       * sweeps the node did not process (back-pressure / stale / no-pose) fall back to the
         non-causal interpolation of --traj, as v0.5's live scoring did for its dropped sweeps.
     `live_lookup_interp` is the same map queried with the non-causal interpolation of --traj
     for EVERY sweep (the v0.5 convention) -- the gap between the two readings is the
     label-level footprint of the causal pose error.
  2. The geometric footprint itself, per processed sweep and pooled: |delta position| of every
     evaluated point between its causal placement and the non-causal interpolation of the same
     recorded stream (p50 / p95 / p99 / max), the fraction of points that change 0.2 m voxel,
     and the rotation angle between the causal and interpolated bin poses, against the
     extrapolated span the node recorded for that sweep (--diag-out).
  3. The replay readings (offline_all ... map_all_lookup) are computed with --traj exactly as
     in v0.5, so a run with --traj <recorded stream> IS the CAUSAL-ISO replay arm.
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
from replay_v05 import Conf, majority, deskew, KeyIndex, K, NAMES, UNM, ABST, SK, NU


def rot_angle(Ra, Rb):
    """(n,3,3),(n,3,3) -> (n,) angle in rad between the rotations."""
    c = (np.einsum("nii->n", np.einsum("nij,nkj->nik", Ra, Rb)) - 1.0) * 0.5
    return np.arccos(np.clip(c, -1.0, 1.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="dir of f%06d.npz with lab (nusc16) / conf")
    ap.add_argument("--pred-order", default="raw", choices=["raw", "bag"])
    ap.add_argument("--live-npz", default=None, help="node-written map .npz to score by lookup")
    ap.add_argument("--pose-used", default=None, help="node --pose-used-out npz (causal bin poses)")
    ap.add_argument("--traj", default="/data/livo_sem/out/kitti_seq07_fastlivo2_tum.txt")
    ap.add_argument("--f0", type=int, default=0)
    ap.add_argument("--f1", type=int, default=1101)
    ap.add_argument("--conf-gate", type=float, default=0.5)
    ap.add_argument("--voxel", type=float, default=0.20)
    ap.add_argument("--deskew-bins", type=int, default=128)
    ap.add_argument("--json-out", required=True)
    ap.add_argument("--diag-out", default=None, help="npz of the per-sweep geometric diagnostics")
    ap.add_argument("--save-map", default=None)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    ts = parse_ts(RAW + "/velodyne_points/timestamps_start.txt")
    te = parse_ts(RAW + "/velodyne_points/timestamps_end.txt")
    traj = S.TrajInterp(a.traj)
    R_IL = np.ascontiguousarray(S.T_I_L[:3, :3]); t_IL = np.ascontiguousarray(S.T_I_L[:3, 3])
    vm = S.SemanticVoxelMap(voxel=a.voxel, cap0=1 << 22, hash_cap=1 << 24, dyn=False)
    CAP = 1 << 22
    hin = np.zeros(CAP * K, np.int32)
    hout = np.zeros(CAP * K, np.int32)
    SC, proj, W, H = seqreg.use("07")

    off_all = {s: Conf() for s in ("outside", "frustum")}
    off_ins = {s: Conf() for s in ("outside", "frustum")}
    live_i = {s: Conf() for s in ("outside", "frustum")}     # interpolated placement (v0.5 convention)
    live_c = {s: Conf() for s in ("outside", "frustum")}     # causal placement (pose-used) + fallback
    g_keys, g_gt, g_in = [], [], []
    nop = {"outside": np.zeros(K, np.int64), "frustum": np.zeros(K, np.int64)}
    n_nopose = n_frames = 0
    n_pts_eval = n_pts_ins = n_pts_gated = n_pts_nopose = 0
    live_idx = None
    hits_i = hits_c = q_i = q_c = 0
    if a.live_npz:
        z = np.load(a.live_npz)
        lk = S.voxel_key(z["xyz"].astype(np.float64), a.voxel)
        lc = NU[z["cls"].astype(np.int64)]
        live_idx = KeyIndex(lk, lc)
        live_meta = dict(path=a.live_npz, voxels=int(len(lk)),
                         unique_keys_from_centroids=int(len(np.unique(lk))))
    pu = None
    if a.pose_used:
        zp = np.load(a.pose_used)
        pu = dict(hdr=zp["hdr"], ctr=zp["ctr"], R=zp["R"], p=zp["p"], t_last=zp["t_last"])
        pu_order = np.argsort(pu["hdr"])
        pu_hdr_sorted = pu["hdr"][pu_order]
    # per-sweep geometric diagnostics
    D = dict(frame=[], span=[], n=[], dp_mean=[], dp_p50=[], dp_p95=[], dp_max=[],
             key_change=[], rot_max_mrad=[], bin_dp_max=[], causal=[])
    dp_all = []          # pooled per-point |delta p| (subsampled 1/8 to bound memory)
    n_causal = n_fallback = 0
    t_ins = []
    t0 = time.time()
    for f in range(a.f0, a.f1):
        sp = RAW + "/velodyne_points/data/%010d.bin" % f
        pp = os.path.join(a.pred, "f%06d.npz" % f)
        lp = LB + "/%06d.label" % f
        if not (os.path.exists(sp) and os.path.exists(pp) and os.path.exists(lp)):
            continue
        pts = np.fromfile(sp, dtype=np.float32).reshape(-1, 4)
        gt9 = SK[(np.fromfile(lp, dtype=np.uint32) & 0xFFFF).astype(np.int64)]
        _, _, _, inm = proj.project(pts[:, :3])
        _, tsyn, order = synth(pts[:, :3], sweep_duration=(te[f] - ts[f]) / 1e9)
        bag = np.ascontiguousarray(pts[order])
        gt_b = gt9[order]; in_b = inm[order]
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

        ok = traj.valid(t_pt)
        if ok.sum() < 100:
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
            n_pts_gated += int(gm.sum())
        if live_idx is not None:
            m = ok & v
            lpos = np.searchsorted(oki, np.flatnonzero(m))
            key_i = S.voxel_key(pw_all[lpos], a.voxel)
            pred_l, hit = live_idx.lookup(key_i)
            hits_i += int(hit.sum()); q_i += len(hit)
            im = in_b[m]
            live_i["frustum"].add(gt_b[m][im], pred_l[im])
            live_i["outside"].add(gt_b[m][~im], pred_l[~im])
            # ---- causal placement from the node's own bin poses
            rec = None
            if pu is not None:
                j = int(np.searchsorted(pu_hdr_sorted, ts[f] * 1e-9))
                for jj in (j - 1, j):
                    if 0 <= jj < len(pu_hdr_sorted) and abs(pu_hdr_sorted[jj] - ts[f] * 1e-9) < 1e-4:
                        rec = int(pu_order[jj])
                        break
            if rec is not None:
                n_causal += 1
                ctr_c = pu["ctr"][rec]
                dlt = ctr_c[1] - ctr_c[0]
                tlo_c = ctr_c[0] - 0.5 * dlt
                thi_c = tlo_c + NB * dlt
                Rc, pc = pu["R"][rec], pu["p"][rec]
                pw_c = deskew(bag[oki, :3], t_pt[oki], Rc, pc, tlo_c, thi_c, NB, R_IL, t_IL)
                # non-causal interpolation of the SAME stream at the SAME bin centres
                Rn, pn, _ = traj.query(ctr_c)
                pw_n = deskew(bag[oki, :3], t_pt[oki], Rn, pn, tlo_c, thi_c, NB, R_IL, t_IL)
                dp = np.linalg.norm(pw_c[lpos] - pw_n[lpos], axis=1)
                key_c = S.voxel_key(pw_c[lpos], a.voxel)
                key_n = S.voxel_key(pw_n[lpos], a.voxel)
                D["frame"].append(f); D["causal"].append(1)
                D["span"].append(float(max(0.0, thi_c - pu["t_last"][rec])) if np.isfinite(pu["t_last"][rec]) else np.nan)
                D["n"].append(int(len(dp)))
                D["dp_mean"].append(float(dp.mean())); D["dp_p50"].append(float(np.percentile(dp, 50)))
                D["dp_p95"].append(float(np.percentile(dp, 95))); D["dp_max"].append(float(dp.max()))
                D["key_change"].append(float(np.mean(key_c != key_n)))
                D["rot_max_mrad"].append(float(rot_angle(Rc, Rn).max() * 1e3))
                D["bin_dp_max"].append(float(np.linalg.norm(pc - pn, axis=1).max()))
                dp_all.append(dp[::8].astype(np.float32))
            else:
                n_fallback += 1
                key_c = key_i
                D["frame"].append(f); D["causal"].append(0); D["span"].append(np.nan); D["n"].append(int(len(key_i)))
                for kk in ("dp_mean", "dp_p50", "dp_p95", "dp_max", "key_change", "rot_max_mrad", "bin_dp_max"):
                    D[kk].append(np.nan)
            pred_c, hit_c = live_idx.lookup(key_c)
            hits_c += int(hit_c.sum()); q_c += len(hit_c)
            live_c["frustum"].add(gt_b[m][im], pred_c[im])
            live_c["outside"].add(gt_b[m][~im], pred_c[~im])
        if n_frames % 100 == 0:
            print("  %s frame %d  voxels %d  %.0f s" % (a.tag, f, vm.n, time.time() - t0), flush=True)

    n = vm.n
    dom = np.argmax(vm.score[:n], axis=1)
    pred_row = NU[dom]
    h_in = hin[:n * K].reshape(n, K); h_out = hout[:n * K].reshape(n, K)
    map_pp = {"frustum": Conf(), "outside": Conf()}
    map_pp["frustum"].add_hist(h_in, pred_row); map_pp["outside"].add_hist(h_out, pred_row)
    map_maj = {"frustum": majority(h_in, pred_row), "outside": majority(h_out, pred_row)}
    map_maj_global_v03 = majority(h_in + h_out, pred_row)
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
    cen32 = (vm.xyz[:n] / np.maximum(vm.n_obs[:n], 1)[:, None]).astype(np.float32)
    cen_ok = float(np.mean(S.voxel_key(cen32.astype(np.float64), a.voxel) == vm.key[:n]))

    def three(d):
        out = {s: d[s].metrics() for s in ("outside", "frustum")}
        out["global"] = d["outside"].merge(d["frustum"]).metrics()
        return out

    res = dict(tag=a.tag, pred=a.pred, pred_order=a.pred_order, traj=a.traj, pose_used=a.pose_used,
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
        res["live_lookup_interp"] = three(live_i)
        res["live_lookup_causal"] = three(live_c)
        res["live_lookup"] = res["live_lookup_causal"] if pu is not None else res["live_lookup_interp"]
        res["live_meta"] = dict(live_meta, query_points=int(q_i), hit_frac_interp=hits_i / max(1, q_i),
                                hit_frac_causal=hits_c / max(1, q_c), sweeps_causal=n_causal,
                                sweeps_fallback_interp=n_fallback)
        if pu is not None and dp_all:
            dpa = np.concatenate(dp_all).astype(np.float64)
            spans = np.array(D["span"], dtype=np.float64); cz = np.array(D["causal"]) == 1
            kc = np.array(D["key_change"], dtype=np.float64)
            res["causal_geometry"] = dict(
                points_sampled=int(len(dpa)),
                dp_m=dict(p50=float(np.percentile(dpa, 50)), p95=float(np.percentile(dpa, 95)),
                          p99=float(np.percentile(dpa, 99)), max=float(dpa.max()), mean=float(dpa.mean())),
                voxel_key_change_frac=float(np.nanmean(kc[cz])),
                voxel_key_change_frac_p95_sweep=float(np.nanpercentile(kc[cz], 95)),
                extrap_span_s=dict(p50=float(np.nanpercentile(spans[cz], 50)), p95=float(np.nanpercentile(spans[cz], 95)),
                                   max=float(np.nanmax(spans[cz]))),
                rot_max_mrad=dict(p50=float(np.nanpercentile(np.array(D["rot_max_mrad"])[cz], 50)),
                                  p95=float(np.nanpercentile(np.array(D["rot_max_mrad"])[cz], 95)),
                                  max=float(np.nanmax(np.array(D["rot_max_mrad"])[cz]))),
                bin_dp_max_m=dict(p50=float(np.nanpercentile(np.array(D["bin_dp_max"])[cz], 50)),
                                  p95=float(np.nanpercentile(np.array(D["bin_dp_max"])[cz], 95)),
                                  max=float(np.nanmax(np.array(D["bin_dp_max"])[cz]))))
    json.dump(res, open(a.json_out, "w"), indent=2)
    if a.diag_out and live_idx is not None:
        np.savez_compressed(a.diag_out, **{k: np.asarray(v) for k, v in D.items()},
                            dp_sample=(np.concatenate(dp_all) if dp_all else np.zeros(0, np.float32)))
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
        print(line("live_lookup_interp", res["live_lookup_interp"]) + "  hit %.4f" % res["live_meta"]["hit_frac_interp"])
        print(line("live_lookup_causal", res["live_lookup_causal"]) + "  hit %.4f  causal sweeps %d, fallback %d"
              % (res["live_meta"]["hit_frac_causal"], n_causal, n_fallback))
        if "causal_geometry" in res:
            g = res["causal_geometry"]
            print("  causal geometry: |dp| p50 %.4f p95 %.4f p99 %.4f max %.4f m; voxel-key change %.4f; span p50 %.3f p95 %.3f s"
                  % (g["dp_m"]["p50"], g["dp_m"]["p95"], g["dp_m"]["p99"], g["dp_m"]["max"],
                     g["voxel_key_change_frac"], g["extrap_span_s"]["p50"], g["extrap_span_s"]["p95"]))


if __name__ == "__main__":
    main()
