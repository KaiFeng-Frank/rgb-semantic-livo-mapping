#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/propagate_check.py -- v0.6 premise check: do camera pseudo-labels propagated through
the accumulated map give MORE supervision at COMPARABLE precision?

Question, instrument, gates, measurements and the decision rule are fixed by
out/v06_propagate/PREREG.md, written before any number existed.  This file implements that
and decides nothing the PREREG did not fix.  No existing file is edited: what is needed from
tools/map_eval.py and opt/replay_v03.py is COPIED below (deskew, parse_ts); everything else
is imported read-only.

CAMERA SIDE (raw .bin order) -- tools/make_pseudo.py verbatim, with the image-space maps read
from the seq-07 caches instead of a GPU forward:
    EoMT-L  label  out/vs2d/seg2d_eomt   seg (481x1594 = the 1226x370 image at 1.30x)
    EoMT-L  conf   out/distil/conf_eomt  p1  (tools/cache_seg2d_conf.hires_top2 -- the conf
                                              make_pseudo stores; same 481x1594 grid)
    M2F-L   label  out/vs2d/seg2d_m2f    seg (370x1226, native)
  projection score_2d_vs_3d.Projector (P3); z-buffer visible_mask on np.rint pixels with
  OCC_WIN (P4b); depth_edge_mask (P7-B2); the sub-pixel index arithmetic of Arm2D._sample
  (sx = sw/W, sy = sh/H read from each map's own shape -- the 1.30 is never typed);
  Cityscapes-19 -> common-9 via city19_to_coarse(rider default, sky "wrong"), so sky and the
  ignore fill are never a candidate.
  t9 = EoMT label on visible points, m9 = M2F label on visible points, conf = EoMT p1 (f16),
  flags = in-frustum | visible | depth-edge | range<50.
  FILTER E = src/filter_e.filter_E_mask with the spec the B0 config carries (parsed from
  src/Pointcept_v151/configs/semantic_kitti/arm_B0.py, checked against
  out/distil08/filterE_08.json and against the PREREG's statement); the pseudo-label of a kept
  point is t9 (DistilSemanticKITTIDataset, label_source="pseudo", supervise="filterE").

MAP SIDE (driver order) -- tools/map_eval.py verbatim: replay_v03.parse_ts timestamps,
  kitti_scan.synth order + per-point time, sem_core.TrajInterp validity, the node's
  whole-sweep gate (ok.sum() < 100 drops the sweep), 128-bin deskew through T_I_L,
  sem_core.voxel_key at 0.20 m.  The ONE difference, fixed by the PREREG: no confidence gate,
  keep = ok, i.e. map_eval --conf-gate 0.  Checked against a map_eval --conf-gate 0
  --save-map run (--ctl-map): same key set, same point counts per key.

PROPAGATION.  Cells = keys of evaluated points (GT not EXCLUDED) with a valid pose.  A cell's
votes = the filter-E labels of its evaluated points over all frames.  A point NOT labelable
in its own frame gets the majority of its cell with its OWN FRAME's votes removed
(leave-own-frame-out), iff >= k votes remain and the top class holds >= 2/3 of them
(3*top >= 2*n, integer).  The propagation scheme's label of a point = its own filter-E label
if it has one, else that propagated label.
"""
import argparse
import ast
import calendar
import hashlib
import json
import os
import resource
import sys
import time

import numpy as np

sys.path.insert(0, "/data/livo_sem/src")
import sem_core as SEMC                       # noqa: E402  TrajInterp, T_I_L, voxel_key
import label_spaces as LSP                    # noqa: E402  sk_lut, COARSE
import seqreg                                 # noqa: E402  frozen scorer pointed at a sequence
from kitti_scan import synth                  # noqa: E402  driver order + per-point time
from filter_e import (filter_E_mask, F_INFRUSTUM, F_VISIBLE,   # noqa: E402
                      F_DEPTHEDGE, F_RANGE_LT50)

ROOT = "/data/livo_sem"
C9 = list(LSP.COARSE)
K9 = len(C9)
NG = K9 + 1                                   # GT bucket 9 = EXCLUDED (signal table layout)
SKLUT = np.asarray(LSP.sk_lut(), np.int32)    # raw SemanticKITTI id -> 0..8 | -1 EXCLUDED
KS = (1, 2, 3)
NCB, NRB = 20, 5
RANGE_EDGES_M = [10.0, 20.0, 30.0, 50.0]
RB_NAMES = ["0-10", "10-20", "20-30", "30-50", "50+"]
PREREG_SPEC = dict(require_visible=True, range_lt50=False, teachers_agree=True,
                   conf_min=0.9, drop_depth_edge=False)
SUBS = ("all", "in", "out")


# ------------------------------------------------------------------ copied helpers
def parse_ts(path):
    """opt/replay_v03.parse_ts verbatim (the parser tools/map_eval.py imports)."""
    out = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        d, c = line.split(" ")
        hms, frac = (c.split(".") + ["0"])[:2]
        y, mo, dd = (int(x) for x in d.split("-"))
        h, mi, s = (int(x) for x in hms.split(":"))
        out.append(calendar.timegm((y, mo, dd, h, mi, s, 0, 0, 0)) * 10**9
                   + int((frac + "000000000")[:9]))
    return np.array(out, dtype=np.int64)


def deskew(p3_f32, tk, Rb, pb, t_lo, t_hi, NB, R_IL, t_IL):
    """tools/map_eval.deskew verbatim (== semantic_map_node.stage_b).  tk sorted."""
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


def sample_idx(u, v, inm, sh, sw, W, H):
    """score_2d_vs_3d.Arm2D._sample index arithmetic verbatim (== make_pseudo.sample_idx)."""
    assert sh >= H and sw >= W, ("seg shape smaller than native", (sh, sw))
    sx, sy = sw / float(W), sh / float(H)
    ui = np.clip(np.rint((u[inm] + 0.5) * sx - 0.5), 0, sw - 1).astype(np.int64)
    vi = np.clip(np.rint((v[inm] + 0.5) * sy - 0.5), 0, sh - 1).astype(np.int64)
    return vi, ui


# ------------------------------------------------------------------ provenance helpers
def load_filter_spec():
    cfg = ROOT + "/src/Pointcept_v151/configs/semantic_kitti/arm_B0.py"
    spec_cfg = None
    for line in open(cfg):
        if line.startswith("filter_spec = "):
            spec_cfg = ast.literal_eval(line.split("=", 1)[1].strip())
    js = ROOT + "/out/distil08/filterE_08.json"
    spec_js = json.load(open(js))["filter_spec"]
    assert spec_cfg == spec_js == PREREG_SPEC, ("filter E spec mismatch", spec_cfg, spec_js,
                                                PREREG_SPEC)
    return dict(spec_cfg), dict(config=cfg, derivation=js, prereg_statement=PREREG_SPEC)


def inventory_seq07_row(path):
    """The seq-07 row of out/pseudo's own audit (INVENTORY_NOTES.md, cross-sequence table)."""
    head = None
    for line in open(path):
        if line.startswith("| seq | rig / image | frames |"):
            head = [c.strip() for c in line.strip().strip("|").split("|")]
        if line.startswith("| 07 (published)"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            assert head is not None and head[3] == "candidate %" \
                and head[4] == "two-teacher agreement %", head
            return dict(candidate_pct=float(cells[3]), agreement_pct=float(cells[4]),
                        frames=int(cells[2].replace(",", "")), row=line.strip())
    raise RuntimeError("seq-07 row not found in %s" % path)


def pct(num, den):
    return (100.0 * float(num) / float(den)) if den else None


def acc2d_add(C, A, gt, pred):
    """score_2d_vs_3d.Acc.add: C[gt, pred] with an UNMAPPED column, A[gt] abstained."""
    if gt.size == 0:
        return
    ab = pred == -1
    if ab.any():
        A += np.bincount(gt[ab].astype(np.int64), minlength=K9)
    g = gt[~ab].astype(np.int64)
    p = pred[~ab].astype(np.int64)
    if g.size:
        p = np.where(p == -2, K9, p)
        C += np.bincount(g * (K9 + 1) + p, minlength=K9 * (K9 + 1)).reshape(K9, K9 + 1)


def acc2d_metrics(C, A):
    """score_2d_vs_3d.Acc.metrics, the abstain-excluded mIoU over the classes present."""
    row = C.sum(1); col = C[:, :K9].sum(0); diag = np.diag(C[:, :K9])
    n_ans = int(C.sum()); n_ab = int(A.sum()); n = n_ans + n_ab
    present = (row + A) > 0
    iou = []
    for k in range(K9):
        if present[k]:
            u = row[k] + col[k] - diag[k]
            iou.append(100.0 * diag[k] / u if u else 0.0)
    return dict(n_eval=n, n_labelled=n_ans, n_unmapped_pred=int(C[:, K9].sum()),
                coverage=(n_ans / n if n else None),
                acc_abstain_excluded=pct(diag.sum(), n_ans),
                miou9_abstain_excluded=float(np.mean(iou)) if iou else None)


def summarize(C, ev_by_gt):
    """C[gt, label] over supervised evaluated points of one subset; ev_by_gt = evaluated
    points per GT class in that subset (the coverage denominator)."""
    C = np.asarray(C, np.int64)
    row = C.sum(1); col = C.sum(0); dg = np.diag(C)
    n_sup = int(C.sum()); n_ev = int(np.asarray(ev_by_gt).sum()); corr = int(dg.sum())
    per_class, ious = {}, []
    for k in range(K9):
        u = row[k] + col[k] - dg[k]
        present = ev_by_gt[k] > 0
        iou = (100.0 * dg[k] / u) if u else 0.0
        if present:
            ious.append(iou)
        per_class[C9[k]] = dict(gt_evaluated=int(ev_by_gt[k]), gt_supervised=int(row[k]),
                                coverage=pct(row[k], ev_by_gt[k]), labelled_as=int(col[k]),
                                correct=int(dg[k]), precision=pct(dg[k], col[k]),
                                iou_supervised=(round(iou, 4) if present else None))
    return dict(evaluated=n_ev, supervised=n_sup, coverage=pct(n_sup, n_ev), correct=corr,
                precision=pct(corr, n_sup),
                miou9_supervised=(float(np.mean(ious)) if (ious and n_sup) else None),
                per_class=per_class)


def cmat(gt, lab, m):
    s = m & (lab >= 0)
    return np.bincount(gt[s].astype(np.int64) * K9 + lab[s].astype(np.int64),
                       minlength=K9 * K9).reshape(K9, K9)


def own_votes(codes, cq):
    """(len(cq), 9) own-frame votes in each query point's cell; codes = cell*9 + class."""
    out = np.zeros((len(cq), K9), np.int64)
    if len(codes) == 0 or len(cq) == 0:
        return out
    uc, cnt = np.unique(codes, return_counts=True)
    ucell = uc // K9
    ucls = uc % K9
    cells_f = np.unique(ucell)
    M = np.zeros((len(cells_f), K9), np.int64)
    M[np.searchsorted(cells_f, ucell), ucls] = cnt
    pos = np.minimum(np.searchsorted(cells_f, cq), len(cells_f) - 1)
    hit = cells_f[pos] == cq
    out[hit] = M[pos[hit]]
    return out


def majority(Tm):
    nv = Tm.sum(1)
    top = Tm.max(1)
    cls = Tm.argmax(1)
    share_ok = 3 * top >= 2 * nv                    # top / n >= 2/3, exact in integers
    return nv, cls, share_ok


def fmt(x, d=2):
    return "n/a" if x is None else ("%.*f" % (d, x))


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="07", choices=["07"])
    ap.add_argument("--traj", default=ROOT + "/out/kitti_seq07_fastlivo2_tum.txt")
    ap.add_argument("--voxel", type=float, default=0.20)
    ap.add_argument("--deskew-bins", type=int, default=128)
    ap.add_argument("--f0", type=int, default=0)
    ap.add_argument("--f1", type=int, default=None)
    ap.add_argument("--eomt", default=ROOT + "/out/vs2d/seg2d_eomt")
    ap.add_argument("--m2f", default=ROOT + "/out/vs2d/seg2d_m2f")
    ap.add_argument("--conf-eomt", dest="conf_eomt", default=ROOT + "/out/distil/conf_eomt")
    ap.add_argument("--conf-m2f", dest="conf_m2f", default=ROOT + "/out/distil/conf_m2f",
                    help="read ONLY for the label-source sensitivity line, never for filter E")
    ap.add_argument("--signal", default=ROOT + "/out/distil/signal_07.npz")
    ap.add_argument("--results-2d", dest="results_2d",
                    default=ROOT + "/out/vs2d/results_eomt.json")
    ap.add_argument("--inventory", default=ROOT + "/out/pseudo/INVENTORY_NOTES.md")
    ap.add_argument("--ctl-map", dest="ctl_map", default="",
                    help="map_eval --conf-gate 0 --save-map npz (cell-key identity check)")
    ap.add_argument("--prereg", default=ROOT + "/out/v06_propagate/PREREG.md")
    ap.add_argument("--json-out", dest="json_out", required=True)
    ap.add_argument("--report", default="")
    ap.add_argument("--smoke", action="store_true",
                    help="partial-sequence code test: gates are not applied, nothing is a result")
    a = ap.parse_args()

    t_start = time.time()
    prereg_sha = hashlib.sha256(open(a.prereg, "rb").read()).hexdigest()
    prereg_stamp = open(a.prereg).read().strip().splitlines()[-1]
    spec, spec_src = load_filter_spec()
    SCR, proj, W, H = seqreg.use(a.seq)
    if a.f1 is None:
        a.f1 = SCR.N_FRAMES_TOTAL
    RAW_S, LB_S = SCR.RAW, SCR.LABELS
    ts = parse_ts(RAW_S + "/velodyne_points/timestamps_start.txt")
    te = parse_ts(RAW_S + "/velodyne_points/timestamps_end.txt")
    traj = SEMC.TrajInterp(a.traj)
    R_IL = np.ascontiguousarray(SEMC.T_I_L[:3, :3])
    t_IL = np.ascontiguousarray(SEMC.T_I_L[:3, 3])
    city_lut = SCR.city19_to_coarse(rider=SCR.RIDER_DEFAULT, sky_policy="wrong")
    NB = a.deskew_bins
    print("propagate_check seq %s frames [%d,%d) img %dx%d voxel %.2f  spec %s"
          % (a.seq, a.f0, a.f1, W, H, a.voxel, json.dumps(spec)), flush=True)

    # ---------------- pass-1 accumulators
    E_key, E_gt, E_lab, E_in, E_rb, E_len = [], [], [], [], [], []
    X_key, X_lab, X_len = [], [], []
    joint = np.zeros((K9, NG, 2, 3, NCB, NRB), np.int64)
    cen = dict(points=0, in_frustum=0, visible=0, candidate=0, sky=0, occluded_or_fill=0,
               agree=0, filterE_all=0, filterE_eval=0, frames=0, gt_excluded_in_candidate=0)
    C2g = np.zeros((K9, K9 + 1), np.int64); A2g = np.zeros(K9, np.int64)
    C2f = np.zeros((K9, K9 + 1), np.int64); A2f = np.zeros(K9, np.int64)
    sens = dict(argmax_mismatch_eomt=0, argmax_mismatch_m2f=0, sampled=0,
                keptE_alt_eomt=0, keptE_alt_m2f=0, keptE_alt_both=0,
                symdiff_alt_eomt=0, symdiff_alt_m2f=0, symdiff_alt_both=0,
                label_differs_in_common_alt_eomt=0)
    missing = []
    nopose_frames, nopose_eval, okfalse_eval, n_frames = 0, 0, 0, 0
    t0 = time.time()
    for f in range(a.f0, a.f1):
        paths = dict(bin=RAW_S + "/velodyne_points/data/%010d.bin" % f,
                     label=LB_S + "/%06d.label" % f,
                     eomt="%s/f%06d.npz" % (a.eomt, f), m2f="%s/f%06d.npz" % (a.m2f, f),
                     conf="%s/f%06d.npz" % (a.conf_eomt, f),
                     m2fc="%s/f%06d.npz" % (a.conf_m2f, f))
        if not all(os.path.exists(p) for p in paths.values()):
            missing.append(f)
            continue
        pts = np.fromfile(paths["bin"], dtype=np.float32).reshape(-1, 4)
        n = len(pts)
        raw = np.fromfile(paths["label"], dtype=np.uint32) & 0xFFFF
        assert len(raw) == n, ("scan/label length", f, n, len(raw))
        gt9 = SKLUT[raw.astype(np.int64)]
        valid = gt9 >= 0

        # ======== camera side, raw order: tools/make_pseudo.py verbatim ========
        xyz = pts[:, :3].astype(np.float64)
        u, v, z, inm = proj.project(xyz)
        rng = np.linalg.norm(xyz, axis=1)
        rbin = np.clip(np.digitize(rng, RANGE_EDGES_M), 0, NRB - 1).astype(np.int8)
        flags = np.zeros(n, np.uint8)
        flags[inm] |= F_INFRUSTUM
        flags[rng < 50.0] |= F_RANGE_LT50
        t9 = np.full(n, -1, np.int8)
        m9 = np.full(n, -1, np.int8)
        conf16 = np.zeros(n, np.float16)
        pred2d = np.full(n, -1, np.int16)        # Arm2D.predict: the published 2D arm
        t9a = np.full(n, -1, np.int8)            # sensitivity: conf-cache argmax labels
        m9a = np.full(n, -1, np.int8)
        de_full = np.zeros(n, bool)
        ninm = int(inm.sum())
        if ninm:
            ui_r = np.rint(u[inm]).astype(np.int64)
            vi_r = np.rint(v[inm]).astype(np.int64)
            zi = z[inm]
            vis = SCR.visible_mask(ui_r, vi_r, zi, w=W, h=H, half_win=SCR.OCC_WIN)
            de = SCR.depth_edge_mask(ui_r, vi_r, zi, w=W, h=H)
            fi = np.zeros(n, np.uint8)
            fi[inm] = vis.astype(np.uint8) * F_VISIBLE + de.astype(np.uint8) * F_DEPTHEDGE
            flags |= fi
            de_full[inm] = de
            with np.load(paths["eomt"]) as zf:
                seg_e = zf["seg"]
            with np.load(paths["conf"]) as zf:
                p1 = zf["p1"]
                seg_ea = zf["seg"]
            with np.load(paths["m2f"]) as zf:
                seg_m = zf["seg"]
            with np.load(paths["m2fc"]) as zf:
                seg_ma = zf["seg"]
            assert p1.shape == seg_e.shape == seg_ea.shape, (f, p1.shape, seg_e.shape)
            assert seg_ma.shape == seg_m.shape, (f, seg_ma.shape, seg_m.shape)
            sh, sw = seg_e.shape
            vi_s, ui_s = sample_idx(u, v, inm, sh, sw, W, H)
            ce_raw = seg_e[vi_s, ui_s].astype(np.int32)
            ce = np.where(ce_raw < 20, ce_raw, 19)
            c9e = city_lut[ce].astype(np.int8)
            sh2, sw2 = seg_m.shape
            vi2, ui2 = sample_idx(u, v, inm, sh2, sw2, W, H)
            cm_raw = seg_m[vi2, ui2].astype(np.int32)
            cm = np.where(cm_raw < 20, cm_raw, 19)
            c9m = city_lut[cm].astype(np.int8)
            # a candidate = in frustum AND z-buffer visible AND the teacher named a
            # common-9 class (sky -> UNMAPPED -2, ignore-fill -> ABSTAIN -1: neither)
            keep = vis & (c9e >= 0)
            tmp = np.full(ninm, -1, np.int8); tmp[keep] = c9e[keep]
            t9[inm] = tmp
            keepm = vis & (c9m >= 0)
            tmp2 = np.full(ninm, -1, np.int8); tmp2[keepm] = c9m[keepm]
            m9[inm] = tmp2
            conf16[inm] = p1[vi_s, ui_s].astype(np.float16)
            pred2d[inm] = np.where(vis, city_lut[ce], -1)
            # ---- label-source sensitivity (argmax of the confidence caches) ----
            ca_raw = seg_ea[vi_s, ui_s].astype(np.int32)
            ma_raw = seg_ma[vi2, ui2].astype(np.int32)
            sens["argmax_mismatch_eomt"] += int((ca_raw != ce_raw).sum())
            sens["argmax_mismatch_m2f"] += int((ma_raw != cm_raw).sum())
            sens["sampled"] += ninm
            c9ea = city_lut[np.where(ca_raw < 20, ca_raw, 19)].astype(np.int8)
            c9ma = city_lut[np.where(ma_raw < 20, ma_raw, 19)].astype(np.int8)
            ka = vis & (c9ea >= 0)
            tmp3 = np.full(ninm, -1, np.int8); tmp3[ka] = c9ea[ka]
            t9a[inm] = tmp3
            kb = vis & (c9ma >= 0)
            tmp4 = np.full(ninm, -1, np.int8); tmp4[kb] = c9ma[kb]
            m9a[inm] = tmp4
        conf32 = conf16.astype(np.float32)
        keepE = filter_E_mask(t9, m9, conf32, flags, spec)
        lab_own = np.where(keepE, t9, -1).astype(np.int8)
        kA = filter_E_mask(t9a, m9, conf32, flags, spec)
        kB = filter_E_mask(t9, m9a, conf32, flags, spec)
        kAB = filter_E_mask(t9a, m9a, conf32, flags, spec)
        sens["keptE_alt_eomt"] += int(kA.sum()); sens["symdiff_alt_eomt"] += int((kA != keepE).sum())
        sens["keptE_alt_m2f"] += int(kB.sum()); sens["symdiff_alt_m2f"] += int((kB != keepE).sum())
        sens["keptE_alt_both"] += int(kAB.sum()); sens["symdiff_alt_both"] += int((kAB != keepE).sum())
        sens["label_differs_in_common_alt_eomt"] += int((kA & keepE & (t9a != t9)).sum())

        # ---- census, joint table (tools/distil_signal.py layout), 2D arm ----
        visib = inm & (pred2d != -1)
        cand = visib & (pred2d >= 0)
        assert np.array_equal(cand, t9 >= 0), f
        cen["points"] += n; cen["in_frustum"] += ninm; cen["visible"] += int(visib.sum())
        cen["candidate"] += int(cand.sum()); cen["sky"] += int((visib & (pred2d == -2)).sum())
        cen["occluded_or_fill"] += int((inm & (pred2d == -1)).sum())
        cen["agree"] += int((cand & (m9 == t9)).sum())
        cen["gt_excluded_in_candidate"] += int((cand & ~valid).sum())
        cen["filterE_all"] += int(keepE.sum()); cen["filterE_eval"] += int((keepE & valid).sum())
        cen["frames"] += 1
        if cand.any():
            tc = t9[cand].astype(np.int64)
            gb = np.where(gt9 >= 0, gt9, K9)[cand].astype(np.int64)
            dq = de_full[cand].astype(np.int64)
            ob = m9[cand]
            ag = np.where(ob < 0, 2, np.where(ob == tc, 1, 0)).astype(np.int64)
            cb = np.clip((conf32[cand] * NCB).astype(np.int64), 0, NCB - 1)
            rq = rbin[cand].astype(np.int64)
            flat = ((((tc * NG + gb) * 2 + dq) * 3 + ag) * NCB + cb) * NRB + rq
            joint += np.bincount(flat, minlength=joint.size).reshape(joint.shape)
        acc2d_add(C2g, A2g, gt9[valid], pred2d[valid])
        acc2d_add(C2f, A2f, gt9[valid & inm], pred2d[valid & inm])

        # ======== map side, driver order: tools/map_eval.py verbatim, keep = ok ========
        _, tsyn, order = synth(pts[:, :3], sweep_duration=(te[f] - ts[f]) / 1e9)
        bag = np.ascontiguousarray(pts[order])
        gt_b = gt9[order]; in_b = inm[order]; lab_b = lab_own[order]; rb_b = rbin[order]
        t_pt = ts[f] * 1e-9 + tsyn
        v_b = gt_b >= 0
        ok = traj.valid(t_pt)
        key_b = np.full(n, -1, np.int64)
        n_frames += 1
        if ok.sum() < 100:                       # the node's pose gate: whole sweep dropped
            nopose_frames += 1
            nopose_eval += int(v_b.sum())
        else:
            oki = np.flatnonzero(ok)             # keep = ok: NO confidence gate (PREREG)
            tk = t_pt[oki]
            t_lo, t_hi = tk[0], tk[-1]
            ctr = t_lo + (np.arange(NB) + 0.5) * (t_hi - t_lo) / NB
            Rb, pb, _ = traj.query(ctr)
            pw_all = deskew(bag[oki, :3], t_pt[oki], Rb, pb, t_lo, t_hi, NB, R_IL, t_IL)
            key_b[oki] = SEMC.voxel_key(pw_all, a.voxel)
            okfalse_eval += int((v_b & ~ok).sum())
        E_key.append(key_b[v_b]); E_gt.append(gt_b[v_b].astype(np.int8))
        E_lab.append(lab_b[v_b]); E_in.append(in_b[v_b]); E_rb.append(rb_b[v_b])
        E_len.append(int(v_b.sum()))
        xm = (~v_b) & (key_b >= 0)               # GT-EXCLUDED, pose-valid: key check + R2
        X_key.append(key_b[xm]); X_lab.append(lab_b[xm]); X_len.append(int(xm.sum()))
        if n_frames % 100 == 0:
            print("  frame %d  %.0f s  rss %.1f GB" % (
                f, time.time() - t0,
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6), flush=True)

    t_pass1 = time.time() - t0
    E_key = np.concatenate(E_key); E_gt = np.concatenate(E_gt); E_lab = np.concatenate(E_lab)
    E_in = np.concatenate(E_in); E_rb = np.concatenate(E_rb)
    X_key = np.concatenate(X_key); X_lab = np.concatenate(X_lab)
    E_off = np.concatenate([[0], np.cumsum(E_len)]).astype(np.int64)
    X_off = np.concatenate([[0], np.cumsum(X_len)]).astype(np.int64)
    N_eval = int(len(E_key))
    print("pass 1: %d frames (%d missing, %d no-pose) %d evaluated points  %.0f s"
          % (n_frames, len(missing), nopose_frames, N_eval, t_pass1), flush=True)

    res = dict(tool="tools/propagate_check.py", seq=a.seq, smoke=bool(a.smoke),
               prereg=dict(path=a.prereg, sha256=prereg_sha, stamp=prereg_stamp),
               cfg=dict(traj=a.traj, voxel=a.voxel, deskew_bins=NB, f0=a.f0, f1=a.f1,
                        conf_gate=None, cells="keys of evaluated points with a valid pose",
                        filter_spec=spec, filter_spec_sources=spec_src,
                        caches=dict(eomt_label=a.eomt, eomt_conf=a.conf_eomt + " (p1)",
                                    m2f_label=a.m2f),
                        majority_rule="n_votes >= k AND 3*top >= 2*n_votes, after removing "
                                      "the point's own frame's votes (leave-own-frame-out)",
                        ks=list(KS), label_space="common-9 (label_spaces.COARSE)",
                        frustum="score_2d_vs_3d.Projector P3 on the raw sensor-frame point"),
               frames=dict(processed=n_frames, missing=missing, no_pose=nopose_frames),
               points=dict(evaluated=N_eval, no_pose_frames_eval=nopose_eval,
                           pose_invalid_in_posed_frames_eval=okfalse_eval,
                           gt_excluded_pose_valid=int(len(X_key))),
               census=cen, sensitivity_label_source=sens)

    # ======================================================== GATE 1 (and exact aux checks)
    inv = inventory_seq07_row(a.inventory)
    cand_pct = pct(cen["candidate"], cen["points"])
    agree_pct = pct(cen["agree"], cen["candidate"])
    g1_cand = abs(cand_pct - inv["candidate_pct"]) <= 0.1
    g1_agree = abs(agree_pct - inv["agreement_pct"]) <= 0.1
    gate1_pass = bool(g1_cand and g1_agree and not a.smoke)
    d2 = json.load(open(a.results_2d))["agg"]["2d"]
    pub_cov = float(d2["global"]["coverage"]["mean"]) * 100.0
    pub_miou = float(d2["frustum"]["miou_all_abstain_excluded"]["mean"])
    pub_nl = int(d2["global"]["n_labelled"]["mean"]) if isinstance(d2["global"]["n_labelled"], dict) \
        else int(d2["global"]["n_labelled"])
    pub_ne = int(d2["global"]["n_eval"]["mean"]) if isinstance(d2["global"]["n_eval"], dict) \
        else int(d2["global"]["n_eval"])
    m2g = acc2d_metrics(C2g, A2g); m2f = acc2d_metrics(C2f, A2f)
    sig = np.load(a.signal)
    J = sig["joint"].astype(np.int64)
    sig_n = [int(x) for x in sig["n"][:8]]
    jE = J[:, :, :, 1, 18:, :]                   # agree AND conf bin >= 18 (conf >= 0.90)
    sigE_all = int(jE.sum()); sigE_eval = int(jE[:, :K9].sum())
    sigE_corr = int(sum(jE[c, c].sum() for c in range(K9)))
    mine_census = [cen["points"], cen["in_frustum"], cen["visible"], cen["candidate"],
                   cen["sky"], cen["occluded_or_fill"], cen["gt_excluded_in_candidate"],
                   cen["frames"]]
    aux = dict(
        two_d_arm=dict(
            why="independent end-to-end check of projection / z-buffer / sub-pixel sampling / "
                "LUT / GT / frustum on the frozen EoMT cache; NOT filter E's operating point",
            published=dict(coverage_pct=pub_cov, frustum_miou9_abstain_excluded=pub_miou,
                           n_labelled=pub_nl, n_eval=pub_ne, source=a.results_2d),
            reproduced=dict(coverage_pct=100.0 * m2g["coverage"] if m2g["coverage"] else None,
                            frustum_miou9_abstain_excluded=m2f["miou9_abstain_excluded"],
                            n_labelled=m2g["n_labelled"], n_eval=m2g["n_eval"],
                            n_unmapped_pred=m2g["n_unmapped_pred"],
                            frustum_acc_abstain_excluded=m2f["acc_abstain_excluded"]),
        ),
        recon_signal_table=dict(
            why="out/distil/signal_07.npz (tools/distil_signal.py, the source of out/pseudo's "
                "seq-07 row) holds every candidate binned by teacher class x GT x depth-edge x "
                "agreement x conf(0.05) x range; filter E = agreement==1 & conf bin >= 18",
            census_published=dict(zip(["points", "in_frustum", "visible", "candidate", "sky",
                                       "occluded_or_fill", "gt_excluded_in_candidate",
                                       "frames"], sig_n)),
            census_equal=bool(all(m is None or m == s for m, s in zip(mine_census, sig_n))),
            joint_table_equal=bool(np.array_equal(J, joint)),
            joint_table_abs_diff=int(np.abs(J - joint).sum()),
            filterE_published=dict(all_candidates=sigE_all, evaluated=sigE_eval,
                                   correct=sigE_corr,
                                   coverage_pct=pct(sigE_eval, pub_ne),
                                   precision_pct=pct(sigE_corr, sigE_eval)),
        ),
    )
    a2 = aux["two_d_arm"]
    a2["pass_within_0.1"] = bool(
        a2["reproduced"]["coverage_pct"] is not None
        and abs(a2["reproduced"]["coverage_pct"] - pub_cov) <= 0.1
        and abs(a2["reproduced"]["frustum_miou9_abstain_excluded"] - pub_miou) <= 0.1)
    a2["exact_counts"] = bool(m2g["n_labelled"] == pub_nl and m2g["n_eval"] == pub_ne)
    res["gate1"] = {"pass": gate1_pass}
    res["gate1"].update(dict(
        reproduced="out/pseudo's own audit, seq-07 row",
        why=("filter E is NOT the 2D arm's operating point: the 2D arm answers every z-buffer-"
             "visible in-frustum point with the EoMT label and counts sky as an answer; filter E "
             "additionally requires EoMT == Mask2Former and EoMT p1 >= 0.90 and never keeps sky. "
             "PREREG: 'otherwise reproduce out/pseudo's own audit numbers'. The only out/pseudo "
             "audit numbers defined on seq 07 are the '07 (published)' row of "
             "out/pseudo/INVENTORY_NOTES.md (candidate % of all points, two-teacher agreement % "
             "of candidates)."),
        target=inv, reproduced_values=dict(candidate_pct=cand_pct, agreement_pct=agree_pct,
                                           candidate=cen["candidate"], points=cen["points"],
                                           agree=cen["agree"]),
        abs_diff=dict(candidate_pct=abs(cand_pct - inv["candidate_pct"]),
                      agreement_pct=abs(agree_pct - inv["agreement_pct"])),
        tolerance=0.1, aux=aux))
    res["gate1"]["aux_all_exact"] = bool(a2["exact_counts"] and a2["pass_within_0.1"]
                                         and aux["recon_signal_table"]["joint_table_equal"]
                                         and aux["recon_signal_table"]["census_equal"])
    print("GATE 1: candidate %.4f%% (target %.2f)  agreement %.4f%% (target %.1f)  -> %s"
          % (cand_pct, inv["candidate_pct"], agree_pct, inv["agreement_pct"],
             "PASS" if gate1_pass else "FAIL"), flush=True)
    print("   aux 2D arm: coverage %.4f%% (pub %.4f)  frustum mIoU %.4f (pub %.4f)  n_lab %d (pub %d)"
          % (a2["reproduced"]["coverage_pct"] or -1, pub_cov,
             a2["reproduced"]["frustum_miou9_abstain_excluded"] or -1, pub_miou,
             m2g["n_labelled"], pub_nl), flush=True)
    print("   aux signal table: joint equal %s (|diff| %d)  filter E all %d / eval %d (pub %d / %d)"
          % (aux["recon_signal_table"]["joint_table_equal"],
             aux["recon_signal_table"]["joint_table_abs_diff"], cen["filterE_all"],
             cen["filterE_eval"], sigE_all, sigE_eval), flush=True)

    if not gate1_pass and not a.smoke:
        res["decision"] = None
        res["stopped"] = "GATE 1 failed: the per-scan reading does not reproduce; no decision"
        json.dump(res, open(a.json_out, "w"), indent=2)
        if a.report:
            write_report(res, a.report)
        print("STOPPED AT GATE 1", flush=True)
        sys.exit(2)

    # ======================================================== cells
    t1 = time.time()
    has_cell = E_key >= 0
    cells = np.unique(E_key[has_cell])
    n_cells = int(len(cells))
    E_cell = np.full(N_eval, -1, np.int32)
    E_cell[has_cell] = np.searchsorted(cells, E_key[has_cell]).astype(np.int32)
    xpos = np.minimum(np.searchsorted(cells, X_key), max(n_cells - 1, 0))
    xhit = (cells[xpos] == X_key) if n_cells else np.zeros(len(X_key), bool)
    X_cell = np.where(xhit, xpos, -1).astype(np.int64)
    res["cells"] = dict(n_cells=n_cells, evaluated_with_cell=int(has_cell.sum()),
                        evaluated_without_cell=int((~has_cell).sum()),
                        gt_excluded_pose_valid_in_evaluated_cell=int(xhit.sum()))

    # ---- cell-key identity against map_eval --conf-gate 0 --save-map
    kc = dict(checked=False)
    if a.ctl_map and os.path.exists(a.ctl_map):
        zc = np.load(a.ctl_map)
        ck = zc["key"].astype(np.int64)
        o = np.argsort(ck, kind="stable")
        ck_s = ck[o]
        all_keys = np.union1d(cells, X_key)
        keys_equal = bool(np.array_equal(ck_s, all_keys))
        kc = dict(checked=True, ctl_map=a.ctl_map, ctl_voxels=int(len(ck)),
                  mine_voxels=int(len(all_keys)), key_set_equal=keys_equal)
        if keys_equal:
            pc = np.searchsorted(all_keys, cells)
            ev_cnt = np.bincount(E_cell[has_cell], minlength=n_cells)
            in_cnt = np.bincount(E_cell[has_cell & E_in], minlength=n_cells)
            ev_all = np.zeros(len(all_keys), np.int64); ev_all[pc] = ev_cnt
            in_all = np.zeros(len(all_keys), np.int64); in_all[pc] = in_cnt
            tot_all = ev_all + np.bincount(np.searchsorted(all_keys, X_key),
                                           minlength=len(all_keys))
            kc["n_obs_equal"] = bool(np.array_equal(zc["n_obs"].astype(np.int64)[o], tot_all))
            kc["gt_cnt_equal"] = bool(np.array_equal(zc["gt_cnt"].astype(np.int64)[o], ev_all))
            kc["gt_in_cnt_equal"] = bool(np.array_equal(zc["gt_in_cnt"].astype(np.int64)[o],
                                                        in_all))
        kc["identical"] = bool(kc["key_set_equal"] and kc.get("n_obs_equal") and
                               kc.get("gt_cnt_equal") and kc.get("gt_in_cnt_equal"))
    res["cell_key_check"] = kc
    print("cells %d  key check %s" % (n_cells, json.dumps(kc)), flush=True)
    del E_key

    # ======================================================== votes
    own = E_lab >= 0
    vote = own & has_cell
    T = np.bincount(E_cell[vote].astype(np.int64) * K9 + E_lab[vote],
                    minlength=n_cells * K9).reshape(n_cells, K9).astype(np.int64)
    xv = (X_cell >= 0) & (X_lab >= 0)
    TX = np.bincount(X_cell[xv] * K9 + X_lab[xv], minlength=n_cells * K9
                     ).reshape(n_cells, K9).astype(np.int64)
    T2 = T + TX
    res["votes"] = dict(votes_evaluated=int(vote.sum()), votes_gt_excluded=int(xv.sum()),
                        labelable_without_cell=int((own & ~has_cell).sum()),
                        cells_with_votes=int((T.sum(1) > 0).sum()))

    # ======================================================== pass 2: leave-own-frame-out
    P_lofo = {k: np.full(N_eval, -1, np.int8) for k in KS}
    P_incl = {k: np.full(N_eval, -1, np.int8) for k in KS}
    P_r2 = {k: np.full(N_eval, -1, np.int8) for k in KS}
    reach = {s: np.zeros(4, np.int64) for s in ("in", "out")}   # query pts with >=0/1/2/3 votes
    nF = len(E_len)
    for i in range(nF):
        s0, s1 = E_off[i], E_off[i + 1]
        ce_ = E_cell[s0:s1]
        le_ = E_lab[s0:s1]
        own_i = (le_ >= 0) & (ce_ >= 0)
        q = (le_ < 0) & (ce_ >= 0)
        if not q.any():
            continue
        cq = ce_[q].astype(np.int64)
        codes = ce_[own_i].astype(np.int64) * K9 + le_[own_i]
        Tq_incl = T[cq]
        Tq = Tq_incl - own_votes(codes, cq)
        x0, x1 = X_off[i], X_off[i + 1]
        xc = X_cell[x0:x1]; xl = X_lab[x0:x1]
        xm_ = (xc >= 0) & (xl >= 0)
        codes2 = np.concatenate([codes, xc[xm_] * K9 + xl[xm_]])
        Tq2 = T2[cq] - own_votes(codes2, cq)
        assert Tq.min() >= 0 and Tq2.min() >= 0, i
        iq = E_in[s0:s1][q]
        for Tm, Pd, is_lofo in ((Tq, P_lofo, True), (Tq_incl, P_incl, False),
                                (Tq2, P_r2, False)):
            nv, cls, share_ok = majority(Tm)
            for k in KS:
                lab_k = np.where((nv >= k) & share_ok, cls, -1).astype(np.int8)
                Pd[k][s0:s1][q] = lab_k
            if is_lofo:
                for s, msk in (("in", iq), ("out", ~iq)):
                    reach[s] += np.array([int(msk.sum()), int((msk & (nv >= 1)).sum()),
                                          int((msk & (nv >= 2)).sum()),
                                          int((msk & (nv >= 3)).sum())], np.int64)
    t_pass2 = time.time() - t1
    print("pass 2 done %.0f s" % t_pass2, flush=True)

    # ---- brute-force self-check of the vectorised vote arithmetic on a random sample:
    # every vote of the sampled point's cell is listed with its frame and counted directly.
    rgen = np.random.default_rng(20260926)
    qidx = np.flatnonzero(~own & has_cell)
    samp = (np.sort(rgen.choice(qidx, size=min(4000, len(qidx)), replace=False))
            if len(qidx) else qidx)
    vidx = np.flatnonzero(vote)
    xidx = np.flatnonzero(xv)
    all_ce = np.concatenate([E_cell[vidx].astype(np.int64), X_cell[xidx]])
    all_fr = np.concatenate([np.searchsorted(E_off, vidx, side="right") - 1,
                             np.searchsorted(X_off, xidx, side="right") - 1])
    all_lb = np.concatenate([E_lab[vidx].astype(np.int64), X_lab[xidx].astype(np.int64)])
    is_x = np.concatenate([np.zeros(len(vidx), bool), np.ones(len(xidx), bool)])
    o_s = np.argsort(all_ce, kind="stable")
    all_ce = all_ce[o_s]; all_fr = all_fr[o_s]; all_lb = all_lb[o_s]; is_x = is_x[o_s]
    mism = dict(lofo=0, incl=0, r2=0)
    for j in samp:
        c = int(E_cell[j])
        fr = int(np.searchsorted(E_off, j, side="right") - 1)
        lo = np.searchsorted(all_ce, c, "left")
        hi = np.searchsorted(all_ce, c, "right")
        fr_c = all_fr[lo:hi]; lb_c = all_lb[lo:hi]; x_c = is_x[lo:hi]
        for name, sel, Pd in (("lofo", (~x_c) & (fr_c != fr), P_lofo),
                              ("incl", ~x_c, P_incl), ("r2", fr_c != fr, P_r2)):
            tl = np.bincount(lb_c[sel], minlength=K9)
            nv = int(tl.sum()); top = int(tl.max()); cl = int(tl.argmax())
            for k in KS:
                expv = cl if (nv >= k and 3 * top >= 2 * nv) else -1
                if expv != int(Pd[k][j]):
                    mism[name] += 1
    del all_ce, all_fr, all_lb, is_x, o_s
    res["lofo_selfcheck"] = dict(sampled=int(len(samp)), mismatches=mism,
                                 ok=bool(sum(mism.values()) == 0),
                                 what="brute-force per-cell recount of LOFO / incl-own-frame / "
                                      "LOFO+GT-excluded votes vs the vectorised pass, k=1,2,3")
    print("LOFO self-check: %s" % json.dumps(res["lofo_selfcheck"]), flush=True)

    # ======================================================== measurements
    ALL = np.ones(N_eval, bool)
    masks = {"all": ALL, "in": E_in, "out": ~E_in}
    ev_by = {s: np.bincount(E_gt[m].astype(np.int64), minlength=K9) for s, m in masks.items()}
    nown_by = {s: np.bincount(E_gt[m & ~own].astype(np.int64), minlength=K9)
               for s, m in masks.items()}

    def block(L):
        return {s: summarize(cmat(E_gt, L, m), ev_by[s]) for s, m in masks.items()}

    def po_block(P):
        out = {}
        for s, m in masks.items():
            sm = summarize(cmat(E_gt, P, m & ~own), nown_by[s])
            sm["share_of_evaluated_in_subset"] = pct(sm["supervised"], int(m.sum()))
            out[s] = sm
        return out

    per_scan = block(E_lab)
    prop = {str(k): block(np.where(own, E_lab, P_lofo[k]).astype(np.int8)) for k in KS}
    po = {str(k): po_block(P_lofo[k]) for k in KS}
    po_incl = {str(k): po_block(P_incl[k]) for k in KS}
    po_r2 = {str(k): po_block(P_r2[k]) for k in KS}
    prop_r2 = {}
    for k in KS:
        bl = block(np.where(own, E_lab, P_r2[k]).astype(np.int8))
        prop_r2[str(k)] = {s: dict(coverage=bl[s]["coverage"], precision=bl[s]["precision"],
                                   supervised=bl[s]["supervised"]) for s in bl}
    # range breakdown of the propagated-only labels and of per-scan E
    rng_po = {}
    for k in KS:
        rng_po[str(k)] = {}
        for s in ("in", "out"):
            rows = []
            for r in range(NRB):
                m = masks[s] & ~own & (E_rb == r)
                C = cmat(E_gt, P_lofo[k], m)
                rows.append(dict(range=RB_NAMES[r], evaluated_not_labelable=int(m.sum()),
                                 propagated=int(C.sum()), precision=pct(np.trace(C), C.sum()),
                                 coverage=pct(C.sum(), m.sum())))
            rng_po[str(k)][s] = rows
    rng_E = []
    for r in range(NRB):
        m = E_in & (E_rb == r)
        C = cmat(E_gt, E_lab, m)
        rng_E.append(dict(range=RB_NAMES[r], evaluated=int(m.sum()), supervised=int(C.sum()),
                          precision=pct(np.trace(C), C.sum()), coverage=pct(C.sum(), m.sum())))

    # ======================================================== GATE 2
    g2 = {}
    for k in KS:
        n_own = int(own.sum())
        n_po = int((~own & (P_lofo[k] >= 0)).sum())
        n_un = int((~own & (P_lofo[k] < 0)).sum())
        by_sub = {}
        for s in ("in", "out"):
            m = masks[s]
            by_sub[s] = dict(labelable_own=int((m & own).sum()),
                             propagated_only=int((m & ~own & (P_lofo[k] >= 0)).sum()),
                             unlabelled=int((m & ~own & (P_lofo[k] < 0)).sum()),
                             evaluated=int(m.sum()))
            by_sub[s]["sum_equals_evaluated"] = bool(
                by_sub[s]["labelable_own"] + by_sub[s]["propagated_only"]
                + by_sub[s]["unlabelled"] == by_sub[s]["evaluated"])
        g2[str(k)] = dict(labelable_own=n_own, propagated_only=n_po, unlabelled=n_un,
                          total=n_own + n_po + n_un, evaluated=N_eval,
                          sum_equals_evaluated=bool(n_own + n_po + n_un == N_eval),
                          by_subset=by_sub)
    gate2_pass = bool(all(g2[str(k)]["sum_equals_evaluated"] and
                          all(g2[str(k)]["by_subset"][s]["sum_equals_evaluated"]
                              for s in ("in", "out")) for k in KS))
    res["gate2"] = {"pass": gate2_pass}
    res["gate2"].update(dict(per_k=g2, evaluated_total=N_eval,
                        evaluated_total_published=pub_ne,
                        evaluated_total_equals_published=bool(N_eval == pub_ne)))

    res["per_scan_filterE"] = per_scan
    res["per_scan_filterE_by_range_in_frustum"] = rng_E
    res["propagation"] = prop
    res["propagated_only_lofo"] = po
    res["propagated_only_lofo_by_range"] = rng_po
    res["lofo_vote_reach"] = {s: dict(query_points=int(reach[s][0]),
                                      ge1=pct(reach[s][1], reach[s][0]),
                                      ge2=pct(reach[s][2], reach[s][0]),
                                      ge3=pct(reach[s][3], reach[s][0]))
                              for s in reach}
    res["diagnostics_not_decision"] = dict(
        propagated_only_including_own_frame_votes=po_incl,
        propagated_only_lofo_votes_incl_gt_excluded_points=po_r2,
        propagation_lofo_votes_incl_gt_excluded_points=prop_r2,
        note=("including-own-frame = the non-LOFO reading (the query point's own sweep votes "
              "for its cell); votes-incl-GT-excluded = filter-E labels of points whose GT is "
              "EXCLUDED also vote (they would at deployment; the PREREG's cells are built from "
              "evaluated points only)"))

    # ======================================================== decision (PREREG, literal)
    cov_E = per_scan["all"]["coverage"]
    prec_E_in = per_scan["in"]["precision"]
    rows_k = {}
    for k in KS:
        cov_P = prop[str(k)]["all"]["coverage"]
        p_po = po[str(k)]["out"]["precision"]
        rows_k[str(k)] = dict(
            coverage_prop_all=cov_P, coverage_prop_over_perscan=(cov_P / cov_E if cov_E else None),
            a_holds=bool(cov_P is not None and cov_E is not None and cov_P >= 2.0 * cov_E),
            propagated_only_out_precision=p_po,
            propagated_only_out_count=po[str(k)]["out"]["supervised"],
            b_threshold=(prec_E_in - 5.0) if prec_E_in is not None else None,
            b_holds=bool(p_po is not None and prec_E_in is not None and p_po >= prec_E_in - 5.0),
            coverage_propagated_only_alone_all=pct(po[str(k)]["all"]["supervised"], N_eval))
    a_h = rows_k["2"]["a_holds"]; b_h = rows_k["2"]["b_holds"]
    decision = "PREMISE HOLDS" if (a_h and b_h) else "PREMISE FAILS"
    k_b = [k for k in KS if rows_k[str(k)]["b_holds"]]
    res["decision_rule"] = dict(
        primary_k=2, per_k=rows_k, perscan_coverage_all=cov_E, perscan_precision_in=prec_E_in,
        a_holds=a_h, b_holds=b_h, smallest_k_where_b_holds=(k_b[0] if k_b else None),
        reading_of_a=("propagated coverage = (labelable-in-own-frame + propagated-only) / "
                      "evaluated, i.e. the coverage of the propagation scheme whose accounting "
                      "is GATE 2; the propagated-only share alone is reported next to it"))
    instrument_ok = bool(res["lofo_selfcheck"]["ok"] and
                         (res["cell_key_check"].get("identical", False)
                          if res["cell_key_check"].get("checked") else a.smoke))
    res["instrument_ok"] = instrument_ok
    res["decision"] = None if (a.smoke or not gate2_pass or not instrument_ok) else decision
    if not a.smoke and not gate2_pass:
        res["stopped"] = "GATE 2 failed: accounting does not close; no decision"
    elif not a.smoke and not instrument_ok:
        res["stopped"] = ("instrument check failed (cell keys vs map_eval, or LOFO self-check); "
                          "no decision")
    r2p = prop_r2["2"]["all"]["coverage"]; r2b = po_r2["2"]["out"]["precision"]
    dec_r2 = "PREMISE HOLDS" if (r2p is not None and r2p >= 2.0 * cov_E and r2b is not None
                                 and r2b >= prec_E_in - 5.0) else "PREMISE FAILS"
    a_strict = rows_k["2"]["coverage_propagated_only_alone_all"] >= 2.0 * cov_E
    dec_strict = "PREMISE HOLDS" if (a_strict and b_h) else "PREMISE FAILS"
    res["caveats_computed"] = [
        "filter-E labels on GT-EXCLUDED points: %d of %d (%.2f %%); never scored, and they vote "
        "only in the 'LOFO + GT-excluded voters' diagnostic, whose decision would read: %s"
        % (cen["filterE_all"] - cen["filterE_eval"], cen["filterE_all"],
           pct(cen["filterE_all"] - cen["filterE_eval"], cen["filterE_all"]), dec_r2),
        "evaluated points without a cell (sweep without pose, or point outside the trajectory "
        "span): %d (%.3f %% of evaluated); they can only be labelled in their own frame"
        % (int((~has_cell).sum()), pct(int((~has_cell).sum()), N_eval)),
        "stricter reading of (a) -- propagated-only share alone >= 2x per-scan: %s -> decision "
        "would read: %s" % (bool(a_strict), dec_strict),
        "B0's loss additionally drops terrain and manmade (arm_B0.py exclude_classes); filter E "
        "itself does not, and the PREREG measures all nine classes",
    ]
    res["timing_s"] = dict(pass1=t_pass1, pass2=t_pass2, total=time.time() - t_start)
    res["max_rss_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    json.dump(res, open(a.json_out, "w"), indent=2)
    if a.report:
        write_report(res, a.report)
    print("per-scan E: coverage %.3f%%  precision(in) %.3f%%" % (cov_E, prec_E_in), flush=True)
    for k in KS:
        r = rows_k[str(k)]
        print("k>=%d: prop coverage %.3f%% (x%.2f, a %s)  PO-out precision %s n %d (b %s)"
              % (k, r["coverage_prop_all"], r["coverage_prop_over_perscan"], r["a_holds"],
                 fmt(r["propagated_only_out_precision"], 3), r["propagated_only_out_count"],
                 r["b_holds"]), flush=True)
    print("GATE 2 %s   DECISION: %s   smallest k with (b): %s"
          % ("PASS" if gate2_pass else "FAIL", res["decision"],
             res["decision_rule"]["smallest_k_where_b_holds"]), flush=True)


# ------------------------------------------------------------------ report
def write_report(res, path):
    L = []
    w = L.append
    g1 = res["gate1"]
    w("# Map-propagated camera pseudo-labels: premise check -- RESULT (seq %s)" % res["seq"])
    w("")
    w("PREREG: `%s` (sha256 %s..., %s). Tool: `tools/propagate_check.py`; run: "
      "`opt/run_propagate.sh`; numbers: `out/v06_propagate/result.json`."
      % (res["prereg"]["path"], res["prereg"]["sha256"][:16], res["prereg"]["stamp"]))
    if res.get("smoke"):
        w("")
        w("**SMOKE RUN on a partial sequence -- nothing here is a result.**")
    c = res["cfg"]
    w("")
    w("Instrument: seq 07, FAST-LIVO2 trajectory `%s`, %.2f m cells, %d-bin deskew, no "
      "confidence gate; cells = keys of evaluated points (GT not EXCLUDED) with a valid pose. "
      "Filter E = `%s` (from the B0 config, equal to `filterE_08.json` and to the PREREG). "
      "Majority rule: %s." % (c["traj"], c["voxel"], c["deskew_bins"],
                             json.dumps(c["filter_spec"]), c["majority_rule"]))
    w("")
    w("## GATE 1 -- the per-scan reading is right: %s" % ("PASS" if g1["pass"] else "FAIL"))
    w("")
    w("What was reproduced and why: %s" % g1["why"])
    w("")
    w("| number | target | reproduced | abs diff | within 0.1 |")
    w("|---|---|---|---|---|")
    w("| candidate %% of all points (%s) | %.2f | %.4f (%d / %d) | %.4f | %s |"
      % ("out/pseudo seq-07 row", g1["target"]["candidate_pct"],
         g1["reproduced_values"]["candidate_pct"], g1["reproduced_values"]["candidate"],
         g1["reproduced_values"]["points"], g1["abs_diff"]["candidate_pct"],
         g1["abs_diff"]["candidate_pct"] <= 0.1))
    w("| two-teacher agreement %% of candidates (%s) | %.1f | %.4f (%d / %d) | %.4f | %s |"
      % ("out/pseudo seq-07 row", g1["target"]["agreement_pct"],
         g1["reproduced_values"]["agreement_pct"], g1["reproduced_values"]["agree"],
         g1["reproduced_values"]["candidate"], g1["abs_diff"]["agreement_pct"],
         g1["abs_diff"]["agreement_pct"] <= 0.1))
    aux = g1["aux"]
    a2 = aux["two_d_arm"]
    w("")
    w("Also reproduced (not the gate; they check the same code path end to end):")
    w("")
    w("| check | published | reproduced | result |")
    w("|---|---|---|---|")
    w("| 2D arm coverage %% of evaluated points | %.4f (%d / %d) | %.4f (%d / %d) | %s |"
      % (a2["published"]["coverage_pct"], a2["published"]["n_labelled"], a2["published"]["n_eval"],
         a2["reproduced"]["coverage_pct"] or -1, a2["reproduced"]["n_labelled"],
         a2["reproduced"]["n_eval"],
         "exact" if a2["exact_counts"] else ("within 0.1" if a2["pass_within_0.1"] else "MISMATCH")))
    w("| 2D arm in-frustum mIoU-9 (abstain excluded) | %.4f | %.4f | %s |"
      % (a2["published"]["frustum_miou9_abstain_excluded"],
         a2["reproduced"]["frustum_miou9_abstain_excluded"] or -1,
         "within 0.1" if a2["pass_within_0.1"] else "MISMATCH"))
    rs = aux["recon_signal_table"]
    w("| recon joint table (9x10x2x3x20x5 counts, out/distil/signal_07.npz) | %d candidates "
      "| sum |diff| = %d | %s |" % (rs["census_published"]["candidate"], rs["joint_table_abs_diff"],
                                    "bit-identical" if rs["joint_table_equal"] else "DIFFERS"))
    w("| filter E on seq 07 from that table: kept / evaluated kept / correct | %d / %d / %d "
      "| %d / %d / (see below) | %s |"
      % (rs["filterE_published"]["all_candidates"], rs["filterE_published"]["evaluated"],
         rs["filterE_published"]["correct"], res["census"]["filterE_all"],
         res["census"]["filterE_eval"],
         "exact" if (res["census"]["filterE_all"] == rs["filterE_published"]["all_candidates"]
                     and res["census"]["filterE_eval"] == rs["filterE_published"]["evaluated"])
         else "DIFFERS"))
    kc = res.get("cell_key_check", {})
    if kc.get("checked"):
        w("| cell keys vs `map_eval --conf-gate 0 --save-map` | %d voxels | %d voxels; key set "
          "equal %s, points/key equal %s, evaluated/key %s, in-frustum/key %s | %s |"
          % (kc["ctl_voxels"], kc["mine_voxels"], kc["key_set_equal"], kc.get("n_obs_equal"),
             kc.get("gt_cnt_equal"), kc.get("gt_in_cnt_equal"),
             "identical" if kc.get("identical") else "DIFFERS"))
    sens = res["sensitivity_label_source"]
    w("")
    w("Label-source sensitivity (PREREG fixes the frozen label caches; the confidence caches "
      "carry their own argmax from a second GPU pass): argmax differs at %d / %d sampled EoMT "
      "points and %d / %d M2F points; filter E with the conf-cache argmax instead changes the "
      "kept set by %d (EoMT), %d (M2F), %d (both) points."
      % (sens["argmax_mismatch_eomt"], sens["sampled"], sens["argmax_mismatch_m2f"],
         sens["sampled"], sens["symdiff_alt_eomt"], sens["symdiff_alt_m2f"],
         sens["symdiff_alt_both"]))
    if res.get("stopped") and "GATE 1" in res["stopped"]:
        w("")
        w("**%s.** Per the PREREG and the task rules nothing further is computed." % res["stopped"])
        open(path, "w").write("\n".join(L) + "\n")
        return

    ls_ = res.get("lofo_selfcheck", {})
    w("")
    w("Vote arithmetic: brute-force recount of %d random non-labelable points (all three vote "
      "readings, k = 1, 2, 3) against the vectorised pass: mismatches %s -> %s."
      % (ls_.get("sampled", 0), json.dumps(ls_.get("mismatches")),
         "OK" if ls_.get("ok") else "FAILED"))
    if res.get("stopped"):
        w("")
        w("**%s.**" % res["stopped"])
    g2 = res["gate2"]
    w("")
    w("## GATE 2 -- accounting: %s" % ("PASS" if g2["pass"] else "FAIL"))
    w("")
    w("Every evaluated point is exactly one of {labelable in own frame, propagated-only, "
      "unlabelled}. Evaluated total %d (published n_eval %d, equal: %s)."
      % (g2["evaluated_total"], g2["evaluated_total_published"],
         g2["evaluated_total_equals_published"]))
    w("")
    w("| k | labelable own | propagated-only | unlabelled | sum | evaluated | closes |")
    w("|---|---|---|---|---|---|---|")
    for k in KS:
        r = g2["per_k"][str(k)]
        w("| >=%d | %d | %d | %d | %d | %d | %s |" % (k, r["labelable_own"], r["propagated_only"],
                                                  r["unlabelled"], r["total"], r["evaluated"],
                                                  r["sum_equals_evaluated"]))

    def cp(d):
        return "%s / %s" % (fmt(d["coverage"]), fmt(d["precision"]))

    ps = res["per_scan_filterE"]
    w("")
    w("## Coverage / precision (%, all evaluated points of seq 07, GT = SemanticKITTI common-9)")
    w("")
    w("| supervision | all: coverage / precision | in-frustum | out-of-frustum | supervised points |")
    w("|---|---|---|---|---|")
    w("| per-scan filter E | %s | %s | %s | %d |" % (cp(ps["all"]), cp(ps["in"]), cp(ps["out"]),
                                                   ps["all"]["supervised"]))
    for k in KS:
        p = res["propagation"][str(k)]
        w("| propagation k>=%d (own label, else LOFO cell majority) | %s | %s | %s | %d |"
          % (k, cp(p["all"]), cp(p["in"]), cp(p["out"]), p["all"]["supervised"]))
    w("")
    w("Propagated-only points (not labelable in own frame), leave-own-frame-out:")
    w("")
    w("| k | in-frustum: count / precision | out-of-frustum: count / precision | all: count / precision |")
    w("|---|---|---|---|")
    for k in KS:
        p = res["propagated_only_lofo"][str(k)]
        w("| >=%d | %d / %s | %d / %s | %d / %s |"
          % (k, p["in"]["supervised"], fmt(p["in"]["precision"]), p["out"]["supervised"],
             fmt(p["out"]["precision"]), p["all"]["supervised"], fmt(p["all"]["precision"])))

    dr = res["decision_rule"]
    w("")
    w("## Decision (PREREG, applied literally; primary operating point k >= 2, share >= 2/3)")
    w("")
    r2 = dr["per_k"]["2"]
    w("- (a) propagated coverage of all evaluated points %.3f %% vs 2 x per-scan %.3f %% = "
      "%.3f %% -> **%s** (ratio x%.3f)"
      % (r2["coverage_prop_all"], dr["perscan_coverage_all"], 2 * dr["perscan_coverage_all"],
         "holds" if r2["a_holds"] else "does not hold", r2["coverage_prop_over_perscan"]))
    w("- (b) propagated-only out-of-frustum precision %s %% (n = %d) vs per-scan filter-E "
      "in-frustum precision %.3f %% - 5 = %.3f %% -> **%s**"
      % (fmt(r2["propagated_only_out_precision"], 3), r2["propagated_only_out_count"],
         dr["perscan_precision_in"], r2["b_threshold"],
         "holds" if r2["b_holds"] else "does not hold"))
    w("")
    w("**DECISION: %s**" % res["decision"])
    w("")
    w("Smallest k at which (b) holds: %s." % dr["smallest_k_where_b_holds"])
    w("")
    w("| k | (a) prop coverage | x per-scan | (a) | (b) PO-out precision | threshold | (b) |")
    w("|---|---|---|---|---|---|---|")
    for k in KS:
        r = dr["per_k"][str(k)]
        w("| >=%d | %.3f | %.3f | %s | %s | %.3f | %s |"
          % (k, r["coverage_prop_all"], r["coverage_prop_over_perscan"], r["a_holds"],
             fmt(r["propagated_only_out_precision"], 3), r["b_threshold"], r["b_holds"]))
    w("")
    w("Reading of (a): %s. Propagated-only share alone: %s."
      % (dr["reading_of_a"], ", ".join("k>=%d %.3f %%" % (k, dr["per_k"][str(k)]
                                                         ["coverage_propagated_only_alone_all"])
                                      for k in KS)))

    w("")
    w("## Per class")
    w("")
    w("Precision by label class (%; n = points labelled as the class) and coverage by GT class.")
    w("")
    w("| class | per-scan E in: prec (n) | per-scan E: GT coverage all | prop k>=2 all: prec (n) "
      "| prop k>=2: GT coverage all | PO k>=2 out: prec (n) | PO k>=2 out: GT coverage of non-labelable |")
    w("|---|---|---|---|---|---|---|")
    pk = res["propagation"]["2"]["all"]["per_class"]
    pok = res["propagated_only_lofo"]["2"]["out"]["per_class"]
    for cname in C9:
        e = ps["in"]["per_class"][cname]; ea = ps["all"]["per_class"][cname]
        w("| %s | %s (%d) | %s | %s (%d) | %s | %s (%d) | %s |"
          % (cname, fmt(e["precision"]), e["labelled_as"], fmt(ea["coverage"]),
             fmt(pk[cname]["precision"]), pk[cname]["labelled_as"], fmt(pk[cname]["coverage"]),
             fmt(pok[cname]["precision"]), pok[cname]["labelled_as"], fmt(pok[cname]["coverage"])))

    w("")
    w("## Diagnostics (not part of the decision)")
    w("")
    w("Propagated-only out-of-frustum precision by LiDAR range (k>=2):")
    w("")
    w("| range m | non-labelable evaluated | propagated | coverage | precision |")
    w("|---|---|---|---|---|")
    for r in res["propagated_only_lofo_by_range"]["2"]["out"]:
        w("| %s | %d | %d | %s | %s |" % (r["range"], r["evaluated_not_labelable"],
                                         r["propagated"], fmt(r["coverage"]), fmt(r["precision"])))
    w("")
    w("Per-scan filter-E in-frustum precision by range:")
    w("")
    w("| range m | evaluated in-frustum | supervised | coverage | precision |")
    w("|---|---|---|---|---|")
    for r in res["per_scan_filterE_by_range_in_frustum"]:
        w("| %s | %d | %d | %s | %s |" % (r["range"], r["evaluated"], r["supervised"],
                                         fmt(r["coverage"]), fmt(r["precision"])))
    dg = res["diagnostics_not_decision"]
    w("")
    w("Propagated-only precision under two other vote readings (count / precision):")
    w("")
    w("| k | LOFO (the instrument) out | incl. own-frame votes out | LOFO + GT-excluded voters out "
      "| LOFO + GT-excluded voters: prop coverage all |")
    w("|---|---|---|---|---|")
    for k in KS:
        p = res["propagated_only_lofo"][str(k)]["out"]
        q = dg["propagated_only_including_own_frame_votes"][str(k)]["out"]
        x = dg["propagated_only_lofo_votes_incl_gt_excluded_points"][str(k)]["out"]
        y = dg["propagation_lofo_votes_incl_gt_excluded_points"][str(k)]["all"]
        w("| >=%d | %d / %s | %d / %s | %d / %s | %s |"
          % (k, p["supervised"], fmt(p["precision"]), q["supervised"], fmt(q["precision"]),
             x["supervised"], fmt(x["precision"]), fmt(y["coverage"], 3)))
    rr = res["lofo_vote_reach"]
    w("")
    w("LOFO vote reach of non-labelable points (cells with >= 1 / 2 / 3 votes from other frames, "
      "before the 2/3 share test): in-frustum %s / %s / %s %%, out-of-frustum %s / %s / %s %%."
      % (fmt(rr["in"]["ge1"]), fmt(rr["in"]["ge2"]), fmt(rr["in"]["ge3"]),
         fmt(rr["out"]["ge1"]), fmt(rr["out"]["ge2"]), fmt(rr["out"]["ge3"])))
    w("")
    w("Points: %s. Cells: %s. Votes: %s."
      % (json.dumps(res["points"]), json.dumps(res["cells"]), json.dumps(res["votes"])))
    w("")
    w("## Caveats (computed)")
    w("")
    for cv in res.get("caveats_computed", []):
        w("- " + cv)
    open(path, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
