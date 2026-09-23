#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_pseudo.py -- run BOTH frozen 2D teachers over a sequence and project their output
onto the LiDAR points, storing the per-point quantities filter E is a function of.

It deliberately stores the RAW per-point signal, not the filtered labels:

    t9     int8   EoMT-L common-9 label at this point, -1 = not a candidate
    m9     int8   Mask2Former-L common-9 label,        -1 = abstain / not a candidate
    conf   f16    EoMT-L normalised top-1 score (the `conf` of the recon)
    flags  uint8  bit0 in_frustum  bit1 z-buffer visible  bit2 depth-edge  bit3 range<50

so that the filter's stratum boundaries -- which the frozen design requires to be
RE-DERIVED ON seq 08 -- can be applied afterwards by tools/apply_filter.py without
touching the GPU again.  The 88 GB of images are read once, ever.

EVERY geometric rule is taken from the audited score_2d_vs_3d code path verbatim:
projection (Projector.project / P3), z-buffer visibility (visible_mask / P4b,
OCC_WIN=2), depth-edge (depth_edge_mask / P7-B2), the sub-pixel index arithmetic of
Arm2D._sample, and the Cityscapes->common-9 LUT with the default rider and
sky_policy="wrong".  The 2D models are built by the FROZEN seg2d_infer.Seg2DSegmenter
at the frozen configuration (EoMT-L scale 1.30 + hflip TTA; Mask2Former-L native, no
TTA) and the EoMT top-2 read-out is tools/cache_seg2d_conf.hires_top2, which was
verified argmax-identical to the frozen label cache.
"""
import os, sys, json, time, argparse
os.environ.setdefault("HF_HOME", "/data/wuyou/hf_cache")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
sys.path.insert(0, "/data/wuyou/livo_sem/src")
sys.path.insert(0, "/data/wuyou/livo_sem/tools")
import numpy as np

F_INFRUSTUM = 1
F_VISIBLE = 2
F_DEPTHEDGE = 4
F_RANGE_LT50 = 8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--frames", default="all")
    ap.add_argument("--out", required=True, help="root; writes <out>/<seq>/f%06d.npz")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--range-cut", type=float, default=50.0)
    a = ap.parse_args()

    import torch
    import seqreg
    from seg2d_infer import Seg2DSegmenter
    from cache_seg2d_conf import hires_top2

    SC, proj, W, H = seqreg.use(a.seq)
    frames = SC.frame_list(a.frames)
    outdir = os.path.join(a.out, a.seq)
    os.makedirs(outdir, exist_ok=True)
    city_lut = SC.city19_to_coarse(rider=SC.RIDER_DEFAULT, sky_policy="wrong")

    te = Seg2DSegmenter("eomt", device=a.device, scale="scale_13", tta="flip",
                        interp="bicubic", verbose=True)
    tm = Seg2DSegmenter("mask2former", device=a.device, scale="native", tta="none",
                        interp="bicubic", verbose=True)
    png0 = "%s/%010d.png" % (SC.IMAGES, frames[0])
    for _ in range(3):
        hires_top2(te, png0); tm._hires(png0)

    def sample_idx(u, v, inm, sh, sw):
        """Verbatim score_2d_vs_3d.Arm2D._sample index arithmetic."""
        sx, sy = sw / float(W), sh / float(H)
        ui = np.clip(np.rint((u[inm] + 0.5) * sx - 0.5), 0, sw - 1).astype(np.int64)
        vi = np.clip(np.rint((v[inm] + 0.5) * sy - 0.5), 0, sh - 1).astype(np.int64)
        return vi, ui

    t0 = time.time()
    tot = dict(n=0, inm=0, vis=0, cand=0, agree=0)
    hist = np.zeros(9, np.int64)
    done = 0
    for i, f in enumerate(frames):
        op = "%s/f%06d.npz" % (outdir, f)
        if os.path.exists(op):
            done += 1
            continue
        png = "%s/%010d.png" % (SC.IMAGES, f)
        if not os.path.isfile(png):
            print("  MISSING IMAGE frame %d -> %s" % (f, png), flush=True)
            continue
        pts = SC.read_scan(f)
        n = pts.shape[0]
        xyz = pts[:, :3].astype(np.float64)
        u, v, z, inm = proj.project(xyz)
        rng = np.linalg.norm(xyz, axis=1)

        flags = np.zeros(n, np.uint8)
        flags[inm] |= F_INFRUSTUM
        flags[rng < a.range_cut] |= F_RANGE_LT50

        t9 = np.full(n, -1, np.int8)
        m9 = np.full(n, -1, np.int8)
        conf = np.zeros(n, np.float16)

        if inm.any():
            ui_r = np.rint(u[inm]).astype(np.int64)
            vi_r = np.rint(v[inm]).astype(np.int64)
            zi = z[inm]
            vis = SC.visible_mask(ui_r, vi_r, zi, w=W, h=H, half_win=SC.OCC_WIN)
            de = SC.depth_edge_mask(ui_r, vi_r, zi, w=W, h=H)
            fi = np.zeros(n, np.uint8)
            fi[inm] = (vis.astype(np.uint8) * F_VISIBLE
                       + de.astype(np.uint8) * F_DEPTHEDGE)
            flags |= fi

            lab_e, p1, _p2 = hires_top2(te, png)
            lab_m = tm._hires(png)
            if isinstance(lab_m, tuple):
                lab_m = lab_m[0]
            lab_m = np.asarray(lab_m)
            sh, sw = lab_e.shape
            vi_s, ui_s = sample_idx(u, v, inm, sh, sw)
            ce = lab_e[vi_s, ui_s].astype(np.int32)
            ce = np.where(ce < 20, ce, 19)
            c9e = city_lut[ce].astype(np.int8)
            sh2, sw2 = lab_m.shape
            vi2, ui2 = sample_idx(u, v, inm, sh2, sw2)
            cm = lab_m[vi2, ui2].astype(np.int32)
            cm = np.where(cm < 20, cm, 19)
            c9m = city_lut[cm].astype(np.int8)

            # a candidate = in frustum AND z-buffer visible AND the teacher named a
            # common-9 class (sky -> UNMAPPED -2, ignore-fill -> ABSTAIN -1: neither)
            keep = vis & (c9e >= 0)
            tmp = np.full(int(inm.sum()), -1, np.int8); tmp[keep] = c9e[keep]
            t9[inm] = tmp
            keepm = vis & (c9m >= 0)
            tmp2 = np.full(int(inm.sum()), -1, np.int8); tmp2[keepm] = c9m[keepm]
            m9[inm] = tmp2
            cf = np.zeros(int(inm.sum()), np.float16)
            cf[:] = p1[vi_s, ui_s].astype(np.float16)
            conf[inm] = cf

            tot["inm"] += int(inm.sum()); tot["vis"] += int(vis.sum())
            c = int((t9 >= 0).sum()); tot["cand"] += c
            tot["agree"] += int(((t9 >= 0) & (t9 == m9)).sum())
            if c:
                hist += np.bincount(t9[t9 >= 0].astype(np.int64), minlength=9)
        tot["n"] += n
        np.savez_compressed(op, t9=t9, m9=m9, conf=conf, flags=flags)
        done += 1
        if done % 200 == 0:
            el = time.time() - t0
            print("  %d/%d  %.1f s  %.3f s/f  eta %.1f min" %
                  (done, len(frames), el, el / max(done, 1),
                   (len(frames) - done) * el / max(done, 1) / 60), flush=True)

    meta = dict(seq=a.seq, n_frames=len(frames), img_wh=[W, H], range_cut=a.range_cut,
                teachers=dict(eomt=dict(scale="scale_13", scale_val=float(te.scale),
                                        tta="flip"),
                              m2f=dict(scale="native", scale_val=float(tm.scale),
                                       tta="none")),
                occ_win=int(SC.OCC_WIN), rider=SC.RIDER_DEFAULT, sky_policy="wrong",
                totals={k: int(x) for k, x in tot.items()},
                teacher_class_hist=hist.tolist(),
                class_names=list(SC.COARSE), seconds=time.time() - t0)
    json.dump(meta, open(os.path.join(outdir, "meta.json"), "w"), indent=2)
    print("DONE seq %s  %d frames  %.1f s  -> %s" %
          (a.seq, len(frames), time.time() - t0, outdir), flush=True)
    print("  points %d  in-frustum %d  visible %d  candidate %d  agree %d" %
          tuple(tot[k] for k in ("n", "inm", "vis", "cand", "agree")), flush=True)


if __name__ == "__main__":
    main()
