#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sweep9_seq04.py -- STEP 1.  Choose the 2D arm's configuration BY MEASUREMENT, on
seq 04, then freeze it.

WHY seq 04 AND NOT seq 07
  Cityscapes fx ~ 2262 px, KITTI cam2 fx = 707.0912 px.  The same object subtends
  3.20x fewer pixels than either network ever saw in training, which is exactly the
  regime a segmentation net is worst in.  Feeding the native 1226x370 frame is a
  HANDICAP, not a neutral choice, so the scale has to be chosen.  Choosing it on
  seq 07 and then reporting seq 07 is tuning on the test set.  seq 04
  (2011_09_30_drive_0016_sync, 271 GT frames) is a DISJOINT drive on the SAME rig,
  the SAME calibration day and the SAME 1226x370 geometry.

WHAT IS SCORED
  Exactly what score_2d_vs_3d.py scores, in the SAME common-9 space
  (label_spaces.COARSE), with the SAME rules -- frustum + GT-not-EXCLUDED, z-buffer
  occlusion, sky -> UNMAPPED (wrong, never dropped).  The argmax of this sweep is
  therefore the argmax of the metric that gets reported, not of a proxy.

  python tools/sweep9_seq04.py --models eomt mask2former --tta none \\
      --scales native d2_maxsize half_focal shortside focal focal_125 \\
      --nframes 30 --stride 9 --json-out out/sweep9_seq04_s1.json
"""
import os, sys, glob, json, time, argparse
os.environ.setdefault("HF_HOME", "/data/hf_cache")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
sys.path.insert(0, "/data/livo_sem/src")
import numpy as np

import label_spaces as LS
import score_2d_vs_3d as SC

ROOT = "/data/livo_sem"
COARSE = list(LS.COARSE)
K = len(COARSE)


def build_frames(seq, nframes, stride, f0=0):
    SC.set_sequence(seq)
    proj = SC.Projector()
    lut = LS.sk_lut()
    idx = [f for f in range(f0, f0 + stride * nframes, stride)
           if os.path.exists("%s/%010d.bin" % (SC.SCANS, f))
           and os.path.exists("%s/%06d.label" % (SC.LABELS, f))]
    out = []
    ghist = np.zeros(K, np.int64)
    n_all = n_fr = n_ex = 0
    for f in idx:
        pts = SC.read_scan(f)
        gt = SC.read_gt_coarse(f, lut)
        assert len(gt) == len(pts), (f, len(gt), len(pts))
        u, v, z, inm = proj.project(pts[:, :3].astype(np.float64))
        vis = SC.visible_mask(np.rint(u[inm]).astype(np.int64),
                              np.rint(v[inm]).astype(np.int64), z[inm])
        gi = gt[inm]
        keep = gi != LS.EXCLUDED
        n_all += len(pts); n_fr += int(inm.sum()); n_ex += int((~keep).sum())
        ghist += np.bincount(gi[keep].astype(np.int64), minlength=K)
        out.append(dict(f=f,
                        png="%s/%010d.png" % (SC.IMAGES, f),
                        uv=np.stack([u[inm], v[inm]], 1),
                        gt=gi, keep=keep, vis=vis))
    return proj, out, ghist, n_all, n_fr, n_ex


def score_cfg(frames, seg, sky_policy="wrong"):
    """-> dict.  One pass of one configuration over the frame set."""
    cs2c = np.full(20, SC.ABSTAIN, np.int16)
    cs2c[:19] = np.asarray(LS.CS_TO_COARSE, np.int16)
    if sky_policy == "abstain":
        cs2c[LS.CS_SKY_ID] = SC.ABSTAIN

    C = np.zeros((K, K + 1), np.int64); A = np.zeros(K, np.int64)
    Cn = np.zeros((K, K + 1), np.int64); An = np.zeros(K, np.int64)   # no occlusion
    sky = n_pts = 0
    ts = []
    for fr in frames:
        t0 = time.perf_counter()
        lab2d, _conf = seg.sample_points(fr["png"], fr["uv"])
        ts.append((time.perf_counter() - t0) * 1e3)
        city = np.where(lab2d < 20, lab2d, 19).astype(np.int64)
        pred = cs2c[city]
        k = fr["keep"]
        g = fr["gt"][k].astype(np.int64)
        sky += int((city[k] == LS.CS_SKY_ID).sum())
        n_pts += int(k.sum())
        for Cx, Ax, pr in ((C, A, np.where(fr["vis"], pred, SC.ABSTAIN)),
                           (Cn, An, pred)):
            pp = pr[k]
            ab = pp == SC.ABSTAIN
            if ab.any():
                Ax += np.bincount(g[ab], minlength=K)
            gg = g[~ab]; qq = pp[~ab].astype(np.int64)
            if gg.size:
                qq = np.where(qq == LS.UNMAPPED, K, qq)
                Cx += np.bincount(gg * (K + 1) + qq,
                                  minlength=K * (K + 1)).reshape(K, K + 1)

    def met(C, A):
        row = C.sum(1); col = C[:, :K].sum(0); diag = np.diag(C[:, :K])
        n_ans = int(C.sum()); n_ab = int(A.sum()); n = n_ans + n_ab
        pres = (row + A) > 0
        iou_e, iou_w = {}, {}
        for c in range(K):
            if not pres[c]:
                continue
            u0 = row[c] + col[c] - diag[c]
            iou_e[COARSE[c]] = 100.0 * diag[c] / u0 if u0 else 0.0
            u1 = u0 + A[c]
            iou_w[COARSE[c]] = 100.0 * diag[c] / u1 if u1 else 0.0
        return dict(n_eval=n, n_answered=n_ans, n_abstained=n_ab,
                    n_unmapped=int(C[:, K].sum()),
                    coverage=100.0 * n_ans / n if n else float("nan"),
                    acc_ex=100.0 * diag.sum() / n_ans if n_ans else float("nan"),
                    acc_wr=100.0 * diag.sum() / n if n else float("nan"),
                    miou_ex=float(np.mean(list(iou_e.values()))),
                    miou_wr=float(np.mean(list(iou_w.values()))),
                    n_classes=len(iou_e),
                    per_class={a: round(b, 2) for a, b in iou_e.items()})
    r = met(C, A)
    r["no_occlusion"] = met(Cn, An)
    r["sky_hit_pct"] = 100.0 * sky / max(n_pts, 1)
    r["ms_per_frame"] = float(np.mean(ts))
    r["ms_p95"] = float(np.percentile(ts, 95))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="04")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--nframes", type=int, default=30)
    ap.add_argument("--stride", type=int, default=9)
    ap.add_argument("--models", nargs="*", default=["eomt", "mask2former"])
    ap.add_argument("--scales", nargs="*",
                    default=["native", "d2_maxsize", "half_focal", "shortside",
                             "focal", "focal_125"])
    ap.add_argument("--tta", nargs="*", default=["none"])
    ap.add_argument("--interp", default="bicubic")
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args()

    import torch
    from seg2d_infer import Seg2DSegmenter, SCALE_PRESETS, MODELS

    proj, frames, ghist, n_all, n_fr, n_ex = build_frames(a.seq, a.nframes, a.stride)
    n_sc = int(ghist.sum())
    print("SEQ %s  %d frames   points %d   in-frustum %d (%.2f %%)   "
          "EXCLUDED in-frustum %d (%.2f %%)   scored %d"
          % (a.seq, len(frames), n_all, n_fr, 100.0 * n_fr / n_all, n_ex,
             100.0 * n_ex / max(n_fr, 1), n_sc), flush=True)
    print("GT inside the frustum (the only thing the 2D arm can be asked about):")
    for c in np.argsort(-ghist):
        print("   %-14s %9d  (%6.3f %%)" % (COARSE[c], ghist[c], 100.0 * ghist[c] / n_sc))

    rows = []
    for mk in a.models:
        pub = MODELS.get(mk, {}).get("cityscapes_miou_ss", "?")
        print("\n=== %s  (published Cityscapes val %s mIoU s.s.) ===" % (mk, pub), flush=True)
        print("  %-11s %-8s %8s %8s %8s %8s %7s %8s %6s"
              % ("scale", "tta", "acc_ex", "acc_wr", "mIoU_ex", "mIoU_wr", "cov%", "ms/f", "GB"))
        for tta in a.tta:
            for sk in a.scales:
                torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
                try:
                    seg = Seg2DSegmenter(mk, device=a.device, scale=sk, tta=tta,
                                         interp=a.interp, verbose=False)
                except Exception as e:
                    print("  %-11s %-8s CONSTRUCT FAILED %s" % (sk, tta, str(e)[:80])); continue
                try:
                    r = score_cfg(frames, seg)
                except RuntimeError as e:
                    print("  %-11s %-8s RUNTIME FAILED %s" % (sk, tta, str(e)[:80]))
                    del seg; torch.cuda.empty_cache(); continue
                r.update(model=mk, published_miou=pub, scale=sk,
                         scale_val=float(SCALE_PRESETS[sk]), tta=tta, interp=a.interp,
                         vram_gb=torch.cuda.max_memory_allocated() / 2 ** 30,
                         seq=a.seq, n_frames=len(frames))
                rows.append(r)
                print("  %-11s %-8s %8.2f %8.2f %8.2f %8.2f %7.2f %8.0f %6.1f"
                      % (sk, tta, r["acc_ex"], r["acc_wr"], r["miou_ex"], r["miou_wr"],
                         r["coverage"], r["ms_per_frame"], r["vram_gb"]), flush=True)
                del seg; torch.cuda.empty_cache()

    if rows:
        b = max(rows, key=lambda r: r["miou_ex"])
        ba = max(rows, key=lambda r: r["acc_ex"])
        print("\n=== ARGMAX in-frustum mIoU_ex : %s scale=%s (%.4f) tta=%s -> %.2f %% "
              "(acc %.2f %%)" % (b["model"], b["scale"], b["scale_val"], b["tta"],
                                 b["miou_ex"], b["acc_ex"]))
        print("=== ARGMAX in-frustum acc_ex  : %s scale=%s tta=%s -> %.2f %% "
              "(mIoU %.2f %%)" % (ba["model"], ba["scale"], ba["tta"], ba["acc_ex"],
                                  ba["miou_ex"]))
        for m in a.models:
            sub = [r for r in rows if r["model"] == m]
            if sub:
                bb = max(sub, key=lambda r: r["miou_ex"])
                print("    best %-12s scale=%-11s tta=%-7s mIoU %.2f  acc %.2f"
                      % (m, bb["scale"], bb["tta"], bb["miou_ex"], bb["acc_ex"]))
    if a.json_out:
        os.makedirs(os.path.dirname(a.json_out) or ".", exist_ok=True)
        json.dump(dict(seq=a.seq, n_frames=len(frames),
                       frames=[int(f["f"]) for f in frames],
                       n_all=int(n_all), n_frustum=int(n_fr),
                       n_excluded_in_frustum=int(n_ex), n_scored=int(n_sc),
                       gt_hist={COARSE[c]: int(ghist[c]) for c in range(K)},
                       rows=rows), open(a.json_out, "w"), indent=2, default=float)
        print("wrote %s" % a.json_out)


if __name__ == "__main__":
    main()
