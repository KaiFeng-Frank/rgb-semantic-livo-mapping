#!/usr/bin/env python3
"""
sweep_seg2d_scale.py -- MEASURE the 2D arm's best configuration.  Do not guess it.

THIS IS THE FAIRNESS RULE MADE EXECUTABLE.  seg2d_infer.py ships two Cityscapes models
and a reasoned default scale for each, all chosen with NO GPU available.  Run this
first, use the argmax, and quote that.  A 2D number reported at a configuration that
was not the best available configuration is not a number worth having: it will be
quoted, then demolished.

Scores the 2D arm exactly the way eval_report.py scores the 3D arm -- sem_core's
13-slot coarse space, the classes present in GT, and the 9-class subset
eval_report.NINE -- with ONE difference that matters more than anything else here:

    only the ~16 % of points that fall inside the camera frustum can be scored at all.

So this tool also prints the GT class histogram of that frustum subset.  Read it before
comparing anything, and RE-SCORE THE 3D ARM ON THE SAME SUBSET.  The published 3D
numbers (87.3 % acc, 53.2 / 61.0 mIoU) are over ALL points of the scan; setting them
against a 2D number computed on the 16 % the camera can see is not a comparison.

    python tools/sweep_seg2d_scale.py --device cuda --nframes 20 --stride 55
    python tools/sweep_seg2d_scale.py --device cuda --models eomt --scales shortside \\
           --tta none flip ms_flip --json-out out/seg2d_sweep.json

DO NOT RUN THIS WHILE ANOTHER AGENT HOLDS THE GPU.
"""
import os, sys, glob, json, time, argparse
os.environ.setdefault("HF_HOME", "/data/hf_cache")
sys.path.insert(0, "/data/livo_sem/src")
import numpy as np

import sem_core as S
from kitti_calib import KittiCalib
from seg2d_infer import Seg2DSegmenter, SCALE_PRESETS, MODELS, DEFAULT_SCALE, COARSE_VOID

D = "/data/livo_sem/data"
DRIVE = D + "/raw/2011_09_30/2011_09_30_drive_0027_sync"
NINE = ["car", "truck", "other_vehicle", "person", "road", "sidewalk",
        "terrain", "vegetation", "manmade"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--nframes", type=int, default=20)
    ap.add_argument("--stride", type=int, default=55)
    ap.add_argument("--models", nargs="*", default=["eomt", "mask2former"])
    ap.add_argument("--scales", nargs="*", default=["d2_maxsize", "shortside", "focal"])
    ap.add_argument("--tta", nargs="*", default=["none", "flip"])
    ap.add_argument("--interp", default="bicubic")
    ap.add_argument("--rider", default="bicycle",
                    choices=["bicycle", "person", "motorcycle", "void"])
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args()

    import torch
    cuda = a.device.startswith("cuda")
    sc = sorted(glob.glob(DRIVE + "/velodyne_points/data/*.bin"))
    im = sorted(glob.glob(DRIVE + "/image_02/data/*.png"))
    lb = sorted(glob.glob(D + "/odometry/dataset/sequences/07/labels/*.label"))
    idx = list(range(0, a.stride * a.nframes, a.stride))
    calib = KittiCalib(os.path.dirname(os.path.normpath(DRIVE)))
    lut = S.sk_lut()
    COARSE, IGNORE, K = S.COARSE, S.IGNORE, len(S.COARSE)

    frames, ghist, n_all = [], np.zeros(K, np.int64), 0
    for i in idx:
        pts = np.fromfile(sc[i], np.float32).reshape(-1, 4)
        g = lut[np.fromfile(lb[i], np.uint32) & 0xFFFF]
        assert len(pts) == len(g), (len(pts), len(g), i)
        uv, _d, m = calib.project_velo_to_cam2(pts[:, :3], img_shape=(370, 1226),
                                               return_mask=True)
        frames.append((im[i], uv, g[m])); n_all += len(pts)
        ghist += np.bincount(g[m][g[m] != IGNORE], minlength=K)
    n_fr = sum(len(f[2]) for f in frames)
    print("frames=%d   points in frustum = %d / %d scanned  (%.1f %%)"
          % (len(frames), n_fr, n_all, 100.0 * n_fr / n_all))
    print("\nGT INSIDE THE FRUSTUM -- all the 2D arm can be asked about:")
    for k in np.argsort(-ghist):
        if ghist[k]:
            print("   %-14s %9d  (%5.2f %%)" % (COARSE[k], ghist[k], 100.0 * ghist[k] / ghist.sum()))
    print("   -> %d coarse classes present: %s"
          % (int((ghist > 0).sum()), [COARSE[k] for k in range(K) if ghist[k]]))
    print("\n*** RE-SCORE THE 3D ARM ON THESE SAME POINTS BEFORE COMPARING. ***\n")

    rows = []
    for mk in a.models:
        pub = MODELS.get(mk, {}).get("cityscapes_miou_ss", "?")
        print("--- %s  (published Cityscapes val %s mIoU s.s., default scale %s) ---"
              % (mk, pub, DEFAULT_SCALE.get(mk)))
        for tta in a.tta:
            for skey in a.scales:
                if cuda:
                    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
                try:
                    seg = Seg2DSegmenter(mk, device=a.device, scale=skey, tta=tta,
                                         interp=a.interp, verbose=False)
                except Exception as e:
                    print("  %-11s tta=%-7s CONSTRUCT FAILED: %s" % (skey, tta, str(e)[:100]))
                    continue
                inter = np.zeros(K, np.int64); pc = np.zeros(K, np.int64); gc = np.zeros(K, np.int64)
                ok = tot = void = 0; ts = []
                try:
                    for png, uv, gm in frames:
                        t0 = time.perf_counter()
                        l2, _c2 = seg.sample_points(png, uv)
                        ts.append((time.perf_counter() - t0) * 1e3)
                        pr = seg.to_coarse(l2, rider=a.rider)
                        void += int((pr == COARSE_VOID).sum())
                        m = gm != IGNORE
                        ok += int((pr[m] == gm[m]).sum()); tot += int(m.sum())
                        for k in range(K):
                            pk = (pr == k) & m; gk = (gm == k) & m
                            inter[k] += int((pk & gk).sum()); pc[k] += int(pk.sum()); gc[k] += int(gk.sum())
                except RuntimeError as e:
                    print("  %-11s tta=%-7s RUNTIME FAILED: %s" % (skey, tta, str(e)[:100]))
                    del seg
                    if cuda: torch.cuda.empty_cache()
                    continue
                ious = {COARSE[k]: 100.0 * inter[k] / max(pc[k] + gc[k] - inter[k], 1)
                        for k in range(K) if gc[k] > 0}
                miou = float(np.mean(list(ious.values())))
                nine = [ious[n] for n in NINE if n in ious]
                miou9 = float(np.mean(nine))
                vram = (torch.cuda.max_memory_allocated() / 2 ** 30) if cuda else 0.0
                r = dict(model=mk, published_miou=pub, scale=skey,
                         scale_val=SCALE_PRESETS.get(skey, skey), tta=tta,
                         interp=a.interp, rider=a.rider, point_acc=100.0 * ok / tot,
                         miou=miou, n_present=len(ious), miou9=miou9, n_of_nine=len(nine),
                         void_frac=100.0 * void / tot, ms=float(np.mean(ts)),
                         vram_gb=vram, per_class={k: round(v, 2) for k, v in ious.items()})
                rows.append(r)
                print("  %-11s tta=%-7s acc %5.2f %%  mIoU %5.2f %% (%2d present) / "
                      "%5.2f %% (%d of NINE)  sky-void %.2f %%  %6.0f ms/f  %4.1f GB"
                      % (skey, tta, r["point_acc"], miou, len(ious), miou9, len(nine),
                         r["void_frac"], r["ms"], vram))
                del seg
                if cuda: torch.cuda.empty_cache()

    if rows:
        best = max(rows, key=lambda r: r["miou"])
        print("\n================ BEST 2D CONFIGURATION ================")
        print("  model=%s  scale=%s (%.4f)  tta=%s  rider=%s"
              % (best["model"], best["scale"], best["scale_val"], best["tta"], best["rider"]))
        print("  point accuracy %.2f %%   coarse mIoU %.2f %% (%d present) / %.2f %% (%d of NINE)"
              % (best["point_acc"], best["miou"], best["n_present"], best["miou9"], best["n_of_nine"]))
        print("  %.0f ms/frame, %.1f GB peak VRAM" % (best["ms"], best["vram_gb"]))
        for k in sorted(best["per_class"], key=lambda x: -best["per_class"][x]):
            print("      %-14s %6.2f %%" % (k, best["per_class"][k]))
        print("\n  Quote THIS configuration.  If it is not seg2d_infer's default, change")
        print("  the default and say so.  Report the spread across the whole sweep too --")
        print("  a 2D number with no configuration spread beside it is not auditable.")
    if a.json_out:
        os.makedirs(os.path.dirname(a.json_out) or ".", exist_ok=True)
        json.dump(dict(frames=len(frames), n_frustum=int(n_fr), n_all=int(n_all),
                       gt_frustum_hist={COARSE[k]: int(ghist[k]) for k in range(K) if ghist[k]},
                       rows=rows), open(a.json_out, "w"), indent=2)
        print("\nwrote %s" % a.json_out)


if __name__ == "__main__":
    main()
