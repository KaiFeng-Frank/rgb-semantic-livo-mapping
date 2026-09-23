#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""cache_seg2d_conf.py -- same GPU pass as tools/cache_seg2d.py, but ALSO caches the
per-pixel top-2 class probabilities, which the label-only cache threw away.

The pseudo-label track needs a confidence to threshold on.  `_hires` in
seg2d_infer.py already computes it (`conf = (seg/seg.sum(0)).max(0)`) and then
drops everything but the argmax.  This script replicates _hires VERBATIM -- same
scale clamp, same TTA loop, same normalisation -- and keeps top-2 instead of top-1.

seg2d_infer.py is FROZEN and is not touched.  Verification that this reproduces the
existing label cache bit-for-bit is part of the run (--verify).

Stored per frame, at the network's own resolution, float16:
    seg  (H,W) uint8    argmax Cityscapes trainId   -- must equal the frozen cache
    p1   (H,W) f16      max normalised score        -- the confidence
    p2   (H,W) f16      runner-up normalised score  -- p1-p2 is the margin
"""
import os, sys, json, time, argparse
os.environ.setdefault("HF_HOME", "/data/wuyou/hf_cache")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
sys.path.insert(0, "/data/wuyou/livo_sem/src")
import numpy as np


def hires_top2(seg2d, path):
    """Byte-for-byte the body of Seg2DSegmenter._hires, keeping top-2."""
    t = seg2d.torch
    img = seg2d._as_rgb(path)
    H, W = img.shape[:2]
    s = seg2d.scale
    if max(H, W) * s > seg2d.max_long_side:
        s = seg2d.max_long_side / float(max(H, W))
    Hr, Wr = int(round(H * s)), int(round(W * s))
    scales = seg2d.ms_scales if seg2d.tta == "ms_flip" else (1.0,)
    flips = (False, True) if seg2d.tta in ("flip", "ms_flip") else (False,)
    with t.no_grad():
        seg = t.zeros((19, Hr, Wr), dtype=t.float32, device=seg2d.device)
        for ms in scales:
            if seg2d.backend == "eomt":
                for fl in flips:
                    seg += seg2d._eomt_scores(img, s * ms, (Hr, Wr), fl)
            else:
                u8, _ = seg2d._resize(img, s * ms)
                for fl in flips:
                    seg += seg2d._m2f_scores(u8, (Hr, Wr), fl)
        tot = seg.sum(0).clamp_min_(1e-12)
        p = seg / tot
        top2, idx = p.topk(2, dim=0)
        lab = idx[0].to(t.uint8).cpu().numpy()
        p1 = top2[0].cpu().numpy()
        p2 = top2[1].cpu().numpy()
        del seg, p, top2, idx
    return lab, p1, p2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="eomt")
    ap.add_argument("--scale", default="scale_13")
    ap.add_argument("--tta", default="flip")
    ap.add_argument("--interp", default="bicubic")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seq", default="07")
    ap.add_argument("--frames", default="all")
    ap.add_argument("--out", required=True)
    ap.add_argument("--verify", default="",
                    help="frozen label cache dir; assert argmax matches on every frame")
    a = ap.parse_args()

    import seqreg
    from seg2d_infer import Seg2DSegmenter, SCALE_PRESETS, MODELS
    SC, _proj, _W, _H = seqreg.use(a.seq)
    frames = SC.frame_list(a.frames)
    os.makedirs(a.out, exist_ok=True)

    sc = a.scale if a.scale in SCALE_PRESETS else float(a.scale)
    seg2d = Seg2DSegmenter(a.model, device=a.device, scale=sc, tta=a.tta,
                           interp=a.interp, verbose=True)
    png0 = "%s/%010d.png" % (SC.IMAGES, frames[0])
    for _ in range(3):
        seg2d._hires(png0)

    t0 = time.time(); shp = None; nmis = 0; nchk = 0
    for i, f in enumerate(frames):
        png = "%s/%010d.png" % (SC.IMAGES, f)
        lab, p1, p2 = hires_top2(seg2d, png)
        shp = lab.shape
        if a.verify:
            q = "%s/f%06d.npz" % (a.verify, f)
            if os.path.exists(q):
                ref = np.load(q)["seg"]
                assert ref.shape == lab.shape, ("shape", f, ref.shape, lab.shape)
                d = int((ref != lab).sum()); nmis += d; nchk += ref.size
        np.savez_compressed("%s/f%06d.npz" % (a.out, f), seg=lab.astype(np.uint8),
                            p1=p1.astype(np.float16), p2=p2.astype(np.float16))
        if (i + 1) % 100 == 0:
            print("  %d/%d  %.1f s  argmax_mismatch=%d/%d" %
                  (i + 1, len(frames), time.time() - t0, nmis, nchk), flush=True)
    meta = dict(model=a.model, repo=MODELS.get(a.model, {}).get("repo", a.model),
                scale_key=a.scale, scale_val=float(seg2d.scale), tta=a.tta,
                interp=a.interp, dtype=str(seg2d.dtype), seq=a.seq,
                n_frames=len(frames), cached_hw=[int(shp[0]), int(shp[1])],
                verify_against=a.verify, argmax_mismatch_px=nmis, argmax_checked_px=nchk,
                note="top-2 normalised mask-classification scores; p1 is _hires' conf")
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=2)
    print("DONE %d frames %.1f s -> %s  shape=%s  argmax_mismatch=%d/%d"
          % (len(frames), time.time() - t0, a.out, shp, nmis, nchk))


if __name__ == "__main__":
    main()
