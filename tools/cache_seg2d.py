#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cache_seg2d.py -- one GPU pass over a sequence, caching the 2D arm's IMAGE-SPACE
label map per frame.

The map is cached at the NETWORK'S OWN resolution (H*s, W*s), not resampled onto the
native 1226x370 grid.  score_2d_vs_3d.Arm2D then samples it at each point's exact
sub-pixel projection, which is identical to seg2d_infer.sample_points and strictly
more information than rounding onto the native grid first.  [PRO-2D]

Nothing about projection, occlusion or the sky rule happens here: those live in the
scorer where they are auditable and can be varied without touching the GPU.

    python tools/cache_seg2d.py --model eomt --scale half_focal --tta none \\
        --seq 07 --out /data/livo_sem/out/vs2d/seg2d_eomt
"""
import os, sys, json, time, argparse
os.environ.setdefault("HF_HOME", "/data/hf_cache")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
sys.path.insert(0, "/data/livo_sem/src")
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="eomt")
    ap.add_argument("--scale", default="half_focal")
    ap.add_argument("--tta", default="none")
    ap.add_argument("--interp", default="bicubic")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seq", default="07")
    ap.add_argument("--frames", default="all")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import score_2d_vs_3d as SC
    from seg2d_infer import Seg2DSegmenter, SCALE_PRESETS, MODELS
    SC.set_sequence(a.seq)
    frames = SC.frame_list(a.frames)
    os.makedirs(a.out, exist_ok=True)

    sc = a.scale if a.scale in SCALE_PRESETS else float(a.scale)
    seg = Seg2DSegmenter(a.model, device=a.device, scale=sc, tta=a.tta,
                         interp=a.interp, verbose=True)
    # warm up so the first frames do not pay cuDNN/attention autotune
    png0 = "%s/%010d.png" % (SC.IMAGES, frames[0])
    for _ in range(3):
        seg._hires(png0)

    t0 = time.time(); shp = None
    for i, f in enumerate(frames):
        png = "%s/%010d.png" % (SC.IMAGES, f)
        lab, conf, sx, sy, (H, W) = seg._hires(png)
        assert (H, W) == (SC.IMG_H, SC.IMG_W), ("native image size", (H, W))
        shp = lab.shape
        np.savez_compressed("%s/f%06d.npz" % (a.out, f), seg=lab.astype(np.uint8))
        if (i + 1) % 100 == 0:
            print("  %d/%d  %.1f s" % (i + 1, len(frames), time.time() - t0), flush=True)
    meta = dict(model=a.model, repo=MODELS.get(a.model, {}).get("repo", a.model),
                published_cityscapes_val_miou_ss=MODELS.get(a.model, {}).get("cityscapes_miou_ss"),
                scale_key=a.scale, scale_val=float(seg.scale), tta=a.tta,
                interp=a.interp, dtype=str(seg.dtype), device=a.device,
                seq=a.seq, n_frames=len(frames),
                native_hw=[SC.IMG_H, SC.IMG_W], cached_hw=[int(shp[0]), int(shp[1])],
                note="image-space Cityscapes trainIds at the network's own resolution; "
                     "sampled sub-pixel by score_2d_vs_3d.Arm2D")
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=2)
    print("DONE %d frames in %.1f s -> %s  (cached %s)"
          % (len(frames), time.time() - t0, a.out, shp))


if __name__ == "__main__":
    main()
