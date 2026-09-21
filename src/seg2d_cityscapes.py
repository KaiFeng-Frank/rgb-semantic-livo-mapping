#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
seg2d_cityscapes.py -- the 2D arm of the 2D-vs-3D experiment, in its STRONGEST
configuration, writing the on-disk contract that score_2d_vs_3d.py consumes.

    <out>/f%06d.npz   seg  uint8 (370, 1226)  Cityscapes trainId, 255 = ignore
                      prob f16 (19, 370, 1226)  OPTIONAL, --prob

WHY A CITYSCAPES MODEL
======================
Cityscapes IS this task's native domain: German urban street scenes, forward-facing
vehicle camera, daytime.  SemanticKITTI's 19 classes were deliberately modelled on
Cityscapes', so the label map is nearly 1:1.  The 3D arm carries nuScenes weights,
a different country, a different sensor and a different class list -- so the 2D arm
is handed the SMALLER domain gap.  That asymmetry is deliberate and must not be
"fixed": the whole point is to see whether the 3D arm wins anyway.

THE TWO WAYS TO SILENTLY CRIPPLE A 2D ARM, AND HOW THIS MODULE AVOIDS THEM
==========================================================================
(1) THE PROCESSOR'S DEFAULT RESIZE.  `Mask2FormerImageProcessor` ships
    size={"height": 384, "width": 384} and _max_size=2048.  Calling it with defaults
    squashes a 1226x370 image to 384x384, destroying both the aspect ratio and the
    resolution.  This module NEVER calls the processor for geometry; it does its own
    aspect-preserving resize, normalisation and size_divisor padding.

(2) SCALE MISMATCH.  A Cityscapes model is trained on 2048x1024 images from a camera
    with fx ~ 2262 px.  KITTI's rectified image_02 has fx = 707.0912 px.  The same
    object therefore subtends 2262/707.09 = 3.2x FEWER pixels in KITTI than in the
    training domain, which is exactly the regime a segmentation network is worst in.
    Feeding the image at its native size is not a neutral choice -- it is a
    handicap.  `--scale` upsamples before the forward and the labels are mapped back
    afterwards; 3.2 restores the training-domain angular resolution.

    The scale is the ONE thing this arm needs chosen.  Choosing it on seq 07 and
    then reporting seq 07 would be tuning on the test set.  So:

        TUNE on seq 04  (2011_09_30_drive_0016_sync, 271 frames, same rig, same
                         calibration day, VERIFIED index-aligned) -- then FREEZE
        REPORT on seq 07 (2011_09_30_drive_0027_sync, 1101 frames)

    `tune_scale.sh` in opt/ runs exactly that.

(3) TEST-TIME AUGMENTATION.  `--tta hflip` averages the logits of the image and its
    mirror.  It is standard, it costs one extra forward, and it only ever helps the
    2D arm.  It is ON by default for the accuracy run; report latency both ways.

MODELS (both already downloaded to /data/hf_cache)
========================================================
  facebook/mask2former-swin-large-cityscapes-semantic   Cityscapes val mIoU 83.3
  nvidia/segformer-b5-finetuned-cityscapes-1024-1024    Cityscapes val mIoU 82.4
The Mask2Former is the stronger of the two and is the default; the SegFormer is the
one to quote if a reviewer objects that the strongest model is not deployable.

USAGE
=====
    export HF_HOME=/data/hf_cache HF_ENDPOINT=https://hf-mirror.com
    python seg2d_cityscapes.py --seq 07 --scale 3.2 --tta hflip \
        --out /data/livo_sem/opt/out2d/seg2d_s32 \
        --timing-out /data/livo_sem/opt/out2d/timing_2d.json

*** NOT YET RUN ON GPU.  The pre/post-processing geometry is covered by
    `python seg2d_cityscapes.py selftest` (CPU, no network); the weights are
    verified to deserialize by `--check-only`.  The first GPU run should be a
    single frame with --debug-dump to eyeball the overlay before the full pass. ***
"""

import os
import sys
import json
import time
import argparse

import numpy as np

CITYSCAPES_19 = ["road", "sidewalk", "building", "wall", "fence", "pole",
                 "traffic light", "traffic sign", "vegetation", "terrain", "sky",
                 "person", "rider", "car", "truck", "bus", "train", "motorcycle",
                 "bicycle"]
IMG_W, IMG_H = 1226, 370

MODELS = {
    "mask2former": "facebook/mask2former-swin-large-cityscapes-semantic",
    "segformer": "nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
}
SEQ_DRIVE = {"07": "2011_09_30_drive_0027_sync", "04": "2011_09_30_drive_0016_sync"}
SEQ_N = {"07": 1101, "04": 271}
RAWROOT = "/data/livo_sem/data/raw/2011_09_30"

# fx of the KITTI rectified colour camera and of the Cityscapes rig.  The ratio is
# the scale at which a Cityscapes-trained network sees KITTI objects at the pixel
# size it was trained on.
KITTI_FX = 707.0912
CITYSCAPES_FX = 2262.52
NATIVE_SCALE = CITYSCAPES_FX / KITTI_FX        # 3.1998...


# --------------------------------------------------------------------------- #
#  geometry: aspect-preserving resize + size_divisor pad, and the exact inverse
# --------------------------------------------------------------------------- #
def plan(scale, divisor=32, w=IMG_W, h=IMG_H):
    """-> (rw, rh, pw, ph): resize target and padded size.  Padding is on the
    bottom/right only, so the resized image occupies [0:rh, 0:rw] of the padded
    tensor and the inverse is a plain crop."""
    rw = max(divisor, int(round(w * scale)))
    rh = max(divisor, int(round(h * scale)))
    pw = int(np.ceil(rw / divisor) * divisor)
    ph = int(np.ceil(rh / divisor) * divisor)
    return rw, rh, pw, ph


def selftest():
    """CPU, no network, no weights: the pre/post geometry must round-trip."""
    import torch
    import torch.nn.functional as F
    fails = []
    n = [0]

    def chk(name, cond, det=""):
        n[0] += 1
        print("  %-56s %s %s" % (name, "PASS" if cond else "FAIL", det))
        if not cond:
            fails.append(name)

    for sc in [1.0, 2.0, NATIVE_SCALE, 3.5]:
        rw, rh, pw, ph = plan(sc)
        chk("plan(%.2f): padded size divisible by 32" % sc,
            pw % 32 == 0 and ph % 32 == 0, "(%dx%d -> pad %dx%d)" % (rw, rh, pw, ph))
        chk("plan(%.2f): padding only grows the image" % sc, pw >= rw and ph >= rh)
        chk("plan(%.2f): aspect preserved to <1 %%" % sc,
            abs((rw / rh) / (IMG_W / IMG_H) - 1.0) < 0.01)

    # a synthetic "logit" volume whose argmax is a known pattern must survive
    # resize -> pad -> crop -> resize-back unchanged in the interior
    sc = 2.0
    rw, rh, pw, ph = plan(sc)
    gt = np.zeros((IMG_H, IMG_W), np.uint8)
    gt[:, IMG_W // 2:] = 13
    gt[:IMG_H // 2, :] = 8
    logits = torch.zeros(1, 19, ph, pw)
    big = torch.from_numpy(gt).float()[None, None]
    big = F.interpolate(big, size=(rh, rw), mode="nearest")[0, 0].long()
    for c in range(19):
        logits[0, c, :rh, :rw] = (big == c).float() * 10.0
    out = unpad_and_resize(logits, rw, rh)
    back = out.argmax(1)[0].numpy().astype(np.uint8)
    agree = float((back == gt).mean())
    chk("resize->pad->crop->resize-back round-trips", agree > 0.995,
        "(%.4f of pixels identical)" % agree)
    chk("output has the image's exact shape", back.shape == (IMG_H, IMG_W),
        str(back.shape))

    m = np.array([0.485, 0.456, 0.406], np.float32)
    s = np.array([0.229, 0.224, 0.225], np.float32)
    img = (np.random.rand(IMG_H, IMG_W, 3) * 255).astype(np.uint8)
    t = normalise(img, m, s)
    chk("normalise: shape (1,3,H,W)", tuple(t.shape) == (1, 3, IMG_H, IMG_W))
    chk("normalise: matches the reference formula",
        bool(torch.allclose(t[0, 0],
                            torch.from_numpy((img[:, :, 0] / 255.0 - m[0]) / s[0]).float(),
                            atol=1e-5)))
    print("\nSEG2D SELFTEST: %s (%d checks)" %
          ("ALL PASS" if not fails else "%d FAILED" % len(fails), n[0]))
    return len(fails) == 0


def normalise(img_rgb, mean, std):
    import torch
    x = img_rgb.astype(np.float32) / 255.0
    x = (x - mean) / std
    return torch.from_numpy(x.transpose(2, 0, 1)[None]).contiguous()


def unpad_and_resize(logits, rw, rh, w=IMG_W, h=IMG_H):
    """logits (1,C,ph,pw) -> (1,C,h,w): crop the size_divisor padding, then resize
    back to the image.  Bilinear on LOGITS, never nearest on labels -- resizing an
    argmax map is what loses thin structures, and losing them would understate the
    2D arm."""
    import torch.nn.functional as F
    return F.interpolate(logits[:, :, :rh, :rw], size=(h, w),
                         mode="bilinear", align_corners=False)


# --------------------------------------------------------------------------- #
#  model
# --------------------------------------------------------------------------- #
class Seg2D(object):
    def __init__(self, family="mask2former", device="cuda", precision="fp16",
                 scale=NATIVE_SCALE, tta="hflip"):
        import torch
        from transformers import AutoConfig
        self.torch = torch
        self.family = family
        self.repo = MODELS[family]
        self.device = device
        self.precision = precision
        self.scale = float(scale)
        self.tta = tta
        cfg = AutoConfig.from_pretrained(self.repo)
        id2l = {int(k): v for k, v in cfg.id2label.items()}
        assert [id2l[i] for i in range(19)] == CITYSCAPES_19, \
            ("model is not a 19-class Cityscapes model", id2l)
        if family == "mask2former":
            from transformers import Mask2FormerForUniversalSegmentation as M
            self.model = M.from_pretrained(self.repo)
            self.divisor = 32
            self.mean = np.array([0.485, 0.456, 0.406], np.float32)
            self.std = np.array([0.229, 0.224, 0.225], np.float32)
        else:
            from transformers import SegformerForSemanticSegmentation as M
            self.model = M.from_pretrained(self.repo)
            self.divisor = 32
            self.mean = np.array([0.485, 0.456, 0.406], np.float32)
            self.std = np.array([0.229, 0.224, 0.225], np.float32)
        self.model.eval()
        self.n_params = sum(p.numel() for p in self.model.parameters())
        if device != "cpu":
            self.model.to(device)
            if precision == "fp16":
                self.model.half()

    # ------------------------------------------------------------------ #
    def _logits(self, x, rw, rh):
        """x (1,3,ph,pw) already on device -> (1,19,H,W) float32 logits."""
        torch = self.torch
        out = self.model(pixel_values=x)
        if self.family == "mask2former":
            # per-pixel class scores, the standard Mask2Former semantic reduction:
            #   softmax over classes (dropping the no-object query) x sigmoid(masks)
            cls = out.class_queries_logits.float().softmax(-1)[..., :-1]   # (1,Q,19)
            msk = out.masks_queries_logits.float().sigmoid()               # (1,Q,h,w)
            seg = torch.einsum("bqc,bqhw->bchw", cls, msk)
            seg = torch.nn.functional.interpolate(
                seg, size=x.shape[-2:], mode="bilinear", align_corners=False)
        else:
            seg = torch.nn.functional.interpolate(
                out.logits.float(), size=x.shape[-2:], mode="bilinear",
                align_corners=False)
        return unpad_and_resize(seg, rw, rh)

    # ------------------------------------------------------------------ #
    def infer(self, img_rgb, want_prob=False):
        """img_rgb uint8 (370,1226,3) -> seg uint8 (370,1226), prob or None, stages."""
        torch = self.torch
        st = {}
        t = time.perf_counter()
        rw, rh, pw, ph = plan(self.scale, self.divisor)
        x = normalise(img_rgb, self.mean, self.std)
        x = torch.nn.functional.interpolate(x, size=(rh, rw), mode="bicubic",
                                            align_corners=False)
        xp = torch.zeros(1, 3, ph, pw, dtype=x.dtype)
        xp[:, :, :rh, :rw] = x
        xp = xp.to(self.device)
        if self.precision == "fp16" and self.device != "cpu":
            xp = xp.half()
        if self.device != "cpu":
            torch.cuda.synchronize()
        st["pre"] = (time.perf_counter() - t) * 1e3

        t = time.perf_counter()
        with torch.no_grad():
            acc = self._logits(xp, rw, rh)
            if self.tta == "hflip":
                acc = acc + torch.flip(self._logits(torch.flip(xp, dims=[3]), rw, rh),
                                       dims=[3])
                acc = acc / 2.0
        if self.device != "cpu":
            torch.cuda.synchronize()
        st["forward"] = (time.perf_counter() - t) * 1e3

        t = time.perf_counter()
        seg = acc.argmax(1)[0].to(torch.uint8).cpu().numpy()
        prob = None
        if want_prob:
            prob = acc.softmax(1)[0].to(torch.float16).cpu().numpy()
        if self.device != "cpu":
            torch.cuda.synchronize()
        st["post"] = (time.perf_counter() - t) * 1e3
        st["total"] = st["pre"] + st["forward"] + st["post"]
        return seg, prob, st


# --------------------------------------------------------------------------- #
def stats(v):
    v = np.asarray(v, np.float64)
    return dict(mean=float(v.mean()), p50=float(np.percentile(v, 50)),
                p95=float(np.percentile(v, 95)), max=float(v.max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="run",
                    choices=["run", "selftest", "check"])
    ap.add_argument("--family", default="mask2former", choices=sorted(MODELS))
    ap.add_argument("--seq", default="07", choices=sorted(SEQ_DRIVE))
    ap.add_argument("--scale", type=float, default=NATIVE_SCALE)
    ap.add_argument("--tta", default="hflip", choices=["none", "hflip"])
    ap.add_argument("--precision", default="fp16", choices=["fp16", "fp32"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--f0", type=int, default=0)
    ap.add_argument("--f1", type=int, default=-1)
    ap.add_argument("--prob", action="store_true",
                    help="also write full-resolution class probabilities (17 MB per "
                         "frame); only worth it on a subset")
    ap.add_argument("--out", default="")
    ap.add_argument("--timing-out", default="")
    a = ap.parse_args()

    if a.cmd == "selftest":
        sys.exit(0 if selftest() else 1)

    if a.cmd == "check":
        s = Seg2D(a.family, device="cpu", precision="fp32", scale=a.scale, tta=a.tta)
        rw, rh, pw, ph = plan(a.scale)
        print("repo            %s" % s.repo)
        print("parameters      %.1f M" % (s.n_params / 1e6))
        print("classes         19, order verified == Cityscapes trainIds")
        print("scale %.4f -> resize %dx%d, padded %dx%d (divisor %d)"
              % (a.scale, rw, rh, pw, ph, s.divisor))
        print("DESERIALIZE OK (CPU, no forward)")
        return

    assert a.out, "--out is required"
    from PIL import Image
    os.makedirs(a.out, exist_ok=True)
    drive = SEQ_DRIVE[a.seq]
    n = SEQ_N[a.seq] if a.f1 < 0 else a.f1
    seg2d = Seg2D(a.family, a.device, a.precision, a.scale, a.tta)
    rw, rh, pw, ph = plan(a.scale, seg2d.divisor)

    # warm-up: never time a cold kernel
    warm = np.zeros((IMG_H, IMG_W, 3), np.uint8)
    for _ in range(3):
        seg2d.infer(warm)

    acc = {}
    t0 = time.time()
    for f in range(a.f0, n):
        p = "%s/%s/image_02/data/%010d.png" % (RAWROOT, drive, f)
        if not os.path.exists(p):
            continue
        t = time.perf_counter()
        img = np.asarray(Image.open(p).convert("RGB"))
        assert img.shape[:2] == (IMG_H, IMG_W), ("image size", img.shape)
        tread = (time.perf_counter() - t) * 1e3
        seg, prob, st = seg2d.infer(img, want_prob=a.prob)
        st["imread"] = tread
        st["total"] += tread
        for k, v in st.items():
            acc.setdefault(k, []).append(v)
        if a.prob:
            np.savez("%s/f%06d.npz" % (a.out, f), seg=seg, prob=prob)
        else:
            np.savez_compressed("%s/f%06d.npz" % (a.out, f), seg=seg)
        if (f + 1) % 100 == 0:
            print("  %d frames, %.1f s" % (f + 1 - a.f0, time.time() - t0), flush=True)

    timing = {
        "arm": "2d", "model": seg2d.repo, "family": a.family,
        "hardware": os.popen("nvidia-smi --query-gpu=name --format=csv,noheader"
                             ).read().strip() or "cpu",
        "precision": a.precision, "input_hw": [ph, pw],
        "resized_hw": [rh, rw], "input_scale": a.scale,
        "native_scale_for_cityscapes": NATIVE_SCALE,
        "tta": a.tta, "batch": 1, "frames": len(acc.get("total", [])),
        "sequence": a.seq,
        "stages_ms": {k: stats(v) for k, v in acc.items()},
        "note": "imread is CPU PNG decode; the per-point projection is done by "
                "score_2d_vs_3d.py and timed separately",
    }
    print(json.dumps({k: v for k, v in timing.items() if k != "stages_ms"}, indent=2))
    print(json.dumps(timing["stages_ms"], indent=2))
    if a.timing_out:
        json.dump(timing, open(a.timing_out, "w"), indent=2)
    print("wrote %d frames -> %s" % (timing["frames"], a.out))


if __name__ == "__main__":
    main()
