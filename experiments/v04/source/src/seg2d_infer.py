#!/usr/bin/env python3
"""
seg2d_infer.py -- the 2D arm of the "why not just project a 2D segmentation?" study.

WHAT THIS IS FOR
----------------
The 3D arm (ptv3_loader_verified.py) segments the point cloud directly.  The obvious
external challenge is "you have a calibrated camera, so run a 2D network on the image
and paint the labels onto the points -- that is the mature and usually more accurate
approach".  This module IS that challenge, built to be as strong as it can be, so that
whatever the comparison says survives being quoted back at us.

TWO MODELS SHIP HERE.  RUN BOTH.  REPORT THE BETTER ONE.
--------------------------------------------------------
  key            checkpoint                                         Cityscapes val mIoU
                                                                    (single-scale, published)
  "eomt"         tue-mps/cityscapes_semantic_eomt_large_1024               84.2     MIT
  "mask2former"  facebook/mask2former-swin-large-cityscapes-semantic       83.3     other

  84.2 : EoMT (Encoder-only Mask Transformer), ViT-L/DINOv2, 319 M params, 1024x1024
         windowed inference.  Kerssies et al., "Your ViT is Secretly an Image
         Segmentation Model", CVPR 2025 (arXiv 2503.19108), Table 5, Cityscapes val.
  83.3 : Mask2Former Swin-L (IN21k), 216 M params, 90k iters.  Official
         facebookresearch/Mask2Former MODEL_ZOO.md, Cityscapes Semantic Segmentation
         (84.3 with ms+flip).  The same 83.3 appears as the Mask2Former row of EoMT's
         Table 5, so the two numbers are measured on the same protocol.

  Why EoMT is the primary and not merely "also available": besides being +0.9 mIoU in
  distribution, its DINOv2 backbone is much better OUT of distribution, which is the
  regime this whole study lives in (Cityscapes-trained, KITTI-evaluated).  EoMT Table 8
  reports DINOv2-based models beating the Swin-based model by >7.8 mIoU out of
  distribution "despite similar in-distribution performance on Cityscapes".  Its
  Appendix C also reports better-calibrated confidence than ViT-Adapter+Mask2Former,
  which matters because this module hands a confidence map to the fusion code.

  HOW STRONG IS THIS, HONESTLY?  The best published Cityscapes val numbers are
  ViT-Adapter-L + Mask2Former with DINOv2+DepthAnything at 84.8 s.s. (EoMT Tab. 5) and
  InternImage-H / ViT-Adapter-L+BEiT at ~85-86 with Mapillary pre-training and
  multi-scale TTA.  Those are mmsegmentation-0.x stacks with custom CUDA ops (DCNv3,
  MSDeformAttn against mmcv-full 1.x) that cannot be installed beside torch 2.5.1
  without rebuilding the env the 3D arm depends on.  So this 2D arm sits about
  0.6 mIoU below the best practically-published single-scale Cityscapes model and
  about 1.5-2 below the absolute leaderboard.  It is NOT a weak 2D arm, and nobody can
  claim the comparison was won by picking a toy segmenter.  Say the 0.6 out loud.

DOMAIN GAP, DELIBERATELY ASYMMETRIC AND IN THE 2D ARM'S FAVOUR
--------------------------------------------------------------
  2D arm : Cityscapes (German urban street, forward vehicle camera, daytime)
           -> KITTI  (German urban street, forward vehicle camera, daytime)
  3D arm : nuScenes  (32-beam, Boston/Singapore) -> KITTI (64-beam, Karlsruhe)
The 2D arm has by far the smaller shift.  SemanticKITTI's 19 classes were deliberately
modelled on Cityscapes' 19, so the label spaces line up almost one-to-one too (see the
LUT below: 12 of 13 coarse classes reachable, and all 11 that actually occur in GT).
That asymmetry is on purpose.  Do not "fix" it.

THE TRAP IN MASK2FORMER'S SHIPPED PREPROCESSOR
-----------------------------------------------
preprocessor_config.json for the Mask2Former checkpoint ships
    size = {"height": 384, "width": 384}
so AutoImageProcessor with its defaults turns a KITTI frame into 384x384: aspect ratio
3.31 -> 1.00 and a 0.31x downscale.  That would be a crippled 2D arm and an invalid
comparison.  This module never uses that processor for Mask2Former; it does its own
resize / normalise / pad and its own semantic post-processing.
EoMT's shipped processor is, by contrast, already correct (do_split_image=True,
shortest_edge=1024, aspect preserved) -- but it ties the resize to the window size, so
the scale cannot be varied through it.  We therefore reimplement its windowing (same
even-overlap rule, same overlap averaging) in 2D, at EoMT's trained 1024x1024 window,
which leaves the scale free.  At scale 2.7676 the tiling reduces exactly to the
shipped 1-row x 4-column split, so the default configuration IS the shipped recipe.

SCALE -- THE ONE PARAMETER THAT MATTERS
---------------------------------------
A Cityscapes model is trained on a *pixels-per-radian* budget, not on an image size.

    Cityscapes  2048x1024, fx = 2262.52 px  -> 2262.5 px/rad   (FOV 48.7 x 25.5 deg)
    KITTI cam2  1226x 370, fx =  707.0912   ->  707.1 px/rad   (FOV 81.9 x 29.3 deg)

so a native KITTI frame shows the same car 3.20x smaller than either network has ever
seen it.  Both were trained with scale jitter 0.5x-2.0x about the Cityscapes scale;
native KITTI at 0.31x is off the bottom end of that range.  Hence:

  SCALE_FOCAL = 2262.52 / 707.0912 = 3.2004   -- object-scale matched.
  At 3.2004 a KITTI object subtends exactly as many pixels as a Cityscapes object at
  the same metric distance, and the 1184-px image height for 29.3 deg sits within 16 %
  of Cityscapes' own 1024 px for 25.5 deg.

  DEFAULTS: mask2former -> scale 3.2004 "focal"     (3923x1184, padded 3936x1184)
            eomt        -> scale 2.7676 "shortside"  (1024x3393, 4 windows, 1 row)
  EoMT defaults to its authors' own Cityscapes recipe (short side 1024) rather than to
  focal-matched, because that geometry is the one the published 84.2 was measured with.
  It is 0.865x of object-scale-matched, comfortably inside the training jitter range.
  EoMT's position embeddings are a fixed 64x64 grid with NO interpolation, so its
  window is hard-locked to 1024x1024; this module tiles in 2D at that window size so
  the SCALE is still free and still sweepable (at "focal" it becomes 2 rows x 4 cols).

  THESE ARE REASONED DEFAULTS, NOT MEASURED ONES -- no GPU was available when this was
  written.  SWEEP THEM: tools/sweep_seg2d_scale.py does it in 20 frames.  Use the
  argmax.  Picking the scale by measurement is the single largest favour you can do
  the 2D arm, and quoting a number obtained at a scale that was not the best available
  scale is exactly the kind of result that gets demolished.

CONFIDENCE
----------
Both models are mask-classification models, and both build the same per-pixel score
    seg[c,h,w] = sum_q softmax(class_logits)[q,c] * sigmoid(mask_logits)[q,h,w]
(Mask2Former: replicated here, chunked over queries so a 3936x1184 forward does not
blow up VRAM.  EoMT: its own post_process_semantic_segmentation(...,
return_segmentation_scores=True), including the overlap-averaged window merge.)
We return
    label = argmax_c seg      conf = max_c seg / sum_c seg   in [0,1]
i.e. a proper per-pixel posterior over the 19 classes.

USAGE
-----
    from seg2d_infer import Seg2DSegmenter, CITYSCAPES19_TO_COARSE
    seg = Seg2DSegmenter("eomt", device="cuda")     # <-- GPU happens HERE and nowhere else
    lab, conf = seg.segment_image(png_path)         # both (370,1226), native KITTI grid

    # preferred path for point labelling -- keeps the full up-sampled detail:
    uv, depth, mask = calib.project_velo_to_cam2(pts, img_shape=(370,1226), return_mask=True)
    lab_pts, conf_pts = seg.sample_points(png_path, uv)    # (M,) uint8, (M,) float32
    coarse = seg.to_coarse(lab_pts)                 # sem_core.COARSE's 13-slot space
"""

import os

import numpy as np

os.environ.setdefault("HF_HOME", "/data/wuyou/hf_cache")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

MODELS = {
    "eomt": dict(
        repo="tue-mps/cityscapes_semantic_eomt_large_1024",
        cityscapes_miou_ss=84.2, licence="MIT", params_m=319,
        source="Kerssies et al., CVPR 2025 (arXiv 2503.19108), Tab. 5, Cityscapes val"),
    "mask2former": dict(
        repo="facebook/mask2former-swin-large-cityscapes-semantic",
        cityscapes_miou_ss=83.3, licence="other", params_m=216,
        source="facebookresearch/Mask2Former MODEL_ZOO.md, Cityscapes Semantic, "
               "Swin-L(IN21k) 90k (84.3 ms+flip)"),
}
DEFAULT_MODEL = "eomt"

# ---------------------------------------------------------------------------- #
#  scale
# ---------------------------------------------------------------------------- #
CITYSCAPES_FX = 2262.52          # cityscapes camera.json, 2048x1024
KITTI_FX = 707.0912              # P_rect_02[0,0], rectified cam2
KITTI_H, KITTI_W = 370, 1226
SCALE_FOCAL = CITYSCAPES_FX / KITTI_FX                 # 3.2004

SCALE_PRESETS = {
    "native":     1.0,                      # 1226x370  -- 0.31x of trained scale
    "d2_maxsize": 2048.0 / KITTI_W,         # 1.670 -> what detectron2's
                                            #   MIN_SIZE_TEST=1024/MAX_SIZE_TEST=2048
                                            #   degenerates to on a 3.31 aspect ratio
    "half_focal": SCALE_FOCAL / 2.0,        # 1.600 -> bottom of the training jitter range
    "shortside":  1024.0 / KITTI_H,         # 2.768 -> short side to Cityscapes' 1024
                                            #          == EoMT's own shipped recipe
    "focal":      SCALE_FOCAL,              # 3.200 -> object-scale matched
    "focal_125":  SCALE_FOCAL * 1.25,       # 4.000 -> in case bigger still helps
    # --- refinement points added 2026-09-22 after the first seq04 sweep showed
    # the optimum is NOT at the focal-matched scale but between native and
    # shortside.  They exist so the 2D arm runs at its true argmax rather than
    # at the best of a coarse grid.  Adding scales can only help the 2D arm.
    "scale_13":   1.30,
    "scale_20":   2.00,
}
DEFAULT_SCALE = {"eomt": "shortside", "mask2former": "focal"}

# ---------------------------------------------------------------------------- #
#  Cityscapes-19 -> the 13-slot coarse space in sem_core.COARSE
#  Keep BOTH arms in ONE label space or the comparison means nothing.
# ---------------------------------------------------------------------------- #
CITYSCAPES19 = ["road", "sidewalk", "building", "wall", "fence", "pole",
                "traffic light", "traffic sign", "vegetation", "terrain", "sky",
                "person", "rider", "car", "truck", "bus", "train", "motorcycle",
                "bicycle"]

# sem_core.COARSE, restated so this module imports without ROS/torch on the path.
COARSE = ["car", "bicycle", "motorcycle", "truck", "bus", "other_vehicle", "person",
          "road", "sidewalk", "other_flat", "terrain", "vegetation", "manmade"]
COARSE_VOID = len(COARSE)        # 13 -- "predicted something the coarse space cannot
                                 #       express".  Never equals any GT label, so it
                                 #       costs recall but creates no false positive in
                                 #       any scored class -- the mildest honest
                                 #       treatment.  Only `sky` lands here, and
                                 #       measured on real projected returns that is
                                 #       0.00 % of points, so it is a non-issue.
_C = {n: i for i, n in enumerate(COARSE)}


def _build_lut(rider="bicycle"):
    """rider: 'bicycle' (default) | 'motorcycle' | 'person' | 'void'.

    Cityscapes splits a cyclist into `rider` (the human) + `bicycle` (the frame).
    SemanticKITTI does not: 31 bicyclist -> coarse bicycle and 32 motorcyclist ->
    coarse motorcycle (sem_core.SK_TO_COARSE), and nuScenes puts the rider inside the
    bicycle box as well, so the 3D arm says 'bicycle' for a cyclist too.  'bicycle' is
    therefore the mapping that agrees with BOTH the GT and the other arm, and seq07 is
    a residential street where cyclists vastly outnumber motorcyclists.  It is still a
    convention call: re-run with rider='person' and report the spread.
    """
    m = {"road": "road", "sidewalk": "sidewalk",
         "building": "manmade", "wall": "manmade", "fence": "manmade",
         "pole": "manmade", "traffic light": "manmade", "traffic sign": "manmade",
         "vegetation": "vegetation", "terrain": "terrain", "person": "person",
         "car": "car", "truck": "truck", "bus": "bus",
         "train": "other_vehicle",        # Cityscapes 'train' == tram; SK 20 other-vehicle
         "motorcycle": "motorcycle", "bicycle": "bicycle"}
    lut = np.full(19, COARSE_VOID, dtype=np.int32)
    for i, n in enumerate(CITYSCAPES19):
        if n in m:
            lut[i] = _C[m[n]]
    lut[CITYSCAPES19.index("sky")] = COARSE_VOID
    if rider != "void":
        lut[CITYSCAPES19.index("rider")] = _C[rider]
    return lut


CITYSCAPES19_TO_COARSE = _build_lut("bicycle")

# COVERAGE, for whoever scores this.  On the 20-frame eval set the GT contains 11
# coarse classes (eval_report.NINE plus bicycle and motorcycle).  Cityscapes-19 covers
# ALL ELEVEN -- the 2D arm is NOT structurally blind to any scored class.  The single
# coarse class it cannot express is `other_flat` (SemanticKITTI 49 other-ground), which
# does not occur in that GT subset.  If it turns up over the full 1101 frames, drop it
# from BOTH arms or report it apart: scoring the 2D arm on a class its label space does
# not contain is exactly the rigged comparison this study must not make.


# ---------------------------------------------------------------------------- #
class Seg2DSegmenter(object):
    """A Cityscapes semantic segmenter with KITTI-correct preprocessing baked in.

    Parameters
    ----------
    model       "eomt" (default) | "mask2former" | an explicit HF repo id
    device      "cuda" | "cuda:0" | "cpu".  NOTHING touches the GPU until you pass one.
    dtype       "fp16" | "fp32" | None (= fp16 on cuda, fp32 on cpu)
    scale       float, or a key of SCALE_PRESETS, or None (= DEFAULT_SCALE[model]).
                Free for both backends.  EoMT is tiled into its trained 1024x1024
                windows (see _eomt_scores), so any scale >= ~0.4 works; below that the
                whole frame fits in one padded window.
    interp      "bicubic" | "bilinear" | "lanczos"   (mask2former path only)
    tta         "none" | "flip" | "ms_flip"
    ms_scales   relative factors used when tta == "ms_flip"
    query_chunk queries expanded to full resolution at once (mask2former path only)
    window_batch 1024x1024 EoMT windows forwarded per batch (VRAM knob)
    """

    def __init__(self, model=DEFAULT_MODEL, device="cuda", dtype=None, scale=None,
                 interp="bicubic", tta="flip", ms_scales=(0.75, 1.0, 1.25),
                 query_chunk=25, window_batch=4, size_divisor=32, max_long_side=5120,
                 verbose=True):
        import torch
        self.torch = torch

        self.key = model if model in MODELS else None
        self.repo = MODELS[model]["repo"] if self.key else model
        self.backend = "eomt" if ("eomt" in self.repo.lower()) else "mask2former"

        self.device = torch.device(device)
        if dtype is None:
            dtype = "fp16" if self.device.type == "cuda" else "fp32"
        self.dtype = {"fp16": torch.float16, "fp32": torch.float32}[dtype]

        if scale is None:
            scale = DEFAULT_SCALE.get(self.key or self.backend, "focal")
        self.scale = float(SCALE_PRESETS[scale]) if isinstance(scale, str) else float(scale)
        self.scale_key = scale if isinstance(scale, str) else None

        self.interp = interp
        self.tta = tta
        self.ms_scales = tuple(ms_scales)
        self.query_chunk = int(query_chunk)
        self.window_batch = int(window_batch)
        self.size_divisor = int(size_divisor)
        self.max_long_side = int(max_long_side)

        if self.backend == "eomt":
            from transformers import EomtForUniversalSegmentation
            self.model = EomtForUniversalSegmentation.from_pretrained(
                self.repo, dtype=self.dtype).to(self.device).eval()
        else:
            from transformers import Mask2FormerForUniversalSegmentation
            self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
                self.repo, dtype=self.dtype).to(self.device).eval()
            # The load report prints swin.layernorm as MISSING.  It is dead weight: its
            # output is discarded before the backbone returns.  Verified bit-exact --
            # filling it with garbage moves max|d logits| by 0.000e+00.

        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        self.num_classes = len(self.id2label)
        assert self.num_classes == 19, self.num_classes
        assert [self.id2label[i] for i in range(19)] == CITYSCAPES19, \
            "checkpoint's own id2label is not the Cityscapes trainId order"

        self.mean = np.array([0.485, 0.456, 0.406], np.float32)
        self.std = np.array([0.229, 0.224, 0.225], np.float32)

        if verbose:
            info = MODELS.get(self.key, {})
            print("[seg2d] %s (%s)  published Cityscapes val mIoU %s s.s."
                  % (self.repo.split("/")[-1], self.backend,
                     info.get("cityscapes_miou_ss", "?")))
            print("[seg2d]   device=%s dtype=%s scale=%.4f%s tta=%s"
                  % (self.device, dtype, self.scale,
                     (" (%s)" % self.scale_key) if self.scale_key else "", self.tta))

    # ------------------------------------------------------------------ io --
    @staticmethod
    def _as_rgb(path_or_array):
        if isinstance(path_or_array, np.ndarray):
            a = path_or_array
            if a.ndim != 3 or a.shape[2] != 3:
                raise ValueError("expected HxWx3 RGB uint8, got %s" % (a.shape,))
            return np.ascontiguousarray(a.astype(np.uint8))
        from PIL import Image
        return np.asarray(Image.open(str(path_or_array)).convert("RGB"))

    def _resize(self, img_u8, s):
        from PIL import Image
        H, W = img_u8.shape[:2]
        tw, th = int(round(W * s)), int(round(H * s))
        if (tw, th) == (W, H):
            return img_u8, (th, tw)
        f = {"bicubic": Image.BICUBIC, "bilinear": Image.BILINEAR,
             "lanczos": Image.LANCZOS}[self.interp]
        return np.asarray(Image.fromarray(img_u8).resize((tw, th), f)), (th, tw)

    # ------------------------------------------------------- mask2former path --
    def _m2f_tensor(self, img_u8):
        t = self.torch
        x = (img_u8.astype(np.float32) / 255.0 - self.mean) / self.std
        x = t.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0)
        Hs, Ws = x.shape[-2:]
        d = self.size_divisor
        ph, pw = (-Hs) % d, (-Ws) % d
        if ph or pw:
            x = t.nn.functional.pad(x, (0, pw, 0, ph), value=0.0)   # 0 == dataset mean
        return x.to(self.device, self.dtype), (Hs, Ws)

    def _m2f_scores(self, img_u8, out_hw, flip):
        t, F = self.torch, self.torch.nn.functional
        x, (Hs, Ws) = self._m2f_tensor(img_u8)
        if flip:
            x = t.flip(x, dims=[-1])
        o = self.model(pixel_values=x)
        cls = o.class_queries_logits[0].float().softmax(-1)[:, :-1]     # (Q,19)
        ml = o.masks_queries_logits[0]
        Hp, Wp = x.shape[-2:]
        seg = t.zeros((19,) + tuple(out_hw), dtype=t.float32, device=x.device)
        for a in range(0, ml.shape[0], self.query_chunk):
            b = min(a + self.query_chunk, ml.shape[0])
            m = F.interpolate(ml[a:b].unsqueeze(0).float(), size=(Hp, Wp),
                              mode="bilinear", align_corners=False)[0][:, :Hs, :Ws]
            if flip:
                m = t.flip(m, dims=[-1])
            m = m.sigmoid_()
            if (Hs, Ws) != tuple(out_hw):
                m = F.interpolate(m.unsqueeze(0), size=tuple(out_hw),
                                  mode="bilinear", align_corners=False)[0]
            seg += t.einsum("qc,qhw->chw", cls[a:b], m)
            del m
        return seg

    # --------------------------------------------------------------- eomt path --
    #  EoMT's position embeddings are a FIXED 64x64 grid (1024/16) with no
    #  interpolation, so every window fed to it must be exactly 1024x1024.  The shipped
    #  processor enforces that by tying the resize to the window size: shortest_edge
    #  sets BOTH, which pins KITTI to scale 1024/370 = 2.7676 and makes the scale
    #  unsweepable.  So we tile ourselves, in 2D, at the trained 1024x1024 window size
    #  and any scale -- the same even-overlap rule EoMT uses along its long axis,
    #  applied to both axes, with overlaps averaged exactly as merge_image_patches does.
    #  At s = 2.7676 this reduces to 1 row x 4 cols, i.e. the shipped behaviour.
    EOMT_WIN = 1024

    def _tiles(self, L, P):
        """EoMT's even-overlap split of a length-L axis into windows of size P."""
        n = max(1, int(np.ceil(L / float(P))))
        ov = (n * P - L) / float(n - 1) if n > 1 else 0.0
        return [int(i * (P - ov)) for i in range(n)]

    def _eomt_scores(self, img_u8_native, s, out_hw, flip):
        t, F = self.torch, self.torch.nn.functional
        P = self.EOMT_WIN
        im = img_u8_native[:, ::-1].copy() if flip else img_u8_native
        u8, (Hs, Ws) = self._resize(im, s)

        x = (u8.astype(np.float32) / 255.0 - self.mean) / self.std
        x = t.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0)
        ph, pw = max(0, P - Hs), max(0, P - Ws)          # a window is never bigger than
        if ph or pw:                                     # the image
            x = F.pad(x, (0, pw, 0, ph), value=0.0)      # 0 == dataset mean colour
        Hp, Wp = x.shape[-2:]
        x = x.to(self.device, self.dtype)

        ys, xs = self._tiles(Hp, P), self._tiles(Wp, P)
        wins = t.cat([x[:, :, y:y + P, xo:xo + P] for y in ys for xo in xs], 0)

        acc = t.zeros((19, Hp, Wp), dtype=t.float32, device=self.device)
        cnt = t.zeros((1, Hp, Wp), dtype=t.float32, device=self.device)
        for a in range(0, wins.shape[0], self.window_batch):
            b = min(a + self.window_batch, wins.shape[0])
            o = self.model(pixel_values=wins[a:b])
            cls = o.class_queries_logits.float().softmax(-1)[..., :-1]      # (n,Q,19)
            ml = F.interpolate(o.masks_queries_logits.float(), size=(P, P),
                               mode="bilinear", align_corners=False).sigmoid_()
            sv = t.einsum("nqc,nqhw->nchw", cls, ml)                        # (n,19,P,P)
            for k in range(a, b):
                y, xo = ys[k // len(xs)], xs[k % len(xs)]
                acc[:, y:y + P, xo:xo + P] += sv[k - a]
                cnt[:, y:y + P, xo:xo + P] += 1.0
            del o, cls, ml, sv
        seg = (acc / cnt.clamp_min(1.0))[:, :Hs, :Ws]
        if (Hs, Ws) != tuple(out_hw):
            seg = F.interpolate(seg.unsqueeze(0), size=tuple(out_hw),
                                mode="bilinear", align_corners=False)[0]
        return t.flip(seg, dims=[-1]) if flip else seg

    # ------------------------------------------------------------- hi-res map --
    def _hires(self, path_or_array):
        """-> (label (Hr,Wr) uint8, conf (Hr,Wr) f32, sx, sy, (H,W) native)"""
        t = self.torch
        img = self._as_rgb(path_or_array)
        H, W = img.shape[:2]

        s = self.scale
        if max(H, W) * s > self.max_long_side:
            s = self.max_long_side / float(max(H, W))
            if not getattr(self, "_warned_clamp", False):
                print("[seg2d] WARNING: scale clamped to %.4f by max_long_side=%d "
                      "-- raise it if you meant the larger scale" % (s, self.max_long_side))
                self._warned_clamp = True
        Hr, Wr = int(round(H * s)), int(round(W * s))
        scales = self.ms_scales if self.tta == "ms_flip" else (1.0,)
        flips = (False, True) if self.tta in ("flip", "ms_flip") else (False,)

        with t.no_grad():
            seg = t.zeros((19, Hr, Wr), dtype=t.float32, device=self.device)
            for ms in scales:
                if self.backend == "eomt":
                    for fl in flips:
                        seg += self._eomt_scores(img, s * ms, (Hr, Wr), fl)
                else:
                    u8, _ = self._resize(img, s * ms)
                    for fl in flips:
                        seg += self._m2f_scores(u8, (Hr, Wr), fl)
            tot = seg.sum(0).clamp_min_(1e-12)
            conf, lab = (seg / tot).max(0)
            lab = lab.to(t.uint8).cpu().numpy()
            conf = conf.float().cpu().numpy()
            del seg
        return lab, conf, Wr / float(W), Hr / float(H), (H, W)

    # ------------------------------------------------------------ public API --
    def segment_image(self, path_or_array):
        """-> (label_map (H,W) uint8 Cityscapes trainIds, conf (H,W) float32 in [0,1])

        Both on the NATIVE KITTI grid (370x1226), so they index straight with the pixel
        coordinates kitti_calib.project_velo_to_cam2 returns.  The up-sampled map is
        read at the native pixel CENTRES -- nothing is averaged away, so thin structures
        (poles, sign posts) survive at the pixel actually queried.

        For point labelling prefer sample_points(): it keeps each LiDAR return's
        sub-pixel location instead of rounding it onto the native grid.
        """
        lab, conf, sx, sy, (H, W) = self._hires(path_or_array)
        if lab.shape == (H, W):
            return lab, conf
        vi = np.clip(np.rint((np.arange(H) + 0.5) * sy - 0.5), 0, lab.shape[0] - 1).astype(np.int32)
        ui = np.clip(np.rint((np.arange(W) + 0.5) * sx - 0.5), 0, lab.shape[1] - 1).astype(np.int32)
        return lab[np.ix_(vi, ui)], conf[np.ix_(vi, ui)]

    def sample_points(self, path_or_array, uv):
        """Label LiDAR returns at their sub-pixel projections.

        uv : (M,2) float pixel coords in the NATIVE 1226x370 image -- exactly what
             kitti_calib.project_velo_to_cam2 returns.
        -> (labels (M,) uint8 Cityscapes trainIds, conf (M,) float32)
        """
        lab, conf, sx, sy, (H, W) = self._hires(path_or_array)
        uv = np.asarray(uv, dtype=np.float64)
        u = np.clip(np.rint((uv[:, 0] + 0.5) * sx - 0.5), 0, lab.shape[1] - 1).astype(np.int32)
        v = np.clip(np.rint((uv[:, 1] + 0.5) * sy - 0.5), 0, lab.shape[0] - 1).astype(np.int32)
        return lab[v, u], conf[v, u]

    def to_coarse(self, cityscapes_labels, rider="bicycle"):
        lut = CITYSCAPES19_TO_COARSE if rider == "bicycle" else _build_lut(rider)
        return lut[np.asarray(cityscapes_labels, dtype=np.int64)]


# ---------------------------------------------------------------------------- #
#  self-test.  CPU BY DEFAULT ON PURPOSE -- this file must never be the thing that
#  takes the GPU out from under a timed benchmark.
#     python seg2d_infer.py [frame] [--model eomt] [--device cuda] [--scale focal]
# ---------------------------------------------------------------------------- #
if __name__ == "__main__":
    import argparse, time
    ap = argparse.ArgumentParser()
    ap.add_argument("frame", nargs="?", type=int, default=0)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--scale", default=None)
    ap.add_argument("--tta", default="none")
    a = ap.parse_args()

    png = ("/data/wuyou/livo_sem/data/raw/2011_09_30/2011_09_30_drive_0027_sync"
           "/image_02/data/%010d.png" % a.frame)
    sc = a.scale
    if sc is not None and sc not in SCALE_PRESETS:
        sc = float(sc)
    s = Seg2DSegmenter(a.model, device=a.device, scale=sc, tta=a.tta)
    t0 = time.time()
    lab, conf = s.segment_image(png)
    print("[self-test] %s -> label %s conf %s in %.1fs"
          % (os.path.basename(png), lab.shape, conf.shape, time.time() - t0))
    H, W = lab.shape
    n = np.bincount(lab.ravel(), minlength=19)
    print("[self-test] class mix (>0.5 %):")
    for i in np.argsort(-n):
        if n[i] / n.sum() > 0.005:
            print("    %-14s %5.1f %%   mean conf %.3f"
                  % (s.id2label[i], 100.0 * n[i] / n.sum(), conf[lab == i].mean()))
    top = np.bincount(lab[:H // 4].ravel(), minlength=19)
    bot = np.bincount(lab[3 * H // 4:].ravel(), minlength=19)
    print("[self-test] SPATIAL SANITY")
    print("    top quarter   : %-12s %.1f %%  (expect sky/vegetation/building)"
          % (s.id2label[top.argmax()], 100.0 * top.max() / top.sum()))
    print("    bottom quarter: %-12s %.1f %%  (expect road)"
          % (s.id2label[bot.argmax()], 100.0 * bot.max() / bot.sum()))
    print("[self-test] mean confidence %.3f, frac conf<0.5 %.3f"
          % (conf.mean(), float((conf < 0.5).mean())))
