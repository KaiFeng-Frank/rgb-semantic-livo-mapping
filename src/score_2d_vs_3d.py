#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
score_2d_vs_3d.py -- scoring instrument for the question

    "You have RGB.  Why not run a 2D segmentation network on the image and
     project the labels onto the points?  Why segment the point cloud?"

It scores three arms on SemanticKITTI seq 07 (== KITTI raw 2011_09_30_drive_0027,
index-aligned, offset 0) in ONE shared coarse label space, over ONE shared set of
evaluated points, with ONE shared ignore rule, so the arms are directly subtractable.

    A. 3D      PTv3 on the point cloud                 (nuScenes-16 -> coarse)
    B. 2D      Cityscapes segmentation of image_02,
               labels projected onto the points        (Cityscapes-19 -> coarse)
    C. HYBRID  2D where the 2D arm speaks, 3D elsewhere

THE FAIRNESS RULE
=================
Every contestable choice below is resolved IN FAVOUR OF THE 2D ARM, and each such
choice is listed in PROTOCOL["pro_2d_concessions"] so a reader can audit it.  A
conclusion that survives a stacked deck is the only kind worth quoting.  Where a
choice could not be resolved in favour of either arm without hiding something, BOTH
variants are computed and BOTH are reported (this is why every accuracy appears
under two abstention conventions, and why the 2D arm is scored with and without
occlusion reasoning).

===========================================================================
PROTOCOL
===========================================================================

P0.  CANONICAL INDEX SPACE
     The index space is the raw `.bin` point order of the frame, which is the order
     the SemanticKITTI `.label` file is indexed in.  Any arm that internally reorders
     points (the deployed PTv3 worker azimuth-sorts) must ship the permutation
     `order` alongside its labels; this module inverts it.  Nothing is scored in any
     other order, ever.

P1.  LABEL SPACE
     The 13-class coarse space of sem_core.COARSE, which is the space the 3D arm was
     already scored in (point acc 87.3 %, coarse mIoU 53.2 % / 61.0 %).  Reusing it
     is what makes the new numbers comparable to the ones already cited.
       nuScenes-16   -> coarse via sem_core.NUSC16_TO_COARSE  (unchanged)
       Cityscapes-19 -> coarse via CITY19_TO_COARSE           (defined here)
       SemanticKITTI -> coarse via sem_core.sk_lut()          (unchanged, GT)

P2.  EVALUATED POINTS
     A point is evaluated iff its GT coarse class is not IGNORE.  Measured on 101
     frames: 3.2 % of all points and 2.5 % of in-frustum points are IGNORE.  The same
     evaluated set is used for every arm and every subset, so coverage differences
     between arms are differences in the arms, not in the denominators.

P3.  FRUSTUM (arm-independent, pure geometry)
     x_cam = T_velo_cam2 @ x_velo;  u = fx*x/z + cx;  v = fy*y/z + cy
     (fx = fy = 707.0912, cx = 601.8873, cy = 183.1104; images are RECTIFIED so there
     is no distortion term).  A point is IN-FRUSTUM iff

         z > MIN_DEPTH (0.5 m)   AND   0 <= u <= W-1   AND   0 <= v <= H-1

     with W,H = 1226,370 taken from S_rect_02.  Points behind the camera and points
     projecting outside the image are NOT in-frustum and are UNLABELLED globally.
     MEASURED: 16.04 % of points are in-frustum (101 frames, per-frame 13.9 %-17.5 %).
     This mask is computed from calibration alone and never from any prediction, so
     every subset built on it is neutral between the arms.

P4.  HOW A POINT GETS ITS 2D LABEL
     (a) SAMPLING.  Default `nearest`: the label of the pixel (round(u), round(v)).
         A label map is categorical, so interpolating it is meaningless.  If the 2D
         arm also ships per-pixel class probabilities (`prob`), `prob_bilinear` is
         used instead: the probability field is sampled bilinearly at the exact
         sub-pixel (u,v) and the argmax taken.  That is strictly more information
         than nearest and can only help the 2D arm; it is the default when `prob`
         is present.                                    [PRO-2D]
     (b) OCCLUSION.  A background point and a foreground point can project to the
         same pixel; naive projection hands the background point the foreground
         label.  Default `zbuffer`: build a min-depth buffer over the in-frustum
         points, dilate it over a (2*OCC_WIN+1)^2 window, and declare a point
         OCCLUDED -- i.e. the 2D arm ABSTAINS on it -- when

             z > zbuf_dilated + max(OCC_TOL_ABS, OCC_TOL_REL * zbuf_dilated)

         with OCC_WIN=2, OCC_TOL_ABS=0.5 m, OCC_TOL_REL=0.02 -- see the comment on
         OCC_WIN: 2 px is the HDL-64E's vertical sampling pitch in camera pixels,
         not a tuned value, and `--occ-win-sweep` reports the sensitivity.  The
         larger the window the more the 2D arm abstains, which raises its accuracy
         under convention (ii) and lowers its coverage; both are reported, so the
         trade is visible rather than chosen.  The tolerance exists
         so that two points on the same surface that happen to share a pixel both
         keep the label; only a genuine foreground/background straddle abstains.
         This is pro-2D (it removes from the denominator exactly the points the
         camera could not see) and it is also the physically honest rule: the 2D
         arm has no evidence about a point it cannot see.  It uses LiDAR depth,
         which a LiDAR-camera system has at fusion time, so it is not GT leakage.
         `occlusion=none` reproduces the naive projection and IS ALSO REPORTED, so
         the concession is visible rather than assumed.          [PRO-2D + REPORTED]
     (c) SKY.  Cityscapes class 10 is `sky`, which no LiDAR return can be.  Default
         `sky_policy=abstain`: a point landing on a sky pixel is treated as the 2D
         arm declining, not as a wrong answer.  Under convention (i) below an
         abstention costs exactly as much as a wrong answer, so this concession is
         bracketed rather than hidden, and the sky-hit rate is reported.   [PRO-2D]
     (d) The 2D arm also abstains wherever the seg map itself is 255 (ignore).

P5.  ABSTENTION CONVENTIONS -- both are always reported
     (i)  STRICT   unlabelled counted as WRONG.  A point with no label is a point
                   the map cannot answer for; this is what a mapping system needs.
                   acc = correct / (labelled + abstained)
                   IoU_k = TP_k / (FP_k + FN_k + TP_k + abstained_k)
                   (an abstention on GT class k is a false negative for k and a
                   false positive for nothing)
     (ii) ABSTAIN  unlabelled EXCLUDED from the denominator.  Judge the arm where it
                   speaks; this is the convention that is fair to the 2D arm.
                   acc = correct / labelled
     The 3D arm never abstains, so (i) and (ii) coincide for it and the identity
     `acc_strict == acc_abstain when coverage == 1` is asserted by the self-test.
     THE GAP BETWEEN (i) AND (ii) FOR THE 2D ARM IS THE FINDING, not a footnote.

P6.  CLASS SUBSETS FOR mIoU (all four reported)
     all    every coarse class present in the GT of the frame set (11 in seq 07:
            bus and other_flat have zero GT points).
     nine   the 9-class subset CRITICAL_CONSTRAINTS quotes (car, truck,
            other_vehicle, person, road, sidewalk, terrain, vegetation, manmade),
            kept so the new numbers dock onto the already-cited 61.0 %.
     expr   `all` minus the classes that ONE taxonomy structurally cannot express.
            Only `other_vehicle` qualifies: nuScenes has construction_vehicle and
            trailer, Cityscapes has only `train`, while SemanticKITTI id 20 is vans
            and caravans.  Scoring the 2D arm on a class its label set cannot name
            is a taxonomy penalty, not a perception result.               [PRO-2D]
     expr8  `nine` minus other_vehicle.                                   [PRO-2D]

P7.  BOUNDARY SETS -- defined WITHOUT reference to any arm's predictions
     B1 `sem_boundary` (global, GT-only).  A point is a semantic-boundary point iff
        any of its k=10 nearest 3D neighbours within r=0.5 m carries a DIFFERENT
        non-ignore GT coarse class.  Complement within the evaluated set is
        `sem_interior`.  Depends only on xyz and GT.
     B2 `depth_edge` (in-frustum only, GEOMETRY-only, no GT).  Build min- and
        max-depth buffers from the in-frustum points, dilate each over a
        (2*EDGE_R_PX+1)^2 = 11x11 window, and call a point a depth-edge point iff

             zmax_nb - zmin_nb > max(EDGE_ABS_M, EDGE_REL * zmin_nb)

        with EDGE_R_PX=5, EDGE_ABS_M=1.0 m, EDGE_REL=0.3.  This is exactly the
        pixel neighbourhood in which a 2D label can slide off a foreground object
        onto a background point.  Complement is `depth_interior`.
     B1 is semantic, B2 is geometric, and they answer different objections; both are
     reported.  Neither looks at a prediction, so neither can be gamed by an arm.

P8.  RANGE STRATIFICATION
     Bins on LiDAR range ||xyz||, not on camera depth: range is the axis both
     sensors degrade along and it is defined for out-of-frustum points too.
     Bins 0-10, 10-20, 20-30, 30-50, 50+ m.  Reported both globally and in-frustum.
     MEASURED in-frustum distribution: 43.4 % / 39.8 % / 10.2 % / 5.2 % / 1.4 %.

P9.  FRAME SET AND REPEATS
     PRIMARY: all 1101 frames, one pass per arm.  This is the set the cited 3D
     numbers were measured on, so the arms dock onto them directly.
     NOISE SET: 100 frames (stride 11), >= 3 passes of the 3D arm.  The 3D model is
     non-deterministic (4-5 % of points change label between two forwards; ~1 point
     of mIoU spread on 20 frames).  Assuming frame-independent noise that spread
     falls as 1/sqrt(N): ~0.45 pt at 100 frames, ~0.13 pt at 1101.  Every headline
     number is therefore quoted with the MEASURED half-range over the repeats, and
     a verdict is only stated when the arm-to-arm gap exceeds 3x that half-range.
     The 2D arm must be deterministic; `--assert-2d-deterministic` checks two passes
     are bitwise identical and fails loudly if they are not.

P10. LATENCY
     Supplied by the measurement agent as JSON, schema in TIMING_SCHEMA.  Stage
     breakdown per arm, mean/p50/p95/max over frames, plus the conditions
     (hardware, precision, input resolution, TTA, batch).  The hybrid needs BOTH
     networks, so its latency is reported twice: `serial` (one GPU, sum of stages)
     and `parallel` (two streams, max of stages) -- the honest bracket.

P11. THE 2D ARM'S CONFIGURATION, AND WHERE IT WAS CHOSEN
     The 2D arm is a Cityscapes-pretrained semantic segmentation network run on
     image_02 (see src/seg2d_cityscapes.py).  Two things about it are contestable
     and both are resolved in its favour:
       * MODEL.  facebook/mask2former-swin-large-cityscapes-semantic, Cityscapes
         val mIoU 83.3, i.e. close to the best published single model on the 2D
         arm's own benchmark.  nvidia/segformer-b5-finetuned-cityscapes (82.4) is
         kept as the deployable-speed alternative and reported alongside if asked.
       * INPUT SCALE.  A Cityscapes model is trained at fx ~ 2262 px; KITTI's
         rectified image_02 has fx = 707.0912.  The same object subtends 3.2x
         fewer pixels in KITTI, which is the regime a segmentation network is
         worst in.  Feeding the native 1226x370 image is a handicap, not a
         neutral choice, so the image is upsampled before the forward.
     Choosing that scale on seq 07 and then reporting seq 07 would be tuning on
     the test set.  Choosing it not at all would cripple the arm.  So it is tuned
     on SEQ 04 (2011_09_30_drive_0016_sync, 271 frames) -- the same rig, the same
     calibration day, a disjoint drive, VERIFIED index-aligned with offset 0 -- and
     then FROZEN.  Four principled landmarks are compared, not a grid: 1.00
     (native), 2.00, 2.7676 (height -> 1024, the Cityscapes training crop height)
     and 3.1998 (angular match).  Horizontal-flip TTA is on: it is standard, costs
     one extra forward and can only help the 2D arm; latency is reported with and
     without it.

P12. WHAT IS *NOT* CLAIMED
     Per-scan coverage (16 %) is not map coverage: a camera sweeps as the vehicle
     drives.  `mapcov` measures the accumulated-map version of metric 1 (fraction of
     occupied map voxels ever seen by the camera over the whole trajectory) so that
     the obvious rebuttal is answered with a number instead of an argument.

===========================================================================
ON-DISK CONTRACT  (this is all the measurement agent has to produce)
===========================================================================

  3D arm     <dir>/f%06d.npz   lab   uint8 (N,)  nuScenes-16 id, 255 = abstain
                               order int32 (N,)  OPTIONAL permutation (see P0)
                               conf  f16   (N,)  OPTIONAL
             -- exactly what opt/cache_pred.py already writes.

  2D arm     <dir>/f%06d.npz   seg   uint8 (H,W) Cityscapes trainId, 255 = ignore
                               prob  f16 (19,h,w) OPTIONAL, any (h,w); sampled
                                                  bilinearly, enables P4(a)
             The 2D arm caches IMAGE-SPACE labels, not point labels.  The projection,
             the occlusion rule and the sky rule then live in this module where they
             are auditable and can be varied without touching the GPU.

  timing     <file>.json       see TIMING_SCHEMA

===========================================================================
USAGE
===========================================================================
    python score_2d_vs_3d.py selftest                      # no GPU, no model, no cache
    python score_2d_vs_3d.py prep   --frames all           # boundary-set cache (CPU)
    python score_2d_vs_3d.py score  --arm3d D3 --arm2d D2 --frames all --out r.json
    python score_2d_vs_3d.py mapcov --frames all --out mapcov.json
    python report_2d_vs_3d.py r.json > table.md
"""

import os
import sys
import json
import glob
import argparse
import hashlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/data/livo_sem/src")

import label_spaces as LS                                # noqa: E402
from kitti_calib import load_calib                       # noqa: E402

# --------------------------------------------------------------------------- #
#  paths
# --------------------------------------------------------------------------- #
ROOT = "/data/livo_sem"
CALIB_DIR = ROOT + "/data/raw/2011_09_30"

# Both sequences were recorded on 2011_09_30, so they share one calibration and one
# image size (1226x370) -- VERIFIED, including point/label length match at frames
# 0, 1, 100, 270 for seq 04.  seq 04 exists so that any choice the 2D arm needs
# tuned (above all the input scale, see seg2d_cityscapes.py) can be tuned on a set
# that is DISJOINT from the evaluation sequence, on the same rig and the same day.
# Tuning the 2D arm on seq 07 and then reporting seq 07 would be indefensible; not
# tuning it at all would cripple it.  seq 04 is how the 2D arm gets its strongest
# configuration honestly.
SEQUENCES = {
    "07": ("2011_09_30_drive_0027_sync", 1101),   # EVALUATION set
    "04": ("2011_09_30_drive_0016_sync", 271),    # TUNING set for the 2D arm
}
SEQ_ID = "07"
RAW = SEQ = LABELS = SCANS = IMAGES = None
N_FRAMES_TOTAL = 0


def set_sequence(seq_id):
    """Point the module at one of SEQUENCES.  Must be called before any read."""
    global SEQ_ID, RAW, SEQ, LABELS, SCANS, IMAGES, N_FRAMES_TOTAL
    assert seq_id in SEQUENCES, ("unknown sequence", seq_id, list(SEQUENCES))
    drive, n = SEQUENCES[seq_id]
    SEQ_ID = seq_id
    RAW = ROOT + "/data/raw/2011_09_30/" + drive
    SEQ = ROOT + "/data/odometry/dataset/sequences/" + seq_id
    LABELS = SEQ + "/labels"
    SCANS = RAW + "/velodyne_points/data"
    IMAGES = RAW + "/image_02/data"
    N_FRAMES_TOTAL = n


set_sequence("07")

# --------------------------------------------------------------------------- #
#  constants of the protocol  (every one of these lands in the results JSON)
# --------------------------------------------------------------------------- #
IMG_W, IMG_H = 1226, 370
MIN_DEPTH = 0.5           # m, camera-frame z; P3

# P4(b) half-window of the z-buffer dilation, in pixels.  NOT a tuned number: it
# is the LiDAR's own angular sampling pitch expressed in camera pixels.  One
# camera pixel is 1/707.0912 rad = 0.081 deg.  The HDL-64E samples ~0.09 deg
# horizontally (~1.1 px) and ~0.4 deg vertically (~5 px).  A foreground surface is
# therefore sampled by LiDAR points roughly 5 px apart vertically, so a z-buffer
# that only looks at the exact pixel misses most real occlusions: the occluding
# surface is dense in the IMAGE but sparse in the CLOUD.  half_win = 2 (a 5x5
# window) is the smallest window that spans that pitch.  Sensitivity over
# {0,1,2,3} is reported by `score --occ-win-sweep`, so the choice is visible.
OCC_WIN = 2
OCC_TOL_ABS = 0.5         # m
OCC_TOL_REL = 0.02        # dimensionless

SEM_K = 10                # P7/B1
SEM_R = 0.5               # m

EDGE_R_PX = 5             # P7/B2 half-window, pixels
EDGE_ABS_M = 1.0          # m
EDGE_REL = 0.3            # dimensionless

RANGE_EDGES = [0.0, 10.0, 20.0, 30.0, 50.0, np.inf]      # P8

# ADJUDICATION R1 (2026-09-21): the common space is label_spaces.COARSE -- NINE
# classes, every one natively expressible in SemanticKITTI, nuScenes-16 AND
# Cityscapes-19, and every one present in seq07 GT.  The older 13-slot space is
# gone: bus / other_vehicle / other_flat do not survive scrutiny against
# Cityscapes-19, and scoring either arm on a class its own taxonomy cannot name is
# a taxonomy penalty, not a perception result.
COARSE = list(LS.COARSE)
K = len(COARSE)                                          # 9
IGNORE = LS.EXCLUDED                                     # -1, GT side
ABSTAIN = -1                                             # prediction side: no answer
UNMAPPED = LS.UNMAPPED                                   # -2, prediction side: an
#   answer the common space cannot name (2D `sky`, 3D `other_flat`).  ALWAYS WRONG,
#   NEVER DROPPED.  Keeping this distinct from ABSTAIN is what closes the abstention
#   leak: otherwise the 2D arm could delete its own errors by predicting sky.
_C = {n: i for i, n in enumerate(COARSE)}

NINE = list(COARSE)                     # the headline subset IS the whole space
FREQUENT = list(LS.COARSE_FREQUENT)     # >=1 % of in-frustum GT points
# Nothing is structurally inexpressible any more -- that is the point of R1.
NOT_EXPRESSIBLE_BY_2D = []

# --------------------------------------------------------------------------- #
#  Cityscapes-19 -> coarse   (P1)
# --------------------------------------------------------------------------- #
CITYSCAPES_19 = ["road", "sidewalk", "building", "wall", "fence", "pole",
                 "traffic light", "traffic sign", "vegetation", "terrain", "sky",
                 "person", "rider", "car", "truck", "bus", "train", "motorcycle",
                 "bicycle"]
SKY_ID = 10

# `rider` is the one ambiguous entry: SemanticKITTI splits it into bicyclist (31,
# 253 -> coarse `bicycle`) and motorcyclist (32, 255 -> coarse `motorcycle`), and
# never into `person` (30 is pedestrians only), so rider->person would be wrong by
# construction.  MEASURED on seq 07 over 101 frames: bicyclist 2115 pts,
# motorcyclist 0 pts.  rider -> bicycle is therefore the mapping that maximises the
# 2D arm's score.  Declared as a pro-2D concession; flip with --rider.
RIDER_DEFAULT = "two_wheeler"


def city19_to_coarse(rider="two_wheeler", sky_policy="wrong"):
    """(20,) int16 LUT: index 0..18 = Cityscapes trainId, index 19 = 255/ignore.

    Body is label_spaces.CS_TO_COARSE verbatim.  Two entries are policy:
      * trainId 255 (the seg map's own ignore fill) -> ABSTAIN, always.
      * `sky` (10).  DEFAULT sky_policy="wrong" -> UNMAPPED, i.e. counted wrong under
        BOTH abstention conventions.  sky_policy="abstain" is the pro-2D variant and
        is reported as a sensitivity, never as the headline, because under convention
        (ii) it would let the 2D arm delete its own errors.  sky_policy="manmade"
        forces the nearest legal answer.
    The `rider` question the 13-class space had DISAPPEARS here: Cityscapes rider(12),
    bicycle(18) and motorcycle(17) all land on two_wheeler, which is also where
    SemanticKITTI puts bicyclist(31)/motorcyclist(32) and where nuScenes puts its
    rider-inclusive bicycle/motorcycle boxes.  All three taxonomies agree.
    """
    lut = np.full(20, ABSTAIN, dtype=np.int16)
    lut[:19] = np.asarray(LS.CS_TO_COARSE, dtype=np.int16)
    if sky_policy == "abstain":
        lut[LS.CS_SKY_ID] = ABSTAIN
    elif sky_policy == "manmade":
        lut[LS.CS_SKY_ID] = _C["manmade"]
    elif sky_policy != "wrong":
        raise ValueError("sky_policy must be wrong|abstain|manmade")
    assert rider in ("two_wheeler", "person"), rider
    if rider == "person":                 # disclosed sensitivity axis only
        lut[12] = _C["person"]
    return lut


# inverse, used only by the self-test to synthesise a 2D arm from GT
COARSE_TO_CITY19 = {
    "car": 13, "large_vehicle": 14, "two_wheeler": 18, "person": 11,
    "road": 0, "sidewalk": 1, "terrain": 9, "vegetation": 8, "manmade": 2,
}

TIMING_SCHEMA = {
    "arm": "3d|2d|hybrid",
    "hardware": "str, e.g. 'RTX 4090, i9-13900K'",
    "precision": "fp16|fp32|bf16",
    "input_hw": "[H, W] actually fed to the network",
    "input_scale": "float, resize factor applied to the 1226x370 image (2D arm)",
    "tta": "none|hflip|multiscale:...",
    "batch": "int",
    "frames": "int, number of timed frames",
    "stages_ms": {
        "<stage name>": {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0},
        "_3d_stages": "pre(voxelize) / forward / post(devoxelize) / total",
        "_2d_stages": "imread / pre(resize+normalise) / forward / post(argmax+upsample)"
                      " / project(handled by this module, CPU) / total",
    },
}


# --------------------------------------------------------------------------- #
#  geometry
# --------------------------------------------------------------------------- #
class Projector(object):
    """velodyne -> rectified image_02 pixels.  P3.  Arm-independent."""

    def __init__(self, calib_dir=CALIB_DIR, w=IMG_W, h=IMG_H, min_depth=MIN_DEPTH):
        cal = load_calib(calib_dir)
        self.cal = cal
        self.w, self.h = int(w), int(h)
        self.min_depth = float(min_depth)
        assert int(cal.img_size[0]) == self.w and int(cal.img_size[1]) == self.h, \
            ("image size disagrees with S_rect_02", cal.img_size, (w, h))

    def project(self, xyz):
        """-> u (N,) f64, v (N,) f64, z (N,) f64, inmask (N,) bool.

        u/v are NaN where z <= min_depth.  inmask is the P3 frustum test.
        """
        xyz = np.asarray(xyz, dtype=np.float64)[:, :3]
        n = xyz.shape[0]
        hom = np.empty((n, 4), np.float64)
        hom[:, :3] = xyz
        hom[:, 3] = 1.0
        uvw = hom @ (self.cal.P_rect_02 @ self.cal.T_velo_rect0).T   # (N,3)
        z = uvw[:, 2]
        ok = z > self.min_depth
        u = np.full(n, np.nan)
        v = np.full(n, np.nan)
        u[ok] = uvw[ok, 0] / z[ok]
        v[ok] = uvw[ok, 1] / z[ok]
        inm = ok & (u >= 0) & (u <= self.w - 1) & (v >= 0) & (v <= self.h - 1)
        return u, v, z, inm


def _sparse_buffers(u_i, v_i, z, w, h, half_win):
    """min- and max-depth pixel buffers from sparse projected points, dilated over
    a (2*half_win+1)^2 window.  Empty pixels stay +inf / -inf."""
    pix = v_i.astype(np.int64) * w + u_i.astype(np.int64)
    zmin = np.full(w * h, np.inf)
    zmax = np.full(w * h, -np.inf)
    np.minimum.at(zmin, pix, z)
    np.maximum.at(zmax, pix, z)
    if half_win > 0:
        from scipy.ndimage import minimum_filter, maximum_filter
        sz = 2 * half_win + 1
        zmin = minimum_filter(zmin.reshape(h, w), size=sz, mode="nearest").ravel()
        zmax = maximum_filter(zmax.reshape(h, w), size=sz, mode="nearest").ravel()
    return zmin, zmax, pix


def visible_mask(u_i, v_i, z, w=IMG_W, h=IMG_H, half_win=OCC_WIN,
                 tol_abs=OCC_TOL_ABS, tol_rel=OCC_TOL_REL, memo=None):
    """P4(b) z-buffer visibility of already-in-frustum points.  True = the camera
    plausibly saw this point, so the 2D arm is allowed to label it."""
    if memo is not None and half_win in memo:
        zmin, _, pix = memo[half_win]
    else:
        zmin, zmax, pix = _sparse_buffers(u_i, v_i, z, w, h, half_win)
        if memo is not None:
            memo[half_win] = (zmin, zmax, pix)
    front = zmin[pix]
    tol = np.maximum(tol_abs, tol_rel * front)
    return z <= front + tol


def depth_edge_mask(u_i, v_i, z, w=IMG_W, h=IMG_H, half_win=EDGE_R_PX,
                    abs_m=EDGE_ABS_M, rel=EDGE_REL):
    """P7/B2 depth-discontinuity set for already-in-frustum points.  GEOMETRY ONLY:
    no GT, no prediction.  True = the pixel neighbourhood straddles a range step
    large enough for a 2D label to slide across it."""
    zmin, zmax, pix = _sparse_buffers(u_i, v_i, z, w, h, half_win)
    lo = zmin[pix]
    hi = zmax[pix]
    good = np.isfinite(lo) & np.isfinite(hi)
    step = np.where(good, hi - lo, 0.0)
    thr = np.maximum(abs_m, rel * np.where(good, lo, 0.0))
    return step > thr


def sem_boundary_mask(xyz, gt, k=SEM_K, r=SEM_R, workers=1):
    """P7/B1 semantic-boundary set.  GT ONLY: no prediction, no projection."""
    from scipy.spatial import cKDTree
    n = xyz.shape[0]
    tree = cKDTree(np.ascontiguousarray(xyz[:, :3], dtype=np.float64))
    _, idx = tree.query(xyz[:, :3], k=k + 1, distance_upper_bound=r, workers=workers)
    idx = np.atleast_2d(idx)[:, 1:]
    miss = idx >= n
    nb = gt[np.where(miss, 0, idx)]
    diff = (nb != gt[:, None]) & (~miss) & (nb != IGNORE)
    return diff.any(axis=1)


# --------------------------------------------------------------------------- #
#  I/O
# --------------------------------------------------------------------------- #
def read_scan(f):
    return np.fromfile("%s/%010d.bin" % (SCANS, f), dtype=np.float32).reshape(-1, 4)


def read_gt_coarse(f, lut):
    raw = np.fromfile("%s/%06d.label" % (LABELS, f), dtype=np.uint32) & 0xFFFF
    return lut[raw].astype(np.int16)


def frame_list(spec):
    """'all' | 'stride:N' | 'a:b:s' | 'list:1,2,3'"""
    if spec == "all":
        fr = list(range(N_FRAMES_TOTAL))
    elif spec.startswith("stride:"):
        fr = list(range(0, N_FRAMES_TOTAL, int(spec.split(":")[1])))
    elif spec.startswith("list:"):
        fr = [int(x) for x in spec.split(":", 1)[1].split(",")]
    else:
        a, b, s = (spec.split(":") + ["1"])[:3]
        fr = list(range(int(a), int(b), int(s)))
    return [f for f in fr if os.path.exists("%s/%010d.bin" % (SCANS, f))]


# --------------------------------------------------------------------------- #
#  arms
# --------------------------------------------------------------------------- #
class Arm3D(object):
    """Per-point cache in the on-disk contract.  Never abstains (unless lab==255)."""
    kind = "3d"

    def __init__(self, d, name=None, space="nusc16"):
        self.dir = d
        self.name = name or os.path.basename(d.rstrip("/"))
        self.space = space
        self.lut = (LS.nusc_lut().astype(np.int16) if space == "nusc16"
                    else np.arange(K, dtype=np.int16))

    def predict(self, f, ctx):
        z = np.load("%s/f%06d.npz" % (self.dir, f))
        lab = z["lab"].astype(np.int32)
        if "order" in z.files:                           # P0
            order = z["order"].astype(np.int64)
            raw = np.empty_like(lab)
            raw[order] = lab
            lab = raw
        n = ctx["n"]
        assert lab.shape[0] == n, ("3D arm length mismatch", f, lab.shape, n)
        out = np.full(n, ABSTAIN, np.int16)
        ok = lab < len(self.lut)
        out[ok] = self.lut[lab[ok]]
        return out


class Arm2D(object):
    """Image-space Cityscapes labels + the projection rules of P3/P4."""
    kind = "2d"

    def __init__(self, d, name=None, rider=RIDER_DEFAULT, sky_policy="wrong",
                 occlusion="zbuffer", sampling="auto", occ_win=OCC_WIN):
        self.dir = d
        self.name = name or os.path.basename(d.rstrip("/"))
        self.occ_win = int(occ_win)
        self.rider = rider
        self.sky_policy = sky_policy
        self.occlusion = occlusion
        self.sampling = sampling
        self.lut = city19_to_coarse(rider, sky_policy)
        self.sky_hits = 0
        self.occluded = 0
        self.in_frustum = 0
        self.ignorefill_hits = 0

    # ---------------------------------------------------------------- #
    def _sample(self, z, u, v, inm):
        """-> (city trainId per in-frustum point).  P4(a)."""
        seg = z["seg"]
        sh, sw = seg.shape
        assert sh >= IMG_H and sw >= IMG_W, ("seg shape smaller than native", seg.shape)
        # The label map may be cached at the network's OWN (up-sampled) resolution
        # rather than on the native 1226x370 grid.  Sampling it at the exact sub-pixel
        # projection then keeps sub-native-pixel precision -- identical to
        # seg2d_infer.sample_points, and strictly more information than rounding onto
        # the native grid first.  [PRO-2D]  When sh,sw == IMG_H,IMG_W this reduces
        # exactly to the nearest-native-pixel rule.
        sx, sy = sw / float(IMG_W), sh / float(IMG_H)
        use_prob = ("prob" in z.files) and self.sampling in ("auto", "prob_bilinear")
        if use_prob and self.sampling == "auto":
            # `auto` must never DOWNGRADE the 2D arm.  A probability field stored at
            # reduced resolution is blurrier than the full-resolution argmax map at
            # exactly the boundaries this experiment measures, so auto only takes the
            # prob path when the field is at least as fine as the label map.
            ph, pw = z["prob"].shape[-2:]
            use_prob = (int(ph) >= sh and int(pw) >= sw)
        ui = np.clip(np.rint((u[inm] + 0.5) * sx - 0.5), 0, sw - 1).astype(np.int64)
        vi = np.clip(np.rint((v[inm] + 0.5) * sy - 0.5), 0, sh - 1).astype(np.int64)
        self.seg_hw = (int(sh), int(sw))
        if not use_prob:
            return seg[vi, ui].astype(np.int32), (
                "subpixel_on_%dx%d" % (sh, sw) if (sh, sw) != (IMG_H, IMG_W)
                else "nearest")
        p = z["prob"].astype(np.float32)                 # (19, h, w)
        c, hh, ww = p.shape
        # bilinear sample the probability field at the exact sub-pixel location
        x = u[inm] * (ww - 1) / (IMG_W - 1)
        y = v[inm] * (hh - 1) / (IMG_H - 1)
        x0 = np.floor(x).astype(np.int64)
        y0 = np.floor(y).astype(np.int64)
        x1 = np.minimum(x0 + 1, ww - 1)
        y1 = np.minimum(y0 + 1, hh - 1)
        ax = (x - x0)[None, :]
        ay = (y - y0)[None, :]
        val = (p[:, y0, x0] * (1 - ax) * (1 - ay) + p[:, y0, x1] * ax * (1 - ay) +
               p[:, y1, x0] * (1 - ax) * ay + p[:, y1, x1] * ax * ay)
        return np.argmax(val, axis=0).astype(np.int32), "prob_bilinear"

    # ---------------------------------------------------------------- #
    def predict(self, f, ctx, occlusion=None):
        occlusion = self.occlusion if occlusion is None else occlusion
        z = np.load("%s/f%06d.npz" % (self.dir, f))
        u, v, depth, inm = ctx["u"], ctx["v"], ctx["z"], ctx["inm"]
        out = np.full(ctx["n"], ABSTAIN, np.int16)
        if not inm.any():
            return out
        city, self.mode = self._sample(z, u, v, inm)
        lab = np.where(city < 20, city, 19)
        coarse = self.lut[lab]
        if occlusion == "zbuffer":                       # P4(b)
            vis = visible_mask(np.rint(u[inm]).astype(np.int64),
                               np.rint(v[inm]).astype(np.int64), depth[inm],
                               half_win=self.occ_win, memo=ctx.setdefault("_zbuf", {}))
            coarse = np.where(vis, coarse, ABSTAIN)
            self.occluded += int((~vis).sum())
        self.sky_hits += int((city == SKY_ID).sum())
        self.ignorefill_hits += int((city >= 19).sum())
        self.in_frustum += int(inm.sum())
        out[inm] = coarse
        return out


class ArmHybrid(object):
    """C: the 2D label wherever the 2D arm speaks, the 3D label everywhere else.
    Coverage is 1.0 by construction whenever the 3D arm never abstains."""
    kind = "hybrid"

    def __init__(self, a2, a3, name="hybrid"):
        self.a2, self.a3, self.name = a2, a3, name

    def predict(self, f, ctx):
        p2 = self.a2.predict(f, ctx)
        p3 = self.a3.predict(f, ctx)
        return np.where(p2 >= 0, p2, p3)


# --------------------------------------------------------------------------- #
#  accumulation and metrics
# --------------------------------------------------------------------------- #
class Acc(object):
    """Confusion matrix over points the arm ANSWERED, plus an abstention count per
    GT class.  Both abstention conventions of P5 are recoverable from (C, A) in one
    pass.

    THREE outcomes per point, not two -- this is the load-bearing detail:
      * ANSWERED, nameable      -> C[gt, pred],  pred in 0..K-1
      * ANSWERED, UNMAPPED(-2)  -> C[gt, K].  The arm DID answer; the common space
                                   just cannot name what it said (2D `sky`, 3D
                                   `other_flat`).  It is WRONG under both conventions
                                   and is NEVER removed from any denominator.
      * ABSTAINED(-1)           -> A[gt].  No answer at all (outside the frustum,
                                   occluded, or the seg map's own ignore fill).
                                   Wrong under (i), removed from the denominator
                                   under (ii).
    Collapsing UNMAPPED into ABSTAIN would open the abstention leak: an arm could
    delete its own errors by emitting its dead class.
    """

    def __init__(self):
        self.C = np.zeros((K, K + 1), np.int64)      # last column = UNMAPPED
        self.A = np.zeros(K, np.int64)

    def add(self, gt, pred):
        if gt.size == 0:
            return
        ab = pred == ABSTAIN
        if ab.any():
            self.A += np.bincount(gt[ab].astype(np.int64), minlength=K)
        g = gt[~ab].astype(np.int64)
        p = pred[~ab].astype(np.int64)
        if g.size:
            p = np.where(p == UNMAPPED, K, p)
            assert p.min() >= 0 and p.max() <= K, ("prediction out of range", p.min(), p.max())
            self.C += np.bincount(g * (K + 1) + p, minlength=K * (K + 1)).reshape(K, K + 1)

    def merge(self, o):
        self.C += o.C
        self.A += o.A
        return self

    # ---------------------------------------------------------------- #
    def metrics(self, class_subsets):
        C, A = self.C, self.A
        row = C.sum(1)                 # answered GT points per class (UNMAPPED incl.)
        col = C[:, :K].sum(0)          # nameable predictions per class
        diag = np.diag(C[:, :K])
        n_ans = int(C.sum())           # points the arm answered
        n_unm = int(C[:, K].sum())     # ...of which the space cannot name
        n_ab = int(A.sum())
        n = n_ans + n_ab
        present = (row + A) > 0

        def _iou(strict):
            out = {}
            for k in range(K):
                if not present[k]:
                    continue
                union = row[k] + col[k] - diag[k] + (A[k] if strict else 0)
                out[COARSE[k]] = (100.0 * diag[k] / union) if union else 0.0
            return out

        iou_ab = _iou(False)
        iou_st = _iou(True)

        def _miou(iou, names):
            v = [iou[x] for x in names if x in iou]
            return float(np.mean(v)) if v else float("nan")

        res = {
            "n_eval": n,
            "n_labelled": n_ans,
            "n_unmapped_pred": n_unm,
            "n_abstained": n_ab,
            "coverage": (n_ans / n) if n else float("nan"),
            "acc_abstain_excluded": (100.0 * diag.sum() / n_ans) if n_ans else float("nan"),
            "acc_abstain_wrong": (100.0 * diag.sum() / n) if n else float("nan"),
            "per_class_iou_abstain_excluded": {k: round(v, 3) for k, v in iou_ab.items()},
            "per_class_iou_abstain_wrong": {k: round(v, 3) for k, v in iou_st.items()},
            "classes_present": [COARSE[k] for k in range(K) if present[k]],
            "gt_count_per_class": {COARSE[k]: int(row[k] + A[k])
                                   for k in range(K) if present[k]},
        }
        for sname, names in class_subsets.items():
            res["miou_%s_abstain_excluded" % sname] = _miou(iou_ab, names)
            res["miou_%s_abstain_wrong" % sname] = _miou(iou_st, names)
            res["n_classes_%s" % sname] = len([x for x in names if x in iou_ab])
        return res


# --------------------------------------------------------------------------- #
#  the scoring pass
# --------------------------------------------------------------------------- #
SUBSETS = ["global", "frustum", "outside",
           "sem_boundary", "sem_interior",
           "depth_edge", "depth_interior"] + \
          ["range_%s" % t for t in ["0_10", "10_20", "20_30", "30_50", "50_inf"]] + \
          ["frustum_range_%s" % t for t in ["0_10", "10_20", "20_30", "30_50", "50_inf"]]

RANGE_TAGS = ["0_10", "10_20", "20_30", "30_50", "50_inf"]


def _bcache_path(bcache, f):
    return "%s/b%06d.npz" % (bcache, f)


def frame_context(f, proj, lut, bcache=None, need_boundary=True, workers=1):
    pts = read_scan(f)
    xyz = pts[:, :3].astype(np.float64)
    gt = read_gt_coarse(f, lut)
    assert len(gt) == len(xyz), ("scan/label length mismatch", f, len(xyz), len(gt))
    u, v, z, inm = proj.project(xyz)
    rng = np.linalg.norm(xyz, axis=1)
    valid = gt != IGNORE                                  # P2

    sem_b = None
    if need_boundary:
        p = _bcache_path(bcache, f) if bcache else None
        if p and os.path.exists(p):
            d = np.load(p)
            sem_b = np.unpackbits(d["sem_b"], count=len(xyz)).astype(bool)
        else:
            sem_b = sem_boundary_mask(xyz, gt, workers=workers)
            if p:
                os.makedirs(bcache, exist_ok=True)
                np.savez_compressed(p, sem_b=np.packbits(sem_b))

    dedge = np.zeros(len(xyz), bool)
    if inm.any():
        dedge[inm] = depth_edge_mask(np.rint(u[inm]).astype(np.int64),
                                     np.rint(v[inm]).astype(np.int64), z[inm])

    masks = {
        "global": valid,
        "frustum": valid & inm,
        "outside": valid & ~inm,
        "depth_edge": valid & inm & dedge,
        "depth_interior": valid & inm & ~dedge,
    }
    if sem_b is not None:
        masks["sem_boundary"] = valid & sem_b
        masks["sem_interior"] = valid & ~sem_b
    for i, t in enumerate(RANGE_TAGS):
        lo, hi = RANGE_EDGES[i], RANGE_EDGES[i + 1]
        rb = (rng >= lo) & (rng < hi)
        masks["range_%s" % t] = valid & rb
        masks["frustum_range_%s" % t] = valid & inm & rb

    return dict(n=len(xyz), xyz=xyz, pts=pts, gt=gt, u=u, v=v, z=z, inm=inm,
                rng=rng, valid=valid, masks=masks)


def score_pass(frames, arms, proj, lut, bcache=None, need_boundary=True,
               workers=1, progress=True):
    accs = {a.name: {s: Acc() for s in SUBSETS} for a in arms}
    n_done = 0
    for f in frames:
        ctx = frame_context(f, proj, lut, bcache, need_boundary, workers)
        for a in arms:
            pred = a.predict(f, ctx)
            gt = ctx["gt"]
            for s, m in ctx["masks"].items():
                accs[a.name][s].add(gt[m], pred[m])
        n_done += 1
        if progress and n_done % 50 == 0:
            print("  %d/%d frames" % (n_done, len(frames)), flush=True)
    return accs


def class_subsets_for(accs_global):
    """Under R1 `all` and `nine` coincide by construction -- every class of the
    common space is expressible in every vocabulary.  `frequent` is the robustness
    variant: mIoU over 9 gives a 0.25 %-support class the same weight as a 27 % one."""
    present = accs_global["classes_present"]
    allc = [c for c in COARSE if c in present]
    freq = [c for c in FREQUENT if c in present]
    return {"all": allc, "nine": allc, "frequent": freq}


# --------------------------------------------------------------------------- #
#  metric 7 -- accumulated map coverage  (P11)
# --------------------------------------------------------------------------- #
def mapcov(frames, proj, voxel=0.2, out=None):
    """Fraction of OCCUPIED MAP VOXELS ever seen by the camera over the trajectory.
    Per-scan coverage is 16 %; this is the number that answers 'but the camera
    sweeps as you drive'.  Poses: KITTI odometry GT (sequences/07/poses.txt),
    X_world = P_i @ Tr @ X_velo with Tr from sequences/07/calib.txt."""
    P = np.loadtxt(SEQ + "/poses.txt").reshape(-1, 3, 4)
    Tr = None
    for line in open(SEQ + "/calib.txt"):
        if line.startswith("Tr:"):
            Tr = np.array([float(x) for x in line.split()[1:]]).reshape(3, 4)
    assert Tr is not None
    Tr4 = np.vstack([Tr, [0, 0, 0, 1.0]])

    seen = {}
    for f in frames:
        pts = read_scan(f)
        xyz = pts[:, :3].astype(np.float64)
        _, _, _, inm = proj.project(xyz)
        T = np.vstack([P[f], [0, 0, 0, 1.0]]) @ Tr4
        wp = (np.hstack([xyz, np.ones((len(xyz), 1))]) @ T.T)[:, :3]
        key = np.floor(wp / voxel).astype(np.int64)
        kk = (key[:, 0] * 1000003 + key[:, 1]) * 1000033 + key[:, 2]
        ku, inv = np.unique(kk, return_inverse=True)
        nobs = np.bincount(inv, minlength=len(ku))
        ncam = np.bincount(inv, weights=inm.astype(np.float64), minlength=len(ku))
        for i, k in enumerate(ku):
            a = seen.get(k)
            if a is None:
                seen[k] = [int(nobs[i]), int(ncam[i])]
            else:
                a[0] += int(nobs[i])
                a[1] += int(ncam[i])
    arr = np.array(list(seen.values()), np.int64)
    ever = float((arr[:, 1] > 0).mean())
    frac = float((arr[:, 1] / arr[:, 0]).mean())
    res = dict(voxel_m=voxel, frames=len(frames), n_voxels=int(len(arr)),
               voxels_ever_camera_covered=ever,
               mean_fraction_of_observations_camera_covered=frac)
    print(json.dumps(res, indent=2))
    if out:
        json.dump(res, open(out, "w"), indent=2)
    return res


# --------------------------------------------------------------------------- #
#  aggregation over repeats
# --------------------------------------------------------------------------- #
def aggregate(reps):
    """reps: list of {arm: {subset: metricdict}} -> {arm:{subset:{metric:
    {mean, half_range, values}}}}.  Half-range, not stddev: with 3 repeats the
    stddev is a worse estimate than the observed spread, and the spread is what a
    reader needs in order to know whether a gap is real (P9)."""
    out = {}
    for arm in reps[0]:
        out[arm] = {}
        for sub in reps[0][arm]:
            out[arm][sub] = {}
            for met, v in reps[0][arm][sub].items():
                if not isinstance(v, (int, float)) or isinstance(v, bool):
                    out[arm][sub][met] = v
                    continue
                vals = [r[arm][sub][met] for r in reps]
                out[arm][sub][met] = {
                    "mean": float(np.mean(vals)),
                    "half_range": float((np.max(vals) - np.min(vals)) / 2.0),
                    "values": [float(x) for x in vals],
                }
    return out


# --------------------------------------------------------------------------- #
#  self-test -- NO GPU, NO MODEL, NO CACHE
# --------------------------------------------------------------------------- #
def _synth_2d_cache(frames, proj, lut, outdir, leak=False):
    """Build a 2D arm out of GT: project GT coarse labels into the image with a
    z-buffer (nearest point wins the pixel) and map them to Cityscapes trainIds.
    An honest scorer must give this arm ~100 % in-frustum accuracy."""
    os.makedirs(outdir, exist_ok=True)
    inv = np.array([COARSE_TO_CITY19[c] for c in COARSE], np.int32)
    for f in frames:
        pts = read_scan(f)
        gt = read_gt_coarse(f, lut)
        u, v, z, inm = proj.project(pts[:, :3].astype(np.float64))
        seg = np.full((IMG_H, IMG_W), 255, np.uint8)
        idx = np.nonzero(inm)[0]
        # paint far points first so the nearest point wins the pixel
        idx = idx[np.argsort(-z[idx])]
        ui = np.rint(u[idx]).astype(np.int64)
        vi = np.rint(v[idx]).astype(np.int64)
        g = gt[idx]
        keep = g != IGNORE
        seg[vi[keep], ui[keep]] = inv[g[keep]].astype(np.uint8)
        if leak:
            # emulate 2D->3D boundary leakage: let labels bleed outward by one
            # pixel.  255 is the ignore fill and grey_dilation takes the max, so
            # it has to be moved out of the way first or it swallows the image.
            from scipy.ndimage import grey_dilation
            t = seg.astype(np.int16)
            t[t == 255] = -1
            t = grey_dilation(t, size=(3, 3))
            t[t < 0] = 255
            seg = t.astype(np.uint8)
        np.savez_compressed("%s/f%06d.npz" % (outdir, f), seg=seg)


def _synth_3d_cache(frames, lut, outdir, mode="oracle", seed=0):
    """mode: oracle (GT in nuScenes space) | constant (always nuScenes 'car')."""
    os.makedirs(outdir, exist_ok=True)
    n2c = LS.nusc_lut()
    back = {}
    for i in range(16):
        if int(n2c[i]) >= 0:                      # skip the 3D arm's dead class
            back.setdefault(int(n2c[i]), i)
    inv = np.array([back.get(k, 14) for k in range(K)], np.uint8)
    rs = np.random.RandomState(seed)
    for f in frames:
        gt = read_gt_coarse(f, lut)
        if mode == "oracle":
            lab = inv[np.where(gt >= 0, gt, 0)]
        elif mode == "constant":
            lab = np.full(len(gt), 3, np.uint8)           # nuScenes 'car'
        else:
            lab = rs.randint(0, 16, len(gt)).astype(np.uint8)
        order = rs.permutation(len(gt)).astype(np.int32)  # exercise P0
        np.savez_compressed("%s/f%06d.npz" % (outdir, f),
                            lab=lab[order], order=order)


def selftest(tmp=None, frames=None, verbose=True):
    import tempfile
    tmp = tmp or tempfile.mkdtemp(prefix="score2d3d_")
    frames = frames or [0, 137, 500, 913]
    proj = Projector()
    lut = LS.sk_lut()
    fails = []

    n_chk = [0]

    def chk(name, cond, detail=""):
        n_chk[0] += 1
        if verbose:
            print("  %-58s %s %s" % (name, "PASS" if cond else "FAIL", detail))
        if not cond:
            fails.append("%s %s" % (name, detail))

    # ---- unit: LUTs ------------------------------------------------- #
    lutc = city19_to_coarse()
    chk("common space has 9 classes", K == 9 and COARSE == list(LS.COARSE))
    chk("city19 LUT names 18 of 19 trainIds", int((lutc[:19] >= 0).sum()) == 18,
        "(only sky is unnameable)")
    chk("city19 sky -> UNMAPPED, not ABSTAIN", lutc[SKY_ID] == UNMAPPED)
    chk("city19 trainId 255 slot -> ABSTAIN", lutc[19] == ABSTAIN)
    chk("city19 rider/bicycle/motorcycle all -> two_wheeler",
        lutc[12] == _C["two_wheeler"] and lutc[18] == _C["two_wheeler"] and
        lutc[17] == _C["two_wheeler"])
    n3 = LS.nusc_lut()
    chk("nusc other_flat -> UNMAPPED", n3[LS.NUSC_OTHER_FLAT_ID] == UNMAPPED)
    chk("exactly one dead class per arm",
        int((n3 == UNMAPPED).sum()) == 1 and int((lutc[:19] == UNMAPPED).sum()) == 1)
    chk("label_spaces self-verify", LS.verify())
    chk("city19 road/sidewalk/veg/terrain 1:1",
        lutc[0] == _C["road"] and lutc[1] == _C["sidewalk"] and
        lutc[8] == _C["vegetation"] and lutc[9] == _C["terrain"])

    # ---- unit: Acc identities --------------------------------------- #
    a = Acc()
    g = np.array([0, 0, 1, 1, 2], np.int16)
    p = np.array([0, 1, 1, -1, 2], np.int16)
    a.add(g, p)
    m = a.metrics({"all": COARSE})
    chk("Acc: n_eval == points in", m["n_eval"] == 5)
    chk("Acc: abstain counted once", m["n_abstained"] == 1)
    chk("Acc: coverage", abs(m["coverage"] - 4.0 / 5.0) < 1e-12)
    chk("Acc: acc_abstain_excluded", abs(m["acc_abstain_excluded"] - 75.0) < 1e-9,
        "(3 of 4 labelled correct)")
    chk("Acc: acc_abstain_wrong", abs(m["acc_abstain_wrong"] - 60.0) < 1e-9,
        "(3 of 5)")
    # class 1: TP=1 FP=1 FN=0 abstain=1 -> IoU_ex = 1/2, IoU_strict = 1/3
    chk("Acc: IoU abstain-excluded",
        abs(m["per_class_iou_abstain_excluded"]["large_vehicle"] - 50.0) < 1e-6)
    chk("Acc: IoU abstain-wrong (abstention is an FN)",
        abs(m["per_class_iou_abstain_wrong"]["large_vehicle"] - 100.0 / 3.0) < 1e-3)
    # THE ABSTENTION LEAK, closed: an UNMAPPED answer is wrong under BOTH conventions
    # and is removed from NEITHER denominator.
    au = Acc()
    au.add(np.array([4, 4, 4, 4], np.int16),
           np.array([4, 4, 4, UNMAPPED], np.int16))
    mu = au.metrics({"all": COARSE})
    chk("Acc: UNMAPPED does not abstain", mu["n_abstained"] == 0 and
        mu["n_unmapped_pred"] == 1 and mu["coverage"] == 1.0)
    chk("Acc: UNMAPPED is wrong under BOTH conventions",
        abs(mu["acc_abstain_excluded"] - 75.0) < 1e-9 and
        abs(mu["acc_abstain_wrong"] - 75.0) < 1e-9)
    chk("Acc: UNMAPPED is an FN for the true class, FP for nothing",
        abs(mu["per_class_iou_abstain_excluded"]["road"] - 75.0) < 1e-9)
    ax = Acc()
    ax.add(np.array([4, 4, 4, 4], np.int16),
           np.array([4, 4, 4, ABSTAIN], np.int16))
    mx = ax.metrics({"all": COARSE})
    chk("Acc: ABSTAIN and UNMAPPED differ exactly under convention (ii)",
        abs(mx["acc_abstain_excluded"] - 100.0) < 1e-9 and
        abs(mx["acc_abstain_wrong"] - mu["acc_abstain_wrong"]) < 1e-9,
        "(abstain 100.00/75.00 vs unmapped 75.00/75.00)")
    a2 = Acc()
    a2.add(g, np.array([0, 0, 1, 1, 2], np.int16))
    m2 = a2.metrics({"all": COARSE})
    chk("Acc: conventions coincide at coverage 1",
        m2["coverage"] == 1.0 and
        m2["acc_abstain_excluded"] == m2["acc_abstain_wrong"] and
        m2["miou_all_abstain_excluded"] == m2["miou_all_abstain_wrong"])

    # ---- unit: geometry --------------------------------------------- #
    u, v, z, inm = proj.project(np.array([[5.0, 0.0, 0.0], [-5.0, 0.0, 0.0]]))
    chk("projector: point in front is in frustum", bool(inm[0]))
    chk("projector: point behind camera is not", not bool(inm[1]))
    chk("projector: principal-ray pixel near (cx,cy)",
        abs(u[0] - 601.8873) < 30 and abs(v[0] - 183.1104) < 30,
        "(u=%.1f v=%.1f)" % (u[0], v[0]))
    # occlusion: two points on the same pixel, 5 m apart -> far one abstains
    vis = visible_mask(np.array([100, 100]), np.array([100, 100]),
                       np.array([5.0, 10.0]))
    chk("occlusion: far point on same pixel is occluded",
        bool(vis[0]) and not bool(vis[1]))
    vis = visible_mask(np.array([100, 100]), np.array([100, 100]),
                       np.array([5.0, 5.2]))
    chk("occlusion: same-surface pair within tolerance both visible",
        bool(vis[0]) and bool(vis[1]))
    de = depth_edge_mask(np.array([100, 103]), np.array([100, 100]),
                         np.array([5.0, 30.0]))
    chk("depth_edge: range step flagged", bool(de[0]) and bool(de[1]))
    de = depth_edge_mask(np.array([100, 103]), np.array([100, 100]),
                         np.array([5.0, 5.1]))
    chk("depth_edge: flat surface not flagged",
        (not bool(de[0])) and (not bool(de[1])))
    xyz = np.array([[0, 0, 0], [0.1, 0, 0], [9, 9, 9]], np.float64)
    gtb = np.array([0, 1, 0], np.int16)
    sb = sem_boundary_mask(xyz, gtb)
    chk("sem_boundary: adjacent different-class pair flagged",
        bool(sb[0]) and bool(sb[1]) and not bool(sb[2]))

    # ---- integration: synthetic arms on real frames ------------------ #
    d2 = os.path.join(tmp, "seg2d")
    d3o = os.path.join(tmp, "p3d_oracle")
    d3c = os.path.join(tmp, "p3d_const")
    _synth_2d_cache(frames, proj, lut, d2)
    _synth_3d_cache(frames, lut, d3o, "oracle")
    _synth_3d_cache(frames, lut, d3c, "constant")

    a2d = Arm2D(d2, "oracle2d")
    a3d = Arm3D(d3o, "oracle3d")
    a3c = Arm3D(d3c, "const3d")
    hyb = ArmHybrid(a2d, a3d, "hybrid")
    accs = score_pass(frames, [a2d, a3d, a3c, hyb], proj, lut,
                      need_boundary=True, progress=False)
    R = {n: {s: accs[n][s].metrics({"all": COARSE, "nine": NINE})
             for s in accs[n]} for n in accs}

    chk("oracle 3D arm: coverage 1.0", R["oracle3d"]["global"]["coverage"] == 1.0)
    chk("oracle 3D arm: 100 % accuracy (P0 permutation inverted)",
        abs(R["oracle3d"]["global"]["acc_abstain_excluded"] - 100.0) < 1e-9)
    chk("oracle 3D arm: 100 % mIoU",
        abs(R["oracle3d"]["global"]["miou_all_abstain_excluded"] - 100.0) < 1e-9)
    chk("oracle 2D arm: in-frustum accuracy > 97 %",
        R["oracle2d"]["frustum"]["acc_abstain_excluded"] > 97.0,
        "(%.2f %%)" % R["oracle2d"]["frustum"]["acc_abstain_excluded"])
    chk("oracle 2D arm: labels nothing outside the frustum",
        R["oracle2d"]["outside"]["n_labelled"] == 0)
    cov = R["oracle2d"]["global"]["coverage"]
    chk("oracle 2D arm: global coverage in [0.12, 0.18]", 0.12 < cov < 0.18,
        "(%.4f)" % cov)
    chk("oracle 2D arm: no UNMAPPED leak in the synthetic arm",
        R["oracle2d"]["frustum"]["n_unmapped_pred"] == 0)
    chk("hybrid: coverage 1.0", R["hybrid"]["global"]["coverage"] == 1.0)
    ng = R["oracle3d"]["global"]["n_eval"]
    chk("subsets partition: frustum + outside == global",
        R["oracle3d"]["frustum"]["n_eval"] + R["oracle3d"]["outside"]["n_eval"] == ng)
    chk("subsets partition: sem_boundary + sem_interior == global",
        R["oracle3d"]["sem_boundary"]["n_eval"] +
        R["oracle3d"]["sem_interior"]["n_eval"] == ng)
    chk("subsets partition: depth_edge + depth_interior == frustum",
        R["oracle3d"]["depth_edge"]["n_eval"] +
        R["oracle3d"]["depth_interior"]["n_eval"] ==
        R["oracle3d"]["frustum"]["n_eval"])
    chk("subsets partition: range bins == global",
        sum(R["oracle3d"]["range_%s" % t]["n_eval"] for t in RANGE_TAGS) == ng)
    chk("2D and 3D arms share the same denominator",
        R["oracle2d"]["global"]["n_eval"] == ng and
        R["hybrid"]["global"]["n_eval"] == ng)
    ac = R["const3d"]["global"]["acc_abstain_excluded"]
    gc = R["const3d"]["global"]["gt_count_per_class"]
    chk("constant arm scores exactly the class frequency",
        abs(ac - 100.0 * gc["car"] / ng) < 1e-6, "(%.3f %%)" % ac)
    chk("constant arm mIoU is low", R["const3d"]["global"]["miou_all_abstain_excluded"] < 15.0)

    # leakage sanity: dilating the synthetic seg must hurt the boundary set more
    d2l = os.path.join(tmp, "seg2d_leak")
    _synth_2d_cache(frames, proj, lut, d2l, leak=True)
    a2l = Arm2D(d2l, "leak2d")
    accs2 = score_pass(frames, [a2l], proj, lut, need_boundary=False, progress=False)
    L = {s: accs2["leak2d"][s].metrics({"all": COARSE}) for s in accs2["leak2d"]}
    de = L["depth_edge"]["acc_abstain_excluded"]
    di = L["depth_interior"]["acc_abstain_excluded"]
    chk("leakage probe: dilated 2D is worse on depth_edge than interior", de < di,
        "(edge %.2f %% vs interior %.2f %%)" % (de, di))

    covs = []
    for wsz in [0, 1, 2, 3]:
        aw = Arm2D(d2, "w%d" % wsz, occ_win=wsz)
        acw = score_pass(frames, [aw], proj, lut, need_boundary=False, progress=False)
        covs.append(acw["w%d" % wsz]["frustum"].metrics({"all": COARSE})["coverage"])
    chk("occ-win sweep: coverage falls monotonically with window",
        all(covs[i] >= covs[i + 1] for i in range(3)),
        "(%s)" % ", ".join("w%d %.4f" % (i, c) for i, c in enumerate(covs)))

    a2n = Arm2D(d2, "naive2d", occlusion="none")
    accs3 = score_pass(frames, [a2n], proj, lut, need_boundary=False, progress=False)
    N = {s: accs3["naive2d"][s].metrics({"all": COARSE}) for s in accs3["naive2d"]}
    chk("occlusion=none labels strictly more points than zbuffer",
        N["frustum"]["n_labelled"] > R["oracle2d"]["frustum"]["n_labelled"],
        "(%d vs %d)" % (N["frustum"]["n_labelled"],
                        R["oracle2d"]["frustum"]["n_labelled"]))
    chk("occlusion=none is less accurate than zbuffer (occluded points are wrong)",
        N["frustum"]["acc_abstain_excluded"] <
        R["oracle2d"]["frustum"]["acc_abstain_excluded"],
        "(%.3f %% vs %.3f %%)" % (N["frustum"]["acc_abstain_excluded"],
                                  R["oracle2d"]["frustum"]["acc_abstain_excluded"]))

    # ---- aggregate ---------------------------------------------------- #
    ag = aggregate([R, R])
    chk("aggregate: zero spread over identical repeats",
        ag["oracle3d"]["global"]["acc_abstain_excluded"]["half_range"] == 0.0)

    print("\nSELFTEST: %s (%d checks)" %
          ("ALL PASS" if not fails else "%d FAILED" % len(fails), n_chk[0]))
    for f in fails:
        print("   FAILED: " + f)
    return len(fails) == 0


# --------------------------------------------------------------------------- #
#  driver
# --------------------------------------------------------------------------- #
def build_protocol(args, frames, arms_meta):
    return {
        "question": "2D Cityscapes segmentation projected onto points vs 3D PTv3 on "
                    "the point cloud, on SemanticKITTI seq 07",
        "sequence": "KITTI raw %s == SemanticKITTI seq %s, index-aligned, offset 0"
                    % (SEQUENCES[SEQ_ID][0], SEQ_ID),
        "sequence_role": ("EVALUATION" if SEQ_ID == "07"
                          else "TUNING (disjoint from the evaluation sequence)"),
        "frames_spec": args.frames,
        "n_frames": len(frames),
        "frames_first_last": [frames[0], frames[-1]] if frames else [],
        "label_space": {
            "name": "common-9 (label_spaces.COARSE)", "classes": COARSE,
            "rule": "every class is natively expressible in ALL THREE vocabularies "
                    "and present in seq07 GT; neither arm is scored on a class its "
                    "own taxonomy cannot name",
            "gt": "SemanticKITTI via label_spaces.sk_lut()",
            "3d": "nuScenes-16 via label_spaces.nusc_lut()",
            "2d": "Cityscapes-19 via label_spaces.cs_lut() / city19_to_coarse()",
            "sentinels": {
                "EXCLUDED(-1)": "GT-side: the point is dropped for BOTH arms",
                "UNMAPPED(-2)": "prediction-side: always wrong, never dropped "
                                "(2D sky, 3D other_flat)",
                "ABSTAIN(-1 in prediction arrays)": "no answer at all"}},
        "ignore_rule": "points whose GT coarse class is IGNORE are excluded from every "
                       "denominator for every arm (P2)",
        "frustum": {"image_wh": [IMG_W, IMG_H], "min_depth_m": MIN_DEPTH,
                    "fx": 707.0912, "fy": 707.0912, "cx": 601.8873, "cy": 183.1104,
                    "distortion": "none (images are rectified)",
                    "rule": "z > min_depth and 0 <= u <= W-1 and 0 <= v <= H-1",
                    "measured_coverage_101_frames": 0.16041},
        "2d_label_rule": {"sampling": args.sampling, "occlusion": args.occlusion,
                          "occ_win_px": getattr(args, "occ_win", OCC_WIN),
                          "occ_tol_abs_m": OCC_TOL_ABS,
                          "occ_tol_rel": OCC_TOL_REL,
                          "sky_policy": args.sky, "rider_maps_to": args.rider},
        "abstention_conventions": {
            "abstain_wrong": "unlabelled counted as wrong; denominator = all evaluated "
                             "points (what a mapping system needs)",
            "abstain_excluded": "unlabelled excluded from the denominator (the "
                                "convention that is fair to the 2D arm)"},
        "class_subsets": {
            "all": "every one of the 9 common classes present in GT",
            "nine": NINE,
            "frequent": "%s -- classes with >=1 %% of in-frustum GT support"
                        % (FREQUENT,)},
        "boundary_sets": {
            "sem_boundary": "GT-only: any of k=%d nearest 3D neighbours within %.2f m "
                            "carries a different non-ignore GT coarse class"
                            % (SEM_K, SEM_R),
            "depth_edge": "geometry-only, in-frustum: max-min LiDAR depth over an "
                          "%dx%d pixel window exceeds max(%.1f m, %.2f * near depth)"
                          % (2 * EDGE_R_PX + 1, 2 * EDGE_R_PX + 1, EDGE_ABS_M, EDGE_REL),
            "independence": "neither set uses any arm's predictions"},
        "range_bins_m": [[0, 10], [10, 20], [20, 30], [30, 50], [50, "inf"]],
        "range_axis": "LiDAR range ||xyz||, not camera depth",
        "repeats": args.repeats_note,
        "instrument_noise": "3D arm is non-deterministic: ~4-5 % of points change label "
                            "between two forwards; ~1 pt mIoU spread on 20 frames. "
                            "Half-range over repeats is reported for every number.",
        "pro_2d_concessions": [
            "Cityscapes-pretrained model, not ADE20K: KITTI is far closer to "
            "Cityscapes than to nuScenes, so the 2D arm gets the SMALLER domain gap.",
            "SemanticKITTI's 19 classes were defined after Cityscapes, so the 2D "
            "taxonomy maps almost 1:1 while nuScenes-16 does not.",
            "occlusion=zbuffer: points the camera cannot see are treated as the 2D "
            "arm abstaining, not as wrong answers.",
            "SKY IS NOT A CONCESSION IN THE HEADLINE RUN: sky_policy=wrong, i.e. a "
            "point landing on a sky pixel is UNMAPPED and counted WRONG under BOTH "
            "conventions.  sky_policy=abstain (which would let the 2D arm delete "
            "those errors) is reported only as a separate sensitivity row.",
            "R1 common-9 label space: SemanticKITTI other-vehicle (20), parking (44), "
            "other-ground (49), other-structure (52) and other-object (99) are all "
            "EXCLUDED from scoring because Cityscapes-19 either has no class for them "
            "or was trained with them ignored.  5.76 % of all points / 5.49 % "
            "in-frustum.  Every one of those exclusions removes points the 2D arm "
            "would have been charged for.",
            "rider / bicycle / motorcycle merged into two_wheeler, which is what all "
            "three taxonomies agree on -- the Cityscapes rider/vehicle split can no "
            "longer cost the 2D arm anything.",
            "prob_bilinear sampling used whenever the 2D arm ships probabilities.",
            "sub-pixel sampling on the UPSAMPLED label map (seg2d_infer.sample_points), "
            "not nearest-neighbour on the native 1226x370 grid.",
            "input scale and TTA chosen by MEASUREMENT on seq 04, a disjoint drive on "
            "the same rig and the same calibration day, then frozen.",
            "both 2D checkpoints run (EoMT-L 84.2 and Mask2Former Swin-L 83.3 "
            "Cityscapes val mIoU, single-scale) and the stronger quoted per metric.",
            "abstain_excluded convention reported alongside abstain_wrong.",
        ],
        "arms": arms_meta,
        "code_sha1": hashlib.sha1(
            open(os.path.abspath(__file__), "rb").read()).hexdigest(),
    }


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    for name in ("selftest", "prep", "mapcov", "score"):
        pass

    p = sub.add_parser("selftest")
    p.add_argument("--frames", default="list:0,137,500,913")
    p.add_argument("--seq", default="07", choices=sorted(SEQUENCES))

    p = sub.add_parser("prep", help="precompute the sem_boundary cache (CPU, slow)")
    p.add_argument("--frames", default="all")
    p.add_argument("--bcache", default=ROOT + "/opt/out2d/bcache")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--seq", default="07", choices=sorted(SEQUENCES))

    p = sub.add_parser("mapcov")
    p.add_argument("--frames", default="all")
    p.add_argument("--seq", default="07", choices=sorted(SEQUENCES))
    p.add_argument("--voxel", type=float, default=0.2)
    p.add_argument("--out", default=None)

    p = sub.add_parser("score")
    p.add_argument("--arm3d", default="", help="comma-separated dirs, one per repeat")
    p.add_argument("--arm2d", default="", help="comma-separated dirs, one per repeat")
    p.add_argument("--frames", default="all")
    p.add_argument("--bcache", default=ROOT + "/opt/out2d/bcache")
    p.add_argument("--no-boundary", action="store_true")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--occlusion", default="zbuffer", choices=["zbuffer", "none"])
    p.add_argument("--occ-win", type=int, default=OCC_WIN,
                   help="half-window of the z-buffer dilation, px (default %d)" % OCC_WIN)
    p.add_argument("--occ-win-sweep", default="",
                   help="comma-separated half-windows, e.g. 0,1,2,3; adds one extra "
                        "2D arm per value so the occlusion/coverage trade is a "
                        "reported sensitivity row rather than a hidden choice")
    p.add_argument("--sampling", default="auto",
                   choices=["auto", "nearest", "prob_bilinear"])
    p.add_argument("--sky", default="wrong",
                   choices=["wrong", "abstain", "manmade"],
                   help="wrong (default, R1): a sky prediction is UNMAPPED, i.e. an "
                        "answer the common space cannot name -- counted wrong under "
                        "BOTH conventions, dropped from neither.  abstain: the pro-2D "
                        "variant, reported as a sensitivity only.")
    p.add_argument("--rider", default=RIDER_DEFAULT,
                   choices=["two_wheeler", "person"])
    p.add_argument("--skip-naive-2d", action="store_true",
                   help="do NOT additionally score the 2D arm with occlusion=none; "
                        "by default the naive-projection variant is always reported "
                        "so the occlusion concession is visible rather than assumed")
    p.add_argument("--assert-2d-deterministic", action="store_true")
    p.add_argument("--timing", default="", help="comma-separated timing JSON files")
    p.add_argument("--seq", default="07", choices=sorted(SEQUENCES),
                   help="07 = the evaluation sequence; 04 = the disjoint set on "
                        "which the 2D arm's input scale is tuned")
    p.add_argument("--out", required=True)

    a = ap.parse_args()
    set_sequence(a.seq)

    if a.cmd == "selftest":
        ok = selftest(frames=frame_list(a.frames))
        sys.exit(0 if ok else 1)

    proj = Projector()
    lut = LS.sk_lut()
    frames = frame_list(a.frames)

    if a.cmd == "prep":
        os.makedirs(a.bcache, exist_ok=True)
        for i, f in enumerate(frames):
            p = _bcache_path(a.bcache, f)
            if os.path.exists(p):
                continue
            pts = read_scan(f)
            gt = read_gt_coarse(f, lut)
            sb = sem_boundary_mask(pts[:, :3].astype(np.float64), gt,
                                   workers=a.workers)
            np.savez_compressed(p, sem_b=np.packbits(sb))
            if (i + 1) % 50 == 0:
                print("  %d/%d" % (i + 1, len(frames)), flush=True)
        print("boundary cache ready: %s" % a.bcache)
        return

    if a.cmd == "mapcov":
        mapcov(frames, proj, a.voxel, a.out)
        return

    # ---- score ------------------------------------------------------- #
    d3 = [x for x in a.arm3d.split(",") if x]
    d2 = [x for x in a.arm2d.split(",") if x]
    assert d3 or d2, "need at least one arm"
    n_rep = max(len(d3), len(d2), 1)

    if a.assert_2d_deterministic and len(d2) >= 2:
        for f in frames[:20]:
            ref = np.load("%s/f%06d.npz" % (d2[0], f))["seg"]
            for d in d2[1:]:
                other = np.load("%s/f%06d.npz" % (d, f))["seg"]
                assert np.array_equal(ref, other), \
                    ("2D arm is NOT deterministic at frame %d" % f)
        print("2D determinism check: PASS (%d frames bitwise identical)" % min(20, len(frames)))

    reps = []
    for r in range(n_rep):
        arms = []
        a2 = a2n = a3 = None
        if d2:
            p2 = d2[min(r, len(d2) - 1)]
            a2 = Arm2D(p2, "2d", rider=a.rider, sky_policy=a.sky,
                       occlusion=a.occlusion, sampling=a.sampling, occ_win=a.occ_win)
            arms.append(a2)
            for wsz in [int(x) for x in a.occ_win_sweep.split(",") if x != ""]:
                if wsz == a.occ_win:
                    continue
                arms.append(Arm2D(p2, "2d_occwin%d" % wsz, rider=a.rider,
                                  sky_policy=a.sky, occlusion="zbuffer",
                                  sampling=a.sampling, occ_win=wsz))
            if (not a.skip_naive_2d) and a.occlusion != "none":
                a2n = Arm2D(p2, "2d_naive_projection", rider=a.rider,
                            sky_policy=a.sky, occlusion="none", sampling=a.sampling)
                arms.append(a2n)
        if d3:
            p3 = d3[min(r, len(d3) - 1)]
            a3 = Arm3D(p3, "3d")
            arms.append(a3)
        if a2 is not None and a3 is not None:
            arms.append(ArmHybrid(a2, a3, "hybrid"))
        print("repeat %d/%d  arms: %s" % (r + 1, n_rep, [x.name for x in arms]))
        accs = score_pass(frames, arms, proj, lut,
                          bcache=None if a.no_boundary else a.bcache,
                          need_boundary=not a.no_boundary, workers=a.workers)
        # class subsets are fixed by GT, so derive them once from the first arm
        first = list(accs)[0]
        cs = class_subsets_for(accs[first]["global"].metrics({"all": COARSE}))
        rep = {n: {s: accs[n][s].metrics(cs) for s in accs[n]} for n in accs}
        for n in rep:
            rep[n]["_class_subsets"] = cs
        if a2 is not None:
            rep["2d"]["_diag"] = {
                "sky_hits": a2.sky_hits, "occluded_points": a2.occluded,
                "in_frustum_points": a2.in_frustum,
                "sampling_used": getattr(a2, "mode", "n/a")}
        reps.append(rep)

    arms_meta = {"3d": {"dirs": d3, "space": "nusc16",
                        "note": "PTv3 Pointcept v1.5.1 nuScenes weights, fp16, "
                                "intensity x0.2, shuffle_orders=False"},
                 "2d": {"dirs": d2, "space": "cityscapes19",
                        "note": "Cityscapes-pretrained semantic segmentation of "
                                "image_02, image-space labels; projection, occlusion "
                                "and sky rules applied by this module"},
                 "2d_naive_projection": {"note": "same 2D cache, occlusion=none"},
                 "hybrid": {"note": "2D where the 2D arm speaks, 3D elsewhere"}}

    timing = {}
    for t in [x for x in a.timing.split(",") if x]:
        timing[os.path.basename(t)] = json.load(open(t))

    out = {
        "protocol": build_protocol(
            argparse.Namespace(frames=a.frames, sampling=a.sampling,
                               occlusion=a.occlusion, sky=a.sky, rider=a.rider,
                               occ_win=a.occ_win, seq=a.seq,
                               repeats_note="%d repeat(s); half-range reported" % n_rep),
            frames, arms_meta),
        "timing": timing,
        "timing_schema": TIMING_SCHEMA,
        "repeats": reps,
        "agg": aggregate([{k: {s: v for s, v in r[k].items()
                               if isinstance(v, dict) and "n_eval" in v}
                           for k in r} for r in reps]),
    }
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2, default=float)
    print("wrote %s" % a.out)


if __name__ == "__main__":
    main()
