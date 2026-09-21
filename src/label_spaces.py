#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
label_spaces.py -- the three-way label bridge for the 2D-vs-3D semantics experiment.

WHAT THIS FILE IS FOR
=====================
We are answering one external challenge with numbers:

    "You have a calibrated camera.  Why not run a 2D segmentation network on the
     image and project the labels onto the points, instead of segmenting the point
     cloud directly?"

Answering it requires scoring two arms against one ground truth:

    GT   SemanticKITTI seq07  (28 raw ids present in this sequence, out of 34 defined)
    3D   PTv3 trained on nuScenes-lidarseg  -> 16 classes
    2D   a Cityscapes-pretrained image segmenter -> 19 classes

Those three vocabularies were written by three different groups with three different
purposes and they DO NOT NEST.  The comparison lives or dies on this file.  A careless
mapping silently converts a vocabulary mismatch into an accuracy difference, and the
resulting number is not a measurement of anything.

THE GOVERNING RULE
==================
The 2D arm must be given its strongest defensible configuration.  Where a mapping
choice could go either way, this file takes the option that helps the 2D arm, and
every such choice is listed in DECISIONS below with its direction stated.  A
conclusion that survives a deck stacked for the 2D arm is the only kind worth quoting.

The 2D arm is already advantaged by the choice of Cityscapes over ADE20K, and that
advantage is deliberate: Cityscapes is daytime forward-facing German urban street
scenes from a car, which is exactly what KITTI is, whereas the 3D arm's nuScenes
weights come from Boston/Singapore with a 32-beam sensor facing a 64-beam HDL-64E.
The 2D arm gets the SMALLER domain gap.  Do not "fix" that.

TWO KINDS OF "NOT IN THIS VOCABULARY"
=====================================
These are different and must never be conflated:

  EXCLUDED  (-1)   A GROUND-TRUTH decision.  The GT class cannot be expressed in all
                   three spaces, so points carrying it are removed from scoring for
                   BOTH arms identically.  Neither arm can win or lose on them.

  UNMAPPED  (-2)   A PREDICTION-side decision.  The arm emitted a native class that
                   has no counterpart in the common space.  Such a point is counted
                   WRONG.  It is never dropped.

Dropping points because of what an arm PREDICTED would let that arm delete its own
errors by predicting an unmappable class -- an abstention leak.  The 2D arm has an
obvious one available to it (`sky`), so this distinction is not academic.  The single
place where a prediction-conditioned drop is offered at all is the `sky` sensitivity
variant, and there it is applied to BOTH arms' scored point set so the two arms are
always compared on an identical set of points.

By construction each arm ends up with exactly one dead native class -- one class it can
emit that can never be correct, because that class's GT counterpart is EXCLUDED:

    3D arm : nuScenes `other_flat`   (GT counterpart SK 49 other-ground, excluded)
    2D arm : Cityscapes `sky`        (no LiDAR return exists for sky at all)

One each.  That symmetry is not engineered; it fell out, and it is reported rather
than tidied away.

SKY
===
Sky is the one class in any of the three spaces with no physical LiDAR counterpart:
a LiDAR return means a surface was hit, so no point is ever truly sky.  A LiDAR point
whose pixel the 2D network calls sky is therefore one of
    (a) a projection/occlusion artefact (the point is behind a silhouette edge), or
    (b) a thin structure -- pole, wire, mast, tree crown -- that the network merged
        into the sky region.
Both are genuine failure modes of "segment the image, then project", so the default
policy counts them WRONG (SKY_POLICY="wrong").  The count is reported separately as
`n_pred_sky`, because it is itself a citable number.  SKY_POLICY="drop" is provided
as the 2D-favourable sensitivity variant; it drops those points from BOTH arms.

WHY THIS MAPPING DOES NOT FAVOUR EITHER ARM
===========================================
Stress-testing the coarsening, arm by arm.

  What the 2D arm loses to the merge.  Six Cityscapes classes -- building, wall,
  fence, pole, traffic light, traffic sign -- collapse into one coarse class,
  `manmade`, because nuScenes-lidarseg has a single `static.manmade` class whose own
  definition explicitly names "buildings, walls, guard rails, fences, poles, ...
  street signs, ... traffic lights".  There is no way to split it.  `manmade` is the
  second-largest coarse class in this sequence (21.4% of in-frustum GT points), so
  this merge is the single biggest act of coarsening in the file.
      Direction of the effect: it HELPS the 2D arm on the metric.  Every
  building-vs-wall, fence-vs-building, pole-vs-sign confusion the 2D network makes
  becomes free.  The 3D arm has nothing to gain there -- it only ever had one class.
      But it also HIDES a real 2D capability.  Separating poles, traffic lights and
  traffic signs is something the 2D arm can genuinely do and the 3D arm, with these
  weights, simply cannot.  The headline number therefore understates the semantic
  richness of the 2D arm.  That has to be said out loud every time the number is
  quoted; it is not captured by any figure in this file.

  What the 3D arm loses to the merge.  Five of its sixteen native classes stop
  existing: barrier and traffic_cone fold into `manmade`; trailer and
  construction_vehicle fold into `large_vehicle`; other_flat becomes UNMAPPED.  Its
  bicycle/motorcycle distinction is merged away.  So 5/16 destroyed against the 2D
  arm's 6/19 merged plus 1 dead -- comparable proportions, and in both cases the
  merges are intra-coarse-class, which is to say forgiving rather than punishing.

  The remaining asymmetry, stated rather than fixed.  `manmade` is a big, easy,
  high-support class for both arms, and folding six fine 2D classes into it raises
  the 2D arm's score on 21% of the points.  We keep it because nuScenes leaves no
  alternative, and we disclose it.  There is no version of this experiment in which
  poles are scored separately, because one of the two arms has never heard of them.

  The two-wheeler cluster is the one place where all three spaces disagree along
  three different axes, and it is handled by a merge rather than by silence -- see
  the EXCLUSIONS entry for it, and MEASURED_RIDER_EVIDENCE for the check that the
  merge rests on real data rather than on a reading of the class list.

WHAT IS *NOT* DECIDED HERE
==========================
This file is a vocabulary bridge.  It says nothing about, and must not be read as
endorsing, the following, which belong to whoever runs the measurement:
  * Which points are scored at all.  Only 16.0% of a scan lands in the camera
    frustum (measured below, 21 389 101 / 133 585 213 points over seq07).  The two
    arms MUST be scored on the same point set, which means the in-frustum set.  The
    coverage figure is a separate headline number, not part of the accuracy number.
  * Occlusion.  A background point can project onto a foreground object's pixels.
    That is a real failure mode of the 2D approach, but a strong 2D implementation
    would use a visibility/z-buffer check, and the fairness rule says the 2D arm gets
    the strong implementation.  Whether one was used must be disclosed.
  * Run-to-run noise.  The 3D model is non-deterministic (~4-5% of point labels move
    between two forwards of the same scan).  Repeats and spread are mandatory.

USAGE
=====
    import numpy as np, label_spaces as LS

    gt_raw  = np.fromfile(label_path, np.uint32) & 0xFFFF     # (N,)
    gt      = LS.sk_lut()[gt_raw]                             # coarse or EXCLUDED

    pred3d  = LS.nusc_lut()[nusc_ids]                         # (N,)  3D arm
    pred2d  = LS.cs_lut()[cs_ids_at_projected_pixel]           # (M,)  2D arm, in-frustum

    # score BOTH arms on the same points: in-frustum AND not EXCLUDED
    m = LS.eval_mask(gt, sky_policy="wrong") & in_frustum
    r3 = LS.score(gt, pred3d, m)
    r2 = LS.score(gt, pred2d_scattered_back_to_N, m)
    # r["point_acc"], r["miou"], r["per_class_iou"], r["n_pred_unmapped"], r["confusion"]

    # the disclosed sensitivity variants
    LS.sk_lut(include_parking_as_road=True)                   # +1.28% of points
    LS.eval_mask(gt, cs_ids, sky_policy="drop") & in_frustum  # 2D-favourable
    LS.score(gt, pred, m, classes=LS.COARSE_FREQUENT)         # drops the tiny classes

Both arms MUST be passed the same `m`.  Do not let one arm be scored on points the
other was not.

SOURCES (all fetched, not recalled)
===================================
  SemanticKITTI  config/semantic-kitti.yaml from the official API repo
                 https://github.com/PRBonn/semantic-kitti-api
                 local copy: /data/livo_sem/ref/semantic-kitti.yaml
  nuScenes-16    class list and the raw-category -> 16-class learning_map that the
                 deployed checkpoint was trained with, read out of the Pointcept
                 v1.5.1 source actually used by the 3D arm:
                   configs/nuscenes/semseg-pt-v3m1-0-base.py            (`names`)
                   pointcept/datasets/nuscenes.py                       (learning_map)
                 class semantics from the official annotator instructions
                 https://github.com/nutonomy/nuscenes-devkit/blob/master/docs/instructions_lidarseg.md
                 local copy: /data/livo_sem/ref/instructions_lidarseg.md
                 rider convention from
                 https://github.com/nutonomy/nuscenes-devkit/blob/master/docs/instructions_nuscenes.md
                 local copy: /data/livo_sem/ref/instructions_nuscenes.md
  Cityscapes-19  https://github.com/mcordts/cityscapesScripts  helpers/labels.py
                 local copy: /data/livo_sem/ref/cityscapes_labels.py
                 cross-checked against the id2label of three candidate checkpoints,
                 which are byte-identical in name and order:
                   nvidia/segformer-b5-finetuned-cityscapes-1024-1024
                   facebook/mask2former-swin-large-cityscapes-semantic
                   shi-labs/oneformer_cityscapes_swin_large
                 local copies: /data/livo_sem/ref/cfg_*.json
                 So the mapping does not depend on which Cityscapes model is chosen.
"""

import numpy as np

# =========================================================================== #
#  SENTINELS
# =========================================================================== #
EXCLUDED = -1     # GT-side: this point is not scored, for either arm
UNMAPPED = -2     # prediction-side: this prediction is always wrong, never dropped


# =========================================================================== #
#  (a) SOURCE VOCABULARY 1 -- SemanticKITTI, the GROUND TRUTH
#      /data/livo_sem/ref/semantic-kitti.yaml  ->  `labels:`
#      34 ids defined.  Verified against the file, not against memory: the id list
#      handed to us omitted 16 "on-rails" and is otherwise correct.
# =========================================================================== #
VOCAB_SEMANTICKITTI = {
    0: "unlabeled",          1: "outlier",
    10: "car",               11: "bicycle",           13: "bus",
    15: "motorcycle",        16: "on-rails",          18: "truck",
    20: "other-vehicle",     30: "person",            31: "bicyclist",
    32: "motorcyclist",      40: "road",              44: "parking",
    48: "sidewalk",          49: "other-ground",      50: "building",
    51: "fence",             52: "other-structure",   60: "lane-marking",
    70: "vegetation",        71: "trunk",             72: "terrain",
    80: "pole",              81: "traffic-sign",      99: "other-object",
    252: "moving-car",       253: "moving-bicyclist", 254: "moving-person",
    255: "moving-motorcyclist", 256: "moving-on-rails", 257: "moving-bus",
    258: "moving-truck",     259: "moving-other-vehicle",
}

# Which of those 34 actually occur in seq07 (measured, see main()).
SK_IDS_PRESENT_IN_SEQ07 = (0, 1, 10, 11, 15, 18, 20, 30, 40, 44, 48, 50, 51, 52,
                           70, 71, 72, 80, 81, 99, 252, 253, 254, 258)
# Absent: 13 bus, 16 on-rails, 31 bicyclist(static), 32 motorcyclist, 49 other-ground,
#         60 lane-marking, 255, 256, 257, 259.


# =========================================================================== #
#  (b) SOURCE VOCABULARY 2 -- nuScenes-lidarseg 16, the 3D ARM's OUTPUT
#      Order is the `names` list in Pointcept configs/nuscenes/semseg-pt-v3m1-0-base.py,
#      i.e. the exact channel order of the deployed checkpoint's logits.
# =========================================================================== #
VOCAB_NUSCENES16 = (
    "barrier", "bicycle", "bus", "car", "construction_vehicle", "motorcycle",
    "pedestrian", "traffic_cone", "trailer", "truck", "driveable_surface",
    "other_flat", "sidewalk", "terrain", "manmade", "vegetation",
)

# Semantics that actually drive the mapping, quoted from instructions_lidarseg.md /
# instructions_nuscenes.md:
#   driveable_surface : "All paved or unpaved surfaces that a car can drive on with no
#                        concern of traffic rules."   -> parking lots included
#   other_flat        : "Horizontal surfaces which cannot be classified as ground plane
#                        / sidewalk / terrain, e.g. water ... traffic islands,
#                        delimiters, rail tracks, stairs with at most 3 steps"
#   terrain           : "ground level horizontal vegetation (< 20 cm tall), grass, soil"
#   vegetation        : "any vegetation higher than the ground ... trees"  -> trunks in
#   manmade           : "buildings, walls, guard rails, fences, poles, drainages,
#                        hydrants, flags, banners, street signs, electric circuit boxes,
#                        traffic lights, parking meters and stairs with >3 steps"
#   bicycle/motorcycle: "If there is a rider and/or passenger, include them in the box."
#                        -> THE RIDER IS PART OF THE VEHICLE CLASS, not of pedestrian.


# =========================================================================== #
#  (c) SOURCE VOCABULARY 3 -- Cityscapes 19 (trainId order), the 2D ARM's OUTPUT
#      cityscapesscripts/helpers/labels.py, the 19 entries with trainId != 255,
#      ordered by trainId.  Identical to config.id2label of all three candidate
#      checkpoints.
# =========================================================================== #
VOCAB_CITYSCAPES19 = (
    "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
    "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car",
    "truck", "bus", "train", "motorcycle", "bicycle",
)

# Cityscapes classes that are ANNOTATED but have trainId 255, i.e. the 19-class model
# was trained with these pixels IGNORED and has no defined behaviour on them.  This
# list is load-bearing for the `parking` decision below.
CITYSCAPES_IGNORED_IN_19 = (
    "unlabeled", "ego vehicle", "rectification border", "out of roi", "static",
    "dynamic", "ground", "parking", "rail track", "guard rail", "bridge", "tunnel",
    "polegroup", "caravan", "trailer", "license plate",
)


# =========================================================================== #
#  2. THE COMMON COARSE SPACE
# =========================================================================== #
# Nine classes.  Every one of them is (i) natively expressible in all three
# vocabularies and (ii) actually present in seq07's ground truth.
#
# Changes from the coarse space the pipeline used before this file existed
# (car, bicycle, motorcycle, truck, bus, other_vehicle, person, road, sidewalk,
#  other_flat, terrain, vegetation, manmade):
#   - bus           MERGED into large_vehicle.  seq07 contains ZERO bus points, and
#                   SemanticKITTI's own benchmark map folds bus into other-vehicle.
#   - other_vehicle EXCLUDED.  Cityscapes-19 has no class for it.
#   - other_flat    EXCLUDED.  Cityscapes-19 has no class for it, and its GT
#                   counterpart (SK 49) does not occur in seq07 anyway.
#   - bicycle,
#     motorcycle    MERGED into two_wheeler, rider-inclusive.  See DECISIONS.
#   - truck         RENAMED large_vehicle, because it now also carries bus, trailer,
#                   construction_vehicle and train.  Calling that "truck" would be a
#                   label that lies about its contents.
COARSE = (
    "car",            # 0
    "large_vehicle",  # 1  truck / bus / trailer / construction vehicle / train
    "two_wheeler",    # 2  bicycle / motorcycle AND their riders
    "person",         # 3  pedestrians NOT riding anything
    "road",           # 4
    "sidewalk",       # 5
    "terrain",        # 6
    "vegetation",     # 7
    "manmade",        # 8  building / wall / fence / pole / sign / light / barrier
)
NUM_COARSE = len(COARSE)
_C = {n: i for i, n in enumerate(COARSE)}

# Classes with >= 1% of in-frustum GT points.  mIoU over the full 9 gives a class with
# 0.25% support (two_wheeler) the same weight as one with 27% (road); this subset is
# the robustness variant, not a replacement for the headline.
COARSE_FREQUENT = ("car", "large_vehicle", "road", "sidewalk", "terrain",
                   "vegetation", "manmade")


# =========================================================================== #
#  MAPPING 1 of 3:  SemanticKITTI raw id  ->  coarse   (GROUND TRUTH side)
# =========================================================================== #
SK_TO_COARSE = {
    # --- excluded, see EXCLUSIONS for the reason attached to each -------------
    0: EXCLUDED,    # unlabeled
    1: EXCLUDED,    # outlier
    16: EXCLUDED,   # on-rails
    256: EXCLUDED,  # moving-on-rails
    20: EXCLUDED,   # other-vehicle
    259: EXCLUDED,  # moving-other-vehicle
    44: EXCLUDED,   # parking
    49: EXCLUDED,   # other-ground
    52: EXCLUDED,   # other-structure
    99: EXCLUDED,   # other-object
    # --- scored ---------------------------------------------------------------
    10: _C["car"],            252: _C["car"],
    18: _C["large_vehicle"],  258: _C["large_vehicle"],
    13: _C["large_vehicle"],  257: _C["large_vehicle"],
    11: _C["two_wheeler"],    15: _C["two_wheeler"],
    31: _C["two_wheeler"],    253: _C["two_wheeler"],
    32: _C["two_wheeler"],    255: _C["two_wheeler"],
    30: _C["person"],         254: _C["person"],
    40: _C["road"],           60: _C["road"],          # lane-marking is painted road
    48: _C["sidewalk"],
    72: _C["terrain"],
    70: _C["vegetation"],     71: _C["vegetation"],    # trunk is part of the tree
    50: _C["manmade"],        51: _C["manmade"],       # building, fence
    80: _C["manmade"],        81: _C["manmade"],       # pole, traffic-sign
}

# =========================================================================== #
#  MAPPING 2 of 3:  nuScenes-16 class index  ->  coarse   (3D ARM prediction side)
# =========================================================================== #
NUSC_TO_COARSE = (
    _C["manmade"],        # 0  barrier            -> nuScenes manmade covers guard rails
    _C["two_wheeler"],    # 1  bicycle            (rider included by annotation rule)
    _C["large_vehicle"],  # 2  bus
    _C["car"],            # 3  car
    _C["large_vehicle"],  # 4  construction_vehicle
    _C["two_wheeler"],    # 5  motorcycle         (rider included by annotation rule)
    _C["person"],         # 6  pedestrian         (riders are NOT in here)
    _C["manmade"],        # 7  traffic_cone
    _C["large_vehicle"],  # 8  trailer
    _C["large_vehicle"],  # 9  truck
    _C["road"],           # 10 driveable_surface
    UNMAPPED,             # 11 other_flat         <-- the 3D arm's dead class
    _C["sidewalk"],       # 12 sidewalk
    _C["terrain"],        # 13 terrain
    _C["manmade"],        # 14 manmade
    _C["vegetation"],     # 15 vegetation
)

# =========================================================================== #
#  MAPPING 3 of 3:  Cityscapes-19 trainId  ->  coarse    (2D ARM prediction side)
# =========================================================================== #
CS_TO_COARSE = (
    _C["road"],           # 0  road
    _C["sidewalk"],       # 1  sidewalk
    _C["manmade"],        # 2  building
    _C["manmade"],        # 3  wall
    _C["manmade"],        # 4  fence
    _C["manmade"],        # 5  pole
    _C["manmade"],        # 6  traffic light
    _C["manmade"],        # 7  traffic sign
    _C["vegetation"],     # 8  vegetation
    _C["terrain"],        # 9  terrain
    UNMAPPED,             # 10 sky               <-- the 2D arm's dead class
    _C["person"],         # 11 person            (riders are NOT in here)
    _C["two_wheeler"],    # 12 rider
    _C["car"],            # 13 car
    _C["large_vehicle"],  # 14 truck
    _C["large_vehicle"],  # 15 bus
    _C["large_vehicle"],  # 16 train
    _C["two_wheeler"],    # 17 motorcycle
    _C["two_wheeler"],    # 18 bicycle
)

CS_SKY_ID = 10
NUSC_OTHER_FLAT_ID = 11


# =========================================================================== #
#  3. THE EXCLUSION LIST -- every class that does not exist in all three spaces,
#     with the decision taken and the direction it pushes the result.
# =========================================================================== #
# fields: (space, native class, decision, reason, which arm the choice helps)
EXCLUSIONS = (
    # ---- ground-truth classes removed from scoring --------------------------
    ("SemanticKITTI", "0 unlabeled / 1 outlier", "EXCLUDED",
     "No semantic content.  SemanticKITTI's own learning_map ignores them.",
     "neutral"),

    ("SemanticKITTI", "20 other-vehicle / 259 moving-other-vehicle", "EXCLUDED",
     "A catch-all for vehicles that are not car/truck/bus/two-wheeler: vans, trams, "
     "caravans, towed trailers.  Cityscapes-19 has NO class for it -- caravan and "
     "trailer are annotated in Cityscapes but carry trainId 255, so the 19-class model "
     "cannot emit them.  nuScenes has trailer and construction_vehicle but neither is "
     "a translation of SemanticKITTI's catch-all.  Scoring it would charge the 2D arm "
     "for a class it structurally cannot produce.",
     "2D_arm"),

    ("SemanticKITTI", "44 parking", "EXCLUDED",
     "The three spaces genuinely disagree.  nuScenes driveable_surface is defined as "
     "'all paved or unpaved surfaces that a car can drive on WITH NO CONCERN OF TRAFFIC "
     "RULES', so a parking lot is driveable_surface and the 3D arm would be scored "
     "correct if parking were mapped to road.  Cityscapes annotates parking as its own "
     "class and then gives it trainId 255, so the 19-class model was trained with "
     "parking pixels IGNORED and its behaviour there is undefined.  Mapping parking to "
     "road would hand the 3D arm 1.28% of the in-frustum points and give the 2D arm a "
     "coin flip.  Excluded.  Sensitivity: INCLUDE_PARKING_AS_ROAD=True restores it.",
     "2D_arm"),

    ("SemanticKITTI", "49 other-ground", "EXCLUDED",
     "Cityscapes-19 has no class for it (its `ground` and `rail track` are trainId 255). "
     "Does not occur in seq07 at all, so the exclusion costs nothing here.",
     "2D_arm"),

    ("SemanticKITTI", "52 other-structure / 99 other-object", "EXCLUDED",
     "Following the dataset authors: SemanticKITTI's own learning_map sends both to "
     "'unlabeled', with the stated reason that they are inconsistent in the ground "
     "truth.  Together 1.83% of in-frustum points.  NOTE: the pipeline's previous "
     "coarse space counted 52 as `manmade`, so the 3D arm's previously cited scores "
     "are NOT comparable to scores computed with this file.",
     "neutral"),

    ("SemanticKITTI", "16 on-rails / 256 moving-on-rails", "EXCLUDED",
     "nuScenes-16 has no rail class.  Cityscapes has `train`.  Two out of three is not "
     "enough.  Absent from seq07 anyway.",
     "neutral"),

    # ---- prediction-side classes with no counterpart ------------------------
    ("Cityscapes", "10 sky", "UNMAPPED (counted wrong, never dropped)",
     "No LiDAR return is ever sky -- a return means a surface was hit.  A point whose "
     "pixel is called sky is either a projection/occlusion artefact or a thin structure "
     "(pole, wire, mast) merged into the sky region.  Both are real failure modes of "
     "'segment the image, then project', so they are errors, not abstentions.  Dropping "
     "them would let the 2D arm delete its own hardest errors.  Reported separately as "
     "n_pred_sky, and SKY_POLICY='drop' gives the 2D-favourable variant -- which drops "
     "those points for BOTH arms so the point sets stay identical.",
     "neutral (the 'drop' variant helps 2D_arm)"),

    ("nuScenes", "11 other_flat", "UNMAPPED (counted wrong, never dropped)",
     "Its GT counterpart is SemanticKITTI 49 other-ground, which is EXCLUDED because "
     "Cityscapes-19 cannot express it.  So other_flat can never be scored correct.  "
     "This is the 3D arm's dead class, exactly mirroring the 2D arm's sky.  It could "
     "have been charitably folded into sidewalk or road; it is not, because the "
     "fairness rule points the other way.  Report n_pred_unmapped for the 3D arm so "
     "the cost is visible.",
     "2D_arm"),

    # ---- classes that exist everywhere but had to be merged -----------------
    ("all three", "the two-wheeler / rider cluster", "MERGED into two_wheeler",
     "Three vocabularies, three incompatible conventions, along three different axes. "
     "SemanticKITTI splits by whether the vehicle is being ridden (11 bicycle and 15 "
     "motorcycle are the unridden vehicles; 31 bicyclist and 32 motorcyclist are the "
     "ridden ones).  Cityscapes splits human from machine (25 rider is the human only; "
     "33 bicycle is the bike, ridden or not).  nuScenes puts the human INSIDE the "
     "vehicle ('if there is a rider and/or passenger, include them in the box').  No "
     "assignment keeping bicycle and motorcycle apart is correct in all three: "
     "Cityscapes `rider` is ambiguous between them, since the same class covers a "
     "cyclist and a motorcyclist.  One rider-inclusive bucket IS consistent in all "
     "three, and it leaves `person` clean, because all three exclude riders from their "
     "pedestrian/person class.  Cost: the bicycle-vs-motorcycle distinction, which all "
     "three could make, is lost -- symmetrically, from both arms.",
     "neutral"),

    ("all three", "bus", "MERGED into large_vehicle",
     "All three vocabularies have a bus class, so this is not a vocabulary failure.  It "
     "is a support failure: seq07 contains ZERO bus points.  A coarse class with no GT "
     "support yields an undefined IoU and turns every bus prediction into a pure "
     "penalty with no possibility of credit.  SemanticKITTI's own benchmark map also "
     "declines to keep bus separate (it folds 13 into other-vehicle).  Merged, "
     "symmetrically, for both arms.",
     "neutral"),

    ("nuScenes", "0 barrier, 7 traffic_cone", "MERGED into manmade",
     "nuScenes's own manmade definition names guard rails; Cityscapes puts cones in "
     "`static` (trainId 255) and barriers in fence/guard rail, of which fence -> "
     "manmade and guard rail is trainId 255.  manmade is the only landing place that "
     "exists in all three.",
     "3D_arm (they would otherwise be dead classes)"),

    ("nuScenes", "4 construction_vehicle, 8 trailer", "MERGED into large_vehicle",
     "Neither has a Cityscapes-19 counterpart (caravan and trailer are trainId 255). "
     "Folding them into large_vehicle alongside Cityscapes `truck`/`bus`/`train` lets "
     "both arms reach the same coarse class for the same objects.  The alternative -- "
     "UNMAPPED -- would penalise the 3D arm for a vocabulary accident rather than for "
     "a mistake.",
     "3D_arm"),

    ("Cityscapes", "16 train", "MERGED into large_vehicle",
     "nuScenes-16 has no rail class.  A `train` prediction in Karlsruhe suburbia is a "
     "hallucination on some large vehicle; large_vehicle is where it can at least be "
     "right.  The alternative -- UNMAPPED -- would penalise the 2D arm for a "
     "vocabulary accident.",
     "2D_arm"),

    ("Cityscapes", "2 building, 3 wall, 4 fence, 5 pole, 6 traffic light, "
     "7 traffic sign", "MERGED into manmade",
     "nuScenes-lidarseg has exactly one class, static.manmade, whose definition "
     "explicitly enumerates buildings, walls, guard rails, fences, poles, street signs "
     "and traffic lights.  There is no split available.  This merge makes the 2D arm's "
     "internal confusions among these six free, on 21.4% of the in-frustum points.  It "
     "also hides a genuine 2D capability the 3D arm does not have.  Both effects must "
     "be stated whenever the headline number is quoted.",
     "2D_arm (on the metric); 2D_arm is under-credited in capability terms"),

    ("SemanticKITTI", "71 trunk", "MERGED into vegetation",
     "nuScenes vegetation is 'any vegetation higher than the ground ... trees'; "
     "Cityscapes vegetation is 'trees and other vertical vegetation' including trunks. "
     "Neither separates the trunk.",
     "neutral"),

    ("SemanticKITTI", "60 lane-marking", "MERGED into road",
     "Painted markings are part of the road surface in both nuScenes and Cityscapes. "
     "SemanticKITTI's own benchmark map does the same.  Absent from seq07.",
     "neutral"),
)


# =========================================================================== #
#  SWITCHES -- every one of these is a disclosed sensitivity axis, not a knob to
#  tune until the answer looks good.  Report the headline at the defaults and
#  report the variants alongside it.
# =========================================================================== #
INCLUDE_PARKING_AS_ROAD = False   # True -> SK 44 becomes `road` instead of EXCLUDED
SKY_POLICY = "wrong"              # "wrong" | "drop"  (see docstring)


# =========================================================================== #
#  LUT BUILDERS
# =========================================================================== #
def sk_lut(include_parking_as_road=None):
    """SemanticKITTI raw id (low 16 bits of the .label word) -> coarse id / EXCLUDED.

    Returns int8[300].  Any id outside the official 34 maps to EXCLUDED; seq07 was
    measured and contains no such id.
    """
    if include_parking_as_road is None:
        include_parking_as_road = INCLUDE_PARKING_AS_ROAD
    lut = np.full(300, EXCLUDED, dtype=np.int8)
    for k, v in SK_TO_COARSE.items():
        lut[k] = v
    if include_parking_as_road:
        lut[44] = _C["road"]
    return lut


def nusc_lut():
    """nuScenes-16 class index -> coarse id / UNMAPPED.  int8[16]."""
    return np.asarray(NUSC_TO_COARSE, dtype=np.int8)


def cs_lut(sky_policy=None):
    """Cityscapes-19 trainId -> coarse id / UNMAPPED.  int8[19].

    sky_policy only affects bookkeeping downstream, never the LUT: sky is UNMAPPED
    either way.  Under "drop" the caller removes sky-predicted points from BOTH arms
    via eval_mask(), which is the only way a prediction-conditioned drop can be fair.
    """
    return np.asarray(CS_TO_COARSE, dtype=np.int8)


# =========================================================================== #
#  SCORING -- the part that keeps the two arms on one point set
# =========================================================================== #
def eval_mask(gt_coarse, pred_2d_cs_raw=None, sky_policy=None):
    """Boolean mask of the points that are scored, IDENTICALLY for every arm.

    gt_coarse       (N,) int   output of sk_lut()[raw_ids]
    pred_2d_cs_raw  (N,) int   the 2D arm's RAW Cityscapes trainIds, needed only for
                               sky_policy="drop"
    """
    if sky_policy is None:
        sky_policy = SKY_POLICY
    m = np.asarray(gt_coarse) >= 0
    if sky_policy == "drop":
        if pred_2d_cs_raw is None:
            raise ValueError("sky_policy='drop' needs the 2D arm's raw Cityscapes ids")
        m &= np.asarray(pred_2d_cs_raw) != CS_SKY_ID
    elif sky_policy != "wrong":
        raise ValueError("sky_policy must be 'wrong' or 'drop'")
    return m


def score(gt_coarse, pred_coarse, mask, classes=None):
    """Point accuracy, per-class IoU and mIoU on the masked point set.

    An UNMAPPED prediction (-2) matches no class, so it is a false negative for the
    true class and contributes to no class's prediction count.  That is exactly
    "always wrong, never dropped".
    """
    if classes is None:
        classes = COARSE
    g = np.asarray(gt_coarse)[mask].astype(np.int64)
    p = np.asarray(pred_coarse)[mask].astype(np.int64)
    n = int(g.size)
    acc = float((p == g).sum()) / n if n else float("nan")

    ious, inter_c, gc_c, pc_c = {}, {}, {}, {}
    for name in classes:
        k = _C[name]
        gk = g == k
        pk = p == k
        i = int((gk & pk).sum())
        u = int(gk.sum()) + int(pk.sum()) - i
        inter_c[name], gc_c[name], pc_c[name] = i, int(gk.sum()), int(pk.sum())
        if gc_c[name] == 0:
            continue                      # no GT support -> IoU undefined, skipped
        ious[name] = 100.0 * i / u if u else 0.0
    miou = float(np.mean(list(ious.values()))) if ious else float("nan")

    # confusion, with one extra column for UNMAPPED so the cost is auditable
    cm = np.zeros((NUM_COARSE, NUM_COARSE + 1), np.int64)
    pi = np.where(p == UNMAPPED, NUM_COARSE, p)
    ok = g >= 0
    np.add.at(cm, (g[ok], pi[ok]), 1)

    return dict(n_scored=n,
                point_acc=100.0 * acc,
                miou=miou,
                per_class_iou={k: round(v, 3) for k, v in ious.items()},
                gt_count=gc_c, pred_count=pc_c, inter=inter_c,
                n_pred_unmapped=int((p == UNMAPPED).sum()),
                confusion=cm)


def verify():
    """Self-check.  Raises on any inconsistency.  No data needed."""
    assert len(VOCAB_SEMANTICKITTI) == 34
    assert len(VOCAB_NUSCENES16) == 16
    assert len(VOCAB_CITYSCAPES19) == 19
    assert len(NUSC_TO_COARSE) == 16 and len(CS_TO_COARSE) == 19
    # every SemanticKITTI id has an explicit decision -- nothing falls through
    missing = set(VOCAB_SEMANTICKITTI) - set(SK_TO_COARSE)
    assert not missing, "SemanticKITTI ids with no decision: %s" % sorted(missing)
    # every coarse class is reachable from all three spaces
    for k, name in enumerate(COARSE):
        assert k in SK_TO_COARSE.values(), "unreachable from SemanticKITTI: " + name
        assert k in NUSC_TO_COARSE, "unreachable from nuScenes: " + name
        assert k in CS_TO_COARSE, "unreachable from Cityscapes: " + name
    # exactly one dead class per arm
    assert sum(1 for v in NUSC_TO_COARSE if v == UNMAPPED) == 1
    assert sum(1 for v in CS_TO_COARSE if v == UNMAPPED) == 1
    assert CS_TO_COARSE[CS_SKY_ID] == UNMAPPED
    assert NUSC_TO_COARSE[NUSC_OTHER_FLAT_ID] == UNMAPPED
    assert set(COARSE_FREQUENT) <= set(COARSE)
    # the recorded measurement must still describe this mapping
    assert set(MEASURED_COARSE_SEQ07) == set(COARSE) | {"-- EXCLUDED --"}
    tot_all = sum(v[0] for v in MEASURED_COARSE_SEQ07.values())
    tot_fov = sum(v[2] for v in MEASURED_COARSE_SEQ07.values())
    assert tot_all == MEASURED_TOTALS["points_total"], tot_all
    assert tot_fov == MEASURED_TOTALS["points_in_frustum"], tot_fov
    ex_all = sum(v[1] for v in MEASURED_EXCLUDED_BREAKDOWN.values())
    assert ex_all == MEASURED_COARSE_SEQ07["-- EXCLUDED --"][0], ex_all
    for i in MEASURED_SK_IDS_ABSENT:
        assert i in VOCAB_SEMANTICKITTI

    # an UNMAPPED prediction must be WRONG, not invisible: 4 points of GT `road`,
    # one of which the arm calls sky.  Accuracy 75%, road IoU 75%, nothing dropped.
    gt = np.array([_C["road"]] * 4)
    pr = np.array([_C["road"], _C["road"], _C["road"], UNMAPPED])
    m = eval_mask(gt, sky_policy="wrong")
    r = score(gt, pr, m)
    assert r["n_scored"] == 4 and abs(r["point_acc"] - 75.0) < 1e-9
    assert abs(r["per_class_iou"]["road"] - 75.0) < 1e-9 and r["n_pred_unmapped"] == 1
    # ...and under the 'drop' variant it disappears from BOTH arms' point set
    cs_raw = np.array([0, 0, 0, CS_SKY_ID])
    m2 = eval_mask(gt, pred_2d_cs_raw=cs_raw, sky_policy="drop")
    assert m2.sum() == 3 and abs(score(gt, pr, m2)["point_acc"] - 100.0) < 1e-9
    return True


# =========================================================================== #
#  5. MEASURED -- all 1101 GT frames of seq07.  Not assumed, not quoted from a
#     paper.  Reproduce with:  python3 src/measure_gt_dist.py && python3 src/label_spaces.py
# =========================================================================== #
MEASURED_TOTALS = dict(
    frames=1101,
    points_total=133585213,
    points_in_frustum=21389101,          # 16.01% -- the 2D arm can never see the rest
    frustum_fraction=0.160116,
    label_count_mismatches=0,            # every .label matched its .bin, offset 0
    unknown_ids_found=0,                 # no id outside the official 34 occurs
)

# (coarse class -> (all points, all %, in-frustum points, in-frustum %)) at the
# defaults, i.e. INCLUDE_PARKING_AS_ROAD=False.
MEASURED_COARSE_SEQ07 = {
    "car":           (12134359,  9.0836,  2698134, 12.6145),
    "large_vehicle": ( 1101484,  0.8246,   227328,  1.0628),
    "two_wheeler":   (  268104,  0.2007,    53382,  0.2496),
    "person":        (  166135,  0.1244,    34639,  0.1619),
    "road":          (25199192, 18.8638,  5775356, 27.0014),
    "sidewalk":      (18248605, 13.6606,  2103194,  9.8330),
    "terrain":       ( 6877242,  5.1482,  1262510,  5.9026),
    "vegetation":    (21146455, 15.8299,  3493785, 16.3344),
    "manmade":       (40744538, 30.5008,  4566733, 21.3507),
    "-- EXCLUDED --":( 7699099,  5.7634,  1174040,  5.4890),
}

# The EXCLUDED bucket, itemised.  IS IT LARGE ENOUGH TO MAKE THE HEADLINE FRAGILE?
# No.  5.49% of in-frustum points, and no single cause exceeds 1.8%.  Two thirds of
# it (unlabeled + other-structure + other-object = 3.62%) is excluded on the dataset
# authors' own instruction, not on ours.  Our two discretionary exclusions cost
# 1.86% between them (parking 1.28% + other-vehicle 0.58%), and both were taken in
# the 2D arm's favour, so the residual risk is that the headline is slightly KIND to
# the 2D arm -- the safe direction.  Turning parking back on (the largest single
# discretionary item) moves the excluded bucket from 5.49% to 4.21% and adds 1.28
# points to `road`; report both.
MEASURED_EXCLUDED_BREAKDOWN = {      # sk id: (name, all pts, all %, fov pts, fov %)
    0:  ("unlabeled",       2800540, 2.0964,  383580, 1.7933),
    1:  ("outlier",           37816, 0.0283,    1327, 0.0062),
    20: ("other-vehicle",    445296, 0.3333,  123237, 0.5762),
    44: ("parking",         2137226, 1.5999,  274444, 1.2831),
    52: ("other-structure", 1074862, 0.8046,  238630, 1.1157),
    99: ("other-object",    1203359, 0.9008,  152822, 0.7145),
}

# Ground truth ids that seq07 does NOT contain at all, so nothing in the headline
# rests on them: 13 bus, 16 on-rails, 31 bicyclist(static), 32 motorcyclist,
# 49 other-ground, 60 lane-marking, 255, 256, 257, 259.
MEASURED_SK_IDS_ABSENT = (13, 16, 31, 32, 49, 60, 255, 256, 257, 259)

# --------------------------------------------------------------------------- #
#  Evidence behind the rider-inclusive two_wheeler merge, and behind the claim
#  that `person` stays clean in all three spaces.
#  src/check_rider_convention.py, 53 frames of seq07 with >= 30 bicyclist points:
#    * bicyclist clusters run from +0.13 m above the local road surface (min -0.08)
#      to +1.78 m, median extent 1.64 m -- i.e. the cluster reaches the ground and
#      is a whole human-on-a-machine object, not a torso.
#    * in NONE of those 53 frames does a separate id-11 `bicycle` point exist.
#    * in NONE of those 53 frames does a separate id-30 `person` point exist.
#  So SemanticKITTI puts the rider and the ridden machine under one id, and never
#  files a rider under `person`.  nuScenes does the same by its written annotation
#  rule ("if there is a rider and/or passenger, include them in the box").
#  Cityscapes splits them into `rider` + `bicycle`, but both land in two_wheeler.
#  All three therefore agree that `person` means "not riding anything", which is
#  what makes `person` safe to score and the two-wheeler cluster safe to merge.
# --------------------------------------------------------------------------- #
MEASURED_RIDER_EVIDENCE = dict(
    frames_with_bicyclist=53,
    cluster_z_above_road_median=(0.13, 1.78),
    cluster_extent_m_median=1.64,
    separate_bicycle_points_in_those_frames=0,
    separate_person_points_in_those_frames=0,
)


def main():
    import json, os, sys, glob
    verify()
    print("verify() ok\n")

    cache = "/data/livo_sem/out/labelspace/gt_raw_hist.json"
    if not os.path.exists(cache):
        print("run src/measure_gt_dist.py first to produce " + cache)
        return
    d = json.load(open(cache))
    h_all = np.array(d["h_all"], np.int64)
    h_fov = np.array(d["h_fov"], np.int64)
    n_all, n_fov = d["n_pts"], d["n_fov"]

    for include_parking in (False, True):
        lut = sk_lut(include_parking_as_road=include_parking)
        ca = np.zeros(NUM_COARSE, np.int64)
        cf = np.zeros(NUM_COARSE, np.int64)
        for i in range(300):
            if lut[i] >= 0:
                ca[lut[i]] += h_all[i]
                cf[lut[i]] += h_fov[i]
        ex_a, ex_f = n_all - ca.sum(), n_fov - cf.sum()
        print("=" * 78)
        print("SemanticKITTI seq07, %d frames -- INCLUDE_PARKING_AS_ROAD=%s"
              % (d["frames"], include_parking))
        print("=" * 78)
        print("%-16s %14s %9s   %14s %9s" %
              ("coarse class", "all pts", "all %", "in-frustum", "frustum %"))
        for k, name in enumerate(COARSE):
            print("%-16s %14d %8.4f%%   %14d %8.4f%%" %
                  (name, ca[k], 100.0 * ca[k] / n_all, cf[k], 100.0 * cf[k] / n_fov))
        print("%-16s %14d %8.4f%%   %14d %8.4f%%" %
              ("-- EXCLUDED --", ex_a, 100.0 * ex_a / n_all, ex_f,
               100.0 * ex_f / n_fov))
        print("%-16s %14d %8.4f%%   %14d %8.4f%%" %
              ("TOTAL", n_all, 100.0, n_fov, 100.0 * n_fov / n_all))
        if not include_parking:            # cross-check against the recorded numbers
            for k, name in enumerate(COARSE):
                assert MEASURED_COARSE_SEQ07[name][0] == ca[k], name
                assert MEASURED_COARSE_SEQ07[name][2] == cf[k], name
            assert MEASURED_COARSE_SEQ07["-- EXCLUDED --"][0] == ex_a
            assert MEASURED_COARSE_SEQ07["-- EXCLUDED --"][2] == ex_f
            print("(matches MEASURED_COARSE_SEQ07 exactly)")
        print()

    print("EXCLUDED bucket, broken down by SemanticKITTI id (defaults):")
    lut = sk_lut(include_parking_as_road=False)
    for i in range(300):
        if lut[i] < 0 and (h_all[i] or h_fov[i]):
            print("   %3d %-22s %14d %8.4f%%   %14d %8.4f%%" %
                  (i, VOCAB_SEMANTICKITTI.get(i, "?"), h_all[i],
                   100.0 * h_all[i] / n_all, h_fov[i], 100.0 * h_fov[i] / n_fov))


if __name__ == "__main__":
    main()
