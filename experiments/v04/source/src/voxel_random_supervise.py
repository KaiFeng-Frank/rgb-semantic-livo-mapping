#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
voxel_random_supervise.py -- arm R-PRIME's supervision selector.

Dependency-free on purpose (numpy only), for exactly the reason filter_e.py and
random_supervise.py are: the training side imports it through distil_ext_rprime (which
drags in pointcept / spconv / torch), the verification side imports it directly.  ONE
definition, both sides -- a verification that re-implemented the rule would prove
nothing about the rule the trainer actually runs.

WHY ARM R-PRIME EXISTS
----------------------
Arm R matches arm D's supervised point count per frame at the MASK level.  Both arms
then pass through GridSample(0.05, mode="train"), which keeps ONE uniformly chosen
representative point per occupied voxel, and only that representative reaches the loss.
Measured survival (out/v04/armR/gridsample_survival.json, 96 frames of the real
training stream): frustum points 0.8461, uniformly-random points 0.7078.  Arm R
therefore feeds 16.3 % FEWER points into the loss than arm D at identical mask size,
and the deficit is wildly uneven across classes (post-GridSample R/D ratio: person
0.316, large_vehicle 0.520, road 0.559, car 0.604 ... terrain 1.033, manmade 1.034).

The cause is itself geometric -- the dense near-ground rings directly under the sensor
lie BELOW the camera's vertical FOV, so the frustum avoids them and its points sit in
less crowded voxels -- but that does not make the deficit harmless: a drop on person or
road in arm R is confounded with arm R having one third the supervision there.

ARM R-PRIME REMOVES EXACTLY THAT CONFOUND BY SAMPLING AT THE VOXEL LEVEL:
voxelise first, then draw the subset from the SURVIVING voxels, matched to the number
of voxels arm D actually contributes to the loss in that same frame and that same
voxelisation.  R' vs D is then a clean geometry contrast; R vs R' prices the artefact.

G1  PER-FRAME COUNT MATCHING AT THE VOXEL LEVEL -- not the mask size, not a global
    average, not a fixed fraction.  k is recomputed HERE, after GridSample, from arm D's
    OWN post-grid supervised set in the frame and voxelisation in hand:

        arm D sets segment := ignore outside the frustum BEFORE GridSample, so the
        voxels arm D supervises are exactly those whose surviving representative is
        (in-frustum AND label-valid).  That is `d_sup` below -- the same predicate,
        evaluated after the sample instead of before it.  tools/verify_arm_Rprime.py
        checks the two against each other through the PRODUCTION pipelines rather than
        trusting this paragraph.

    THE POOL IS THE LABEL-VALID SURVIVING VOXELS, NOT ALL OF THEM.  A voxel whose
    representative carries an EXCLUDED SemanticKITTI class holds ignore_index in every
    arm and can never contribute to the loss; drawing such a voxel would leave arm R'
    with FEWER loss-carrying voxels than arm D and break the very count match this arm
    exists to establish.  With the pool restricted this way,
    |sel| == |sel & valid| == k == |d_sup| exactly, for every frame, by construction.
    This is the same restriction, and the same reason, as arm R's F1 at point level.
    The size of the alternative reading is MEASURED, not argued: verify_arm_Rprime.py
    reports how many surviving voxels are label-invalid.

    k <= pool always holds without a clip, because d_sup = (frustum & valid) is a
    SUBSET of valid.  It is asserted rather than clipped: a clip would silently
    under-supervise arm R'.

G2  THE SUBSET IS FIXED, NOT RESAMPLED PER EPOCH.
    `frame_priority` gives every point of a frame a rank drawn from a 128-bit BLAKE2b
    digest of (salt, sequence, frame) and NOTHING else -- not the epoch, not the
    process, not the wall clock, not PYTHONHASHSEED, not any global RNG state.  It
    travels through GridSample as an ordinary per-point key, so each surviving voxel
    inherits its representative's rank, and the selection is "the k lowest-ranked
    label-valid survivors".

    `voxel_random_select` DRAWS NOTHING.  It contains no RNG call at all, so there is
    no per-epoch resampling to suppress: the selection is a pure function of the frame's
    frozen priority vector and the voxelisation it is handed.  verify_arm_Rprime.py
    proves this the direct way, by showing numpy's global RNG state is bit-identical
    before and after the transform.

    NOTE, PLAINLY: the voxel SET itself cannot be epoch-invariant, because
    RandomRotate / RandomScale / RandomFlip / RandomJitter run BEFORE GridSample and
    move every point, so each epoch partitions the sweep differently.  Arm D is in
    exactly the same position -- its frustum is a fixed POINT set whose post-grid
    realisation changes every epoch.  What is fixed, in both arms, is the RULE and the
    per-frame randomness it consumes; what varies, in both arms, is the voxelisation.
    Arm R' is fixed in the same sense arm D is, and no stronger claim is made.

    The draw is argpartition over a PRE-DRAWN uniform vector, not rng.choice: NEP 19
    guarantees the BitGenerator stream but not that a distribution method keeps its
    algorithm between numpy releases, and this project runs two interpreters.  "The k
    smallest of a fixed vector" is the same SET under any argpartition implementation.

G3  THE CLASS DISTRIBUTION IS NOT MATCHED.  The frustum's class mix is part of what
    "geometry" means here.  Both distributions are reported; neither is equalised.
"""
import numpy as np

from random_supervise import frame_seed        # ONE definition of the seeding rule

# Arm R' draws INDEPENDENTLY of arm R.  A shared salt would make R' a nested
# sub-selection of R's own draw (R' would be the k lowest-ranked of the same ranking
# that produced R), and the R-vs-R' contrast would then carry an undisclosed dependency
# between the two arms.  Distinct salt, frozen here rather than passed in from a config
# that could drift between runs.
VOXEL_RANDOM_SALT = "armRprime-v04-20260923"


def frame_priority(seq, frame, n, salt=VOXEL_RANDOM_SALT):
    """Per-point rank in [0, 1) for one frame, in RAW point order.

    A property of the frame, not of the epoch: same sequence and frame -> same vector,
    in any process, under any PYTHONHASHSEED, on either machine.  It is carried through
    GridSample as an ordinary per-point key so that each surviving voxel inherits the
    rank of the representative point the sampler happened to keep.
    """
    return np.random.default_rng(frame_seed(seq, frame, salt)).random(int(n))


def voxel_random_select(prio, frustum, valid):
    """(sel, k, n_pool) for ONE frame, AFTER GridSample.  One entry per surviving voxel.

    prio    float[v]  the rank each surviving voxel inherited from its representative
    frustum bool[v]   the CAMERA frustum, i.e. arm D's mask, after GridSample
    valid   bool[v]   segment >= 0, i.e. the voxels any arm is allowed to supervise

    returns sel bool[v] with sel.sum() == k == (frustum & valid).sum() == arm D's
            post-GridSample supervised voxel count for this frame and this voxelisation.

    NO RANDOM NUMBER IS DRAWN HERE.  See G2.
    """
    prio = np.asarray(prio).reshape(-1)
    frustum = np.asarray(frustum).reshape(-1).astype(bool)
    valid = np.asarray(valid).reshape(-1).astype(bool)
    assert prio.shape == frustum.shape == valid.shape, \
        (prio.shape, frustum.shape, valid.shape)
    k = int(np.count_nonzero(frustum & valid))      # <- arm D's count for THIS frame
    sel = np.zeros(frustum.shape[0], dtype=bool)
    pool = np.flatnonzero(valid)
    if k == 0 or pool.size == 0:
        return sel, k, int(pool.size)
    # k <= pool.size always, since (frustum & valid) is a subset of valid.  Asserted
    # rather than clipped: a clip would silently under-supervise arm R'.
    assert k <= pool.size, (k, pool.size)
    if k == pool.size:
        sel[pool] = True
        return sel, k, int(pool.size)
    sel[pool[np.argpartition(prio[pool], k - 1)[:k]]] = True
    return sel, k, int(pool.size)
