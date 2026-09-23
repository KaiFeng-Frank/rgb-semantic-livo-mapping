#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
random_supervise.py -- arm R's supervision selector.

Dependency-free on purpose (numpy only), for exactly the reason filter_e.py is: the
training side imports it through distil_ext (which drags in pointcept / spconv /
torch), the verification side imports it directly from an environment that has no CUDA
extension at all.  ONE definition, both sides -- a verification that re-implemented the
rule would prove nothing about the rule the trainer actually runs.

WHAT ARM R IS
-------------
Arm D supervises the camera-frustum points.  That set is NOT a random 15 % of the
sweep: it is spatially contiguous, azimuthally biased (forward only) and
range-dependent.  The frustum experiment therefore cannot separate

    "sparse supervision propagates"   from   "sparse supervision OF THIS GEOMETRY
                                               propagates".

Arm R holds the COUNT fixed and destroys the GEOMETRY: the same number of supervised
points per frame, drawn uniformly at random over the full 360 deg sweep.

F1  PER-FRAME COUNT MATCHING -- not a global average, not a fixed fraction.
    k is recomputed HERE, for the frame in hand, from arm D's OWN mask
    (frustum & label-valid).  There is no precomputed count table, so there is nothing
    that can go stale or drift away from what arm D actually did.

    The draw is from the LABEL-VALID points, not from all points.  A point whose GT
    class is EXCLUDED from common-9 carries ignore_index in EVERY arm and can never be
    supervised by anybody; drawing such a point would silently leave arm R with FEWER
    supervised points than arm D and break F1.  With the pool restricted this way,
    |selected| == |selected & valid| == k exactly, for every frame, by construction.

F2  THE SUBSET IS FIXED ACROSS EPOCHS, NOT RESAMPLED.
    The seed is a 128-bit BLAKE2b digest of (salt, sequence, frame) and nothing else.
    It does not depend on the epoch, the process, the wall clock, PYTHONHASHSEED or any
    global RNG state, so all 10 epochs of arm R see the SAME point set -- exactly as
    arm D's frustum is the same set in all 10 epochs.  If R redrew each epoch it would
    see far more distinct points over training and would win for the wrong reason.

    The draw is rng.random() + argpartition, NOT rng.choice(replace=False).  NEP 19
    guarantees the BitGenerator stream; it does not guarantee that a distribution
    method keeps its algorithm between numpy releases, and this project runs two
    interpreters (ags: numpy 1.26.4, ptv3: numpy 2.2.6).  "The k smallest of a uniform
    draw" is the same SET under any argpartition implementation, so the mask is
    reproducible across both -- which tools/verify_arm_R.py checks rather than assumes.
"""
import hashlib

import numpy as np

# Salt for arm R's per-frame seeds.  Changing it changes every frame's subset, so it is
# frozen here rather than passed in from a config that could drift between runs.
RANDOM_SUPERVISE_SALT = "armR-v04-20260923"


def frame_seed(seq, frame, salt=RANDOM_SUPERVISE_SALT):
    """Deterministic 128-bit seed for one frame.  Stable across processes, machines,
    python versions and PYTHONHASHSEED -- python's builtin hash() is none of those."""
    key = ("%s|%s|%s" % (salt, seq, frame)).encode("ascii")
    return int.from_bytes(hashlib.blake2b(key, digest_size=16).digest(), "big")


def random_supervise_mask(frustum, valid, seq, frame, salt=RANDOM_SUPERVISE_SALT):
    """(sel, k) for one frame.

    frustum : bool[n]  the camera-frustum mask, i.e. arm D's selection
    valid   : bool[n]  segment >= 0, i.e. the points any arm is allowed to supervise
    returns   sel bool[n] with sel.sum() == k == (frustum & valid).sum()
    """
    frustum = np.asarray(frustum).reshape(-1).astype(bool)
    valid = np.asarray(valid).reshape(-1).astype(bool)
    assert frustum.shape == valid.shape, (frustum.shape, valid.shape)
    k = int(np.count_nonzero(frustum & valid))          # <- arm D's count for THIS frame
    sel = np.zeros(frustum.shape[0], dtype=bool)
    pool = np.flatnonzero(valid)
    if k == 0 or pool.size == 0:
        return sel, k
    # k <= pool.size always, since (frustum & valid) is a subset of valid.  Asserted
    # rather than clipped: a clip would silently under-supervise arm R.
    assert k <= pool.size, (seq, frame, k, pool.size)
    if k == pool.size:
        sel[pool] = True
        return sel, k
    rng = np.random.default_rng(frame_seed(seq, frame, salt))
    u = rng.random(pool.size)
    sel[pool[np.argpartition(u, k - 1)[:k]]] = True
    return sel, k
