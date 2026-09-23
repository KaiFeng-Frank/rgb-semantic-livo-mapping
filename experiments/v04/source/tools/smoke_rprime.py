#!/usr/bin/env python3
"""Standalone smoke test of arm R''s transform chain on SYNTHETIC points.
Checks the plumbing (registry, key survival through GridSample, count match,
no-RNG property) before any real data exists.  Not a substitute for
tools/verify_arm_Rprime.py, which runs the production pipelines."""
import sys, random, copy
sys.path.insert(0, "/data/wuyou/livo_sem/src")
import numpy as np
import pointcept_ext, distil_ext, distil_ext_rprime          # noqa: F401
from pointcept.datasets.transform import Compose
from voxel_random_supervise import frame_priority, voxel_random_select

rng = np.random.default_rng(0)
n = 40000
coord = rng.normal(0, 8, (n, 3)).astype(np.float32)
strength = rng.random((n, 1)).astype(np.float32) * 0.2
segment = rng.integers(-1, 9, n).astype(np.int32)      # -1 == excluded class
frustum = (np.arctan2(coord[:, 1], coord[:, 0]) > 1.2).astype(np.int32)
prio = frame_priority("00", "000000", n)

BASE = [dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
        dict(type="RandomScale", scale=[0.9, 1.1]),
        dict(type="RandomFlip", p=0.5),
        dict(type="RandomJitter", sigma=0.005, clip=0.02)]
GS = lambda keys: dict(type="GridSample", grid_size=0.05, hash_type="fnv", mode="train",
                       keys=keys, return_grid_coord=True)
TAIL = [dict(type="ToTensor"),
        dict(type="Collect", keys=("coord", "grid_coord", "segment", "frustum"),
             feat_keys=("coord", "strength"))]

pipeD = Compose(BASE + [GS(("coord", "strength", "segment", "frustum"))] + TAIL)
pipeR = Compose(BASE + [GS(("coord", "strength", "segment", "frustum", "prio")),
                        dict(type="VoxelRandomSupervise", ignore_index=-1)] + TAIL)
pipeW = Compose(BASE + [GS(("coord", "strength", "segment", "frustum", "prio"))])

def base_dict(mask_like_armD):
    seg = np.where(frustum > 0, segment, -1).astype(np.int32) if mask_like_armD else segment.copy()
    d = dict(coord=coord.copy(), strength=strength.copy(), segment=seg,
             frustum=frustum.copy())
    if not mask_like_armD:
        d["prio"] = prio.copy()
    return d

S = 4242
random.seed(S); np.random.seed(S); dD = pipeD(base_dict(True))
random.seed(S); np.random.seed(S); dR = pipeR(base_dict(False))
random.seed(S); np.random.seed(S); dW = pipeW(base_dict(False))

cD, cR = dD["coord"].numpy(), dR["coord"].numpy()
segD, segR = dD["segment"].numpy(), dR["segment"].numpy()
selR = dR["frustum"].numpy().astype(bool)
kD, kR = int((segD >= 0).sum()), int((segR >= 0).sum())

st0 = np.random.get_state(); py0 = random.getstate()
sel2, k2, pool2 = voxel_random_select(dW["prio"], dW["frustum"].astype(bool), dW["segment"] >= 0)
st1 = np.random.get_state(); py1 = random.getstate()
rng_same = (st0[0] == st1[0] and np.array_equal(st0[1], st1[1])
            and st0[2:] == st1[2:] and py0 == py1)

print("voxels after GridSample      %d   (of %d raw points)" % (cR.shape[0], n))
print("same voxelisation D vs R'    %s" % np.array_equal(cD, cR))
print("prio survived GridSample     %s  shape %s" % ("prio" in dW, dW["prio"].shape))
print("k arm D (post-grid)          %d" % kD)
print("k arm R' (post-grid)         %d   match %s" % (kR, kD == kR))
print("k recomputed by selector     %d   match %s" % (k2, k2 == kD))
print("selection == recomputation   %s" % np.array_equal(sel2, selR))
print("emitted frustum IS the sel   %s" % (int(selR.sum()) == kR))
print("pool (label-valid survivors) %d ; invalid survivors %d"
      % (pool2, cR.shape[0] - pool2))
print("selector left RNG untouched  %s" % rng_same)
print("R' labels are the real GT    %s"
      % np.array_equal(segR[selR], dW["segment"][selR]))
print("prio/seq_frame dropped       %s" % ("prio" not in dR and "seq_frame" not in dR))
ok = all([np.array_equal(cD, cR), kD == kR == k2, np.array_equal(sel2, selR),
          int(selR.sum()) == kR, rng_same,
          np.array_equal(segR[selR], dW["segment"][selR]),
          "prio" not in dR])
print("\nSMOKE %s" % ("PASS" if ok else "*** FAIL ***"))
sys.exit(0 if ok else 1)
