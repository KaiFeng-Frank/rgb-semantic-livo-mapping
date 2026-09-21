#!/usr/bin/env python3
"""Bit-identity + timing of ptv3_fast.voxelize_gpu and hilbert_encode_fast."""
import os, sys, glob, time
os.environ.setdefault("PTV3_SHUFFLE", "0"); os.environ.setdefault("PTV3_HALF", "1")
sys.path.insert(0, "/data/livo_sem/src")
import numpy as np, torch
import ptv3_worker as W
import ptv3_fast as F

D = "/data/livo_sem/data/raw/2011_09_30/2011_09_30_drive_0027_sync/velodyne_points/data/"
files = sorted(glob.glob(D + "*.bin"))
seg = W.Segmenter(half=True, shuffle=False, fast_voxel=True)   # brings pointcept onto sys.path
import importlib
HREF = importlib.import_module("pointcept.models.utils.serialization.hilbert")
DEF = importlib.import_module("pointcept.models.utils.serialization.default")

print("=== hilbert_encode_fast vs reference ===")
bad = 0; nchk = 0
rng = np.random.default_rng(0)
for depth in (10, 11, 12, 13):
    for trial in range(3):
        g = torch.from_numpy(rng.integers(0, 1 << depth, size=(60000, 3)).astype(np.int64)).cuda()
        a = HREF.encode(g, num_dims=3, num_bits=depth)
        b = F.hilbert_encode_fast(g, 3, depth)
        d = int((a != b).sum()); bad += d; nchk += g.shape[0]
        # the transposed variant too
        gt = g[:, [1, 0, 2]]
        a2 = HREF.encode(gt, num_dims=3, num_bits=depth)
        b2 = F.hilbert_encode_fast(gt, 3, depth)
        bad += int((a2 != b2).sum()); nchk += g.shape[0]
print("  random coords: %d mismatches out of %d codes (depths 10-13, xyz + yxz)" % (bad, nchk))

# real grid coords
bad = 0; nchk = 0
for i in range(0, 1100, 137):
    p = np.fromfile(files[i], dtype=np.float32).reshape(-1, 4)
    c = np.ascontiguousarray(p[:, :3], dtype=np.float32)
    s = np.ascontiguousarray(p[:, 3:4] * 0.2, dtype=np.float32)
    cv, sv, gv, inv = F.voxelize_gpu(c, s, 0.05)
    depth = int(gv.max()).bit_length()
    a = HREF.encode(gv.long(), num_dims=3, num_bits=depth)
    b = F.hilbert_encode_fast(gv.long(), 3, depth)
    bad += int((a != b).sum()); nchk += gv.shape[0]
print("  real scans   : %d mismatches out of %d codes (depth %d)" % (bad, nchk, depth))

print("\n=== voxelize_gpu vs the shipped numpy+GPU path ===")
badc = bads = badg = badi = 0
for i in range(0, 1100, 50):
    p = np.fromfile(files[i], dtype=np.float32).reshape(-1, 4)
    c = np.ascontiguousarray(p[:, :3], dtype=np.float32)
    s = np.ascontiguousarray(p[:, 3:4] * 0.2, dtype=np.float32)
    a = W.voxelize(c, s, 0.05, torch, fast=True)
    b = F.voxelize_gpu(c, s, 0.05)
    badc += 0 if np.array_equal(a[0], b[0].cpu().numpy()) else 1
    bads += 0 if np.array_equal(a[1], b[1].cpu().numpy()) else 1
    badg += 0 if np.array_equal(a[2], b[2].cpu().numpy()) else 1
    badi += 0 if torch.equal(a[3], b[3]) else 1
print("  22 scans: coord %d / strength %d / grid_coord %d / inverse %d mismatching frames"
      % (badc, bads, badg, badi))

print("\n=== timing ===")
p = np.fromfile(files[0], dtype=np.float32).reshape(-1, 4)
c = np.ascontiguousarray(p[:, :3], dtype=np.float32)
s = np.ascontiguousarray(p[:, 3:4] * 0.2, dtype=np.float32)
for name, fn in (("numpy+gpu (shipped)", lambda: W.voxelize(c, s, 0.05, torch, fast=True)),
                 ("voxelize_gpu       ", lambda: F.voxelize_gpu(c, s, 0.05))):
    for _ in range(5): fn()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(30): fn()
    torch.cuda.synchronize()
    print("  %s %6.2f ms" % (name, (time.perf_counter()-t0)/30*1e3))
gv = F.voxelize_gpu(c, s, 0.05)[2].long()
for name, fn in (("hilbert reference", lambda: HREF.encode(gv, num_dims=3, num_bits=12)),
                 ("hilbert fast     ", lambda: F.hilbert_encode_fast(gv, 3, 12))):
    for _ in range(3): fn()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(20): fn()
    torch.cuda.synchronize()
    print("  %s %6.2f ms" % (name, (time.perf_counter()-t0)/20*1e3))
