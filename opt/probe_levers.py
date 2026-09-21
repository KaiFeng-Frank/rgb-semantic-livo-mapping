#!/usr/bin/env python3
"""(1) Is CUDA float64 divide+floor bit-identical to numpy's on real scans?
   (2) How much of the 56 ms forward is the pure-python Hilbert bit loop?"""
import os, sys, glob, time
os.environ.setdefault("PTV3_SHUFFLE", "0"); os.environ.setdefault("PTV3_HALF", "1")
sys.path.insert(0, "/data/livo_sem/src")
import numpy as np, torch
import ptv3_worker as W

D = "/data/livo_sem/data/raw/2011_09_30/2011_09_30_drive_0027_sync/velodyne_points/data/"
files = sorted(glob.glob(D + "*.bin"))

print("=== (1) CUDA float64 divide/floor vs numpy, real scans, grid 0.05 ===")
bad = 0; tot = 0
for i in range(0, 1100, 50):
    p = np.fromfile(files[i], dtype=np.float32).reshape(-1, 4)
    c = np.ascontiguousarray(p[:, :3], dtype=np.float32)
    gs = 0.05
    cpu = np.floor(c.astype(np.float64) / gs).astype(np.int64)
    t = torch.from_numpy(c).cuda().to(torch.float64)
    gpu = torch.floor(t / gs).to(torch.int64).cpu().numpy()
    d = int((cpu != gpu).sum()); bad += d; tot += cpu.size
print("  differing cell indices: %d of %d  (%.3e)" % (bad, tot, bad / tot))

print("\n=== (2) serialization cost inside the forward ===")
seg = W.Segmenter(half=True, shuffle=False, fast_voxel=True)
import importlib
ST = importlib.import_module("pointcept.models.utils.structure")
SER = importlib.import_module("pointcept.models.utils.serialization")
pts = np.fromfile(files[0], dtype=np.float32).reshape(-1, 4)
for _ in range(4): seg.segment(pts)
torch.cuda.synchronize()

acc = {}
orig_ser = ST.Point.serialization
def timed_ser(self, order=("z",), depth=None, shuffle_orders=False):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    r = orig_ser(self, order=order, depth=depth, shuffle_orders=shuffle_orders)
    torch.cuda.synchronize(); acc.setdefault("serialization", []).append((time.perf_counter()-t0)*1e3)
    return r
ST.Point.serialization = timed_ser
orig_enc = SER.encode
def timed_enc(grid_coord, batch=None, depth=16, order="z"):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    r = orig_enc(grid_coord, batch, depth, order)
    torch.cuda.synchronize(); acc.setdefault("enc_"+order, []).append((time.perf_counter()-t0)*1e3)
    return r
SER.encode = timed_enc
ST.encode = timed_enc

for _ in range(6): seg.segment(pts)
for k in sorted(acc):
    v = np.array(acc[k]); print("  %-22s calls/frame %4.1f   total %6.3f ms/frame" % (k, len(v)/6.0, v.sum()/6.0))
