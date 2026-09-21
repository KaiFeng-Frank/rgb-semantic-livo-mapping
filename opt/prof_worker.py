#!/usr/bin/env python3
"""STEP 2: one profiling run. Splits the worker's 'inference' ms into
(a) numpy voxel prep, (b) H2D + GPU unique/argsort, (c) model forward,
(d) softmax + scatter-back + D2H.  Then torch.profiler for the kernel breakdown."""
import os, sys, glob, time, json
os.environ.setdefault("PTV3_SHUFFLE", "0")
sys.path.insert(0, "/data/livo_sem/src")
import numpy as np, torch
import ptv3_worker as W

HALF = int(os.environ.get("H", "1"))
os.environ["PTV3_HALF"] = str(HALF)
D = "/data/livo_sem/data/raw/2011_09_30/2011_09_30_drive_0027_sync/velodyne_points/data/"
files = sorted(glob.glob(D + "*.bin"))
frames = [np.fromfile(files[i], dtype=np.float32).reshape(-1, 4) for i in (0, 250, 500, 750, 1000)]
seg = W.Segmenter(half=bool(HALF), shuffle=False, fast_voxel=True)
for _ in range(5):
    seg.segment(frames[0])
torch.cuda.synchronize()

def timed(pts):
    t = {}
    torch.cuda.synchronize(); t0 = time.perf_counter()
    coord = np.ascontiguousarray(pts[:, :3], dtype=np.float32)
    strength = np.ascontiguousarray(pts[:, 3:4] * seg.intensity_scale, dtype=np.float32)
    # --- (a) numpy prep only (the float64 divide/floor/min/max/ravel)
    scaled = coord.astype(np.float64); scaled /= seg.grid_size
    np.floor(scaled, out=scaled)
    gT = scaled.T.astype(np.int32); gT -= gT.min(axis=1)[:, None]
    ext = (gT.max(axis=1) + 1).astype(np.int64)
    key = gT[0].astype(np.int64) * ext[1]; key += gT[1]; key *= ext[2]; key += gT[2]
    t["a_numpy_prep"] = (time.perf_counter() - t0) * 1e3
    # --- (b) GPU unique/argsort
    t1 = time.perf_counter()
    kt = torch.from_numpy(key).cuda(non_blocking=True)
    _, inverse = torch.unique(kt, sorted=True, return_inverse=True)
    order = torch.argsort(inverse, stable=True)
    sc = inverse[order]
    first = torch.ones_like(sc, dtype=torch.bool); first[1:] = sc[1:] != sc[:-1]
    idx_first = order[first].cpu().numpy()
    t["b_gpu_unique"] = (time.perf_counter() - t1) * 1e3
    t1 = time.perf_counter()
    cv = coord[idx_first]; sv = strength[idx_first]
    gv = np.ascontiguousarray(gT[:, idx_first].T)
    t["c_gather"] = (time.perf_counter() - t1) * 1e3
    # --- (d) forward
    fd = seg.feat_dtype
    with torch.inference_mode():
        t1 = time.perf_counter()
        c = torch.from_numpy(cv).cuda(); s = torch.from_numpy(sv).cuda(); gc = torch.from_numpy(gv).cuda()
        feat = torch.cat([c.to(fd), s.to(fd)], dim=1)
        inp = dict(coord=c.to(fd), grid_coord=gc, feat=feat,
                   offset=torch.tensor([c.shape[0]], device="cuda", dtype=torch.long))
        torch.cuda.synchronize(); t["d_h2d"] = (time.perf_counter() - t1) * 1e3
        t1 = time.perf_counter()
        logits = seg.model(inp)["seg_logits"]
        torch.cuda.synchronize(); t["e_forward"] = (time.perf_counter() - t1) * 1e3
        t1 = time.perf_counter()
        prob = torch.softmax(logits.float(), dim=-1)
        conf_v, lab_v = prob.max(dim=-1)
        lab = lab_v[inverse].to(torch.int16).cpu().numpy()
        conf = conf_v[inverse].cpu().numpy()
        torch.cuda.synchronize(); t["f_scatter_d2h"] = (time.perf_counter() - t1) * 1e3
    t["TOTAL"] = sum(v for k, v in t.items())
    t["voxels"] = len(cv)
    return t

acc = {}
N = 6
for rep in range(N):
    for fr in frames:
        r = timed(fr)
        for k, v in r.items():
            acc.setdefault(k, []).append(v)
print("=== worker stage split (half=%d), %d frames x %d reps ===" % (HALF, len(frames), N))
for k in ["a_numpy_prep", "b_gpu_unique", "c_gather", "d_h2d", "e_forward", "f_scatter_d2h", "TOTAL"]:
    v = np.array(acc[k]); print("  %-16s mean %7.3f  p95 %7.3f  max %7.3f" % (k, v.mean(), np.percentile(v, 95), v.max()))
print("  voxels mean %.0f" % np.mean(acc["voxels"]))

# ---- kernel-level breakdown on one warm frame
from torch.profiler import profile, ProfilerActivity
pts = frames[0]
seg.segment(pts); torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    seg.segment(pts)
    torch.cuda.synchronize()
ka = prof.key_averages()
rows = sorted(ka, key=lambda e: -e.self_device_time_total)[:18]
tot = sum(e.self_device_time_total for e in ka)
print("\n=== top CUDA kernels (self device us), total device %.1f ms ===" % (tot / 1e3))
for e in rows:
    if e.self_device_time_total <= 0: continue
    print("  %-52s %8.0f us  %5.1f%%  n=%d" % (e.key[:52], e.self_device_time_total,
          100 * e.self_device_time_total / max(tot, 1), e.count))
cpu = sorted(ka, key=lambda e: -e.self_cpu_time_total)[:10]
print("\n=== top CPU self time (us) ===")
for e in cpu:
    print("  %-52s %8.0f us  n=%d" % (e.key[:52], e.self_cpu_time_total, e.count))
