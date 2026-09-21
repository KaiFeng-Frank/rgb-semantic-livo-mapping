#!/usr/bin/env python3
"""End-to-end timing of Segmenter.segment() across configurations. GPU exclusive."""
import os, sys, glob, time, json
os.environ.setdefault("PTV3_SHUFFLE", "0")
sys.path.insert(0, "/data/livo_sem/src")
import numpy as np, torch

cfgs = json.loads(sys.argv[1])
D = "/data/livo_sem/data/raw/2011_09_30/2011_09_30_drive_0027_sync/velodyne_points/data/"
files = sorted(glob.glob(D + "*.bin"))
idx = list(range(0, 1100, 9))          # 123 real frames
frames = [np.fromfile(files[i], dtype=np.float32).reshape(-1, 4) for i in idx]

import ptv3_worker as W
for cfg in cfgs:
    seg = W.Segmenter(**{k: v for k, v in cfg.items() if k != "tag"})
    for _ in range(8):
        seg.segment(frames[0])
    torch.cuda.synchronize()
    ts = []
    for f in frames:
        t0 = time.perf_counter(); seg.segment(f); ts.append((time.perf_counter() - t0) * 1e3)
    ts = np.array(ts)
    print("%-28s n=%d  mean %6.2f  p50 %6.2f  p95 %6.2f  max %6.2f  voxels %d"
          % (cfg.get("tag", ""), len(ts), ts.mean(), np.percentile(ts, 50),
             np.percentile(ts, 95), ts.max(), seg.last_voxels))
    del seg
    torch.cuda.empty_cache()
