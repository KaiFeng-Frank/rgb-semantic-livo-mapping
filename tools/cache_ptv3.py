#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cache_ptv3.py -- one GPU pass over a sequence caching the 3D arm's per-point argmax.

Loader: src/ptv3_loader_verified.build_ptv3 -- Pointcept v1.5.1 (the version the
released nuScenes checkpoint was trained against; 488/488 tensors, strict=True).
NEVER src/ptv3_infer.py (Pointcept HEAD, DO-NOT-USE, car IoU 0).

MEASURED CONSTANTS, not guesses (CRITICAL_CONSTRAINTS.md):
    intensity x 0.2     KITTI .bin intensity is [0,1]; nuScenes trains on /255 values
                        concentrated near 0.06.  Feeding it unscaled collapses road
                        IoU from 84 to 8.
    grid_size 0.05      one representative point per voxel, scattered back via the
                        inverse index.
    fp16 weights        the checkpoint was TRAINED under fp16 autocast and flash-attn
                        casts qkv to fp16 inside every block anyway.
    shuffle_orders=False  v1.5.1 applies shuffle_orders inside forward() ungated by
                        self.training, so .eval() does NOT disable it.  Pinning it
                        removes ONE source of run-to-run non-determinism; the model
                        is still non-deterministic, which is why --tag exists and why
                        every number is quoted with the spread over repeats.
    TTA off             the config ships a 10-way multi-scale+flip TTA that would
                        multiply latency by 10.

Points are fed in the RAW .bin order, which is the order the .label file indexes,
so no permutation is written and P0 is trivially satisfied.

    python tools/cache_ptv3.py --seq 07 --out .../ptv3_r0
"""
import os, sys, time, argparse
sys.path.insert(0, "/data/livo_sem/src")
import numpy as np

INTENSITY_SCALE = 0.2
GRID = 0.05


def voxelize(coord, strength, grid, torch):
    g = np.floor(coord / grid).astype(np.int64)
    g -= g.min(0)
    key = (g[:, 0] * (g[:, 1].max() + 1) + g[:, 1]) * (g[:, 2].max() + 1) + g[:, 2]
    uk, first, inv = np.unique(key, return_index=True, return_inverse=True)
    return (np.ascontiguousarray(coord[first]), np.ascontiguousarray(strength[first]),
            np.ascontiguousarray(g[first]), inv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="07")
    ap.add_argument("--frames", default="all")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--warmup", type=int, default=5)
    a = ap.parse_args()

    import torch
    import score_2d_vs_3d as SC
    from ptv3_loader_verified import build_ptv3
    SC.set_sequence(a.seq)
    frames = SC.frame_list(a.frames)
    os.makedirs(a.out, exist_ok=True)

    model, cfg = build_ptv3(device=a.device, shuffle_orders=False, half=True)
    assert int(cfg.model.num_classes) == 16, cfg.model.num_classes

    def run(pts):
        coord = np.ascontiguousarray(pts[:, :3], dtype=np.float32)
        stren = np.ascontiguousarray(pts[:, 3:4] * INTENSITY_SCALE, dtype=np.float32)
        cv, sv, gv, inv = voxelize(coord, stren, GRID, torch)
        with torch.inference_mode():
            c = torch.from_numpy(cv).to(a.device)
            s = torch.from_numpy(sv).to(a.device)
            g = torch.from_numpy(gv).to(a.device)
            feat = torch.cat([c.half(), s.half()], 1)
            out = model(dict(coord=c.half(), grid_coord=g, feat=feat,
                             offset=torch.tensor([c.shape[0]], device=a.device,
                                                 dtype=torch.long)))["seg_logits"]
            prob = torch.softmax(out.float(), -1)
            cf, lb = prob.max(-1)
            return (lb.cpu().numpy()[inv].astype(np.uint8),
                    cf.cpu().numpy()[inv].astype(np.float16))

    for i in range(a.warmup):
        run(SC.read_scan(frames[i % len(frames)]))
    torch.cuda.synchronize()

    t0 = time.time()
    for i, f in enumerate(frames):
        pts = SC.read_scan(f)
        lab, conf = run(pts)
        assert len(lab) == len(pts)
        np.savez("%s/f%06d.npz" % (a.out, f), lab=lab, conf=conf)
        if (i + 1) % 200 == 0:
            print("  %d/%d  %.1f s" % (i + 1, len(frames), time.time() - t0), flush=True)
    print("DONE %d frames in %.1f s -> %s" % (len(frames), time.time() - t0, a.out))


if __name__ == "__main__":
    main()
