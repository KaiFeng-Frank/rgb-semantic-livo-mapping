#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""STEP 7 -- per-frame latency of each arm on the 4090.  EXCLUSIVE GPU.

Every stage is timed with torch.cuda.synchronize() around it, after a warm-up, over
the same frames for every arm.  mean / p50 / p95 / max are reported.

The 2D arm's cost INCLUDES the projection, the z-buffer visibility test and the
per-point sampling, because a deployed 2D->3D arm has to pay them.  They are timed as
their own stage so a reader who disagrees can subtract them.
"""
import os, sys, json, time, argparse
os.environ.setdefault("HF_HOME", "/data/hf_cache")
sys.path.insert(0, "/data/livo_sem/src")
import numpy as np

INTENSITY_SCALE, GRID = 0.2, 0.05


def stats(v):
    v = np.asarray(v, float)
    return dict(mean=float(v.mean()), p50=float(np.percentile(v, 50)),
                p95=float(np.percentile(v, 95)), max=float(v.max()), n=int(v.size))


def voxelize(coord, stren):
    g = np.floor(coord / GRID).astype(np.int64)
    g -= g.min(0)
    key = (g[:, 0] * (g[:, 1].max() + 1) + g[:, 1]) * (g[:, 2].max() + 1) + g[:, 2]
    _uk, first, inv = np.unique(key, return_index=True, return_inverse=True)
    return (np.ascontiguousarray(coord[first]), np.ascontiguousarray(stren[first]),
            np.ascontiguousarray(g[first]), inv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["3d", "2d"])
    ap.add_argument("--model", default="eomt")
    ap.add_argument("--scale", default="half_focal")
    ap.add_argument("--tta", default="none")
    ap.add_argument("--seq", default="07")
    ap.add_argument("--nframes", type=int, default=100)
    ap.add_argument("--stride", type=int, default=11)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import torch
    import score_2d_vs_3d as SC
    SC.set_sequence(a.seq)
    frames = [f for f in range(0, a.stride * a.nframes, a.stride)
              if os.path.exists("%s/%010d.bin" % (SC.SCANS, f))]
    proj = SC.Projector()
    sync = torch.cuda.synchronize
    T, INFO = {}, {}

    def rec(k, dt):
        T.setdefault(k, []).append(dt * 1e3)

    if a.arm == "3d":
        from ptv3_loader_verified import build_ptv3
        model, _cfg = build_ptv3(device="cuda", shuffle_orders=False, half=True)

        def one(f, timed):
            t0 = time.perf_counter()
            pts = SC.read_scan(f)
            t1 = time.perf_counter()
            coord = np.ascontiguousarray(pts[:, :3], np.float32)
            stren = np.ascontiguousarray(pts[:, 3:4] * INTENSITY_SCALE, np.float32)
            cv, sv, gv, inv = voxelize(coord, stren)
            with torch.inference_mode():
                c = torch.from_numpy(cv).cuda(); s = torch.from_numpy(sv).cuda()
                g = torch.from_numpy(gv).cuda()
                feat = torch.cat([c.half(), s.half()], 1)
                sync(); t2 = time.perf_counter()
                out = model(dict(coord=c.half(), grid_coord=g, feat=feat,
                                 offset=torch.tensor([c.shape[0]], device="cuda",
                                                     dtype=torch.long)))["seg_logits"]
                sync(); t3 = time.perf_counter()
                prob = torch.softmax(out.float(), -1)
                cf, lb = prob.max(-1)
                _lab = lb.cpu().numpy()[inv]
                _cf = cf.cpu().numpy()[inv]
                sync(); t4 = time.perf_counter()
            if timed:
                rec("read_scan", t1 - t0); rec("pre_voxelize", t2 - t1)
                rec("forward", t3 - t2); rec("post_devoxelize", t4 - t3)
                rec("total_excl_read", t4 - t1); rec("total", t4 - t0)
            INFO.update(points=int(len(pts)), voxels=int(cv.shape[0]))
    else:
        from seg2d_infer import Seg2DSegmenter, SCALE_PRESETS, MODELS
        sc = a.scale if a.scale in SCALE_PRESETS else float(a.scale)
        seg = Seg2DSegmenter(a.model, device="cuda", scale=sc, tta=a.tta, verbose=True)
        lut = SC.city19_to_coarse()

        def one(f, timed):
            t0 = time.perf_counter()
            img = seg._as_rgb("%s/%010d.png" % (SC.IMAGES, f))
            pts = SC.read_scan(f)
            t1 = time.perf_counter()
            lab, _conf, sx, sy, _hw = seg._hires(img)
            sync(); t2 = time.perf_counter()
            u, v, z, inm = proj.project(pts[:, :3].astype(np.float64))
            ui = np.clip(np.rint((u[inm] + 0.5) * sx - 0.5), 0, lab.shape[1] - 1).astype(np.int64)
            vi = np.clip(np.rint((v[inm] + 0.5) * sy - 0.5), 0, lab.shape[0] - 1).astype(np.int64)
            city = lab[vi, ui].astype(np.int64)
            vis = SC.visible_mask(np.rint(u[inm]).astype(np.int64),
                                  np.rint(v[inm]).astype(np.int64), z[inm])
            pr = np.full(len(pts), SC.ABSTAIN, np.int16)
            pr[inm] = np.where(vis, lut[np.where(city < 20, city, 19)], SC.ABSTAIN)
            t3 = time.perf_counter()
            if timed:
                rec("read_image_and_scan", t1 - t0)
                rec("forward_incl_pre_post", t2 - t1)
                rec("project_zbuffer_sample_cpu", t3 - t2)
                rec("total_excl_read", t3 - t1); rec("total", t3 - t0)
            INFO.update(points=int(len(pts)), network_hw=[int(lab.shape[0]), int(lab.shape[1])],
                        in_frustum=int(inm.sum()))

    for i in range(a.warmup):
        one(frames[i % len(frames)], False)
    sync()
    for f in frames:
        one(f, True)

    meta = dict(arm=a.arm, hardware="NVIDIA RTX 4090 (exclusive)", precision="fp16",
                frames=len(frames), seq=a.seq, batch=1, info=INFO,
                frame_ids=[int(f) for f in frames],
                stages_ms={k: stats(v) for k, v in T.items()},
                # RAW per-frame series, in frame_ids order.  Needed because the hybrid
                # bracket is a per-frame max (parallel) / per-frame sum (serial) of the
                # two arms, and a max of two p95s is not the p95 of the per-frame max.
                raw_ms={k: [float(x) for x in v] for k, v in T.items()})
    if a.arm == "2d":
        meta.update(model=a.model, repo=MODELS.get(a.model, {}).get("repo"),
                    input_scale=float(seg.scale), scale_key=a.scale, tta=a.tta,
                    input_hw=INFO.get("network_hw"))
    else:
        meta.update(model="PTv3 Pointcept v1.5.1 nuScenes-16", tta="none",
                    intensity_scale=INTENSITY_SCALE, grid_size=GRID)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(meta, open(a.out, "w"), indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
