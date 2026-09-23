#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cache_trained.py -- put a TRAINED arm into the SAME on-disk contract arm A uses, so
src/score_2d_vs_3d.py can score them against each other without a single change.

Hard rules it obeys (from the v0.4 handoff):
  * the cache is `space="nusc16"`: `lab` is the argmax over the 16-way nuScenes head,
    NOT a 9-way label.  Writing a 9-way cache would bypass Arm3D's UNMAPPED accounting
    for nuScenes other_flat and silently delete a class of error.
  * points in the RAW .bin order, so no `order` array is needed.
  * identical preprocessing to tools/cache_ptv3.py: intensity x0.2, grid 0.05, fp16,
    shuffle_orders=False, single forward, no TTA.  Changing any of these between arms
    is a confound.

  python tools/cache_trained.py --ckpt exp/sk/armB1/model/model_best.pth \
         --seq 07 --out out/v04/B1_r1
"""
import os, sys, time, json, argparse
sys.path.insert(0, "/data/wuyou/livo_sem/src")
import numpy as np

INTENSITY_SCALE = 0.2      # MEASURED, CRITICAL_CONSTRAINTS C3
GRID = 0.05


def voxelize(coord, strength, grid):
    g = np.floor(coord / grid).astype(np.int64)
    g -= g.min(0)
    key = (g[:, 0] * (g[:, 1].max() + 1) + g[:, 1]) * (g[:, 2].max() + 1) + g[:, 2]
    uk, first, inv = np.unique(key, return_index=True, return_inverse=True)
    return (np.ascontiguousarray(coord[first]), np.ascontiguousarray(strength[first]),
            np.ascontiguousarray(g[first]), inv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seq", default="07")
    ap.add_argument("--frames", default="all")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--warmup", type=int, default=5)
    a = ap.parse_args()

    import torch
    import pointcept_ext          # noqa: F401  stubs first
    import distil_ext             # noqa: F401  registers DistilSegmentorMiB
    import seqreg
    from pointcept.utils.config import Config
    from pointcept.models import build_model

    CFG = ("/data/wuyou/livo_sem/src/Pointcept_v151/configs/semantic_kitti/"
           "semseg-pt-v3m1-distil-common9.py")
    cfg = Config.fromfile(CFG)
    cfg.model.backbone.shuffle_orders = False      # pinned, as arm A was measured
    cfg.model.kl_enabled = False                   # inference: no anchor needed
    model = build_model(cfg.model)

    sd = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd)
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    sd = {k: v for k, v in sd.items() if not k.startswith("frozen_")}
    info = model.load_state_dict(sd, strict=False)
    missing = [k for k in info.missing_keys if not k.startswith("frozen_")]
    assert not missing, ("missing non-frozen keys", missing[:10])
    assert not info.unexpected_keys, ("unexpected keys", info.unexpected_keys[:10])
    print("loaded %d tensors; %d frozen-anchor keys left at init (unused at inference)"
          % (len(sd), len(info.missing_keys) - len(missing)), flush=True)

    model = model.to(a.device).eval().half()
    for p in model.parameters():
        p.requires_grad_(False)

    SC, _proj, _W, _H = seqreg.use(a.seq)
    frames = SC.frame_list(a.frames)
    os.makedirs(a.out, exist_ok=True)

    def run(pts):
        coord = np.ascontiguousarray(pts[:, :3], dtype=np.float32)
        stren = np.ascontiguousarray(pts[:, 3:4] * INTENSITY_SCALE, dtype=np.float32)
        cv, sv, gv, inv = voxelize(coord, stren, GRID)
        with torch.inference_mode():
            c = torch.from_numpy(cv).to(a.device)
            s = torch.from_numpy(sv).to(a.device)
            g = torch.from_numpy(gv).to(a.device)
            feat = torch.cat([c.half(), s.half()], 1)
            d = dict(coord=c.half(), grid_coord=g, feat=feat,
                     offset=torch.tensor([c.shape[0]], device=a.device, dtype=torch.long))
            logits16 = model._logits16(d)          # the 16-way head, exactly as arm A
            prob = torch.softmax(logits16.float(), -1)
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
    json.dump(dict(ckpt=a.ckpt, seq=a.seq, n_frames=len(frames), space="nusc16",
                   intensity_scale=INTENSITY_SCALE, grid=GRID, shuffle_orders=False,
                   tta=False, seconds=time.time() - t0),
              open(os.path.join(a.out, "meta.json"), "w"), indent=2)
    print("DONE %d frames in %.1f s -> %s" % (len(frames), time.time() - t0, a.out))


if __name__ == "__main__":
    main()
