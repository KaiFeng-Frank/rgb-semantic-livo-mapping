#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/v05_zoom_region.py -- find the map region where two models' LIVE maps disagree most,
so the RViz zoom pair (M3) is chosen by measurement, not by eye.

  --a map_ZS.npz --b map_B0.npz     the two node-written maps (xyz float32 centroids, cls uint16)
  --gt replaymap.npz                a replay_v05 --save-map file (key, gt_cls, gt_cnt): per-voxel
                                    SemanticKITTI GT majority in common-9, joined by voxel key
  --tile 10.0                       xy tile edge (m)

Voxels are matched by their voxel key (0.20 m world grid).  For every tile: number of matched
voxels, number whose fused class differs between A and B, and -- where GT is available -- the
point-weighted accuracy of A and of B inside the tile.  Prints the top tiles.
"""
import argparse, sys, json
import numpy as np
sys.path.insert(0, "/data/livo_sem/src")
import sem_core as S
import label_spaces as LS

NU = np.asarray(LS.nusc_lut(), np.int64)


def load(p, voxel):
    z = np.load(p)
    xyz = z["xyz"].astype(np.float64)
    key = S.voxel_key(xyz, voxel)
    cls = z["cls"].astype(np.int64)
    o = np.argsort(key, kind="stable")
    return xyz[o], key[o], cls[o]


def join(ka, va, kb):
    i = np.searchsorted(ka, kb); i = np.minimum(i, len(ka) - 1)
    hit = ka[i] == kb
    return i, hit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
    ap.add_argument("--gt", default=None)
    ap.add_argument("--voxel", type=float, default=0.20)
    ap.add_argument("--tile", type=float, default=10.0)
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args()
    xa, ka, ca = load(a.a, a.voxel)
    xb, kb, cb = load(a.b, a.voxel)
    i, hit = join(ka, ca, kb)
    xyz = xb[hit]; k = kb[hit]; A = ca[i[hit]]; B = cb[hit]
    print("A %d voxels, B %d voxels, matched by key %d (%.1f %% of B)" % (len(ka), len(kb), hit.sum(), 100.0 * hit.mean()))
    A9 = NU[A]; B9 = NU[B]
    diff = A9 != B9
    gt = None
    if a.gt:
        z = np.load(a.gt)
        gk = z["key"].astype(np.int64); o = np.argsort(gk, kind="stable")
        gk = gk[o]; gc = z["gt_cls"].astype(np.int64)[o]; gn = z["gt_cnt"].astype(np.int64)[o]
        j, h2 = join(gk, gc, k)
        gt = np.full(len(k), -1, np.int64); gcnt = np.zeros(len(k), np.int64)
        gt[h2] = gc[j[h2]]; gcnt[h2] = gn[j[h2]]
        print("GT joined for %d of %d matched voxels (%.1f %%)" % (h2.sum(), len(k), 100.0 * h2.mean()))
    tx = np.floor(xyz[:, 0] / a.tile).astype(np.int64); ty = np.floor(xyz[:, 1] / a.tile).astype(np.int64)
    tid = tx * 100000 + ty
    u, inv = np.unique(tid, return_inverse=True)
    n_vox = np.bincount(inv); n_diff = np.bincount(inv, weights=diff).astype(np.int64)
    rows = []
    for t in np.argsort(-n_diff)[:a.top]:
        m = inv == t
        cx, cy, cz = xyz[m].mean(0)
        rec = dict(tile=int(u[t]), center=[round(float(cx), 1), round(float(cy), 1), round(float(cz), 1)],
                   voxels=int(n_vox[t]), differing=int(n_diff[t]), frac=float(n_diff[t] / n_vox[t]))
        pairs = {}
        for pa, pb in zip(A9[m & diff], B9[m & diff]):
            kk = "%s->%s" % (LS.COARSE[pa] if pa >= 0 else "other_flat", LS.COARSE[pb] if pb >= 0 else "other_flat")
            pairs[kk] = pairs.get(kk, 0) + 1
        rec["top_changes"] = sorted(pairs.items(), key=lambda x: -x[1])[:5]
        if gt is not None:
            mm = m & (gt >= 0)
            w = gcnt[mm]
            rec["acc_A"] = float(100.0 * w[A9[mm] == gt[mm]].sum() / max(1, w.sum()))
            rec["acc_B"] = float(100.0 * w[B9[mm] == gt[mm]].sum() / max(1, w.sum()))
            rec["gt_voxels"] = int(mm.sum())
        rows.append(rec)
        print("tile %s  centre (%.1f, %.1f, %.1f)  voxels %6d  differ %5d (%.1f %%)  %s  %s"
              % (rec["tile"], cx, cy, cz, rec["voxels"], rec["differing"], 100 * rec["frac"],
                 ("accA %.1f accB %.1f" % (rec["acc_A"], rec["acc_B"])) if gt is not None else "",
                 rec["top_changes"][:3]))
    if a.json_out:
        json.dump(dict(a=a.a, b=a.b, gt=a.gt, tile=a.tile, matched=int(hit.sum()),
                       differing_total=int(diff.sum()), tiles=rows), open(a.json_out, "w"), indent=2)


if __name__ == "__main__":
    main()
