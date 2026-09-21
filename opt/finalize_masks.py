#!/usr/bin/env python3
"""
finalize_masks.py -- turn the node's per-sweep dump into the mask directories
opt/score_dynamic.py consumes, for all three arms of the same run.

The v0.3 decision is a property of the FINAL map -- a voxel is withheld from the
published map when it ends the run with >= k_free free-space votes -- so `keep`
cannot be written while the sweep is being processed.  The node instead dumps the
map ROW every point landed in, plus the final n_free vector, and this script
resolves

    keep[i] = n_free[row[i]] < k_free

which is EXACTLY the published map's membership: a point is "kept out of the map"
iff the voxel it fell in is withheld.

Three arms from the one dump, on identical frames and identical eligible sets:
    main   the v0.3 policy
    noop   v0.2 as shipped -- nothing withheld (S3, the ribbon baseline)
    naive  "drop every point whose PREDICTED class is potentially-movable",
           using the node's own PTv3 argmax (S2, the control arm)
"""
import argparse, os, sys, glob
import numpy as np
sys.path.insert(0, "/data/livo_sem/src")
import sem_core as S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", default="main", choices=["main", "noop", "naive"])
    ap.add_argument("--k-free", type=int, default=None)
    a = ap.parse_args()
    st = np.load(os.path.join(a.dump, "_mapstate.npz"))
    n_free = st["n_free"]
    k = a.k_free if a.k_free is not None else int(st["k_free"])
    withheld = n_free >= k if k > 0 else np.zeros(len(n_free), bool)
    os.makedirs(a.out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.dump, "f??????.npz")))
    nk = nt = 0
    for f in files:
        fi = int(os.path.basename(f)[1:7])
        with np.load(f) as z:
            idx, xyz, pred, row = z["idx"], z["xyz"], z["pred"], z["row"]
        if a.arm == "noop":
            keep = np.ones(len(idx), bool)
        elif a.arm == "naive":
            keep = ~S.MOVABLE_NUSC[pred]
        else:
            keep = ~withheld[row]
        np.savez(os.path.join(a.out, "frame_%06d.npz" % fi),
                 keep=keep, eligible=np.ones(len(idx), bool), idx=idx,
                 xyz=xyz, pred=pred, order="bag")
        nk += int(keep.sum()); nt += len(keep)
    print("%s arm: %d frames, k_free=%d, %d/%d points kept (%.3f %% withheld) -> %s"
          % (a.arm, len(files), k, nk, nt, 100.0 * (1 - nk / max(nt, 1)), a.out))


if __name__ == "__main__":
    main()
