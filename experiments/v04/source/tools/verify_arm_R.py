#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_arm_R.py -- the F1/F2/F3 evidence for arm R, measured through the PRODUCTION
dataset class, not through a re-implementation of it.

F1  per-frame supervised count of arm R == per-frame supervised count of arm D, exactly.
F2  the random subset is FIXED: the same frame yields the same point indices in a second
    call, in a second process, and against a frozen reference dump.
F3  the two class distributions, reported side by side and NOT equalised.

  # dump the reference (run once, before arm R trains)
  python tools/verify_arm_R.py --frames 64 --out out/v04/armR/verify_armR.json
  # re-check against it (a second process, and again at training time)
  python tools/verify_arm_R.py --frames 64 --out /tmp/x.json --ref out/v04/armR/verify_armR.json

Exits non-zero on ANY mismatch, so it can gate the runner.
"""
import os, sys, json, random, hashlib, argparse

import numpy as np

sys.path.insert(0, "/data/wuyou/livo_sem/src")
import pointcept_ext as PX                                          # noqa: E402
import distil_ext as DX                                             # noqa: E402

DATA_ROOT = "/data/wuyou/livo_sem/data/pointcept_sk"
PSEUDO_ROOT = "/data/wuyou/livo_sem/out/pseudo"
WEIGHTS = "/data/wuyou/livo_sem/out/pseudo/rare_weights.json"
FILTER_SPEC = {"require_visible": True, "range_lt50": False, "teachers_agree": True,
               "conf_min": 0.9, "drop_depth_edge": False}
NAMES = PX.COMMON9_NAMES


def build(supervise):
    return DX.DistilSemanticKITTIDataset(
        split="train", data_root=DATA_ROOT, pseudo_root=PSEUDO_ROOT,
        filter_spec=FILTER_SPEC, label_source="gt", supervise=supervise,
        sample_weights_json=WEIGHTS, transform=None, ignore_index=-1)


def idx_sha(mask):
    return hashlib.sha256(np.flatnonzero(mask).astype(np.int64).tobytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=64)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref", default="")
    a = ap.parse_args()

    dsD, dsR = build("frustum"), build("random")
    assert dsD.data_list == dsR.data_list, "the two arms must walk the same frame order"
    n_list = len(dsD.data_list)
    idxs = np.unique(np.linspace(0, n_list - 1, a.frames).astype(int)).tolist()

    rows, bad = [], []
    histD = np.zeros(len(NAMES), np.int64)
    histR = np.zeros(len(NAMES), np.int64)
    for i in idxs:
        name = dsD.get_data_name(i)
        assert name == dsR.get_data_name(i), (i, name)
        dD, dR = dsD.get_data(i), dsR.get_data(i)

        segD, segR = dD["segment"], dR["segment"]
        fruD, fruR = dD["frustum"].astype(bool), dR["frustum"].astype(bool)
        n = segD.shape[0]
        nD, nR = int((segD >= 0).sum()), int((segR >= 0).sum())

        # --- F2, within-process: a second call, with the global RNGs deliberately
        #     disturbed in between, must reproduce the subset bit for bit.  This is the
        #     "epoch 2" case: Pointcept re-enters get_data once per epoch.
        np.random.seed(12345); np.random.random(10000); random.random()
        dR2 = dsR.get_data(i)
        same_epoch2 = bool(np.array_equal(dR2["frustum"], dR["frustum"]) and
                           np.array_equal(dR2["segment"], segR))

        # coord / strength must be untouched by the supervision mode
        same_input = bool(np.array_equal(dD["coord"], dR["coord"]) and
                          np.array_equal(dD["strength"], dR["strength"]))
        # the emitted mask IS the supervised set in arm R
        sel_is_sup = bool(int(fruR.sum()) == nR)
        # the supervised labels themselves are untouched GT
        gt_same = bool(np.array_equal(segD[fruD & (segD >= 0)],
                                      dsD.raw_lut[np.fromfile(
                                          os.path.join(os.path.dirname(os.path.dirname(
                                              dsD.data_list[i])), "labels",
                                              dsD.get_data_name(i).split("_")[1] + ".label"),
                                          dtype=np.uint32) & 0xFFFF][fruD & (segD >= 0)]))

        rows.append(dict(i=int(i), name=name, n_points=n,
                         n_sup_D=nD, n_sup_R=nR, match=bool(nD == nR),
                         n_camera_frustum=int(fruD.sum()),
                         n_kl_D=int(n - fruD.sum()), n_kl_R=int(n - fruR.sum()),
                         sel_sha_R=idx_sha(fruR), sel_sha_D=idx_sha(fruD),
                         fixed_across_epochs=same_epoch2,
                         same_input=same_input, sel_is_sup=sel_is_sup, gt_same=gt_same))
        if not (nD == nR and same_epoch2 and same_input and sel_is_sup and gt_same):
            bad.append(rows[-1])
        histD += np.bincount(segD[segD >= 0], minlength=len(NAMES))
        histR += np.bincount(segR[segR >= 0], minlength=len(NAMES))

    out = dict(frames=len(idxs), rows=rows, names=NAMES,
               histD=histD.tolist(), histR=histR.tolist(),
               salt=__import__("random_supervise").RANDOM_SUPERVISE_SALT,
               numpy=np.__version__, python=sys.version.split()[0])
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)

    tD, tR = int(histD.sum()), int(histR.sum())
    print("=" * 92)
    print("ARM R VERIFICATION   %d frames   numpy %s   python %s   salt %s"
          % (len(idxs), out["numpy"], out["python"], out["salt"]))
    print("=" * 92)
    print("F1  per-frame supervised count, arm D vs arm R")
    print("    %-14s %10s %10s %10s %10s %6s" %
          ("frame", "points", "n_sup_D", "n_sup_R", "cam_frust", "match"))
    for r in rows[:12] + (["..."] if len(rows) > 24 else []) + rows[-12:]:
        if r == "...":
            print("    ...")
            continue
        print("    %-14s %10d %10d %10d %10d %6s" %
              (r["name"], r["n_points"], r["n_sup_D"], r["n_sup_R"],
               r["n_camera_frustum"], "OK" if r["match"] else "*** NO ***"))
    print("    totals over the sample: D %d   R %d   %s"
          % (tD, tR, "EXACT MATCH" if tD == tR else "*** DIFFER ***"))
    print()
    print("F2  fixed subset:  identical on a second call after the global RNGs were "
          "disturbed:  %d / %d frames" % (sum(r["fixed_across_epochs"] for r in rows), len(rows)))
    print("    per-frame sha256 of the selected point indices is in the dump; compare "
          "it across processes with --ref.")
    print()
    print("F3  class distribution of the SUPERVISED points (NOT equalised, by design)")
    print("    %-16s %14s %8s   %14s %8s" % ("class", "arm D", "%", "arm R", "%"))
    for j, nm in enumerate(NAMES):
        print("    %-16s %14d %7.3f   %14d %7.3f"
              % (nm, histD[j], 100.0 * histD[j] / max(tD, 1),
                 histR[j], 100.0 * histR[j] / max(tR, 1)))
    print()
    kD = sum(r["n_kl_D"] for r in rows); kR = sum(r["n_kl_R"] for r in rows)
    np_ = sum(r["n_points"] for r in rows)
    print("F5  KL-anchor set size:  D %d (%.3f %% of points)   R %d (%.3f %%)   "
          "R is larger by %.3f pp" % (kD, 100.0 * kD / np_, kR, 100.0 * kR / np_,
                                      100.0 * (kR - kD) / np_))
    print("    D's KL set is the OUT-OF-FRUSTUM region = the evaluation region.")
    print("    R's KL set is scattered and interleaved with the supervised points.")

    if a.ref:
        ref = json.load(open(a.ref))
        rr = {r["name"]: r for r in ref["rows"]}
        diff = [r["name"] for r in rows
                if r["name"] not in rr or rr[r["name"]]["sel_sha_R"] != r["sel_sha_R"]]
        print()
        print("REF %s  -> %s" % (a.ref, "IDENTICAL SUBSETS" if not diff
                                 else "*** %d FRAMES DIFFER: %s ***" % (len(diff), diff[:8])))
        if diff:
            bad.append(dict(ref_mismatch=diff))
    if bad:
        print("\n*** %d FAILING FRAMES ***" % len(bad)); sys.exit(1)
    print("\nALL CHECKS PASS")


if __name__ == "__main__":
    main()
