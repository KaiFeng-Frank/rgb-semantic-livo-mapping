#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_arm_Rprime.py -- the G1/G2/G3/G6 evidence for arm R', MEASURED through the
PRODUCTION pipelines that the trainer itself runs, not through a re-implementation.

It builds four datasets straight out of the real config files:

    dsC     arm_C.py        supervise="all"          -> the FULL post-GridSample GT,
                                                        which gives the pool size and
                                                        the label-invalid survivors
    dsD     arm_D.py        supervise="frustum"      -> arm D's post-grid supervision
    dsR     arm_Rprime.py   supervise="voxel_random" -> arm R''s post-grid supervision
    dsRaw   arm_Rprime.py, transform TRUNCATED right after GridSample
                                                     -> the raw post-grid arrays, so the
                                                        selection can be recomputed
                                                        independently of the transform

and calls them on the SAME frame with the SAME global RNG seed.  Every augmentation in
this project draws from python's `random` and numpy's GLOBAL RNG (RandomRotate uses
random.random(), RandomScale/Flip/Jitter use np.random.*, and GridSample picks its voxel
representatives with np.random.randint), and none of them depends on the supervision
mode, so seeding both RNGs identically before each call hands all four pipelines the
same augmented cloud and the SAME voxelisation.  That is asserted on the coordinates,
not assumed.

WHAT IS CHECKED
---------------
G1  arm D's post-GridSample supervised VOXEL count == arm R''s, per frame, exactly.
    Three independent numbers must agree: the count arm D's own pipeline produces, the
    count arm R''s pipeline produces, and the count voxel_random_select computes from
    the truncated pipeline's raw arrays.  Reported per frame; any mismatch fails.
    Also reported, because it is the quantity the whole arm exists to fix: arm D's
    CAMERA-frustum survivor count, and the label-invalid survivors that separate "all
    surviving voxels" from "all LABEL-VALID surviving voxels" as the draw pool.

G2  the selection draws nothing and is reproducible.
      G2a  numpy's and python's global RNG states are BIT-IDENTICAL across a call to
           voxel_random_select -- there is no draw to make it epoch-dependent.
      G2b  a second call on the same frame, with the global RNGs deliberately disturbed
           in between and then re-seeded, reproduces the selection bit for bit.
      G2c  --ref compares per-frame sha256 of the selected voxel indices, of the
           post-grid priority vector and of the voxelised coordinates against a frozen
           dump, which is how the SECOND PROCESS and the different PYTHONHASHSEED are
           checked.
      G2d  --dump-inputs writes (prio, frustum, valid) per probe so the selector alone,
           which depends on nothing but numpy, can be re-run under a DIFFERENT numpy
           build with --selector-only.
    Two different augmentation seeds are probed, so determinism is shown on two
    different voxelisations rather than on one lucky draw.

G3  both class distributions of the supervised voxels, side by side, NOT equalised.

G6  the KL-anchor set sizes, which differ in SHAPE between the arms and cannot be made
    to agree.  Reported, not corrected.

  # freeze the reference (run once, BEFORE arm R' trains)
  python tools/verify_arm_Rprime.py --frames 64 --out out/v04/armRprime/verify_Rprime.json \
         --dump-inputs out/v04/armRprime/selector_inputs
  # re-check from a second process with a different PYTHONHASHSEED
  PYTHONHASHSEED=31337 python tools/verify_arm_Rprime.py --frames 64 --out /tmp/x.json \
         --ref out/v04/armRprime/verify_Rprime.json
  # re-check the selector alone under a different numpy
  python tools/verify_arm_Rprime.py --selector-only out/v04/armRprime/selector_inputs \
         --ref out/v04/armRprime/verify_Rprime.json

Exits non-zero on ANY mismatch, so it can gate the runner.
"""
import os, sys, json, copy, random, hashlib, argparse

import numpy as np

sys.path.insert(0, "/data/wuyou/livo_sem/src")

CFGDIR = "/data/wuyou/livo_sem/src/Pointcept_v151/configs/semantic_kitti"
SEEDS = [20260923, 777]          # two distinct "epochs" = two distinct voxelisations


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def idx_sha(mask):
    return sha(np.flatnonzero(mask).astype(np.int64))


# ===================================================================== #
#  SELECTOR-ONLY MODE -- numpy and nothing else.  No torch, no pointcept.
# ===================================================================== #
def selector_only(inp_dir, ref_path):
    from voxel_random_supervise import voxel_random_select, VOXEL_RANDOM_SALT
    ref = json.load(open(ref_path))
    want = {(r["name"], r["seed"]): r for r in ref["rows"]}
    files = sorted(f for f in os.listdir(inp_dir) if f.endswith(".npz"))
    if not files:
        print("*** no .npz in %s ***" % inp_dir); return 1
    print("=" * 92)
    print("G2d  SELECTOR-ONLY RE-RUN   numpy %s   python %s   salt %s"
          % (np.__version__, sys.version.split()[0], VOXEL_RANDOM_SALT))
    print("     reference: %s  (written under numpy %s / python %s)"
          % (ref_path, ref["numpy"], ref["python"]))
    print("=" * 92)
    bad = []
    for f in files:
        z = np.load(os.path.join(inp_dir, f))
        name, seed = str(z["name"]), int(z["seed"])
        sel, k, pool = voxel_random_select(z["prio"], z["frustum"], z["valid"])
        r = want.get((name, seed))
        ok = (r is not None and idx_sha(sel) == r["sel_sha"] and k == r["k_recomputed"])
        print("  %-16s seed %-9d voxels %7d  k %7d  pool %7d  sel_sha %s  %s"
              % (name, seed, sel.shape[0], k, pool, idx_sha(sel)[:16],
                 "MATCH" if ok else "*** DIFFERS ***"))
        if not ok:
            bad.append((name, seed))
    if bad:
        print("\n*** %d PROBE(S) DIFFER: %s ***" % (len(bad), bad[:8])); return 1
    print("\nG2d PASS: the selector is bit-identical under numpy %s." % np.__version__)
    return 0


# ===================================================================== #
#  FULL MODE
# ===================================================================== #
def build(cfg_file, truncate_after_gridsample=False):
    from pointcept.utils.config import Config
    from pointcept.datasets.builder import build_dataset
    cfg = Config.fromfile(os.path.join(CFGDIR, cfg_file))
    tr = copy.deepcopy(dict(cfg.data.train))
    tl = [dict(t) for t in tr["transform"]]
    if truncate_after_gridsample:
        gi = [i for i, t in enumerate(tl) if t["type"] == "GridSample"][0]
        tl = tl[:gi + 1]
    for t in tl:
        if t["type"] == "VoxelRandomSupervise":
            t["audit_path"] = None       # verification must not append to the run's audit
    tr["transform"] = tl
    return build_dataset(tr)


def _np(x):
    return x.numpy() if hasattr(x, "numpy") else np.asarray(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=64)
    ap.add_argument("--out", default="")
    ap.add_argument("--ref", default="")
    ap.add_argument("--dump-inputs", default="")
    ap.add_argument("--selector-only", default="")
    a = ap.parse_args()

    if a.selector_only:
        sys.exit(selector_only(a.selector_only, a.ref))

    import pointcept_ext as PX                                    # noqa: F401
    import distil_ext as DX                                       # noqa: F401
    import distil_ext_rprime as RX                                # noqa: F401
    from voxel_random_supervise import (voxel_random_select, frame_priority,
                                        VOXEL_RANDOM_SALT)
    NAMES = PX.COMMON9_NAMES

    dsC = build("arm_C.py")
    dsD = build("arm_D.py")
    dsR = build("arm_Rprime.py")
    dsRaw = build("arm_Rprime.py", truncate_after_gridsample=True)
    assert dsD.data_list == dsR.data_list == dsC.data_list == dsRaw.data_list, \
        "the arms must walk the same frame order"
    n_list = len(dsD.data_list)
    idxs = np.unique(np.linspace(0, n_list - 1, a.frames).astype(int)).tolist()
    if a.dump_inputs:
        os.makedirs(a.dump_inputs, exist_ok=True)

    def run(ds, i, S):
        random.seed(S); np.random.seed(S)
        return ds[i]

    rows, bad = [], []
    histD = np.zeros(len(NAMES), np.int64)
    histR = np.zeros(len(NAMES), np.int64)
    n_dumped = 0
    frame_prio_sha = {}
    for i in idxs:
        name = dsD.get_data_name(i)
        # THE FRAME-LEVEL ranking: a property of (sequence, frame) alone.  This is the
        # object G2 says is fixed.  The post-grid `prio` vector is this vector sampled at
        # whichever representatives the voxeliser kept, so it necessarily changes with the
        # voxelisation -- that is not a failure of G2, it is the voxelisation moving.
        pth = dsD.data_list[i % len(dsD.data_list)]
        sq, fr = dsD._seq_frame(pth)
        frame_prio_sha[name] = sha(frame_priority(sq, fr, os.path.getsize(pth) // 16))
        for S in SEEDS:
            dC = run(dsC, i, S); dD = run(dsD, i, S)
            dR = run(dsR, i, S); dW = run(dsRaw, i, S)

            # --- the four pipelines really did see the same voxelisation ----------
            # Compared on grid_coord, the INTEGER voxel index, not on coord: ToTensor
            # casts float64 coordinates down to float32, so the three Collect-ed
            # pipelines carry float32 while the truncated one still carries float64 and
            # a value comparison would fail on the cast alone.  grid_coord is exact and
            # IS the identity of the voxelisation.
            gC, gD, gR, gW = (_np(dC["grid_coord"]), _np(dD["grid_coord"]),
                              _np(dR["grid_coord"]), np.asarray(dW["grid_coord"]))
            cR = _np(dR["coord"])
            same_vox = bool(np.array_equal(gC, gD) and np.array_equal(gD, gR)
                            and np.array_equal(gR, gW))

            segC = _np(dC["segment"]); segD = _np(dD["segment"]); segR = _np(dR["segment"])
            fruD = _np(dD["frustum"]).astype(bool)
            selR = _np(dR["frustum"]).astype(bool)
            v = int(segC.shape[0])
            pool = int((segC >= 0).sum())
            invalid = v - pool
            camfru = int(fruD.sum())
            kD = int((segD >= 0).sum())
            kR = int((segR >= 0).sum())

            # --- G2a: the selector draws NOTHING ---------------------------------
            st_np0, st_py0 = np.random.get_state(), random.getstate()
            sel2, k2, pool2 = voxel_random_select(
                dW["prio"], np.asarray(dW["frustum"]).astype(bool), dW["segment"] >= 0)
            st_np1, st_py1 = np.random.get_state(), random.getstate()
            rng_untouched = bool(st_np0[0] == st_np1[0]
                                 and np.array_equal(st_np0[1], st_np1[1])
                                 and st_np0[2:] == st_np1[2:] and st_py0 == st_py1)

            # --- G2b: re-run the production pipeline after disturbing the RNGs ----
            np.random.seed(999); np.random.random(10000); random.random()
            dR2 = run(dsR, i, S)
            repeatable = bool(np.array_equal(_np(dR2["frustum"]), _np(dR["frustum"]))
                              and np.array_equal(_np(dR2["segment"]), segR))

            # --- label integrity: the supervised labels are untouched GT ----------
            lab_R_is_gt = bool(np.array_equal(segR[selR], segC[selR]))
            lab_D_is_gt = bool(np.array_equal(segD[segD >= 0], segC[segD >= 0]))
            sel_is_sup = bool(int(selR.sum()) == kR)
            # arm D's post-grid supervision IS (camera frustum AND label-valid)
            d_pred_ok = bool(np.array_equal(segD >= 0, fruD & (segC >= 0)))

            ok = (same_vox and kD == kR == k2 and np.array_equal(sel2, selR)
                  and rng_untouched and repeatable and lab_R_is_gt and lab_D_is_gt
                  and sel_is_sup and d_pred_ok and pool == pool2)
            row = dict(i=int(i), name=name, seed=int(S), voxels=v,
                       pool_valid=pool, invalid_survivors=invalid,
                       camera_frustum_survivors=camfru,
                       k_armD=kD, k_armRprime=kR, k_recomputed=int(k2),
                       match=bool(kD == kR == k2),
                       kl_voxels_D=v - camfru, kl_voxels_Rprime=v - kR,
                       sel_sha=idx_sha(selR), selD_sha=idx_sha(segD >= 0),
                       prio_sha=sha(np.asarray(dW["prio"])),
                       prio_frame_sha=frame_prio_sha[name],
                       coord_sha=sha(gR),
                       same_voxelisation=same_vox, rng_untouched=rng_untouched,
                       repeatable=repeatable, sel_equals_recomputed=bool(
                           np.array_equal(sel2, selR)),
                       labels_are_gt_Rprime=lab_R_is_gt, labels_are_gt_D=lab_D_is_gt,
                       sel_is_supervised_set=sel_is_sup,
                       armD_postgrid_is_frustum_and_valid=d_pred_ok, ok=bool(ok))
            rows.append(row)
            if not ok:
                bad.append(row)
            histD += np.bincount(segD[segD >= 0], minlength=len(NAMES))
            histR += np.bincount(segR[selR], minlength=len(NAMES))

            if a.dump_inputs and n_dumped < 16:
                np.savez_compressed(
                    os.path.join(a.dump_inputs, "%s_s%d.npz" % (name, S)),
                    prio=np.asarray(dW["prio"]),
                    frustum=np.asarray(dW["frustum"]).astype(bool),
                    valid=(dW["segment"] >= 0), name=name, seed=S)
                n_dumped += 1

    out = dict(frames=len(idxs), seeds=SEEDS, rows=rows, names=NAMES,
               histD=histD.tolist(), histR=histR.tolist(),
               salt=VOXEL_RANDOM_SALT, numpy=np.__version__,
               python=sys.version.split()[0],
               pythonhashseed=os.environ.get("PYTHONHASHSEED", "<unset>"))

    tD, tR = int(histD.sum()), int(histR.sum())
    P = []
    P.append("=" * 100)
    P.append("ARM R-PRIME VERIFICATION   %d frames x %d voxelisations   numpy %s   "
             "python %s   PYTHONHASHSEED %s" % (len(idxs), len(SEEDS), out["numpy"],
                                                out["python"], out["pythonhashseed"]))
    P.append("salt %s" % VOXEL_RANDOM_SALT)
    P.append("=" * 100)
    P.append("G1  POST-GridSample SUPERVISED VOXEL COUNT, arm D vs arm R'")
    P.append("    k_armD        arm D's own pipeline: voxels whose surviving "
             "representative is in-frustum AND label-valid")
    P.append("    k_armRprime   arm R''s own pipeline")
    P.append("    k_recomp      voxel_random_select, from the truncated pipeline's raw "
             "arrays, independently of the transform")
    P.append("")
    P.append("    %-14s %6s %8s %8s %8s %8s %8s %8s %6s" %
             ("frame", "seed", "voxels", "valid", "camfrust", "k_armD", "k_armR'",
              "k_recomp", "match"))
    show = rows[:10] + (["..."] if len(rows) > 20 else []) + rows[-10:]
    for r in show:
        if r == "...":
            P.append("    ..."); continue
        P.append("    %-14s %6d %8d %8d %8d %8d %8d %8d %6s" %
                 (r["name"], r["seed"], r["voxels"], r["pool_valid"],
                  r["camera_frustum_survivors"], r["k_armD"], r["k_armRprime"],
                  r["k_recomputed"], "OK" if r["match"] else "*** NO ***"))
    sD = sum(r["k_armD"] for r in rows); sR = sum(r["k_armRprime"] for r in rows)
    P.append("    totals over the sample:  arm D %d   arm R' %d   %s"
             % (sD, sR, "EXACT MATCH" if sD == sR else "*** DIFFER ***"))
    P.append("    per-frame mismatches: %d / %d"
             % (sum(not r["match"] for r in rows), len(rows)))
    P.append("")
    sv = sum(r["voxels"] for r in rows); si = sum(r["invalid_survivors"] for r in rows)
    sc = sum(r["camera_frustum_survivors"] for r in rows)
    P.append("    THE DRAW POOL.  Arm R' draws from the LABEL-VALID survivors, not from")
    P.append("    every survivor, for the same reason arm R draws from label-valid")
    P.append("    points: a voxel holding an EXCLUDED class carries ignore_index in")
    P.append("    every arm and can never enter the loss, so drawing it would leave")
    P.append("    arm R' with fewer loss-carrying voxels than arm D and break G1.")
    P.append("    Size of that decision, measured here: %d of %d surviving voxels "
             "(%.3f %%) are label-invalid." % (si, sv, 100.0 * si / max(sv, 1)))
    P.append("    Arm D's supervision is %.3f %% of survivors; its camera frustum "
             "covers %.3f %%." % (100.0 * sD / max(sv, 1), 100.0 * sc / max(sv, 1)))
    P.append("")
    P.append("G2  THE SELECTION DRAWS NOTHING AND IS REPRODUCIBLE")
    P.append("    G2a  global RNG state (numpy AND python) bit-identical across the "
             "selector: %d / %d" % (sum(r["rng_untouched"] for r in rows), len(rows)))
    P.append("    G2b  production pipeline re-run after the global RNGs were disturbed "
             "and re-seeded: %d / %d" % (sum(r["repeatable"] for r in rows), len(rows)))
    P.append("         the transform's output equals the independent recomputation: "
             "%d / %d" % (sum(r["sel_equals_recomputed"] for r in rows), len(rows)))
    P.append("    G2c  per-frame sha256 of the selected voxel indices, of the post-grid "
             "priority vector and of")
    P.append("         the voxelised coordinates are in the dump; --ref compares them "
             "across processes.")
    P.append("    NOTE, STATED PLAINLY: the voxel SET cannot be epoch-invariant.  "
             "RandomRotate / RandomScale /")
    P.append("    RandomFlip / RandomJitter run BEFORE GridSample and move every point, "
             "so each epoch partitions")
    P.append("    the sweep differently.  Arm D is in exactly the same position: its "
             "frustum is a fixed POINT set")
    P.append("    whose post-grid realisation changes every epoch.  What is fixed in "
             "both arms is the RULE and the")
    P.append("    per-frame randomness it consumes -- for arm R' a BLAKE2b(salt|seq|"
             "frame) ranking that is a property")
    P.append("    of the frame and of nothing else.  Two voxelisations are probed here "
             "for exactly that reason.")
    P.append("")
    P.append("    sanity: the same frame under the two seeds must give a DIFFERENT "
             "voxelisation (different augmentation)")
    P.append("            and the SAME frame-level priority vector (the ranking "
             "depends on seq and frame only).")
    pr_ok = cd_ok = 0; npair = 0
    for j in range(0, len(rows), len(SEEDS)):
        grp = rows[j:j + len(SEEDS)]
        if len(grp) < 2:
            continue
        npair += 1
        pr_ok += (grp[0]["prio_frame_sha"] == grp[1]["prio_frame_sha"])
        cd_ok += (grp[0]["coord_sha"] != grp[1]["coord_sha"])
    P.append("            frame-level priority vector identical across the two "
             "voxelisations: %d / %d frames" % (pr_ok, npair))
    P.append("            voxel grids differ across the two voxelisations:            "
             "%d / %d frames" % (cd_ok, npair))
    P.append("")
    P.append("    label integrity: arm R''s supervised labels are untouched GT "
             "%d / %d ; arm D's %d / %d"
             % (sum(r["labels_are_gt_Rprime"] for r in rows), len(rows),
                sum(r["labels_are_gt_D"] for r in rows), len(rows)))
    P.append("    arm D's post-grid supervision == (camera frustum AND label-valid): "
             "%d / %d" % (sum(r["armD_postgrid_is_frustum_and_valid"] for r in rows),
                          len(rows)))
    P.append("    the emitted `frustum` key IS arm R''s supervised set: %d / %d"
             % (sum(r["sel_is_supervised_set"] for r in rows), len(rows)))
    P.append("    all four pipelines saw the same voxelisation: %d / %d"
             % (sum(r["same_voxelisation"] for r in rows), len(rows)))
    P.append("")
    P.append("G3  CLASS DISTRIBUTION OF THE SUPERVISED VOXELS (post-GridSample, NOT "
             "equalised, by design)")
    P.append("    %-16s %14s %8s   %14s %8s   %8s" %
             ("class", "arm D", "%", "arm R'", "%", "R'/D"))
    for j, nm in enumerate(NAMES):
        P.append("    %-16s %14d %7.3f   %14d %7.3f   %8.3f"
                 % (nm, histD[j], 100.0 * histD[j] / max(tD, 1),
                    histR[j], 100.0 * histR[j] / max(tR, 1),
                    (histR[j] / histD[j]) if histD[j] > 0 else float("nan")))
    P.append("    totals %d vs %d" % (tD, tR))
    P.append("")
    P.append("G6  KL-ANCHOR SET, THE RESIDUAL ASYMMETRY.  STATED, NOT FIXABLE.")
    kd = sum(r["kl_voxels_D"] for r in rows); kr = sum(r["kl_voxels_Rprime"] for r in rows)
    P.append("    arm D anchors %d voxels (%.3f %% of survivors) -- the OUT-OF-FRUSTUM "
             "region, which IS the" % (kd, 100.0 * kd / max(sv, 1)))
    P.append("    evaluation region: arm D is anchored precisely where it is scored.")
    P.append("    arm R' anchors %d voxels (%.3f %%) -- scattered and interleaved with "
             "its supervised voxels," % (kr, 100.0 * kr / max(sv, 1)))
    P.append("    overlapping the evaluation region only partially.  arm R' anchors "
             "%.3f pp more of the cloud," % (100.0 * (kr - kd) / max(sv, 1)))
    P.append("    because arm D's frustum also contains label-EXCLUDED voxels that "
             "receive neither loss nor anchor,")
    P.append("    while arm R' has no such gap.  This cannot be removed without either "
             "changing what the KL means")
    P.append("    or breaking the per-frame count match.")

    if a.ref:
        ref = json.load(open(a.ref))
        rr = {(r["name"], r["seed"]): r for r in ref["rows"]}
        diff = []
        for r in rows:
            q = rr.get((r["name"], r["seed"]))
            if q is None or q["sel_sha"] != r["sel_sha"] or q["prio_sha"] != r["prio_sha"] \
               or q["prio_frame_sha"] != r["prio_frame_sha"] \
               or q["coord_sha"] != r["coord_sha"] or q["k_armD"] != r["k_armD"]:
                diff.append((r["name"], r["seed"]))
        P.append("")
        P.append("G2c  REF %s" % a.ref)
        P.append("     written under numpy %s / python %s / PYTHONHASHSEED %s"
                 % (ref["numpy"], ref["python"], ref.get("pythonhashseed")))
        P.append("     this run  numpy %s / python %s / PYTHONHASHSEED %s"
                 % (out["numpy"], out["python"], out["pythonhashseed"]))
        P.append("     -> %s" % ("IDENTICAL SELECTIONS, PRIORITIES AND VOXELISATIONS"
                                 if not diff else
                                 "*** %d PROBES DIFFER: %s ***" % (len(diff), diff[:8])))
        if diff:
            bad.append(dict(ref_mismatch=diff))

    if npair and (pr_ok != npair or cd_ok != npair):
        bad.append(dict(sanity=dict(frame_prio_identical=pr_ok,
                                    voxelgrid_differs=cd_ok, of=npair)))
    for fld in ("same_voxelisation", "rng_untouched", "repeatable",
                "sel_equals_recomputed", "labels_are_gt_Rprime", "labels_are_gt_D",
                "sel_is_supervised_set", "armD_postgrid_is_frustum_and_valid"):
        if sum(r[fld] for r in rows) != len(rows):
            bad.append(dict(failed_field=fld))

    txt = "\n".join(P)
    print(txt)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
        json.dump(out, open(a.out, "w"), indent=1)
        open(os.path.splitext(a.out)[0] + ".txt", "w").write(txt + "\n")
        print("\n-> %s" % a.out)
    if bad:
        print("\n*** %d FAILING PROBES ***" % len(bad)); sys.exit(1)
    print("\nALL CHECKS PASS")


if __name__ == "__main__":
    main()
