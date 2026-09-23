#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pseudo_inventory.py -- the Step-4 deliverable: what the training signal actually is.

Walks the per-point teacher caches written by tools/make_pseudo.py, applies filter E
with the boundaries re-derived on seq 08, and reports

  * total points, in-frustum points, candidate points, surviving points
  * per-class surviving counts -- above all person / two_wheeler / large_vehicle,
    whose ABSOLUTE counts decide whether those classes can be learned at all
  * the frame-level rare-class presence table, written out as rare_weights.json

RARE-CLASS SAMPLING, DECLARED.  The frozen design fixes the mechanism ("frame-level
resampling by teacher-predicted presence of person / two_wheeler / large_vehicle") and
says the identical sampling is applied to arms B, C and D.  It does NOT state the
multiplier or the presence threshold.  Nothing is invented here silently: the rule is
  weight(frame) = 1 + #{rare classes with at least `--min-pts` teacher-predicted points}
with --min-pts defaulting to 1, i.e. literal presence.  Both numbers are written into
the JSON so the constant lives in the artefact, not in someone's head.
"""
import os, sys, json, argparse
sys.path.insert(0, "/data/wuyou/livo_sem/src")
import numpy as np

C9 = ["car", "large_vehicle", "two_wheeler", "person",
      "road", "sidewalk", "terrain", "vegetation", "manmade"]
RARE = ["large_vehicle", "two_wheeler", "person"]
RARE_IDX = [C9.index(c) for c in RARE]
F_INFRUSTUM, F_VISIBLE, F_DEPTHEDGE, F_RANGE_LT50 = 1, 2, 4, 8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/wuyou/livo_sem/out/pseudo")
    ap.add_argument("--seqs", default="00,01,02,04,05,06,09,10")
    ap.add_argument("--spec", default="", help="filter-E JSON; default = the seq08 fit")
    ap.add_argument("--min-pts", type=int, default=1)
    ap.add_argument("--out", default="/data/wuyou/livo_sem/out/pseudo/inventory.json")
    ap.add_argument("--weights-out", default="/data/wuyou/livo_sem/out/pseudo/rare_weights.json")
    a = ap.parse_args()

    from filter_e import filter_E_mask
    spec = (json.load(open(a.spec))["filter_spec"] if a.spec else
            dict(require_visible=True, range_lt50=True, teachers_agree=True,
                 conf_min=0.90, drop_depth_edge=True))
    print("filter_spec = %s" % json.dumps(spec), flush=True)

    seqs = [s for s in a.seqs.split(",") if s]
    tot = dict(points=0, in_frustum=0, visible=0, candidate=0, agree=0, kept=0, frames=0)
    cls_cand = np.zeros(9, np.int64)
    cls_kept = np.zeros(9, np.int64)
    per_seq = {}
    weights = {}
    wh = np.zeros(len(RARE) + 1, np.int64)
    frames_with = np.zeros(len(RARE), np.int64)

    for seq in seqs:
        d = os.path.join(a.root, seq)
        if not os.path.isdir(d):
            print("  seq %s: MISSING %s" % (seq, d)); continue
        fs = sorted(f for f in os.listdir(d) if f.startswith("f") and f.endswith(".npz"))
        s_tot = dict(points=0, candidate=0, kept=0, frames=len(fs))
        for fn in fs:
            z = np.load(os.path.join(d, fn))
            t9 = z["t9"]; m9 = z["m9"]; conf = z["conf"].astype(np.float32); fl = z["flags"]
            n = t9.shape[0]
            cand = t9 >= 0
            keep = filter_E_mask(t9, m9, conf, fl, spec)
            tot["points"] += n; s_tot["points"] += n
            tot["in_frustum"] += int(((fl & F_INFRUSTUM) > 0).sum())
            tot["visible"] += int(((fl & F_VISIBLE) > 0).sum())
            tot["candidate"] += int(cand.sum()); s_tot["candidate"] += int(cand.sum())
            tot["agree"] += int((cand & (t9 == m9)).sum())
            tot["kept"] += int(keep.sum()); s_tot["kept"] += int(keep.sum())
            tot["frames"] += 1
            if cand.any():
                cls_cand += np.bincount(t9[cand].astype(np.int64), minlength=9)
            if keep.any():
                cls_kept += np.bincount(t9[keep].astype(np.int64), minlength=9)
            # frame-level rare-class presence, on the TEACHER's raw prediction
            present = []
            for j, k in enumerate(RARE_IDX):
                c = int((t9 == k).sum())
                if c >= a.min_pts:
                    present.append(j); frames_with[j] += 1
            wh[len(present)] += 1
            weights["%s/%s" % (seq, fn[1:7])] = 1 + len(present)
        per_seq[seq] = s_tot
        print("  seq %s  frames=%d  points=%d  cand=%d  kept=%d" %
              (seq, s_tot["frames"], s_tot["points"], s_tot["candidate"], s_tot["kept"]),
              flush=True)

    print("\n" + "=" * 92)
    print("PSEUDO-LABEL INVENTORY  (train split: %s)" % ",".join(seqs))
    print("=" * 92)
    P = max(tot["points"], 1)
    for k in ("frames", "points", "in_frustum", "visible", "candidate", "agree", "kept"):
        print("  %-12s %14d   %6.2f %% of all points" %
              (k, tot[k], 100.0 * tot[k] / P if k != "frames" else float("nan")))
    print("\n  %-14s %14s %14s %8s %10s" %
          ("class", "candidate", "after filter E", "kept%", "share%"))
    K = max(cls_kept.sum(), 1)
    for i, c in enumerate(C9):
        print("  %-14s %14d %14d %7.1f%% %9.3f%%" %
              (c, cls_cand[i], cls_kept[i],
               100.0 * cls_kept[i] / max(cls_cand[i], 1), 100.0 * cls_kept[i] / K))
    print("\n  RARE-CLASS FRAME PRESENCE (>= %d teacher points), %d frames:" %
          (a.min_pts, tot["frames"]))
    for j, c in enumerate(RARE):
        print("    %-14s present in %6d frames (%.1f %%)" %
              (c, frames_with[j], 100.0 * frames_with[j] / max(tot["frames"], 1)))
    print("    frames by #rare classes present -> weight:")
    for k in range(len(RARE) + 1):
        print("      %d present -> weight %d : %6d frames" % (k, k + 1, wh[k]))
    eff = sum((k + 1) * wh[k] for k in range(len(RARE) + 1))
    print("    resampled epoch length: %d -> %d frames (x%.3f)" %
          (tot["frames"], eff, eff / max(tot["frames"], 1)))

    json.dump(dict(spec=spec, seqs=seqs, totals={k: int(v) for k, v in tot.items()},
                   per_seq=per_seq, class_names=C9,
                   candidate_per_class=cls_cand.tolist(),
                   kept_per_class=cls_kept.tolist(),
                   rare_frames={c: int(frames_with[j]) for j, c in enumerate(RARE)},
                   weight_hist=wh.tolist(), resampled_len=int(eff)),
              open(a.out, "w"), indent=2)
    json.dump(dict(rule="weight = 1 + #rare classes present",
                   rare=RARE, min_pts=a.min_pts, seqs=seqs,
                   n_frames=int(tot["frames"]), resampled_len=int(eff),
                   weights=weights),
              open(a.weights_out, "w"))
    print("\nJSON -> %s\nweights -> %s" % (a.out, a.weights_out))


if __name__ == "__main__":
    main()
