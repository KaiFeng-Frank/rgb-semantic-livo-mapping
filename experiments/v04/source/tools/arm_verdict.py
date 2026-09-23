#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
arm_verdict.py -- apply the PRE-REGISTERED decision rule to one trained arm.

The rule was written to out/v04_decision_rule.md BEFORE any arm was trained and is not
re-derived here; the thresholds are READ from out/v04/armA_thresholds.json.

  PRIMARY METRIC  out-of-frustum-only mIoU over common-9, arm minus arm A.
      >= +0.60  PROPAGATION      <= -0.60  FORGETTING      inside  NEITHER (P4)
  The band is +-0.60: 3*se(delta) = 0.524, rounded out to cover arm B's own spread.
  two_wheeler is an ABSTAIN CELL in all three subsets (P2) -- not quoted either way.
  The same delta without two_wheeler is published beside it, as DECOMPOSITION ONLY.
  Per-class signs (P3) are tested each against its OWN per-class 3-sigma, not zero.

P3, as recorded before any number was seen: under filter E, arm B vs arm A should show
  car, sidewalk, vegetation, person, large_vehicle UP ; road FLAT ;
  terrain and manmade DOWN if they were not excluded from the loss.
two_wheeler was declared untestable, so P3 is an EIGHT-cell test.
"""
import os, sys, json, argparse
import numpy as np

CLS = ["car", "large_vehicle", "two_wheeler", "person", "road", "sidewalk",
       "terrain", "vegetation", "manmade"]
SUB = [("outside", "OUT-OF-FRUSTUM (PRIMARY)"), ("frustum", "in-frustum"),
       ("global", "global")]
K = "per_class_iou_abstain_excluded"
P3_UP = ["car", "sidewalk", "vegetation", "person", "large_vehicle"]
P3_FLAT = ["road"]
P3_DOWN = ["terrain", "manmade"]
BAND = 0.60


def draws(path):
    d = json.load(open(path))
    return [r["3d"] for r in d["repeats"]]


def stat(v):
    v = np.asarray(v, float)
    return dict(mean=float(v.mean()), sd=float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                half_range=float((v.max() - v.min()) / 2),
                values=[float(x) for x in v])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, help="the arm's score JSON")
    ap.add_argument("--name", required=True)
    ap.add_argument("--armA", default="", help="comma-separated arm-A score JSONs")
    ap.add_argument("--thresholds", default="/data/wuyou/livo_sem/out/v04/armA_thresholds.json")
    ap.add_argument("--out", default="")
    ap.add_argument("--note", default="", help="a PRE-REGISTERED text file, written "
                    "before the arm was trained, appended verbatim to the verdict. "
                    "Empty (the default) leaves the verdict byte-identical to before "
                    "this argument existed.")
    a = ap.parse_args()

    TH = json.load(open(a.thresholds))
    apaths = [p for p in (a.armA or
                          "/data/wuyou/livo_sem/out/vs2d/results_eomt.json,"
                          "/data/wuyou/livo_sem/out/v04/armA_spread.json").split(",") if p]
    A = []
    for p in apaths:
        A += draws(p)
    B = draws(a.arm)
    L = []
    L.append("=" * 100)
    L.append("ARM %s  vs  ARM A   (%d draws vs %d draws)" % (a.name, len(B), len(A)))
    L.append("=" * 100)

    for s, lab in SUB:
        na = {r[s]["n_eval"] for r in A} | {r[s]["n_eval"] for r in B}
        L.append("  %-26s n_eval %s%s" % (lab, sorted(na),
                 "   *** n_eval DIFFERS -- NOT COMPARABLE ***" if len(na) > 1 else ""))

    out = dict(name=a.name, n_draws=len(B), subsets={})
    for s, lab in SUB:
        ma = stat([r[s]["miou_nine_abstain_excluded"] for r in A])
        mb = stat([r[s]["miou_nine_abstain_excluded"] for r in B])
        aa = stat([r[s]["acc_abstain_excluded"] for r in A])
        ab = stat([r[s]["acc_abstain_excluded"] for r in B])
        m8a = stat([float(np.mean([r[s][K][c] for c in CLS if c != "two_wheeler"]))
                    for r in A])
        m8b = stat([float(np.mean([r[s][K][c] for c in CLS if c != "two_wheeler"]))
                    for r in B])
        L.append("")
        L.append("-" * 100)
        L.append("%s" % lab)
        L.append("-" * 100)
        L.append("  mIoU-9      A %8.4f (sd %.4f)   %s %8.4f (sd %.4f)   delta %+8.4f"
                 % (ma["mean"], ma["sd"], a.name, mb["mean"], mb["sd"],
                    mb["mean"] - ma["mean"]))
        L.append("  mIoU-8 (no two_wheeler, DECOMPOSITION ONLY)"
                 "   A %8.4f   %s %8.4f   delta %+8.4f"
                 % (m8a["mean"], a.name, m8b["mean"], m8b["mean"] - m8a["mean"]))
        L.append("  point acc   A %8.4f            %s %8.4f            delta %+8.4f"
                 % (aa["mean"], a.name, ab["mean"], ab["mean"] - aa["mean"]))
        L.append("  per-class IoU (delta vs its OWN 3-sigma threshold):")
        L.append("    %-15s %9s %9s %9s %8s  %s" %
                 ("class", "A", a.name, "delta", "3sigma", "call"))
        pc = {}
        for c in CLS:
            ca = stat([r[s][K][c] for r in A]); cb = stat([r[s][K][c] for r in B])
            dl = cb["mean"] - ca["mean"]
            th = TH["per_class"][c]["thr_3sigma"]
            abst = TH["per_class"][c]["abstain"]
            call = ("ABSTAIN CELL" if abst else
                    ("UP" if dl >= th else ("DOWN" if dl <= -th else "flat")))
            pc[c] = dict(A=ca["mean"], arm=cb["mean"], delta=dl, thr=th, call=call)
            L.append("    %-15s %9.4f %9.4f %+9.4f %8.3f  %s" %
                     (c, ca["mean"], cb["mean"], dl, th, call))
        out["subsets"][s] = dict(miou9_A=ma, miou9_arm=mb, delta_miou9=mb["mean"] - ma["mean"],
                                 miou8_A=m8a, miou8_arm=m8b,
                                 delta_miou8=m8b["mean"] - m8a["mean"],
                                 acc_A=aa, acc_arm=ab, per_class=pc)

    d = out["subsets"]["outside"]["delta_miou9"]
    d8 = out["subsets"]["outside"]["delta_miou8"]
    verdict = ("PROPAGATION" if d >= BAND else
               ("FORGETTING" if d <= -BAND else "NEITHER"))
    L.append("")
    L.append("=" * 100)
    L.append("PRIMARY METRIC (out-of-frustum mIoU-9):  delta = %+0.4f   band = +-%.2f"
             % (d, BAND))
    L.append("VERDICT: %s" % verdict)
    if verdict == "NEITHER":
        L.append("  Read per P4: \"a signal of this magnitude is not enough\", "
                 "NOT \"distillation does not work\".")
        L.append("  Only arm D can separate those two.")
    L.append("DECOMPOSITION ONLY (never the verdict): the same delta without "
             "two_wheeler = %+0.4f" % d8)

    # ---- P3, the eight testable cells ------------------------------------- #
    pc = out["subsets"]["outside"]["per_class"]
    L.append("")
    L.append("P3 SIGN TEST, 8 testable cells (two_wheeler abstains):")
    hits = 0; tested = 0
    for c in CLS:
        if c == "two_wheeler":
            L.append("    %-15s ABSTAIN CELL -- not testable in v0.4" % c); continue
        want = "UP" if c in P3_UP else ("flat" if c in P3_FLAT else "DOWN")
        got = pc[c]["call"]
        tested += 1
        ok = (got == want)
        hits += ok
        L.append("    %-15s predicted %-5s  observed %-5s  %s" %
                 (c, want, got, "OK" if ok else "MISS"))
    L.append("  P3: %d / %d cells agree." % (hits, tested))
    if hits < tested:
        L.append("  Per P3, a disagreeing sign pattern IS the finding. "
                 "No explanation is retrofitted here.")
    out["verdict"] = verdict
    out["delta_primary"] = d
    out["delta_primary_no_two_wheeler"] = d8
    out["p3"] = dict(hits=hits, tested=tested)
    if a.note:
        note = open(a.note).read().rstrip("\n")
        L.append("")
        L.append(note)
        out["prereg_note"] = note
        out["prereg_note_file"] = os.path.abspath(a.note)
    L.append("=" * 100)
    txt = "\n".join(L)
    print(txt)
    if a.out:
        json.dump(out, open(a.out, "w"), indent=2)
        open(os.path.splitext(a.out)[0] + ".txt", "w").write(txt + "\n")
        print("\n-> %s" % a.out)


if __name__ == "__main__":
    main()
