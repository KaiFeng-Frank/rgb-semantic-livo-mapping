#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""audit_results.py -- post-hoc invariants on the SHIPPED numbers, not on synthetic
data.  If any of these fails the table must not be quoted."""
import sys, json
import numpy as np

R = json.load(open(sys.argv[1]))
agg = R["agg"]
ok = True


def chk(name, cond, detail=""):
    global ok
    ok = ok and bool(cond)
    print("  %-62s %s %s" % (name, "PASS" if cond else "**FAIL**", detail))


def m(arm, sub, met):
    v = agg[arm][sub][met]
    return v["mean"] if isinstance(v, dict) else v


arms = [a for a in agg if a in ("3d", "2d", "hybrid", "2d_naive_projection")]
subs = [s for s in agg["3d"] if isinstance(agg["3d"][s], dict) and "n_eval" in agg["3d"][s]]

print("ARMS   :", arms)
print("SUBSETS:", len(subs))
print()

# 1. THE thing that invalidates everything: identical denominators.
bad = []
for s in subs:
    ref = m("3d", s, "n_eval")
    for a in arms:
        if abs(m(a, s, "n_eval") - ref) > 0.5:
            bad.append((s, a, ref, m(a, s, "n_eval")))
chk("every arm is scored on the IDENTICAL point set, every subset", not bad, str(bad[:3]))

# 2. partitions
for parts, whole in ((["frustum", "outside"], "global"),
                     (["sem_boundary", "sem_interior"], "global"),
                     (["depth_edge", "depth_interior"], "frustum"),
                     (["range_0_10", "range_10_20", "range_20_30", "range_30_50",
                       "range_50_inf"], "global"),
                     (["frustum_range_0_10", "frustum_range_10_20", "frustum_range_20_30",
                       "frustum_range_30_50", "frustum_range_50_inf"], "frustum")):
    got = sum(m("3d", p, "n_eval") for p in parts if p in subs)
    chk("partition %-42s" % ("+".join(p.replace("frustum_range_", "fr").replace("range_", "r")
                                      for p in parts) + " == " + whole),
        abs(got - m("3d", whole, "n_eval")) < 0.5, "%d vs %d" % (got, m("3d", whole, "n_eval")))

# 3. the 3D arm never abstains -> both conventions coincide
chk("3D arm coverage == 1", abs(m("3d", "global", "coverage") - 1.0) < 1e-9)
chk("3D arm: convention (i) == convention (ii)",
    abs(m("3d", "global", "acc_abstain_excluded") - m("3d", "global", "acc_abstain_wrong")) < 1e-6)
if "hybrid" in arms:
    chk("hybrid coverage == 1", abs(m("hybrid", "global", "coverage") - 1.0) < 1e-9)

# 4. the 2D arm answers only inside the frustum
n2 = m("2d", "global", "n_labelled")
nf = m("2d", "frustum", "n_eval")
chk("2D arm labels nothing outside the frustum",
    m("2d", "outside", "n_labelled") == 0 and n2 <= nf + 0.5, "%d labelled <= %d in frustum" % (n2, nf))

# 5. abstention leak closed: UNMAPPED must appear in the ANSWERED bucket
nu = m("2d", "frustum", "n_unmapped_pred")
chk("2D `sky` predictions are ANSWERED-and-wrong, not abstentions", nu > 0,
    "%d unmapped predictions counted wrong" % nu)
chk("2D convention (i) <= convention (ii) at coverage < 1",
    m("2d", "global", "acc_abstain_wrong") <= m("2d", "global", "acc_abstain_excluded") + 1e-9)

# 6. occlusion concession direction
if "2d_naive_projection" in arms:
    chk("z-buffer HELPS the 2D arm's accuracy vs naive projection (it is a concession)",
        m("2d", "frustum", "acc_abstain_excluded") >=
        m("2d_naive_projection", "frustum", "acc_abstain_excluded") - 1e-9,
        "%.3f vs %.3f" % (m("2d", "frustum", "acc_abstain_excluded"),
                          m("2d_naive_projection", "frustum", "acc_abstain_excluded")))
    chk("naive projection labels MORE points than z-buffer",
        m("2d_naive_projection", "global", "n_labelled") > n2)

# 7. repeat spread, so no gap is quoted below the instrument's own noise
sp = []
for a in arms:
    for met in ("acc_abstain_excluded", "miou_nine_abstain_excluded"):
        v = agg[a]["frustum"][met]
        if isinstance(v, dict):
            sp.append((a, met, v["half_range"]))
print()
print("MEASURED instrument noise (half-range over repeats, in-frustum):")
for a, met, h in sp:
    print("   %-24s %-32s ±%.3f" % (a, met, h))
print()
print("AUDIT:", "ALL PASS" if ok else "FAILED")
sys.exit(0 if ok else 1)
