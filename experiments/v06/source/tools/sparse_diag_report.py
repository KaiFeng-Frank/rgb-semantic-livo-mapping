#!/usr/bin/env python3
"""tools/sparse_diag_report.py -- executes out/v06_sparsediag/PREREG.md (+ A1) on the runs."""
import json
import os
import sys

O = "/data/livo_sem/out/v06_sparsediag"
CTL = "/data/livo_sem/out/v06_mapeval/ctl_B0_r1.json"
TAGS = ["B0_r1", "B0_r2", "ZS_r1"]
J = {t: json.load(open("%s/diag_%s.json" % (O, t))) for t in TAGS if os.path.exists("%s/diag_%s.json" % (O, t))}
if len(J) < 3:
    print("missing:", [t for t in TAGS if t not in J]); sys.exit(1)

print("=" * 84)
print("0. GATES")
ctl = json.load(open(CTL))
same = all(J["B0_r1"][k] == ctl[k] for k in ("map_all_lookup", "anatomy", "anatomy_voxel", "reweighted_map_vs_scan"))
print("  B0_r1 shared sections == map_eval control run: %s" % same)
acc_ok = True
for t in TAGS:
    iso = J[t]["sparse_diag"]["isolation"]
    a = all(iso[s]["accounting_n_equal"] and iso[s]["accounting_correct_equal"] for s in ("frustum", "outside"))
    print("  %-6s isolation accounting == offline_all: %s" % (t, a))
    acc_ok &= a
if not (same and acc_ok):
    print("  -> WITHHELD"); sys.exit(2)


def P(t, pop, sub="global"):
    return J[t]["sparse_diag"]["populations"][sub][pop]


def pct(a, b):
    return 100.0 * a / b if b else float("nan")


print()
print("=" * 84)
print("1. SPARSE WRONG CELLS (n_obs<=4, fused != GT majority), all cells")
print("  %-6s %9s | %7s %10s %7s | %9s %13s | %10s %10s"
      % ("arm", "cells", "SINGLE", "CONSISTENT", "SPLIT", "cam-reach", "nonsplit-cam", "d8min-wrg", "d8min-cor"))
for t in TAGS:
    w = P(t, "sparse_wrong"); c = P(t, "sparse_correct")
    print("  %-6s %9d | %6.1f%% %9.1f%% %6.1f%% | %8.1f%% %12.1f%% | %9.3fm %9.3fm"
          % (t, w["cells"], pct(w["single"], w["cells"]), pct(w["consistent"], w["cells"]),
             pct(w["split"], w["cells"]), pct(w["camera_labelable"], w["cells"]),
             pct(w["nonsplit_camera_labelable"], w["nonsplit"]),
             w["nonsplit_d8_min_median"], c["d8_min_q"][1]))
print("  (d8min-wrg = median d8_min of non-SPLIT sparse wrong cells; d8min-cor = of sparse correct)")
print("  dense wrong cells for contrast:")
for t in TAGS:
    w = P(t, "dense_wrong")
    print("  %-6s %9d | %6.1f%% %9.1f%% %6.1f%% | %8.1f%% camera-reachable"
          % (t, w["cells"], pct(w["single"], w["cells"]), pct(w["consistent"], w["cells"]),
             pct(w["split"], w["cells"]), pct(w["camera_labelable"], w["cells"])))

print()
print("=" * 84)
print("2. PER-SCAN ACCURACY vs ISOLATION (d8 = distance to 8th nearest same-scan neighbour)")
for t in TAGS:
    iso = J[t]["sparse_diag"]["isolation"]["outside"]
    print("  %-6s out-of-frustum: least isolated quartile (d8<=%.3fm) acc %.2f%%   most isolated (d8>=%.3fm) acc %.2f%%"
          % (t, iso["q1_d8"], iso["least_isolated"]["acc"], iso["q3_d8"], iso["most_isolated"]["acc"]))
iso = J["B0_r1"]["sparse_diag"]["isolation"]["outside"]
print("  B0_r1 curve (out-of-frustum):")
for b in iso["curve"]:
    if b["n"]:
        print("    d8 %5.2f-%-6s  %11d pts  acc %6.2f%%" % (b["lo"], "inf" if b["hi"] > 1e8 else "%.2f" % b["hi"], b["n"], b["acc"]))

print()
print("=" * 84)
print("3. SUPERVISION REACH (camera pseudo-label reachability, z-buffer visible)")
for t in ("B0_r1",):
    r = J[t]["sparse_diag"]["supervision_reach"]
    print("  per scan: %.1f%% of scored points are camera-labelable in their own scan" % (100 * r["per_scan_camera_labelable_frac"]))
    print("  via the map: %.1f%% of out-of-frustum points sit in a cell the camera labelled at SOME time"
          % (100 * r["out_of_frustum_points_in_cells_labelable_sometime"]))
    print("  %.1f%% of labelled cells are camera-labelable at some time" % (100 * r["labelled_cells_labelable_sometime"]))

print()
print("=" * 84)
print("4. DECISION (PREREG + A1)")
res = {}
for t in ("B0_r1", "B0_r2"):
    w = P(t, "sparse_wrong"); c = P(t, "sparse_correct")
    iso = J[t]["sparse_diag"]["isolation"]["outside"]
    fusion = pct(w["split"], w["cells"]) >= 50.0
    sup = pct(w["nonsplit_camera_labelable"], w["nonsplit"]) >= 50.0
    ctx_i = (iso["least_isolated"]["acc"] - iso["most_isolated"]["acc"]) >= 5.0
    ctx_ii = w["nonsplit_d8_min_median"] > c["d8_min_q"][1]
    res[t] = (fusion, sup, ctx_i and ctx_ii)
    print("  %s  FUSION primary: %-5s | SUPERVISION: %-5s | CONTEXT: %-5s (i %s, ii %s)"
          % (t, fusion, sup, ctx_i and ctx_ii, ctx_i, ctx_ii))
agree = res["B0_r1"] == res["B0_r2"]
print("  B0_r1 and B0_r2 agree: %s" % agree)
if agree:
    fusion, sup, ctx = res["B0_r1"]
    if fusion:
        print("  -> FUSION is the primary lever; the network is not where sparse cells fail.")
    elif sup and ctx:
        print("  -> SUPERVISION and CONTEXT both supported: the method combines map-propagated")
        print("     pseudo-labels with a context mechanism (completion head / multi-scan input).")
    elif sup:
        print("  -> SUPERVISION supported, CONTEXT not: propagate pseudo-labels through the map;")
        print("     a completion head is not supported by this evidence.")
    elif ctx:
        print("  -> CONTEXT supported, SUPERVISION not: a context mechanism (completion head /")
        print("     multi-scan input) is the lever.")
    else:
        print("  -> NEITHER: these three causes do not explain the sparse errors; back to analysis.")
