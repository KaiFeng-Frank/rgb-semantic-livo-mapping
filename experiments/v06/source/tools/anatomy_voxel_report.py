#!/usr/bin/env python3
"""tools/anatomy_voxel_report.py -- POST-HOC voxel-weighted check of the completion verdict.
Rules fixed in out/v06_anatomy/PREREG_voxel.md before any voxel-weighted number existed."""
import json
import os
import sys

O = "/data/livo_sem/out/v06_anatomy"
TAGS = ["ZS_r1", "B0_r1", "B0_r2", "RP_r1"]
M = "miou9_abstain_excluded"
NAMES = ["car", "large_vehicle", "two_wheeler", "person", "road", "sidewalk",
         "terrain", "vegetation", "manmade"]


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


NEW = {t: load("%s/anatv_%s.json" % (O, t)) for t in TAGS}
OLD = {t: load("%s/anat_%s.json" % (O, t)) for t in TAGS}
if any(NEW[t] is None or OLD[t] is None for t in TAGS):
    print("missing json")
    sys.exit(1)

print("=" * 86)
print("0. INSTRUMENT (post-hoc run must reproduce the preregistered one exactly)")
ok = True
for t in TAGS:
    n, o = NEW[t], OLD[t]
    a = n["map_all_lookup"] == o["map_all_lookup"]
    b = n["anatomy"] == o["anatomy"]
    ce = all(abs(v - 100.0) < 1e-9 for s in ("outside", "global")
             for v in n["anatomy_voxel"][s]["ceiling"]["per_class_iou_abstain_excluded"].values())
    print("  %-6s map_all_lookup same %-5s  point-weighted anatomy same %-5s  voxel ceiling==100 %-5s"
          % (t, a, b, ce))
    ok &= a and b and ce
if not ok:
    print("  -> WITHHELD")
    sys.exit(2)


def V(t, s="outside"):
    return NEW[t]["anatomy_voxel"][s]


def P(t):
    return NEW[t]["anatomy"]["outside"]


print()
print("=" * 86)
print("1. VOXEL-WEIGHTED, outside (voxels majority out-of-frustum); point-weighted beside it")
print("%-6s | %9s %9s %7s %8s %8s | %8s %8s || %8s %8s"
      % ("arm", "voxels", "wrong", "wrong%", "SPARSE%", "DENSE%", "vox-mIoU", "sp-orcl", "pt-mIoU", "pt-orcl"))
for t in TAGS:
    d = V(t); w = d["wrong"]; sp = d["split"]["4"]; p = P(t)
    print("%-6s | %9d %9d %6.2f%% %7.1f%% %7.1f%% | %8.2f %8.2f || %8.2f %8.2f"
          % (t, d["voxels"], w, 100.0 * w / d["voxels"], 100.0 * sp["sparse_wrong"] / w,
             100.0 * sp["dense_wrong"] / w, d["model"][M], sp["sparse_oracle_miou"],
             p["model"][M], p["split"]["4"]["sparse_oracle_miou"]))
print("  sparse voxels (n_obs<=4) are %.1f%% of B0's outside voxels"
      % (100.0 * V("B0_r1")["split"]["4"]["sparse_voxels"] / V("B0_r1")["voxels"]))

print()
print("=" * 86)
print("2. VERDICT (post-hoc rules, PREREG_voxel.md)")
d1 = abs(V("B0_r1")["model"][M] - V("B0_r2")["model"][M])
band = max(0.60, 2.1 * d1)
print("  pass-to-pass |B0_r1 - B0_r2| = %.3f  ->  band_vox = max(0.60, 2.1 x %.3f) = %.3f" % (d1, d1, band))
verdicts = []
for t in ("B0_r1", "B0_r2"):
    d = V(t)
    gain = d["split"]["4"]["sparse_oracle_miou"] - d["model"][M]
    largest = {}
    for T in ("2", "4", "8"):
        sp = d["split"][T]
        largest[T] = "SPARSE" if sp["sparse_wrong"] > sp["dense_wrong"] else "DENSE"
    agree = len(set(largest.values())) == 1
    r1 = gain < band
    print("  %s  model %.2f  sparse-oracle %.2f  (best case %.2f vs band %.2f)  largest %s%s"
          % (t, d["model"][M], d["split"]["4"]["sparse_oracle_miou"], gain, band, largest,
             "" if agree else "  THRESHOLD-SENSITIVE"))
    print("         rule 1: %s   rule 2: %s"
          % ("NO TARGET" if r1 else "TARGET", largest["4"] if agree else "undecided"))
    verdicts.append((r1, largest["4"] if agree else None))
robust = all(v[0] and v[1] == "DENSE" for v in verdicts)
target = any((not v[0]) or v[1] == "SPARSE" for v in verdicts)
print()
if robust:
    print("  -> ROBUST: no completion target per point AND per voxel.  The paper may say so")
    print("     without qualifying the evaluation unit.")
elif target:
    print("  -> EVALUATION-DEPENDENT: no target in the point-weighted headline, a real one per")
    print("     map cell.  Reason 2 does not cover these voxels (still sparse after the whole")
    print("     drive); only reason 3 (novelty: JS3C-Net) remains.  The choice is which unit")
    print("     the paper's claim lives in.")
else:
    print("  -> MIXED: see the lines above.")
print("  expectation stated in PREREG_voxel.md: flip.  %s"
      % ("MET" if target else "NOT MET -- the expectation was wrong"))

print()
print("=" * 86)
print("3. B0_r1, outside: wrong-voxel rate by n_obs and by closest-approach range")
d = V("B0_r1")
print("    %-10s %9s %9s %8s" % ("n_obs", "voxels", "wrong", "wrong%"))
for b in d["nobs_curve"]:
    if b["voxels"]:
        hi = "inf" if b["hi"] > 1e9 else "%d" % (b["hi"] - 1)
        print("    %-10s %9d %9d %7.2f%%" % ("%d-%s" % (b["lo"], hi), b["voxels"], b["wrong"],
                                           100.0 * b["wrong"] / b["voxels"]))
print("    %-10s %9s %9s %8s" % ("range m", "voxels", "wrong", "wrong%"))
for b in d["range_curve"]:
    if b["voxels"]:
        hi = "inf" if b["hi"] > 1e8 else "%.0f" % b["hi"]
        print("    %-10s %9d %9d %7.2f%%" % ("%.0f-%s" % (b["lo"], hi), b["voxels"], b["wrong"],
                                           100.0 * b["wrong"] / b["voxels"]))

print()
print("=" * 86)
print("4. PER CLASS, outside: voxel IoU and where the wrong voxels sit (sparse := n_obs<=4)")
for t in ("B0_r1", "RP_r1"):
    d = V(t)
    print("  %s" % t)
    print("    %-14s %8s %9s %12s %11s" % ("class", "vox-IoU", "voxels", "sparse-wrong", "dense-wrong"))
    for c in NAMES:
        iou = d["model"]["per_class_iou_abstain_excluded"].get(c)
        w = d["wrong_by_class"][c]
        if iou is None:
            continue
        print("    %-14s %8.2f %9d %12d %11d" % (c, iou, w["voxels"], w["sparse_wrong"], w["dense_wrong"]))
