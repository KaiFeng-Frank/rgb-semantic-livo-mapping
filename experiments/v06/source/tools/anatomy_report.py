#!/usr/bin/env python3
"""tools/anatomy_report.py -- verdict for the completion-head question.

Reads out/v06_anatomy/anat_<tag>.json (tools/residual_anatomy.py) and the v0.5 jsons of
the same caches.  Every rule it applies was fixed in out/v06_anatomy/PREREG.md before
the runs; this file only executes them.
"""
import json
import os
import sys

ROOT = "/data/livo_sem"
O = ROOT + "/out/v06_anatomy"
V05 = ROOT + "/out/v05"
TAGS = ["ZS_r1", "B0_r1", "B0_r2", "RP_r1"]
M = "miou9_abstain_excluded"
BAND = 0.60
NAMES = ["car", "large_vehicle", "two_wheeler", "person", "road", "sidewalk",
         "terrain", "vegetation", "manmade"]


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def real(p):
    return os.path.realpath(p if os.path.isabs(p) else os.path.join(ROOT, p))


A = {t: load("%s/anat_%s.json" % (O, t)) for t in TAGS}
R = {t: load("%s/map_%s.json" % (V05, t)) for t in TAGS}
miss = [t for t in TAGS if A[t] is None or R[t] is None]
if miss:
    print("missing:", miss)
    sys.exit(1)

# ------------------------------------------------------------------ 0. instrument
print("=" * 84)
print("0. INSTRUMENT -- negative control against v0.5, and full error accounting")
ok_all = True
for t in TAGS:
    a, r = A[t], R[t]
    same_pred = real(a["pred"]) == real(r["pred"])
    same = all(a["map_all_lookup"][s][k] == r["map_all_lookup"][s][k]
               for s in ("outside", "frustum", "global")
               for k in (M, "acc_abstain_excluded", "n_eval", "n_labelled"))
    acct = all(all(a["anatomy"][s]["accounting"].values()) for s in ("outside", "frustum"))
    print("  %-6s same cache %-5s  map_all_lookup == v0.5 %-5s  accounting %-5s  (out %s %.4f)"
          % (t, same_pred, same, acct, M, a["map_all_lookup"]["outside"][M]))
    ok_all &= same_pred and same and acct
if not ok_all:
    print("  -> WITHHELD: not the v0.5 map, or not every error accounted for.")
    sys.exit(2)
print("  -> every number below is the v0.5 map, and every counted error is classified")


def out(t):
    return A[t]["anatomy"]["outside"]


# ------------------------------------------------------------------ 1. table
print()
print("=" * 84)
print("1. OUT-OF-FRUSTUM RESIDUAL (headline subset), sparse := n_obs <= 4")
print("%-6s | %-7s %-7s %-12s | %-9s %-6s %-7s %-6s | %-9s"
      % ("arm", "model", "ceiling", "sparse-orcl", "errors", "MIX", "SPARSE", "DENSE", "floor"))
for t in TAGS:
    d = out(t); e = d["errors"]; sp = d["split"]["4"]
    print("%-6s | %7.2f %7.2f %12.2f | %9d %5.1f%% %6.1f%% %5.1f%% | %9d"
          % (t, d["model"][M], d["ceiling"][M], sp["sparse_oracle_miou"], e,
             100.0 * d["mix"] / e, 100.0 * sp["sparse"] / e, 100.0 * sp["dense"] / e, d["floor"]))
print("  model = headline map mIoU; ceiling = every voxel labelled with its own GT majority;")
print("  sparse-orcl = only sparse voxels relabelled perfectly (the most completion could buy)")

# ------------------------------------------------------------------ 2. verdict
print()
print("=" * 84)
print("2. VERDICT (preregistered; B0 = the deployable arm)")
for t in ("B0_r1", "B0_r2"):
    d = out(t)
    largest = {}
    for T in ("2", "4", "8"):
        sp = d["split"][T]
        c = {"MIX": d["mix"], "SPARSE": sp["sparse"], "DENSE": sp["dense"]}
        largest[T] = max(c, key=c.get)
    lab = set(largest.values())
    what = largest["4"] if len(lab) == 1 else "THRESHOLD-SENSITIVE %s" % largest
    so = d["split"]["4"]["sparse_oracle_miou"]; mo = d["model"][M]
    rule1 = (so - mo) < BAND
    print("  %s  model %.2f  sparse-oracle %.2f  -> completion's best case %s the %.2f band"
          % (t, mo, so, "is BELOW" if rule1 else "EXCEEDS", BAND))
    print("         largest residual category at T=2/4/8: %s" % largest)
    if rule1:
        print("         RULE 1: NO TARGET -- even perfect classification of every sparse voxel")
        print("                 cannot move the headline measurably.")
    msg = {"MIX": "R2: MIX largest -- reason 1 stands as stated (discretization).",
           "SPARSE": "R1: SPARSE largest -- reason 1 is REFUTED; completion has a target.",
           "DENSE": "R3: DENSE largest -- no completion target, but reason 1 is worded wrongly for "
                    "B0: its residual is systematic classification on well-observed geometry."}
    print("         " + msg.get(what, "rule 2 undecided: %s" % what))

# ------------------------------------------------------------------ 3. consistency
print()
print("=" * 84)
print("3. CONSISTENCY PREDICTIONS (the anatomy is trusted only if these hold)")
fl = {t: out(t)["floor"] for t in ("ZS_r1", "B0_r1", "RP_r1")}
mean = sum(fl.values()) / 3.0
dev = max(abs(v - mean) / mean for v in fl.values())
print("  P1 floor model-independent (<=10%%): %s   max deviation %.1f%%   %s"
      % ("PASS" if dev <= 0.10 else "FAIL", 100 * dev, fl))
sh = {t: out(t)["mix"] / out(t)["errors"] for t in ("ZS_r1", "B0_r1", "RP_r1")}
p2 = sh["ZS_r1"] < sh["B0_r1"] < sh["RP_r1"]
print("  P2 MIX share rises ZS->B0->RP:      %s   %s"
      % ("PASS" if p2 else "FAIL", {k: "%.1f%%" % (100 * v) for k, v in sh.items()}))
rp = out("RP_r1")
p3 = all(rp["mix_by_class"][c] > rp["cls_by_class"][c] for c in ("road", "sidewalk"))
print("  P3 Rprime road/sidewalk errors majority MIX: %s   road MIX %d vs CLS %d,  sidewalk MIX %d vs CLS %d"
      % ("PASS" if p3 else "FAIL", rp["mix_by_class"]["road"], rp["cls_by_class"]["road"],
         rp["mix_by_class"]["sidewalk"], rp["cls_by_class"]["sidewalk"]))

# ------------------------------------------------------------------ 4. shape
print()
print("=" * 84)
print("4. WHERE B0's ERRORS SIT (B0_r1, out-of-frustum)")
d = out("B0_r1")
print("  by n_obs (confident hits over the whole drive):")
print("    %-10s %9s %11s %8s %8s %8s" % ("n_obs", "voxels", "scored pts", "CLS %", "MIX %", "floor %"))
for b in d["nobs_curve"]:
    if b["scored"] == 0:
        continue
    hi = "inf" if b["hi"] > 1e9 else "%d" % (b["hi"] - 1)
    print("    %-10s %9d %11d %7.2f%% %7.2f%% %7.2f%%"
          % ("%d-%s" % (b["lo"], hi), b["voxels"], b["scored"], 100.0 * b["cls"] / b["scored"],
             100.0 * b["mix"] / b["scored"], 100.0 * b["floor"] / b["scored"]))
print("  by closest-approach range:")
print("    %-10s %9s %11s %8s %8s" % ("range m", "voxels", "scored pts", "CLS %", "MIX %"))
for b in d["range_curve"]:
    if b["scored"] == 0:
        continue
    hi = "inf" if b["hi"] > 1e8 else "%.0f" % b["hi"]
    print("    %-10s %9d %11d %7.2f%% %7.2f%%"
          % ("%.0f-%s" % (b["lo"], hi), b["voxels"], b["scored"], 100.0 * b["cls"] / b["scored"],
             100.0 * b["mix"] / b["scored"]))

# ------------------------------------------------------------------ 5. per class
print()
print("=" * 84)
print("5. PER CLASS, out-of-frustum IoU: model vs ceiling, and what the errors are")
for t in ("B0_r1", "RP_r1"):
    d = out(t)
    print("  %s" % t)
    print("    %-14s %7s %8s %10s %10s" % ("class", "model", "ceiling", "MIX pts", "CLS pts"))
    for c in NAMES:
        mi = d["model"]["per_class_iou_abstain_excluded"].get(c)
        ce = d["ceiling"]["per_class_iou_abstain_excluded"].get(c)
        if mi is None:
            continue
        print("    %-14s %7.2f %8.2f %10d %10d" % (c, mi, ce, d["mix_by_class"][c], d["cls_by_class"][c]))
