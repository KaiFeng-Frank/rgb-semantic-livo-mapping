#!/usr/bin/env python3
"""tools/v06_report.py -- the v0.6 table.

Four questions, in order of how much they can hurt:

 1. DECISION CONTAMINATION.  How much of the v0.4/v0.5 headline was bought by having
    looked at seq 07 while choosing things?  Difference-in-differences against the
    zero-shot arm, which no decision was tuned against:
        [B0(07) - B0(09)] - [ZS(07) - ZS(09)]
    The raw seq07-minus-seq09 gap is NOT this number -- it also contains the two scenes
    being differently hard -- so the raw gap is printed too, and the difference between
    them is the whole point.

 2. WHAT THE KL ANCHOR COST.  B0 vs B0_noKL, same seeds, one factor apart.

 3. DOES THE SEED MATTER MORE THAN THE NOISE?  Three seeds x two forward passes lets the
    seed-to-seed spread be compared against the pass-to-pass spread of one checkpoint.
    If they are the same size, "three seeds" bought nothing and the honest statement is
    that the arm's variance is dominated by the model's own non-determinism.

 4. DO v0.5's THREE STRUCTURAL FINDINGS SURVIVE ON A CLEAN SEQUENCE?
      (a) the map scores better than the per-scan predictions it is built from
      (b) that gain shrinks as the classifier gets stronger
      (c) the strongest arm LOSES on planar classes at map level
    These were measured on seq 07 only.  (c) is the one that matters most -- it is the
    problem this baseline exposed, and if it does not reproduce it was a seq-07 artefact.
"""
import glob
import json
import os
import re
import statistics as st
import sys

OUT = "/data/livo_sem/out/v06_score"
M = "miou9_abstain_excluded"
PC = "per_class_iou_abstain_excluded"
PLANAR = ("road", "sidewalk")
TAG_RE = re.compile(r"^map_(?P<arm>.+)_(?P<seq>\d{2})_r(?P<rep>\d+)\.json$")


def load_all():
    """-> {(arm, seq): [json, ...]}"""
    out = {}
    for f in sorted(glob.glob(os.path.join(OUT, "map_*.json"))):
        m = TAG_RE.match(os.path.basename(f))
        if not m:
            continue
        try:
            d = json.load(open(f))
        except Exception as e:
            print("  ! unreadable %s: %s" % (os.path.basename(f), e))
            continue
        out.setdefault((m["arm"], m["seq"]), []).append(d)
    return out


def val(d, key, field=M, cls=None):
    node = d.get(key, {}).get("outside", {})
    return node.get(PC, {}).get(cls) if cls else node.get(field)


def agg(ds, key, cls=None):
    vs = [val(d, key, cls=cls) for d in ds]
    vs = [v for v in vs if v is not None]
    if not vs:
        return None, None, 0
    return st.mean(vs), (st.pstdev(vs) if len(vs) > 1 else 0.0), len(vs)


def fam(data, prefix, seq):
    """all runs of an arm family (e.g. every armB0_s* seed) on one sequence"""
    out = []
    for (arm, s), ds in data.items():
        if s == seq and arm.startswith(prefix) and "noKL" not in arm[len(prefix):]:
            out.append((arm, ds))
    return sorted(out)


def f(m, s=None):
    if m is None:
        return "  --  "
    return "%6.2f" % m if s is None else "%6.2f±%.2f" % (m, s)


data = load_all()
if not data:
    print("no scored json in %s -- run opt/score_v06.sh first" % OUT)
    sys.exit(1)
print("loaded %d (arm, seq) groups, %d runs\n" % (len(data), sum(len(v) for v in data.values())))

FAMS = [("ZS", "ZS"), ("armB0_s", "B0"), ("armB0_noKL_s", "B0_noKL"),
        ("armRprime_noKL_s", "Rprime_noKL")]


def family_mean(prefix, seq, key, cls=None):
    """mean over every seed and every pass of a family"""
    ds = []
    for _, runs in fam(data, prefix, seq):
        ds += runs
    return agg(ds, key, cls=cls)


# ------------------------------------------------------------------ main table
print("=" * 80)
print("MAIN TABLE -- out-of-frustum mIoU-9, mean±sd over seeds and passes")
print("%-13s | %-16s | %-16s" % ("", "seq 07 (design-seen)", "seq 09 (clean)"))
print("%-13s | %-7s %-8s | %-7s %-8s" % ("arm", "offline", "map", "offline", "map"))
for pre, name in FAMS:
    r = [family_mean(pre, s, k) for s in ("07", "09") for k in ("offline_all", "map_all_lookup")]
    print("%-13s | %-7s %-8s | %-7s %-8s"
          % (name, f(*r[0][:2]), f(*r[1][:2]), f(*r[2][:2]), f(*r[3][:2])))

# ------------------------------------------------------------------ 1. DiD
print()
print("=" * 80)
print("1. DECISION CONTAMINATION (difference-in-differences vs the zero-shot arm)")
zs07 = family_mean("ZS", "07", "offline_all")[0]
zs09 = family_mean("ZS", "09", "offline_all")[0]
if zs07 is None or zs09 is None:
    print("  zero-shot arm missing on one sequence -- DiD cannot be computed")
else:
    print("  scene difficulty (ZS, untuned):      seq07 %s  seq09 %s   gap %+.2f"
          % (f(zs07), f(zs09), zs07 - zs09))
    for pre, name in FAMS[1:]:
        a07 = family_mean(pre, "07", "offline_all")[0]
        a09 = family_mean(pre, "09", "offline_all")[0]
        if a07 is None or a09 is None:
            continue
        raw = a07 - a09
        did = raw - (zs07 - zs09)
        print("  %-12s raw gap %+6.2f   minus scene difficulty  ->  contamination %+6.2f"
              % (name, raw, did))
    print()
    print("  ASSUMPTION, stated not hidden: that the two scenes are equally hard for a")
    print("  tuned arm as for an untuned one.  If a tuned arm is differentially better at")
    print("  seq 07's particular scene content, some of that lands here as contamination.")

# ------------------------------------------------------------------ 2. KL cost
print()
print("=" * 80)
print("2. WHAT THE ANTI-FORGETTING KL ANCHOR COST (B0 vs B0_noKL, one factor apart)")
for seq in ("07", "09"):
    for key, label in (("offline_all", "per-point"), ("map_all_lookup", "map")):
        a = family_mean("armB0_s", seq, key)[0]
        b = family_mean("armB0_noKL_s", seq, key)[0]
        if a is None or b is None:
            continue
        print("  seq%s %-9s  with KL %s   without %s   cost of the anchor %+.2f"
              % (seq, label, f(a), f(b), a - b))

# ------------------------------------------------------------------ 3. variance
print()
print("=" * 80)
print("3. SEED SPREAD vs FORWARD-PASS NOISE (does running 3 seeds buy anything?)")
for pre, name in FAMS[1:]:
    for seq in ("07", "09"):
        per_seed, within = [], []
        for arm, runs in fam(data, pre, seq):
            vs = [val(d, "offline_all") for d in runs]
            vs = [v for v in vs if v is not None]
            if not vs:
                continue
            per_seed.append(st.mean(vs))
            if len(vs) > 1:
                within.append(st.pstdev(vs))
        if len(per_seed) < 2:
            continue
        bs = st.pstdev(per_seed)
        ws = st.mean(within) if within else 0.0
        note = "seed spread is real" if bs > 2 * ws else "SEED SPREAD IS WITHIN NOISE"
        print("  %-12s seq%s  between-seed %.3f   within-seed(pass) %.3f   -> %s"
              % (name, seq, bs, ws, note))

# ------------------------------------------------------------------ 4. v0.5 findings
print()
print("=" * 80)
print("4. DO v0.5's STRUCTURAL FINDINGS SURVIVE ON seq 09?")
gains = {}
for seq in ("07", "09"):
    print("  -- seq %s --" % seq)
    for pre, name in FAMS:
        off = family_mean(pre, seq, "offline_all")[0]
        mp = family_mean(pre, seq, "map_all_lookup")[0]
        if off is None or mp is None:
            continue
        gains.setdefault(seq, []).append((name, off, mp - off))
        print("     (a) %-12s offline %s -> map %s   %s"
              % (name, f(off), f(mp), "map WINS" if mp > off else "map LOSES"))
    g = gains.get(seq, [])
    if len(g) >= 2:
        g2 = sorted(g, key=lambda t: t[1])          # by classifier strength
        shrink = g2[0][2] > g2[-1][2]
        print("     (b) gain %s (weakest %s: %+.2f  ->  strongest %s: %+.2f)"
              % ("SHRINKS as the classifier strengthens" if shrink else "DOES NOT shrink",
                 g2[0][0], g2[0][2], g2[-1][0], g2[-1][2]))
    for cls in PLANAR:
        off = family_mean("armRprime_noKL_s", seq, "offline_all", cls=cls)[0]
        mp = family_mean("armRprime_noKL_s", seq, "map_all_lookup", cls=cls)[0]
        if off is None or mp is None:
            continue
        print("     (c) Rprime %-9s offline %s -> map %s   %s"
              % (cls, f(off), f(mp), "LOSES (v0.5 reproduces)" if mp < off else "does not lose"))
print()
print("  (c) is the finding this baseline was built to produce.  If it does not reproduce")
print("  on seq 09, it was a seq-07 artefact and must not be claimed.")

# ------------------------------------------------------------------ 5. voxel-weighted (map claims)
# map_eval.py writes anatomy_voxel (each map cell once, SSC style) and reweighted_map_vs_scan
# (each cell weight 1 spread over its points, same GT/weights on both sides).  These are the
# primary unit for map-level claims; the point-weighted readings above stay for comparability.
print()
print("=" * 80)
print("5. VOXEL-WEIGHTED MAP READINGS, out-of-frustum (each map cell counts once)")
print("%-13s | %-8s %-9s %-7s | %-8s %-9s %-7s"
      % ("", "seq 07", "", "", "seq 09", "", ""))
print("%-13s | %-8s %-9s %-7s | %-8s %-9s %-7s"
      % ("arm", "vox-mIoU", "sp-oracle", "wrong%", "vox-mIoU", "sp-oracle", "wrong%"))


def vox_stats(prefix, seq):
    ds = []
    for _, runs in fam(data, prefix, seq):
        ds += runs
    ds = [d for d in ds if "anatomy_voxel" in d]
    if not ds:
        return ("  --", "  --", "  --")
    m = st.mean(d["anatomy_voxel"]["outside"]["model"][M] for d in ds)
    so = st.mean(d["anatomy_voxel"]["outside"]["split"]["4"]["sparse_oracle_miou"] for d in ds)
    w = st.mean(100.0 * d["anatomy_voxel"]["outside"]["wrong"] / max(1, d["anatomy_voxel"]["outside"]["voxels"]) for d in ds)
    return ("%.2f" % m, "%.2f" % so, "%.2f" % w)


for pre, name in FAMS:
    a7 = vox_stats(pre, "07"); a9 = vox_stats(pre, "09")
    print("%-13s | %8s %9s %7s | %8s %9s %7s" % (name, a7[0], a7[1], a7[2], a9[0], a9[1], a9[2]))
print()
print("  map vs per-scan with each cell weight 1 spread over its points (same points, same GT):")
for seq in ("07", "09"):
    for pre, name in FAMS:
        ds = []
        for _, runs in fam(data, pre, seq):
            ds += runs
        ds = [d for d in ds if "reweighted_map_vs_scan" in d]
        if not ds:
            continue
        mp = st.mean(d["reweighted_map_vs_scan"]["outside"]["map"]["miou9"] for d in ds)
        ps = st.mean(d["reweighted_map_vs_scan"]["outside"]["per_scan"]["miou9"] for d in ds)
        print("     seq%s %-12s per-scan %.2f -> map %.2f   %s"
              % (seq, name, ps, mp, "map WINS" if mp > ps else "map LOSES"))
