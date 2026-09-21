#!/usr/bin/env python3
"""
score_dynamic.py -- the v0.3 acceptance instrument for dynamic-object removal.

SemanticKITTI labels MOVING objects with their own ids (252 moving-car,
253 moving-bicyclist, 254 moving-person, 255 moving-motorcyclist,
256 moving-on-rails, 257 moving-bus, 258 moving-truck, 259 moving-other-vehicle)
and their parked counterparts with the ordinary ones (10 car, 11 bicycle,
13 bus, 15 motorcycle, 16 on-rails, 18 truck, 20 other-vehicle, 30 person,
31 bicyclist, 32 motorcyclist).  A dynamic-removal policy can therefore be
SCORED rather than eyeballed.

WHAT IT REPORTS
---------------
  dynamic_recall        fraction of GT-moving points the policy kept OUT of the map
  static_false_kill     fraction of GT-static points the policy wrongly kept out
                        (the number that matters: false kills punch holes)
  vp_static_false_kill  the same, restricted to the static vehicle/person classes,
                        where the decision is genuinely hard (a parked car looks
                        exactly like a stopped one)
  damage_ratio          static points killed per moving point killed
  ribbon_m              MAP-LEVEL: principal-axis length of the trail the tracked
                        moving vehicle still leaves in the map over a frame window
                        (default 755..787, where GT instances moving-car #7 and #4
                        drive through at 11.2 and 12.4 m/s), measured on the
                        RETAINED GT-moving points only.  MEASURED BASELINE with
                        nothing removed: 38.98 m / 2228 voxels for instance 7
                        (43.84 m for instance 4), against a 2.38 m one-sweep
                        footprint -- the map stretches one car over 16x its length.
  voxels_retained       the ribbon's voxel count at the deployed 0.20 m.  READ THIS
                        ONE, NOT ONLY ribbon_m: ribbon_m is an EXTENT and therefore
                        saturates.  Measured on this window: removing a random 90 %
                        of the car's points still leaves a 38.6 m ribbon, 99 % still
                        leaves 36.4 m.  Only removing whole OBSERVATIONS collapses it
                        (90 % of sweeps -> 16.6 m, 95 % -> 6.5 m).  voxels_retained
                        falls linearly with what was actually removed, so gate on
                        voxels_retained and quote ribbon_m as the honesty statement.

THE INTERFACE THE PIPELINE MUST EMIT
------------------------------------
One file per processed sweep:

    <mask-dir>/frame_%06d.npz          (%06d = the RAW KITTI frame index, 0..1100)

with these arrays:

  keep      (N,) bool or uint8   REQUIRED.  True  = this point was inserted into
                                 the map.  False = it was not, for ANY reason.
  eligible  (N,) bool or uint8   RECOMMENDED.  True = the point passed every gate
                                 that ALREADY existed in v0.2 (pose-validity gate,
                                 confidence gate).  The v0.3 decision is then
                                 exactly `eligible & ~keep`, and the metrics are
                                 computed on the eligible set, so a pose-gate drop
                                 at the end of the trajectory is not charged to the
                                 dynamic policy.  If absent, every non-kept point is
                                 charged to the policy and the report says so.
  idx       (N,) int32           OPTIONAL.  If `keep` covers only a SUBSET of the
                                 sweep (e.g. the post-gate slice the node carries
                                 around), `idx` gives each entry's position in the
                                 FULL sweep.  Points of the sweep not listed are
                                 treated as not-kept and not-eligible.
  xyz       (N,3) float32        RECOMMENDED.  Sensor-frame coordinates of those
                                 same points, used only to VERIFY the alignment.
                                 Alignment is checked on every frame; a frame that
                                 cannot be aligned is reported, never silently
                                 scored.
  pred      (N,) uint8           OPTIONAL.  The nuScenes-16 argmax the node used.
                                 If present, the naive class-delete control arm is
                                 scored on exactly the same frames for free.
  order     () str               OPTIONAL, default "bag".  "bag" = the point order
                                 the node received from the rosbag (azimuth-sorted
                                 by kitti_scan.synth).  "bin" = raw KITTI .bin /
                                 .label order.  The scorer converts.

WHY `order` MATTERS: the bag is azimuth-sorted, the .label file is laser-major
.bin order.  kitti_scan.synth() returns that permutation and this scorer rebuilds
it with the per-frame sweep duration (t_end - t_start), which reproduces the bag
byte-for-byte except for a handful of exact-tie points per sweep (<= 10 of 120000
measured); those are repaired by exact xyz match when `xyz` is supplied.

REFERENCE POLICIES (need no run, computed from GT alone)
--------------------------------------------------------
    --policy keep-all         v0.2 as shipped: nothing is dropped
    --policy oracle-dynamic   perfect dynamic removal (drop exactly GT-moving)
    --policy oracle-naive     "delete every vehicle/person class" with a PERFECT
                              classifier -- the analytic control arm
    --policy oracle-naive-wide  as above, plus other-vehicle/on-rails

USAGE
-----
    python3 opt/score_dynamic.py --mask-dir out/masks_v03            # score a run
    python3 opt/score_dynamic.py --policy oracle-naive               # control arm
    python3 opt/score_dynamic.py --mask-dir A --repeat-dirs A B C    # K4 spread
    python3 opt/score_dynamic.py --mask-dir A --compare-dir BASE     # delta+spread

Constraint K4: the model is non-deterministic run to run.  Any verdict must be
taken over many frames, and repeats must be reported.  --repeat-dirs prints
mean / min / max per metric; --compare-dir prints the delta next to that spread.
CPU only: this script never touches the GPU.
"""
import argparse, calendar, glob, json, os, sys
import numpy as np

sys.path.insert(0, "/data/livo_sem/src")
import sem_core as S
from kitti_scan import synth

D = "/data/livo_sem/data"
RAW = D + "/raw/2011_09_30/2011_09_30_drive_0027_sync"
LB = D + "/odometry/dataset/sequences/07/labels"

MOVING_IDS = list(range(252, 260))
STATIC_VP_IDS = [10, 11, 13, 15, 16, 18, 20, 30, 31, 32]
IGNORE_IDS = [0, 1, 99]
# GT classes a perfect classifier would map onto the nuScenes classes
# car / truck / bus / bicycle / motorcycle / pedestrian
NAIVE_GT = [10, 11, 13, 15, 18, 30]
NAIVE_GT_WIDE = NAIVE_GT + [16, 20]
# nuScenes-16 ids of those same classes, for a `pred`-driven naive arm
NAIVE_NUSC = [1, 2, 3, 5, 6, 9]
NAIVE_NUSC_WIDE = NAIVE_NUSC + [4, 8]
SK_NAMES = {10: "car", 11: "bicycle", 13: "bus", 15: "motorcycle", 16: "on-rails",
            18: "truck", 20: "other-vehicle", 30: "person", 31: "bicyclist",
            32: "motorcyclist", 40: "road", 44: "parking", 48: "sidewalk",
            49: "other-ground", 50: "building", 51: "fence", 52: "other-structure",
            60: "lane-marking", 70: "vegetation", 71: "trunk", 72: "terrain",
            80: "pole", 81: "traffic-sign", 252: "moving-car",
            253: "moving-bicyclist", 254: "moving-person",
            255: "moving-motorcyclist", 256: "moving-on-rails", 257: "moving-bus",
            258: "moving-truck", 259: "moving-other-vehicle"}


def parse_ts(path):
    out = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        date, clock = line.split(" ")
        hms, frac = (clock.split(".") + ["0"])[:2]
        y, mo, d = (int(v) for v in date.split("-"))
        h, mi, s = (int(v) for v in hms.split(":"))
        out.append(calendar.timegm((y, mo, d, h, mi, s, 0, 0, 0)) * 10**9
                   + int((frac + "000000000")[:9]))
    return np.array(out, dtype=np.int64)


def pca_extent(P):
    if len(P) < 3:
        return (0.0, 0.0, 0.0)
    c = P - P.mean(0)
    _, V = np.linalg.eigh(c.T @ c / len(c))
    pr = c @ V[:, ::-1]
    return tuple(float(pr[:, k].max() - pr[:, k].min()) for k in range(3))


class Frames(object):
    """Scan / label / geometry access, cached per frame."""

    def __init__(self, traj_path):
        self.ts = parse_ts(RAW + "/velodyne_points/timestamps_start.txt")
        self.te = parse_ts(RAW + "/velodyne_points/timestamps_end.txt")
        self.traj = S.TrajInterp(traj_path)
        self.R_IL, self.t_IL = S.T_I_L[:3, :3], S.T_I_L[:3, 3]

    def load(self, f):
        sp = RAW + "/velodyne_points/data/%010d.bin" % f
        lp = LB + "/%06d.label" % f
        if not (os.path.exists(sp) and os.path.exists(lp)):
            return None
        p = np.fromfile(sp, dtype=np.float32).reshape(-1, 4)
        lab = np.fromfile(lp, dtype=np.uint32)
        if len(p) != len(lab):
            return None
        _, tsyn, order = synth(p[:, :3], sweep_duration=(self.te[f] - self.ts[f]) / 1e9)
        t_pt = np.empty(len(p))
        t_pt[order] = tsyn
        t_pt += self.ts[f] * 1e-9
        return dict(xyz=p[:, :3], sem=(lab & 0xFFFF).astype(np.int32),
                    inst=(lab >> 16).astype(np.int32), order=order, t_pt=t_pt)

    def world(self, fr, sel):
        """sel: boolean or index array in .bin order -> world xyz (rows where the
        trajectory is valid) and the validity mask."""
        x = fr["xyz"][sel].astype(np.float64)
        R, p, ok = self.traj.query(fr["t_pt"][sel])
        return np.einsum("nij,nj->ni", R, x @ self.R_IL.T + self.t_IL) + p, ok


def align(fr, m, frame):
    """Return (keep_bin, elig_bin, pred_bin, n_repaired, n_unaligned) in .bin order."""
    n_scan = len(fr["sem"])
    keep = np.asarray(m["keep"]).astype(bool).ravel()
    elig = np.asarray(m["eligible"]).astype(bool).ravel() if "eligible" in m else None
    pred = np.asarray(m["pred"]).astype(np.int64).ravel() if "pred" in m else None
    order_kind = str(m["order"]) if "order" in m else "bag"
    idx = np.asarray(m["idx"]).astype(np.int64).ravel() if "idx" in m else None

    # positions in the node's own ordering
    pos = idx if idx is not None else np.arange(len(keep))
    if len(keep) != len(pos):
        raise SystemExit("frame %d: keep has %d entries, idx has %d"
                         % (frame, len(keep), len(pos)))

    n_rep = n_un = 0
    if order_kind == "bin":
        bin_idx = pos.copy()
    else:
        o = fr["order"]
        if pos.max(initial=-1) >= len(o):
            raise SystemExit("frame %d: idx out of range (%d >= %d)"
                             % (frame, pos.max(), len(o)))
        bin_idx = o[pos]
        if "xyz" in m:                       # verify, and repair exact ties
            xr = np.asarray(m["xyz"], dtype=np.float32).reshape(-1, 3)
            bad = np.flatnonzero((fr["xyz"][bin_idx] != xr).any(1))
            if bad.size:
                # exact-value lookup over the whole scan for the few bad rows
                tbl = {}
                for i in range(n_scan):
                    tbl.setdefault(fr["xyz"][i].tobytes(), i)
                for b in bad:
                    j = tbl.get(xr[b].tobytes())
                    if j is None:
                        n_un += 1
                        keep[b] = False
                        if elig is not None:
                            elig[b] = False
                    else:
                        bin_idx[b] = j
                        n_rep += 1
    keep_b = np.zeros(n_scan, bool)
    elig_b = np.zeros(n_scan, bool)
    pred_b = np.full(n_scan, -1, np.int64)
    keep_b[bin_idx] = keep
    elig_b[bin_idx] = elig if elig is not None else True
    if pred is not None:
        pred_b[bin_idx] = pred
    return keep_b, elig_b, pred_b, n_rep, n_un


def synth_policy(fr, kind):
    sem = fr["sem"]
    keep = np.ones(len(sem), bool)
    if kind == "keep-all":
        pass
    elif kind == "oracle-dynamic":
        keep &= ~np.isin(sem, MOVING_IDS)
    elif kind == "oracle-naive":
        keep &= ~np.isin(sem, MOVING_IDS + NAIVE_GT)
    elif kind == "oracle-naive-wide":
        keep &= ~np.isin(sem, MOVING_IDS + NAIVE_GT_WIDE)
    else:
        raise SystemExit("unknown policy %s" % kind)
    return keep, np.ones(len(sem), bool), np.full(len(sem), -1, np.int64), 0, 0


def score(args, mask_dir, policy, F):
    w0, w1 = (int(v) for v in args.window.split(":"))
    acc = dict(mv=0, mv_drop=0, st=0, st_drop=0, vp=0, vp_drop=0, ign=0,
               kept=0, elig=0, total=0, frames=0, repaired=0, unaligned=0,
               nv_mv_drop=0, nv_st_drop=0, nv_vp_drop=0, has_pred=0)
    per_class_drop = {}
    per_class_tot = {}
    mv_class_drop = {}
    mv_class_tot = {}
    ribbon_pts = {}          # inst -> list of retained world pts inside window
    ribbon_all = {}          # inst -> list of ALL world pts inside window
    mv_keys_kept = []
    missing = []
    frames = range(args.f0, args.f1, args.stride)
    for f in frames:
        fr = F.load(f)
        if fr is None:
            continue
        if policy is not None:
            keep, elig, pred, rep, un = synth_policy(fr, policy)
        else:
            p = os.path.join(mask_dir, "frame_%06d.npz" % f)
            if not os.path.exists(p):
                missing.append(f)
                continue
            with np.load(p, allow_pickle=False) as m:
                keep, elig, pred, rep, un = align(fr, m, f)
        acc["frames"] += 1
        acc["repaired"] += rep
        acc["unaligned"] += un
        sem = fr["sem"]
        mv = np.isin(sem, MOVING_IDS)
        ign = np.isin(sem, IGNORE_IDS)
        st = ~mv & ~ign
        vp = np.isin(sem, STATIC_VP_IDS)
        base = elig if args.denominator == "eligible" else np.ones(len(sem), bool)
        dropped = base & ~keep
        acc["total"] += int(len(sem))
        acc["elig"] += int(elig.sum())
        acc["kept"] += int(keep.sum())
        acc["ign"] += int(ign.sum())
        acc["mv"] += int((mv & base).sum())
        acc["mv_drop"] += int((mv & dropped).sum())
        acc["st"] += int((st & base).sum())
        acc["st_drop"] += int((st & dropped).sum())
        acc["vp"] += int((vp & base).sum())
        acc["vp_drop"] += int((vp & dropped).sum())
        for c in np.unique(sem[mv & base]):
            mv_class_tot[int(c)] = mv_class_tot.get(int(c), 0) + int(((sem == c) & base).sum())
            mv_class_drop[int(c)] = mv_class_drop.get(int(c), 0) + int(((sem == c) & dropped).sum())
        for c in np.unique(sem[st & base]):
            per_class_tot[int(c)] = per_class_tot.get(int(c), 0) + int(((sem == c) & base).sum())
            per_class_drop[int(c)] = per_class_drop.get(int(c), 0) + int(((sem == c) & dropped).sum())
        if (pred >= 0).any():
            acc["has_pred"] += 1
            nd = np.isin(pred, NAIVE_NUSC) & base
            acc["nv_mv_drop"] += int((mv & nd).sum())
            acc["nv_st_drop"] += int((st & nd).sum())
            acc["nv_vp_drop"] += int((vp & nd).sum())
        # ---- map-level ribbon, GT-moving points of each instance in the window
        if w0 <= f < w1 and mv.any():
            sel = np.flatnonzero(mv)
            pw, ok = F.world(fr, sel)
            sel, pw = sel[ok], pw[ok]
            k = keep[sel]
            for i in np.unique(fr["inst"][sel]):
                s = fr["inst"][sel] == i
                ribbon_all.setdefault(int(i), []).append(pw[s])
                if k[s].any():
                    ribbon_pts.setdefault(int(i), []).append(pw[s & k])
        if mv.any():
            sel = np.flatnonzero(mv & keep)
            if sel.size:
                pw, ok = F.world(fr, sel)
                if ok.any():
                    mv_keys_kept.append(np.unique(S.voxel_key(pw[ok], args.voxel)))

    r = dict(frames_scored=acc["frames"], frames_missing=len(missing),
             points_total=acc["total"], points_eligible=acc["elig"],
             points_kept=acc["kept"], points_ignored_gt=acc["ign"],
             alignment_repaired=acc["repaired"], alignment_failed=acc["unaligned"],
             denominator=args.denominator)
    r["moving_points"] = acc["mv"]
    r["static_points"] = acc["st"]
    r["static_vp_points"] = acc["vp"]
    r["dynamic_recall"] = 100.0 * acc["mv_drop"] / max(1, acc["mv"])
    r["static_false_kill"] = 100.0 * acc["st_drop"] / max(1, acc["st"])
    r["vp_static_false_kill"] = 100.0 * acc["vp_drop"] / max(1, acc["vp"])
    r["moving_dropped"] = acc["mv_drop"]
    r["static_dropped"] = acc["st_drop"]
    r["damage_ratio"] = (acc["st_drop"] / acc["mv_drop"]) if acc["mv_drop"] else None
    r["per_moving_class_recall_pct"] = {
        SK_NAMES.get(c, str(c)): [round(100.0 * mv_class_drop[c] / max(1, mv_class_tot[c]), 3),
                                  mv_class_tot[c]]
        for c in sorted(mv_class_tot) if mv_class_tot[c] > 0}
    r["per_class_false_kill_pct"] = {
        SK_NAMES.get(c, str(c)): round(100.0 * per_class_drop[c] / max(1, per_class_tot[c]), 3)
        for c in sorted(per_class_tot) if per_class_tot[c] > 0}
    if acc["has_pred"]:
        r["naive_arm_on_same_frames"] = dict(
            frames=acc["has_pred"],
            dynamic_recall=100.0 * acc["nv_mv_drop"] / max(1, acc["mv"]),
            static_false_kill=100.0 * acc["nv_st_drop"] / max(1, acc["st"]),
            vp_static_false_kill=100.0 * acc["nv_vp_drop"] / max(1, acc["vp"]))
    if mv_keys_kept:
        uk = np.unique(np.concatenate(mv_keys_kept))
        r["moving_voxels_still_in_map"] = int(len(uk))
    else:
        r["moving_voxels_still_in_map"] = 0

    # ---- ribbon
    if ribbon_all:
        inst = args.ribbon_instance
        if inst < 0:
            inst = max(ribbon_all, key=lambda i: sum(len(x) for x in ribbon_all[i]))
        allp = np.concatenate(ribbon_all[inst])
        kept = np.concatenate(ribbon_pts[inst]) if inst in ribbon_pts else np.zeros((0, 3))
        r["ribbon"] = dict(
            window=[w0, w1], instance=int(inst),
            points_gt=int(len(allp)), points_retained=int(len(kept)),
            retained_frac=float(len(kept) / max(1, len(allp))),
            ribbon_m=pca_extent(kept)[0], baseline_ribbon_m=pca_extent(allp)[0],
            voxels_retained=int(len(np.unique(S.voxel_key(kept, args.voxel)))) if len(kept) else 0,
            voxels_baseline=int(len(np.unique(S.voxel_key(allp, args.voxel)))),
            voxel_reduction_pct=float(100.0 * (1.0 - (len(np.unique(S.voxel_key(kept, args.voxel))) if len(kept) else 0)
                                               / max(1, len(np.unique(S.voxel_key(allp, args.voxel)))))),
            other_instances={str(i): round(pca_extent(np.concatenate(v))[0], 2)
                             for i, v in ribbon_all.items() if i != inst})
    if missing:
        r["missing_frames_head"] = missing[:10]
    return r


def fmt(r, tag):
    print("=== %s ===" % tag)
    print("  frames %d (missing %d)  points %d  eligible %d  kept %d  denom=%s"
          % (r["frames_scored"], r["frames_missing"], r["points_total"],
             r["points_eligible"], r["points_kept"], r["denominator"]))
    if r["alignment_repaired"] or r["alignment_failed"]:
        print("  alignment: repaired %d, FAILED %d"
              % (r["alignment_repaired"], r["alignment_failed"]))
    print("  dynamic recall        %7.3f %%   (%d / %d moving pts kept out)"
          % (r["dynamic_recall"], r["moving_dropped"], r["moving_points"]))
    print("  static false-kill     %7.3f %%   (%d / %d static pts lost)"
          % (r["static_false_kill"], r["static_dropped"], r["static_points"]))
    print("  vehicle/person false-kill %3.3f %%   (of %d parked-vehicle/person pts)"
          % (r["vp_static_false_kill"], r["static_vp_points"]))
    if r["damage_ratio"] is not None:
        print("  damage ratio          %7.2f static pts destroyed per moving pt removed"
              % r["damage_ratio"])
    if "ribbon" in r:
        b = r["ribbon"]
        print("  ribbon (frames %d-%d, GT instance %d): %.2f m  (baseline %.2f m), "
              "%d/%d moving pts retained, %d/%d voxels (-%.1f %%)"
              % (b["window"][0], b["window"][1] - 1, b["instance"], b["ribbon_m"],
                 b["baseline_ribbon_m"], b["points_retained"], b["points_gt"],
                 b["voxels_retained"], b["voxels_baseline"], b["voxel_reduction_pct"]))
    if r.get("per_moving_class_recall_pct"):
        print("  dynamic recall PER MOVING CLASS (pooled recall hides the worst one):")
        for k, v in r["per_moving_class_recall_pct"].items():
            print("      %-22s %7.3f %%   (%d pts)" % (k, v[0], v[1]))
    print("  moving voxels still in the map: %d" % r["moving_voxels_still_in_map"])
    if "naive_arm_on_same_frames" in r:
        n = r["naive_arm_on_same_frames"]
        print("  [naive class-delete arm on the same frames] recall %.3f %%  "
              "false-kill %.3f %%  vp %.3f %%"
              % (n["dynamic_recall"], n["static_false_kill"], n["vp_static_false_kill"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask-dir", default=None, help="directory of frame_%%06d.npz")
    ap.add_argument("--policy", default=None,
                    choices=["keep-all", "oracle-dynamic", "oracle-naive",
                             "oracle-naive-wide"],
                    help="score a GT-derived reference policy instead of a run")
    ap.add_argument("--repeat-dirs", nargs="*", default=None,
                    help="several mask dirs of the SAME configuration: report spread (K4)")
    ap.add_argument("--compare-dir", default=None, help="baseline run to diff against")
    ap.add_argument("--traj", default="/data/livo_sem/out/kitti_seq07_fastlivo2_tum.txt")
    ap.add_argument("--f0", type=int, default=0)
    ap.add_argument("--f1", type=int, default=1101)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--window", default="755:788", help="ribbon window f0:f1")
    ap.add_argument("--ribbon-instance", type=int, default=-1,
                    help="GT instance id for the ribbon; -1 = the busiest in the window")
    ap.add_argument("--voxel", type=float, default=0.20)
    ap.add_argument("--denominator", default="eligible", choices=["eligible", "all"],
                    help="eligible = charge only what the v0.3 decision removed")
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args()
    if not a.mask_dir and not a.policy and not a.repeat_dirs:
        ap.error("give --mask-dir, --policy or --repeat-dirs")
    F = Frames(a.traj)

    out = {}
    if a.policy:
        out["main"] = score(a, None, a.policy, F)
        fmt(out["main"], "policy " + a.policy)
    if a.mask_dir:
        out["main"] = score(a, a.mask_dir, None, F)
        fmt(out["main"], a.mask_dir)
    if a.repeat_dirs:
        reps = [score(a, d, None, F) for d in a.repeat_dirs]
        out["repeats"] = reps
        print("\n=== repeat spread over %d runs (K4: the model is non-deterministic) ==="
              % len(reps))
        for k in ("dynamic_recall", "static_false_kill", "vp_static_false_kill"):
            v = np.array([x[k] for x in reps])
            print("  %-22s mean %7.3f  min %7.3f  max %7.3f  spread %.3f"
                  % (k, v.mean(), v.min(), v.max(), v.max() - v.min()))
        if "ribbon" in reps[0]:
            v = np.array([x["ribbon"]["ribbon_m"] for x in reps])
            print("  %-22s mean %7.3f  min %7.3f  max %7.3f  spread %.3f"
                  % ("ribbon_m", v.mean(), v.min(), v.max(), v.max() - v.min()))
    if a.compare_dir:
        base = score(a, a.compare_dir, None, F)
        out["baseline"] = base
        cur = out.get("main") or out["repeats"][0]
        print("\n=== delta vs %s ===" % a.compare_dir)
        sp = None
        if a.repeat_dirs:
            sp = {k: float(np.ptp([x[k] for x in out["repeats"]]))
                  for k in ("dynamic_recall", "static_false_kill", "vp_static_false_kill")}
        for k in ("dynamic_recall", "static_false_kill", "vp_static_false_kill"):
            s = "  (repeat spread %.3f)" % sp[k] if sp else ""
            print("  %-22s %7.3f -> %7.3f%s" % (k, base[k], cur[k], s))
    if a.json_out:
        json.dump(out, open(a.json_out, "w"), indent=2)
        print("\n-> %s" % a.json_out)


if __name__ == "__main__":
    main()
