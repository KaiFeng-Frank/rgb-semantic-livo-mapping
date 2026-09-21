#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
report_2d_vs_3d.py -- turn a score_2d_vs_3d.py results JSON into README-ready
markdown.  Every number carries its measurement conditions; nothing is rounded
away and nothing that was measured is left out.

    python report_2d_vs_3d.py results.json > TABLE.md
    python report_2d_vs_3d.py results.json --mapcov mapcov.json > TABLE.md
"""
import sys
import json
import argparse

ARM_LABEL = {
    "2d": "**B. 2D** Cityscapes -> projected",
    "2d_naive_projection": "B'. 2D, naive projection (no occlusion reasoning)",
    "3d": "**A. 3D** PTv3 on the point cloud",
    "hybrid": "**C. Hybrid** 2D in frustum, 3D outside",
}
ARM_ORDER = ["3d", "2d", "2d_naive_projection", "hybrid"]
RANGE_TAGS = ["0_10", "10_20", "20_30", "30_50", "50_inf"]
RANGE_LABEL = {"0_10": "0-10 m", "10_20": "10-20 m", "20_30": "20-30 m",
               "30_50": "30-50 m", "50_inf": "50+ m"}


def g(agg, arm, sub, met):
    try:
        v = agg[arm][sub][met]
    except KeyError:
        return None
    return v


def fmt(v, nd=2, pct=True):
    if v is None:
        return "-"
    if isinstance(v, dict):
        m, h = v.get("mean"), v.get("half_range", 0.0)
        if m is None or m != m:
            return "-"
        s = ("%." + str(nd) + "f") % m
        if h and h > 5e-4:
            s += " ±%.2f" % h
        return s
    if isinstance(v, (int, float)):
        if v != v:
            return "-"
        return ("%." + str(nd) + "f") % v
    return str(v)


def fmt_int(v):
    if v is None:
        return "-"
    if isinstance(v, dict):
        v = v.get("mean")
    return "{:,}".format(int(round(v)))


def table(rows, head):
    out = ["| " + " | ".join(head) + " |",
           "|" + "|".join(["---"] * len(head)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--mapcov", default=None)
    ap.add_argument("--miou-subset", default="nine",
                    choices=["all", "nine", "frequent"],
                    help="which class subset the headline mIoU column uses")
    a = ap.parse_args()

    R = json.load(open(a.results))
    P = R["protocol"]
    agg = R["agg"]
    arms = [x for x in ARM_ORDER if x in agg]
    sub = a.miou_subset
    O = []
    w = O.append

    # ----------------------------------------------------------------- #
    w("# 2D-projection vs 3D point-cloud semantics on SemanticKITTI seq 07")
    w("")
    w("> *Why segment the point cloud when you have a calibrated camera and a mature")
    w("> 2D segmentation stack?*  Answered with measurements, not arguments.")
    w("")
    w("**The 2D arm was given every advantage.**  It runs a Cityscapes-pretrained")
    w("network, not an ADE20K one: KITTI is German urban street scenes from a")
    w("forward-facing vehicle camera, i.e. the 2D arm's *native* domain, while the 3D")
    w("arm carries nuScenes weights and therefore the *larger* domain gap.  Every")
    w("other contestable choice was also resolved in the 2D arm's favour; they are")
    w("listed in full at the bottom.")
    w("")

    # ---- conditions -------------------------------------------------- #
    w("## Measurement conditions")
    w("")
    cond = [
        ["Sequence", P["sequence"]],
        ["Frames", "%d (`%s`), first=%s last=%s" %
         (P["n_frames"], P["frames_spec"],
          P.get("frames_first_last", ["?", "?"])[0],
          P.get("frames_first_last", ["?", "?"])[-1])],
        ["Repeats", P["repeats"]],
        ["Label space", "%s: %s" % (P["label_space"]["name"],
                                    ", ".join(P["label_space"]["classes"]))],
        ["GT", P["label_space"]["gt"]],
        ["3D arm label map", P["label_space"]["3d"]],
        ["2D arm label map", P["label_space"]["2d"]],
        ["Ignore rule", P["ignore_rule"]],
        ["Camera model", "rectified image_02, %dx%d, fx=fy=%.4f, cx=%.4f, cy=%.4f, %s"
         % (P["frustum"]["image_wh"][0], P["frustum"]["image_wh"][1],
            P["frustum"]["fx"], P["frustum"]["cx"], P["frustum"]["cy"],
            P["frustum"]["distortion"])],
        ["Frustum test", P["frustum"]["rule"] + " (min_depth %.2f m)"
         % P["frustum"]["min_depth_m"]],
        ["2D point-labelling rule",
         "sampling=%s, occlusion=%s (win %d px, tol max(%.2f m, %.0f %%)), sky=%s, rider->%s"
         % (P["2d_label_rule"]["sampling"], P["2d_label_rule"]["occlusion"],
            P["2d_label_rule"]["occ_win_px"], P["2d_label_rule"]["occ_tol_abs_m"],
            100 * P["2d_label_rule"]["occ_tol_rel"], P["2d_label_rule"]["sky_policy"],
            P["2d_label_rule"]["rider_maps_to"])],
        ["Range axis", P["range_axis"]],
        ["Instrument noise", P["instrument_noise"]],
        ["Scorer sha1", "`%s`" % P["code_sha1"][:12]],
    ]
    w(table(cond, ["Condition", "Value"]))
    w("")
    w("`±` is the **half-range over repeats**, not a standard deviation: with a small")
    w("number of repeats the observed spread is the honest quantity.  A gap between")
    w("arms should only be called real when it exceeds roughly 3x this figure.")
    w("")

    cs = None
    for rep in R.get("repeats", []):
        for k in rep:
            if isinstance(rep[k], dict) and "_class_subsets" in rep[k]:
                cs = rep[k]["_class_subsets"]
                break
        if cs:
            break
    if cs:
        w("### Class subsets used for mIoU")
        w("")
        w(table([[k, str(len(v)), ", ".join(v)] for k, v in cs.items()],
                ["Subset", "N classes", "Members"]))
        w("")

    # ---- 1. coverage + headline -------------------------------------- #
    w("## 1. Coverage -- how much of the scan each arm can label at all")
    w("")
    w("This is a first-class result, not a footnote.  The camera sees a %.1f deg x"
      % 81.9)
    w("%.1f deg cone; the HDL-64E sweeps 360 deg.  No amount of 2D accuracy recovers a"
      % 29.3)
    w("point the camera never saw.")
    w("")
    rows = []
    for arm in arms:
        rows.append([
            ARM_LABEL.get(arm, arm),
            fmt({"mean": 100 * g(agg, arm, "global", "coverage")["mean"],
                 "half_range": 100 * g(agg, arm, "global", "coverage")["half_range"]}),
            fmt({"mean": 100 * g(agg, arm, "frustum", "coverage")["mean"],
                 "half_range": 100 * g(agg, arm, "frustum", "coverage")["half_range"]}),
            fmt({"mean": 100 * g(agg, arm, "outside", "coverage")["mean"],
                 "half_range": 100 * g(agg, arm, "outside", "coverage")["half_range"]}),
            fmt_int(g(agg, arm, "global", "n_labelled")),
        ])
    w(table(rows, ["Arm", "Coverage, all points (%)", "Coverage, in-frustum (%)",
                   "Coverage, outside frustum (%)", "Points labelled"]))
    w("")
    w("Evaluated points per frame set: **%s** (points whose GT class is `ignore` are"
      % fmt_int(g(agg, arms[0], "global", "n_eval")))
    w("excluded from every denominator, for every arm).  In-frustum: **%s**."
      % fmt_int(g(agg, arms[0], "frustum", "n_eval")))
    w("")

    # ---- 2. in-frustum ------------------------------------------------ #
    w("## 2. In-frustum accuracy -- the 2D arm's home turf")
    w("")
    w("Restricted to the points that fall inside the camera frustum.  If the 2D arm is")
    w("going to win anywhere, it wins here, and it is reported first for that reason.")
    w("")
    rows = []
    for arm in arms:
        rows.append([
            ARM_LABEL.get(arm, arm),
            fmt(g(agg, arm, "frustum", "acc_abstain_excluded")),
            fmt(g(agg, arm, "frustum", "acc_abstain_wrong")),
            fmt(g(agg, arm, "frustum", "miou_%s_abstain_excluded" % sub)),
            fmt(g(agg, arm, "frustum", "miou_nine_abstain_excluded")),
            fmt(g(agg, arm, "frustum", "miou_all_abstain_excluded")),
        ])
    w(table(rows, ["Arm", "Point acc, abstain excluded (%)",
                   "Point acc, abstain wrong (%)",
                   "mIoU `%s` (%%)" % sub, "mIoU `nine` (%)", "mIoU `all` (%)"]))
    w("")
    w("Class subsets: `all` = every coarse class present in GT; `nine` = the subset the")
    w("existing 3D number is quoted over; `expr` / `expr8` drop `other_vehicle`, which")
    w("Cityscapes-19 structurally cannot name (it has only `train`, while SemanticKITTI")
    w("id 20 is vans and caravans) -- dropping it is a concession to the 2D arm.")
    w("")

    # ---- 3. global ---------------------------------------------------- #
    w("## 3. Global accuracy -- over every point in the scan")
    w("")
    w("The 2D arm is scored under **both** conventions, because the choice is")
    w("contestable and a reader will suspect whichever one is picked alone.")
    w("")
    w("* **abstain excluded** -- judge the arm where it speaks.  Fair to the 2D arm.")
    w("* **abstain wrong** -- a point with no label is a point the map cannot answer")
    w("  for.  This is what a mapping system actually needs.")
    w("")
    w("**The gap between the two columns is the finding.**")
    w("")
    rows = []
    for arm in arms:
        ae = g(agg, arm, "global", "acc_abstain_excluded")
        aw = g(agg, arm, "global", "acc_abstain_wrong")
        gap = (ae["mean"] - aw["mean"]) if (ae and aw) else None
        rows.append([
            ARM_LABEL.get(arm, arm),
            fmt(ae), fmt(aw), fmt(gap),
            fmt(g(agg, arm, "global", "miou_%s_abstain_excluded" % sub)),
            fmt(g(agg, arm, "global", "miou_%s_abstain_wrong" % sub)),
            fmt(g(agg, arm, "global", "miou_nine_abstain_excluded")),
            fmt(g(agg, arm, "global", "miou_nine_abstain_wrong")),
        ])
    w(table(rows, ["Arm", "Acc, abstain excluded (%)", "Acc, abstain wrong (%)",
                   "Gap (pp)",
                   "mIoU `%s`, abstain excl. (%%)" % sub,
                   "mIoU `%s`, abstain wrong (%%)" % sub,
                   "mIoU `nine`, abstain excl. (%)",
                   "mIoU `nine`, abstain wrong (%)"]))
    w("")

    # ---- 4. boundary -------------------------------------------------- #
    w("## 4. Boundary leakage")
    w("")
    w("The structural weakness of 2D->3D projection: at a depth discontinuity a")
    w("foreground pixel's label lands on a background point.  Two boundary sets are")
    w("used, and **neither is defined with reference to any arm's predictions**, so")
    w("neither can be gamed:")
    w("")
    w("* `sem_boundary` (global, **GT only**) -- %s" % P["boundary_sets"]["sem_boundary"])
    w("* `depth_edge` (in-frustum, **geometry only, no GT**) -- %s"
      % P["boundary_sets"]["depth_edge"])
    w("")
    rows = []
    for arm in arms:
        for name, (bs, ins) in [("semantic", ("sem_boundary", "sem_interior")),
                                ("depth", ("depth_edge", "depth_interior"))]:
            b = g(agg, arm, bs, "acc_abstain_excluded")
            i = g(agg, arm, ins, "acc_abstain_excluded")
            drop = (i["mean"] - b["mean"]) if (b and i) else None
            rows.append([
                ARM_LABEL.get(arm, arm), name,
                fmt(b), fmt(i), fmt(drop),
                fmt(g(agg, arm, bs, "acc_abstain_wrong")),
                fmt_int(g(agg, arm, bs, "n_eval")),
            ])
    w(table(rows, ["Arm", "Boundary set", "Acc on boundary, abstain excl. (%)",
                   "Acc on interior, abstain excl. (%)", "Interior - boundary (pp)",
                   "Acc on boundary, abstain wrong (%)", "Boundary points"]))
    w("")

    # ---- 5. range ----------------------------------------------------- #
    w("## 5. Range stratification")
    w("")
    w("Bins on LiDAR range `||xyz||`, not camera depth: range is the axis both sensors")
    w("degrade along, and it is defined for points the camera never saw.  A camera")
    w("loses angular resolution with distance; a LiDAR loses point density.")
    w("")
    for scope, title in [("range", "All points"), ("frustum_range", "In-frustum only")]:
        w("**%s**" % title)
        w("")
        rows = []
        for arm in arms:
            r = [ARM_LABEL.get(arm, arm)]
            for t in RANGE_TAGS:
                r.append(fmt(g(agg, arm, "%s_%s" % (scope, t),
                               "acc_abstain_excluded")))
            for t in RANGE_TAGS:
                r.append(fmt({"mean": 100 * g(agg, arm, "%s_%s" % (scope, t),
                                              "coverage")["mean"],
                              "half_range": 100 * g(agg, arm, "%s_%s" % (scope, t),
                                                    "coverage")["half_range"]}))
            rows.append(r)
        head = (["Arm"] + ["Acc %s (%%)" % RANGE_LABEL[t] for t in RANGE_TAGS] +
                ["Cov %s (%%)" % RANGE_LABEL[t] for t in RANGE_TAGS])
        w(table(rows, head))
        w("")
        n = [fmt_int(g(agg, arms[0], "%s_%s" % (scope, t), "n_eval"))
             for t in RANGE_TAGS]
        w("Evaluated points per bin: " +
          ", ".join("%s %s" % (RANGE_LABEL[t], n[i]) for i, t in enumerate(RANGE_TAGS)))
        w("")

    # ---- 6. per class -------------------------------------------------- #
    w("## 6. Per-class IoU (global, abstain excluded)")
    w("")
    classes = []
    for arm in arms:
        d = g(agg, arm, "global", "per_class_iou_abstain_excluded")
        if isinstance(d, dict):
            for k in d:
                if k not in classes:
                    classes.append(k)
    rows = []
    for c in classes:
        r = [c]
        for arm in arms:
            d = g(agg, arm, "global", "per_class_iou_abstain_excluded") or {}
            r.append(fmt(d.get(c)))
        rows.append(r)
    w(table(rows, ["Class"] + [ARM_LABEL.get(x, x) for x in arms]))
    w("")
    gtc = g(agg, arms[0], "global", "gt_count_per_class")
    if isinstance(gtc, dict):
        w("GT points per class: " +
          ", ".join("%s %s" % (k, "{:,}".format(int(v))) for k, v in gtc.items()))
        w("")

    # ---- 7. latency ---------------------------------------------------- #
    w("## 7. Latency")
    w("")
    T = R.get("timing") or {}
    if not T:
        w("*No timing file supplied.  Schema the measurement agent must fill:*")
        w("")
        w("```json")
        w(json.dumps(R.get("timing_schema", {}), indent=2))
        w("```")
    else:
        rows = []
        for name, t in T.items():
            st = t.get("stages_ms", {})
            stages = [k for k in st if not k.startswith("_")]
            rows.append([
                t.get("arm", name),
                t.get("hardware", "-"), t.get("precision", "-"),
                "x".join(str(x) for x in t.get("input_hw", [])) or "-",
                str(t.get("input_scale", "-")), t.get("tta", "-"),
                str(t.get("frames", "-")),
                "; ".join("%s %.1f" % (k, st[k].get("mean", float("nan")))
                          for k in stages),
                fmt(st.get("total", {}).get("mean"), 1),
                fmt(st.get("total", {}).get("p95"), 1),
            ])
        w(table(rows, ["Arm", "Hardware", "Precision", "Input HxW", "Scale", "TTA",
                       "Frames", "Stage means (ms)", "Total mean (ms)",
                       "Total p95 (ms)"]))
        w("")
        t2 = next((t for t in T.values() if t.get("arm") == "2d"), None)
        t3 = next((t for t in T.values() if t.get("arm") == "3d"), None)
        if t2 and t3:
            a2t = t2["stages_ms"]["total"]
            a3t = t3["stages_ms"]["total"]
            w("")
            w("**Hybrid cost.**  The hybrid arm needs *both* networks, so its latency")
            w("is a bracket, not a number: `serial` is one GPU running them one after")
            w("the other, `parallel` is two streams and is a lower bound that assumes")
            w("no contention.  The merge itself is a numpy `where` over the scan.")
            w("")
            w(table([["serial (one GPU)", fmt(a2t["mean"] + a3t["mean"], 1),
                      fmt(a2t["p95"] + a3t["p95"], 1)],
                     ["parallel (two streams, lower bound)",
                      fmt(max(a2t["mean"], a3t["mean"]), 1),
                      fmt(max(a2t["p95"], a3t["p95"]), 1)],
                     ["3D arm alone, for reference", fmt(a3t["mean"], 1),
                      fmt(a3t["p95"], 1)]],
                    ["Hybrid schedule", "Total mean (ms)", "Total p95 (ms)"]))
        else:
            w("The hybrid arm needs **both** networks, so its cost is bracketed:")
            w("`serial` (one GPU, stages summed) and `parallel` (two streams, maxed).")
            w("Supply timing JSONs for both the 2D and the 3D arm and this table fills")
            w("itself in.")
    w("")

    # ---- 8. diagnostics ------------------------------------------------ #
    dg = None
    for rep in R.get("repeats", []):
        if "2d" in rep and "_diag" in rep["2d"]:
            dg = rep["2d"]["_diag"]
            break
    if dg:
        w("## 8. 2D-arm diagnostics")
        w("")
        w(table([["in-frustum points", "{:,}".format(dg["in_frustum_points"])],
                 ["points abstained as occluded",
                  "{:,} ({:.2f} % of in-frustum)".format(
                      dg["occluded_points"],
                      100.0 * dg["occluded_points"] / max(1, dg["in_frustum_points"]))],
                 ["points landing on a `sky` pixel",
                  "{:,} ({:.2f} % of in-frustum)".format(
                      dg["sky_hits"],
                      100.0 * dg["sky_hits"] / max(1, dg["in_frustum_points"]))],
                 ["sampling rule used", dg["sampling_used"]]],
                ["Diagnostic", "Value"]))
        w("")

    # ---- 8b. occlusion-window sensitivity -------------------------------- #
    sweep = sorted([k for k in agg if k.startswith("2d_occwin")],
                   key=lambda x: int(x.replace("2d_occwin", "")))
    if sweep:
        main_win = P["2d_label_rule"]["occ_win_px"]
        w("## 8b. Sensitivity to the occlusion window")
        w("")
        w("The z-buffer half-window is the one protocol parameter with a real")
        w("accuracy/coverage trade, so it is reported as a row rather than chosen")
        w("silently.  It is set to **%d px** by default because that is the HDL-64E's"
          % main_win)
        w("vertical sampling pitch expressed in camera pixels (one pixel is 0.081 deg,")
        w("the LiDAR's vertical spacing is ~0.4 deg), not because it scored best: an")
        w("occluding surface is dense in the image but sparse in the cloud, so a")
        w("z-buffer that only looks at the exact pixel misses most real occlusions.")
        w("A larger window abstains more, which raises accuracy under *abstain")
        w("excluded* and lowers coverage.  Both move together; the reader can pick.")
        w("")
        rows = []
        order = sorted(sweep + ["2d"],
                       key=lambda x: main_win if x == "2d"
                       else int(x.replace("2d_occwin", "")))
        for arm in order:
            wv = main_win if arm == "2d" else int(arm.replace("2d_occwin", ""))
            rows.append([
                "%d px%s" % (wv, " (default)" if arm == "2d" else ""),
                fmt({"mean": 100 * g(agg, arm, "frustum", "coverage")["mean"],
                     "half_range": 100 * g(agg, arm, "frustum", "coverage")["half_range"]}),
                fmt(g(agg, arm, "frustum", "acc_abstain_excluded")),
                fmt(g(agg, arm, "frustum", "acc_abstain_wrong")),
                fmt(g(agg, arm, "global", "acc_abstain_excluded")),
                fmt(g(agg, arm, "global", "acc_abstain_wrong")),
                fmt(g(agg, arm, "depth_edge", "acc_abstain_excluded")),
            ])
        if "2d_naive_projection" in agg:
            arm = "2d_naive_projection"
            rows.append([
                "none (naive)",
                fmt({"mean": 100 * g(agg, arm, "frustum", "coverage")["mean"],
                     "half_range": 0.0}),
                fmt(g(agg, arm, "frustum", "acc_abstain_excluded")),
                fmt(g(agg, arm, "frustum", "acc_abstain_wrong")),
                fmt(g(agg, arm, "global", "acc_abstain_excluded")),
                fmt(g(agg, arm, "global", "acc_abstain_wrong")),
                fmt(g(agg, arm, "depth_edge", "acc_abstain_excluded")),
            ])
        w(table(rows, ["z-buffer half-window", "In-frustum coverage (%)",
                       "In-frustum acc, abstain excl. (%)",
                       "In-frustum acc, abstain wrong (%)",
                       "Global acc, abstain excl. (%)",
                       "Global acc, abstain wrong (%)",
                       "Acc on depth_edge, abstain excl. (%)"]))
        w("")

    # ---- 9. map coverage ------------------------------------------------ #
    if a.mapcov:
        M = json.load(open(a.mapcov))
        w("## 9. Accumulated map coverage -- the obvious rebuttal, answered")
        w("")
        w("*\"Per-scan coverage is 16 %, but the camera sweeps as the vehicle drives, so")
        w("the accumulated map does get camera coverage everywhere.\"*  Measured over the")
        w("whole trajectory (KITTI odometry GT poses, %.2f m voxels, %d frames,"
          % (M["voxel_m"], M["frames"]))
        w("%s occupied voxels):" % "{:,}".format(M["n_voxels"]))
        w("")
        w(table([["Occupied map voxels ever seen by the camera",
                  "%.2f %%" % (100 * M["voxels_ever_camera_covered"])],
                 ["Mean fraction of a voxel's observations that were camera-covered",
                  "%.2f %%" % (100 * M["mean_fraction_of_observations_camera_covered"])]],
                ["Map-level coverage", "Value"]))
        w("")

    # ---- concessions ---------------------------------------------------- #
    w("## Every choice made in favour of the 2D arm")
    w("")
    for c in P["pro_2d_concessions"]:
        w("* %s" % c)
    w("")
    w("If the 3D arm still wins after all of these, the result is not an artefact of")
    w("the setup.  Where a choice could not be resolved in either arm's favour without")
    w("hiding something, both variants are reported instead (the two abstention")
    w("conventions, and the 2D arm with and without occlusion reasoning).")
    w("")

    print("\n".join(O))


if __name__ == "__main__":
    main()
