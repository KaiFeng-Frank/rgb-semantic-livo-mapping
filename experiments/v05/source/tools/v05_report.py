#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/v05_report.py -- assemble the v0.5 re-qualification tables from the measured files.
Every number printed here is read from a file a command produced; nothing is typed in.

  M1  opt/out/stats_v05_*.json  (node stats)  +  logs/res_v05_*.csv  (nvidia-smi / ps samples)
  M2  out/v05/map_<model>_r<k>.json (replay_v05, offline caches)  +  out/v05/live_<model>.json
  S   out/v05/extract_<tag>.json
Writes out/v05/REPORT.md and out/v05/summary.json.
"""
import json, os, glob, sys
import numpy as np

R = "/data/livo_sem"
MODELS = [("ZS", "zero-shot (released nuScenes PTv3-m1)", "ZS"),
          ("B0", "B0 -- pseudo-labels from the 2D teacher, zero human 3D labels (DEPLOYABLE)", "B0"),
          ("RP", "Rprime_noKL -- randomly-scattered GT supervision (UPPER BOUND ONLY, not deployable)", "Rprime_noKL")]
SUBS = ("outside", "frustum", "global")
OFFLINE_REF = {  # frozen-harness means quoted in the task / verdict files (for cross-checking only)
    "ZS": dict(outside=65.0125, frustum=65.3469, globalx=65.0648),
    "B0": dict(outside=72.9754, frustum=70.6079, globalx=72.6075),
    "RP": dict(outside=84.8993, frustum=86.9664, globalx=85.3676)}
V02 = dict(sat=dict(period_mean=60.98, period_p95=65.99), stage=dict(stage_a=2.71, gate=2.79, deskew=5.07,
           proj=4.14, map=22.19, pub=4.12, ptv3=56.43, ptv3_p95=58.58), rt=dict(processed="1092/1092", vram=1459, rss=1.171, voxels=2700844))


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def res_peaks(tag):
    p = R + "/logs/res_%s.csv" % tag
    if not os.path.exists(p):
        return None
    rows = [l.strip().split(",") for l in open(p).readlines()[1:] if l.strip()]
    v = np.array([[float(x) for x in r] for r in rows if len(r) == 5])
    return dict(vram_mib=int(v[:, 1].max()), node_rss_gb=float(v[:, 2].max() / 1e6),
                worker_rss_gb=float(v[:, 3].max() / 1e6), gpu_util_mean=float(v[:, 4].mean()), samples=len(v))


def m1():
    out = {}
    lines = ["## M1 -- performance (re-measured; identical node arguments, only --ptv3-ckpt differs)", "",
             "Saturated runs: bag rate 2.0, RELIABLE, conf-gate 0.5, expect-voxels 4M, three reps each (a, b, c), "
             "interleaved ZS/B0/RP (a,b first, then c). Frame period = node-intrinsic time between consecutive stage-B completions "
             "(in-callback perf_counter), i.e. the throughput ceiling, not bag delivery.", "",
             "| model | rep | period mean / p95 / max (ms) | ceiling Hz | PTv3 worker mean / p95 (ms) | stage A | gate | de-skew | proj | map | pub | stage B | recv / proc / bp-drop |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for key, _, _ in MODELS:
        reps = []
        for rep in ("a", "b", "c"):
            d = load(R + "/opt/out/stats_v05_sat_%s_%s.json" % (key, rep))
            if not d:
                continue
            reps.append(d)
            fp = d["frame_period_ms"]
            lines.append("| %s | %s | %.2f / %.2f / %.2f | %.1f | %.2f / %.2f | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f | %d / %d / %d |" % (
                key, rep, fp["mean"], fp["p95"], fp["max"], 1000.0 / fp["mean"],
                d["ptv3_gpu_ms"]["mean"], d["ptv3_gpu_ms"]["p95"], d["stage_a_ms"]["mean"], d["gate_ms"]["mean"],
                d["deskew_ms"]["mean"], d["proj_ms"]["mean"], d["map_ms"]["mean"], d["scanpub_ms"]["mean"],
                d["stage_b_ms"]["mean"], d["scans_received"], d["scans_processed"], d["scans_dropped_backpressure"]))
            out["sat_%s_%s" % (key, rep)] = dict(period=fp, ptv3=d["ptv3_gpu_ms"], stage_b=d["stage_b_ms"],
                                                 stages=dict(stage_a=d["stage_a_ms"]["mean"], gate=d["gate_ms"]["mean"], deskew=d["deskew_ms"]["mean"],
                                                             proj=d["proj_ms"]["mean"], map=d["map_ms"]["mean"], pub=d["scanpub_ms"]["mean"]),
                                                 recv=d["scans_received"], proc=d["scans_processed"], drop=d["scans_dropped_backpressure"],
                                                 ckpt=d.get("ptv3_ckpt"))
        # step-change check from the node's own 1 Hz ticks: worker ms in the first 15 ticks vs the rest
        import re
        for rep in ("a", "b", "c"):
            lp = R + "/logs/node_v05_sat_%s_%s.log" % (key, rep)
            if not os.path.exists(lp):
                continue
            g = [int(x) for x in re.findall(r"gpu (\d+) ms", open(lp).read())]
            if len(g) > 20:
                lines.append("| %s | %s ticks | worker ms first 15 s %.1f, after %.1f (%d ticks) | | | | | | | | | | |" % (
                    key, rep, np.mean(g[:15]), np.mean(g[15:]), len(g)))
                out.setdefault("sat_%s_%s" % (key, rep), {})["worker_first15_vs_rest"] = [float(np.mean(g[:15])), float(np.mean(g[15:]))]
        if reps:
            pm = [r["frame_period_ms"]["mean"] for r in reps]; pp = [r["frame_period_ms"]["p95"] for r in reps]
            wm = [r["ptv3_gpu_ms"]["mean"] for r in reps]; sb = [r["stage_b_ms"]["mean"] for r in reps]
            lines.append("| **%s** | **mean of %d / median** | **%.2f / %.2f (median %.2f / %.2f)** | **%.1f** | **%.2f (median %.2f)** | | | | | | | **%.2f** | |" % (
                key, len(reps), np.mean(pm), np.mean(pp), np.median(pm), np.median(pp), 1000.0 / np.median(pm), np.mean(wm), np.median(wm), np.mean(sb)))
            out["sat_%s_summary" % key] = dict(period_mean=pm, period_p95=pp, worker_mean=wm, stage_b_mean=sb)
    lines += ["", "v0.2 reference (opt/out/stats_cap.json, zero-shot): period %.2f / %.2f ms; stage breakdown (stats_after3.json) A %.2f / gate %.2f / de-skew %.2f / proj %.2f / map %.2f / pub %.2f; worker %.2f / %.2f." % (
        V02["sat"]["period_mean"], V02["sat"]["period_p95"], V02["stage"]["stage_a"], V02["stage"]["gate"], V02["stage"]["deskew"],
        V02["stage"]["proj"], V02["stage"]["map"], V02["stage"]["pub"], V02["stage"]["ptv3"], V02["stage"]["ptv3_p95"]), "",
        "Full bag at rate 1.0 (the qualification run; the map .npz used for M2 live scoring and M3):", "",
        "| model | recv / processed / bp-drop / no-pose | published est. | map voxels | peak VRAM (MiB) | node peak RSS (GB) | worker peak RSS (GB) | wall (s) | PTv3 mean (ms) | period mean (ms) | checkpoint | drop timing |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for key, _, _ in MODELS:
        d = load(R + "/opt/out/stats_v05_rt_%s.json" % key)
        if not d:
            continue
        pk = res_peaks("v05_rt_%s" % key) or {}
        lines.append("| %s | %d / %d / %d / %d | %d | %d | %s | %.3f | %s | %.1f | %.2f | %.2f | %s |" % (
            key, d["scans_received"], d["scans_processed"], d["scans_dropped_backpressure"], d["scans_no_pose"],
            d["scans_published_est"], d["map_voxels"], pk.get("vram_mib", "?"), d.get("peak_rss_gb", float("nan")),
            ("%.3f" % pk["worker_rss_gb"]) if pk else "?", d["wall_clock_s"], d["ptv3_gpu_ms"]["mean"], d["frame_period_ms"]["mean"],
            os.path.basename(d.get("ptv3_ckpt") or "released")))
        try:
            import re
            ticks = re.findall(r"recv (\d+) \| processed (\d+) \| bp-drop (\d+)", open(R + "/logs/node_v05_rt_%s.log" % key).read())
            first = next((t for t in ticks if int(t[2]) > 0), None)
            lines[-1] += " drops: first tick with drops at recv %s (bp-drop %s), final bp-drop %s over %d ticks; qdepth mean %.4f |" % (
                first[0] if first else "-", first[2] if first else 0, ticks[-1][2], len(ticks), d["qdepth"]["mean"])
        except Exception as e:
            lines[-1] += " (drop timing unavailable: %s)" % e
        out["rt_%s" % key] = dict(recv=d["scans_received"], proc=d["scans_processed"], drop=d["scans_dropped_backpressure"],
                                  no_pose=d["scans_no_pose"], voxels=d["map_voxels"], peaks=pk, node_rss=d.get("peak_rss_gb"),
                                  hist=d["class_histogram"], ckpt=d.get("ptv3_ckpt"))
    lines += ["", "v0.2 reference: %s processed, 0 dropped, peak VRAM %d MiB, node RSS %.3f GB, %d voxels." % (
        V02["rt"]["processed"], V02["rt"]["vram"], V02["rt"]["rss"], V02["rt"]["voxels"])]
    return out, lines


def agg(files, key, sub, field):
    v = [json.load(open(f))[key][sub][field] for f in files]
    return float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, v


def m2():
    out = {}
    lines = ["## M2 -- map-level semantic quality (common-9, SemanticKITTI seq07 GT)", "",
             "Replay = the pipeline's stage B run on the CPU from the SAME cached per-point predictions the offline numbers were "
             "scored from (`opt/replay_v05.py`, extends `opt/replay_v03.py`'s map_accuracy). Per subset: out-of-frustum / "
             "in-frustum / global. mIoU-9 conventions: `offline_*` and `map_*_inserted` use abstain-excluded (no abstention exists there); "
             "`map_all_lookup` and `live` use abstain-WRONG (a point with no voxel is a miss for a map consumer).", ""]
    for key, name, _ in MODELS:
        files = sorted(glob.glob(R + "/out/v05/map_%s_r*.json" % key))
        if not files:
            continue
        d0 = json.load(open(files[0]))
        lines += ["### %s  (%d replay draws: %s)" % (name, len(files), ", ".join(os.path.basename(f)[4:-5] for f in files)), "",
                  "frames %d (no-pose %d), map voxels %s, inserted fraction of evaluated points %.4f, gated-out points found in a voxel %.4f" % (
                      d0["frames"], d0["frames_no_pose"], "/".join(str(json.load(open(f))["map_voxels"]) for f in files),
                      np.mean([json.load(open(f))["points"]["inserted_frac"] for f in files]),
                      np.mean([json.load(open(f))["points"]["gated_out_found_in_map"] / max(1, json.load(open(f))["points"]["gated_out"]) for f in files])), "",
                  "| reading | out mIoU-9 | out acc | in mIoU-9 | in acc | global mIoU-9 | global acc |", "|---|---|---|---|---|---|---|"]
        rows = [("offline_all (== frozen harness)", "offline_all", "abstain_excluded"),
                ("offline_inserted (pose ok, conf>=0.5)", "offline_inserted", "abstain_excluded"),
                ("map, per-point GT, inserted pts", "map_pointgt_inserted", "abstain_excluded"),
                ("map, voxel-majority GT (v0.3 def.)", "map_majority_inserted", "abstain_excluded"),
                ("map, ALL evaluated pts (lookup)", "map_all_lookup", "abstain_wrong")]
        rec = {}
        for label, k, conv in rows:
            cells = []
            for s in SUBS:
                mi, sdi, _ = agg(files, k, s, "miou9_" + conv)
                ac, sda, _ = agg(files, k, s, "acc_" + conv)
                cells += ["%.2f +- %.2f" % (mi, sdi), "%.2f +- %.2f" % (ac, sda)]
                rec.setdefault(k, {})[s] = dict(miou9=mi, miou9_sd=sdi, acc=ac, acc_sd=sda)
            lines.append("| %s | %s |" % (label, " | ".join(cells)))
        lv = load(R + "/out/v05/live_%s.json" % key)
        if lv:
            cells = []
            for s in SUBS:
                cells += ["%.2f" % lv["live_lookup"][s]["miou9_abstain_wrong"], "%.2f" % lv["live_lookup"][s]["acc_abstain_wrong"]]
                rec.setdefault("live_lookup", {})[s] = dict(miou9=lv["live_lookup"][s]["miou9_abstain_wrong"], acc=lv["live_lookup"][s]["acc_abstain_wrong"])
            lines.append("| LIVE map (rate-1.0 ROS run, lookup; one draw) | %s |" % " | ".join(cells))
            lines.append("")
            lines.append("live map: %d voxels, lookup hit fraction %.4f" % (lv["live_meta"]["voxels"], lv["live_meta"]["hit_frac"]))
        g_off = rec["offline_all"]["outside"]["miou9"]; g_map = rec["map_all_lookup"]["outside"]["miou9"]
        g_ins = rec["offline_inserted"]["outside"]["miou9"]; g_fus = rec["map_pointgt_inserted"]["outside"]["miou9"]
        lines += ["", "**Out-of-frustum GAP, map-level minus offline (positive = the map is better than the per-scan prediction):** "
                  "end-to-end %+.2f (all points: %.2f -> %.2f); of which gate selection %+.2f (%.2f -> %.2f), fusion on the inserted points %+.2f (%.2f -> %.2f), "
                  "coverage of gated-out/no-pose points %+.2f (%.2f -> %.2f)." % (
                      g_map - g_off, g_off, g_map, g_ins - g_off, g_off, g_ins, g_fus - g_ins, g_ins, g_fus, g_map - g_fus, g_fus, g_map)]
        if lv:
            lines.append("LIVE map vs offline (out-of-frustum): %+.2f mIoU-9 (%.2f -> %.2f)." % (rec["live_lookup"]["outside"]["miou9"] - g_off, g_off, rec["live_lookup"]["outside"]["miou9"]))
        # per-class out-of-frustum: offline vs map_all
        pc_off = {c: np.mean([json.load(open(f))["offline_all"]["outside"]["per_class_iou_abstain_excluded"][c] for f in files]) for c in d0["offline_all"]["outside"]["per_class_iou_abstain_excluded"]}
        pc_map = {c: np.mean([json.load(open(f))["map_all_lookup"]["outside"]["per_class_iou_abstain_wrong"][c] for f in files]) for c in pc_off}
        lines += ["", "per-class IoU, out-of-frustum, offline -> map (all points): " + "; ".join("%s %.1f -> %.1f" % (c, pc_off[c], pc_map[c]) for c in pc_off), ""]
        worse = [c for c in pc_off if pc_map[c] < pc_off[c]]
        lines += ["classes the map scores LOWER than the per-scan prediction (out-of-frustum): %s" % (
            ", ".join("%s (%.1f -> %.1f)" % (c, pc_off[c], pc_map[c]) for c in worse) if worse else "none"), ""]
        # abstain-excluded variant of the all-points reading (no-pose frames' points dropped instead of counted wrong)
        mi_ex, _, _ = agg(files, "map_all_lookup", "outside", "miou9_abstain_excluded")
        ac_ex, _, _ = agg(files, "map_all_lookup", "outside", "acc_abstain_excluded")
        lines += ["map, ALL evaluated pts, abstain-EXCLUDED convention (out-of-frustum): mIoU-9 %.2f, acc %.2f" % (mi_ex, ac_ex), ""]
        # in-frustum vs out-of-frustum, offline vs map
        lines += ["in-frustum minus out-of-frustum point accuracy: offline %+.2f, map (all points) %+.2f -- every voxel is voted on from many "
                  "viewpoints, so the camera-frustum distinction largely dissolves at map level" % (
                      rec["offline_all"]["frustum"]["acc"] - rec["offline_all"]["outside"]["acc"],
                      rec["map_all_lookup"]["frustum"]["acc"] - rec["map_all_lookup"]["outside"]["acc"]), ""]
        rec["per_class_out"] = dict(offline=pc_off, map=pc_map, worse=worse)
        rec["map_all_lookup_abstain_excluded_out"] = dict(miou9=mi_ex, acc=ac_ex)
        out[key] = rec
    return out, lines


def sec_extract():
    lines = ["## Checkpoint extraction and verification (tools/extract_student.py)", "",
             "| tag | source epoch / seq08 val | tensors (student / anchor / other) | S1 strict 488/488 | S2 tensors equal fp32 / fp16 | S3 tree | S4 cross vs within agreement (all; margin>2) | NEG control (all) | GT acc distil / plain / worker | verdict |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    out = {}
    for tag in ("ZS", "B0", "Rprime_noKL"):
        d = load(R + "/out/v05/extract_%s.json" % tag)
        if not d:
            continue
        ag = d["S4"]["agreement"]; vi = d["S4"]["verdict_inputs"]; gs = d["S4"]["gt_score"]
        lines.append("| %s | %s / %s | %d / %d / %d | %d missing, %d unexpected | %d / %d | %s | %.5f vs %.5f; %.5f vs %.5f | %s | %.2f / %.2f / %.2f | %s |" % (
            tag, d["source"]["epoch"], ("%.4f" % d["source"]["best_metric_value"]) if d["source"]["best_metric_value"] is not None else "-",
            d["source"]["n_student"], d["source"]["n_anchor"], d["source"]["n_other"],
            len(d["S1"]["missing"]), len(d["S1"]["unexpected"]), d["S2"]["n_equal_fp32"], d["S2"]["n_equal_fp16"],
            "equal" if d["S3"]["backbone_repr_equal"] and d["S3"]["head_repr_equal"] else "DIFF",
            vi["all"]["cross"], vi["all"]["within"], vi["margin>2"]["cross"], vi["margin>2"]["within"],
            ("%.5f" % vi["neg"]["neg_all"]) if "neg" in vi else "-",
            gs["distil"]["point_acc"], gs["plain"]["point_acc"], gs["worker"]["point_acc"], "PASS" if d["PASS"] else "FAIL: " + "; ".join(d["fails"])))
        out[tag] = dict(out=d["out"], out_sha256=d["out_sha256"], PASS=d["PASS"], frames=len(d["frames"]))
    return out, lines


def m3():
    lines = ["## M3 -- RViz2 screenshots (DISPLAY=:0, captured from the latched TRANSIENT_LOCAL map after each run)", "",
             "Files: out/v05/rviz_<model>_<view>_<rgb|class>.png with model in {ZS, B0, RP}, view in {top (overview, focal 20,93,-3, "
             "distance 360, pitch 1.50), obl (street view, focal 4.6,21.7,-0.5, distance 55, pitch 0.42), zA, zB (10 m tiles below)}. "
             "Class view = RViz Intensity transformer on the `class` channel, bounds 0..15, rainbow (legend: out/v05/rviz_class_legend_rainbow0-15.png; "
             "car yellow, driveable_surface cyan-blue, sidewalk blue, terrain violet, manmade purple, vegetation magenta, truck cyan).", ""]
    out = {}
    for f, name in (("zoom_ZS_B0", "ZS vs B0"), ("zoom_B0_RP", "B0 vs Rprime_noKL")):
        d = load(R + "/out/v05/%s.json" % f)
        if not d:
            continue
        lines += ["Live-map disagreement %s: %d voxels matched by key, %d differ (%.2f %%). Top 10 m tiles (centre; voxels; differing; "
                  "accuracy of A / B on GT-labelled voxels; dominant changes):" % (name, d["matched"], d["differing_total"], 100.0 * d["differing_total"] / d["matched"])]
        for t in d["tiles"][:4]:
            lines.append("- (%.1f, %.1f, %.1f): %d voxels, %d differ (%.1f %%), acc A %.1f / B %.1f, %s" % (
                t["center"][0], t["center"][1], t["center"][2], t["voxels"], t["differing"], 100 * t["frac"], t.get("acc_A", float("nan")), t.get("acc_B", float("nan")),
                ", ".join("%s x%d" % (k, v) for k, v in t["top_changes"][:3])))
        lines.append("")
        out[f] = dict(matched=d["matched"], differing=d["differing_total"], tiles=d["tiles"][:4])
    pngs = sorted(os.path.basename(p) for p in glob.glob(R + "/out/v05/rviz_*.png"))
    lines += ["captured PNGs (%d): %s" % (len(pngs), ", ".join(pngs)), ""]
    return out, lines


def main():
    ex, l0 = sec_extract()
    m3o, l3 = m3()
    p, l1 = m1()
    q, l2 = m2()
    md = ["# v0.5 re-qualification: trained semantic models in the live mapping pipeline", "",
          "Machine: RTX 4090, Ubuntu 24.04, ROS 2 Jazzy; Pointcept v1.5.1; fp16, shuffle_orders=False, intensity x0.2, grid 0.05; "
          "seq07 held-out (read at scoring time only). Files: out/v05/.", ""] + l0 + [""] + l1 + [""] + l2 + [""] + l3
    open(R + "/out/v05/REPORT.md", "w").write("\n".join(md) + "\n")
    json.dump(dict(extract=ex, m1=p, m2=q, m3=m3o), open(R + "/out/v05/summary.json", "w"), indent=2)
    print("\n".join(md))


if __name__ == "__main__":
    main()
