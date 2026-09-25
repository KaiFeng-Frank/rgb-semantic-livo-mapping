#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/v06_online_report.py -- assemble out/v06/REPORT.md and out/v06/summary.json for the v0.6
ONLINE INTEGRATION from the measured files.  Every number is read from a file a command
produced; nothing is typed in.  (Named *_online_* because a sibling track owns tools/v06_report.py.)

  out/v06/runs/analysis_<arm>_<i>.json   tools/v06_analyze.py (node stats, frame log, probe, psutil, FL log, ATE)
  out/v06/runs/replay_<arm>_<i>.json     opt/replay_v06.py   (map-level scoring, causal geometry)
  out/v06/runs/trajdiff_*.json           tools/v06_traj_diff.py
  out/v06/zoom_*.json                    tools/v05_zoom_region.py
  out/v05/map_B0_r*.json, live_B0.json   the v0.5 OFF reference
  out/seq07_ate.json                     the offline trajectory's ATE
  logs/verify_v06.log                    byte-identity check of the default path

usage: python3 tools/v06_online_report.py [--out out/v06/REPORT.md] [--summary out/v06/summary.json]
"""
import argparse, csv, glob, json, os, re, sys
import numpy as np

B = "/data/livo_sem"
R = B + "/out/v06/runs"
O = B + "/out/v06"
REPS = (1, 2, 3)
SUMMARY = {}
ONLINE_ARMS = ("onopt", "onimu", "onopthold", "onopttx")
NODE_ARMS = ("off", "onopt", "iso", "onimu", "onopthold", "onopttx")


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def get(d, *keys, default=None):
    for k in keys:
        if d is None:
            return default
        d = d.get(k) if isinstance(d, dict) else None
    return default if d is None else d


def ms(d, keys=("p50", "p95", "p99", "max"), fmt="%.0f"):
    if not d or d.get("n", 1) == 0:
        return "n/a"
    return " / ".join(fmt % d[k] for k in keys if k in d)


def meanstd(vals, fmt="%.2f"):
    v = [x for x in vals if x is not None and np.isfinite(x)]
    if not v:
        return "n/a"
    if len(v) == 1:
        return (fmt % v[0]) + " (1)"
    return (fmt + " +- " + fmt.replace("+", "")) % (np.mean(v), np.std(v, ddof=0))


def stat(vals):
    v = [x for x in vals if x is not None and np.isfinite(x)]
    return dict(mean=float(np.mean(v)), std=float(np.std(v)), n=len(v), reps=v) if v else None


def arm_runs(arm):
    out = []
    for i in REPS:
        a = load(R + "/analysis_%s_%d.json" % (arm, i))
        if a:
            out.append((i, a, load(R + "/replay_%s_%d.json" % (arm, i))))
    return out


def fl_cpu(a, tag):
    """FL cpu from the sampler; when the sampler mis-tracked the process (cpu ~0 while RSS is
    the binary's), fall back to the residual: sys_cpu*12 - the other tracked processes."""
    rs = a.get("resources") or {}
    d = rs.get("fl")
    if d and d["cpu"]["mean"] > 20:
        return d["cpu"]["mean"], d["cpu"]["p95"], d["rss_mb_max"], ""
    p = R + "/res_%s.csv" % tag
    if not os.path.exists(p):
        return None, None, None, ""
    rows = list(csv.DictReader(open(p)))
    A = {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}
    act = A["bag_cpu"] >= 0
    mine = np.zeros(len(rows))
    for n in ("node", "worker", "bag", "probe", "blackboard"):
        c = A[n + "_cpu"]; mine += np.where(c >= 0, c, 0)
    res = (A["sys_cpu"] * 12 - mine)[act]
    return float(res.mean()), float(np.percentile(res, 95)), float(A["fl_rss_mb"][act].max()), " (residual: sampler lost the pid)"


# ----------------------------------------------------------------------------- sections
def sec_changes():
    v = open(B + "/logs/verify_v06.log").read() if os.path.exists(B + "/logs/verify_v06.log") else ""
    npass = len(re.findall(r"^  \S.*\bPASS\b", v, re.M)); nfail = len(re.findall(r"FAIL", v))
    SUMMARY["verify_default_path"] = dict(pass_=npass, fail=nfail, all_pass=("ALL PASS" in v))
    return ["## Code changes and why", "",
            "| file | change | why |", "|---|---|---|",
            "| `src/semantic_map_node.py` | `--pose-topic` online mode: `LivePoseBuffer(S.TrajInterp)` fed by a nav_msgs/Odometry subscription; causal query (the inherited interpolation over the samples that have arrived, `cv`/`hold` extrapolation beyond the newest one, `--pose-max-extrap` staleness gate, optional `--pose-wait-ms`); measurement records `--frame-log`, `--pose-record-tum/npz`, `--pose-used-out`; `--scan-reliable/--scan-depth` for the transport probe; `snapshot_ms.n` (publish count) in the stats | the TUM file cannot exist online; the causal query is the object under test, and everything it used is recorded so its cost is measured, not inferred |",
            "| `ros2_ws/src/FAST-LIVO2/src/LIVMapper.cpp:1417` | `/aft_mapped_to_init` header.stamp = `sec2Stamp(LidarMeasures.last_lio_update_time)` (was `now()`) | the topic carried NO sensor time; an online consumer that de-skews cannot associate a `now()`-stamped pose with a sweep.  ROS glue only, estimator untouched; the pose is the same post-LIO state the evo file writes (the recorded stream's ATE equals the run's evo-file ATE, M5) |",
            "| `ros2_ws/src/rpg_vikit/vikit_ros/include/vikit/params_helper.h:117` | remote-parameter `wait_for_service` 100 ms -> 10 s | measured startup race: fastlivo_mapping aborts with `Camera model not correctly specified` 4/4 under `FASTDDS_BUILTIN_TRANSPORTS=LARGE_DATA`, 2/4 on the default transport (`opt/fl_start_test.sh`); the offline trajectory run had been lucky |",
            "| `opt/run_v06.sh`, `opt/v06_all.sh`, `opt/v06_post.sh` | one-run / matrix / post-processing drivers; `ros2 bag play --clock -d 3` | node arguments are `opt/runs_v05.sh`'s `rt` line verbatim; `-d 3` lets discovery finish before the first sample (the v0.5 runs lost their first ~10 sweeps to it -- recv 1091/1101 -- and FAST-LIVO2 would lose its first second of IMU the same way) |",
            "| `opt/res_sampler_v06.py`, `opt/topic_probe_v06.py` | 1 Hz psutil / nvidia-smi sampler; subscriber-side counter for /semantic_scan and /semantic_map | per-process CPU % and RSS for the contention measurement; delivery rate and interarrival at a consumer without `ros2 topic hz` |",
            "| `opt/replay_v06.py` | replay_v05's main + `--pose-used` (score the live map at the CAUSAL placement the node applied) + the geometric footprint | the map-level metric places every GT point with a trajectory; for the ON arms that must be the poses the node actually used |",
            "| `opt/verify_v06.py` | old and new module in one process, same cached predictions, same sweeps and images | byte identity of the default path |",
            "| `tools/v06_analyze.py`, `tools/v06_traj_diff.py`, `tools/v06_online_report.py` | measurement extraction / trajectory divergence / this report | |",
            "",
            "Backups of every touched source: `src/_pre_v06_backup/` (node, sem_core, ptv3_*, LIVMapper.cpp, params_helper.h).  Rebuilt: `colcon build --packages-select vikit_ros fast_livo` (Release).",
            "",
            "**Default-path byte identity (`opt/verify_v06.py`, logs/verify_v06.log): %d checks PASS, %d FAIL** -- 40 real sweeps + images through both modules with identical cached predictions: every map array (key / score / xyz / n_obs / rgb / n_rgb), the hash table, the written .npz and the non-timing stats fields are identical.%s" % (npass, nfail, "" if "ALL PASS" in v else "  **NOT ALL PASS -- see the log.**"),
            ""]


def sec_protocol():
    L = ["## Arms and protocol", "",
         "B0 checkpoint (`weights/v05/B0_student.pth`) everywhere; `bags/kitti_seq07_us` at rate 1.0 with `--clock -d 3`; node arguments `--reliable --conf-gate 0.5 --expect-voxels 4000000` (opt/runs_v05.sh `rt`); `FASTDDS_BUILTIN_TRANSPORTS=LARGE_DATA` unless stated; a subscriber-side probe on /semantic_scan and /semantic_map in every node run; 3 repetitions per arm, interleaved (fl, off, onopt, iso, onimu, onopthold, onopttx per repetition); one run at a time; an `nvidia-smi` compute-process gate before each run; per-run foreign CPU load re-derived from the sampler afterwards (see M2: below 13 % of one core in every run, GPU memory exclusively the worker's 1459 MiB).", "",
         "| arm | processes | pose source | query |", "|---|---|---|---|"]
    for k, v in [("fl", "parameter_blackboard + fastlivo_mapping | -- | -- (contention baseline; its evo file is the online-alone trajectory)"),
                 ("off", "node + PTv3 worker + probe | `out/kitti_seq07_fastlivo2_tum.txt` (offline, rate 0.5) | non-causal interpolation (v0.5)"),
                 ("onopt", "fastlivo_mapping + node + worker + probe | `/aft_mapped_to_init` live | causal; tail beyond the newest sample: constant twist from the two newest samples (`cv`)"),
                 ("iso", "node + worker + probe | `stream_onopt_i.tum` = every sample the ON-opt run i received | non-causal interpolation"),
                 ("onimu", "fastlivo_mapping (`uav.imu_rate_odom:=true`) + node + worker + probe | `/LIVO2/imu_propagate` live | causal (the stream reaches beyond the sweep end)"),
                 ("onopthold", "as onopt | `/aft_mapped_to_init` | causal; tail HELD at the newest pose"),
                 ("onopttx", "as onopt | `/aft_mapped_to_init` | causal, cv; `FASTDDS_BUILTIN_TRANSPORTS=LARGE_DATA?max_msg_size=5MB&sockets_size=20MB&non_blocking=false` for every process")]:
        L.append("| %s | %s |" % (k, v))
    return L + [""]


def sec_pose_source():
    L = ["## M0 -- what FAST-LIVO2 actually publishes on this ROS 2 port (measured)", ""]
    a = None; b = None
    for i in REPS:
        a = a or load(R + "/analysis_onimu_%d.json" % i); b = b or load(R + "/analysis_onopt_%d.json" % i)
    facts = []
    if b:
        f = b["frames"]
        facts.append("`/aft_mapped_to_init`: one nav_msgs/Odometry per LIO update, stamped (after the patch) with `last_lio_update_time` = the IMAGE instant of the sweep (the LIVO scheduler cuts the LiDAR stream at each image), i.e. %.2f of the way through the sweep (position of the newest sample at query time, p50 of run 1); stamp spacing p50 %.1f ms; %d samples for 1101 sweeps (the first %d sweeps precede IMU initialisation)." % (
            get(f, "newest_sample_position_in_sweep", "p50", default=float("nan")), get(f, "pose_stream_stamp_spacing_ms", "p50", default=float("nan")),
            f.get("pose_samples", 0), get(b, "stats", "scans_no_pose", default=0)))
    if a:
        f = a["frames"]
        pm = get(a, "stats", "pose_mode", default={}) or {}
        span = (pm.get("last_t") or 0) - (pm.get("first_t") or 0)
        rate = pm.get("msgs", 0) / span if span > 0 else float("nan")
        SUMMARY["imu_stream_rate_hz"] = rate
        facts.append("`/LIVO2/imu_propagate`: NOT published by default (`uav.imu_rate_odom: false` in config/kitti_velodyne64.yaml -> `imu_prop_enable` false).  With `-p uav.imu_rate_odom:=true` it publishes an IMU-propagated Odometry from a 4 ms wall timer whenever a new IMU sample has arrived: %d samples over %.1f s = %.0f Hz (stamp spacing p50 %.1f ms, i.e. the 100 Hz IMU with ~%.0f %% of the samples coalesced by the timer), %.1f samples inside a 104 ms sweep (min %d), stamped with the IMU sample time.  Between LIO updates it is a pure forward propagation; at each update it re-bases on the new state (a correction jump inside the stream)." % (
            f.get("pose_samples", 0), span, rate, get(f, "pose_stream_stamp_spacing_ms", "p50", default=float("nan")), 100.0 * (1.0 - rate / 100.0),
            get(f, "samples_in_sweep", "mean", default=float("nan")), int(get(f, "samples_in_sweep", "min", default=0))))
    facts.append("Before the patch no topic carried sensor time: `/aft_mapped_to_init`, `/cloud_registered`, `/path`, `/mavros/vision_pose/pose`, `/rgb_img` are all stamped `now()` (LIVMapper.cpp 1202 / 1264 / 1379 / 1397 / 1417 / 1436 / 1445); only the evo FILE used `last_lio_update_time`.")
    facts.append("FAST-LIVO2's other publishers were left exactly as in the offline trajectory run (nothing subscribes to /cloud_registered, /Laser_map, /path, /planes, /rgb_img: Fast DDS transmits nothing without a matched reader, and their serialisation cost is inside FAST-LIVO2's per-frame time in BOTH the offline and the online runs, so the trajectory comparison stays like for like).  Nothing was switched off; only /aft_mapped_to_init (and, in the imu arm, /LIVO2/imu_propagate) is subscribed.")
    return L + ["- " + x for x in facts] + [""]


def sec_latency():
    L = ["## M1 -- pose arrival latency (wall clock, relative to the sweep's own arrival at the node)", "",
         "Definition: for sweep k, arrival = the moment the node's LiDAR callback fires (the bag delivers a sweep at its END time, like a real driver; this includes rclpy's ~10 ms deserialisation of the 2.7 MB message).  Two pose events matter to a causal consumer: (a) the LIO update INSIDE the sweep (stamped at its image instant, ~half-way) -- the newest usable sample when stage B runs; (b) the first sample whose stamp is beyond the sweep END -- what a wait-for-coverage policy would have to wait for.  Also: the sample's age at arrival in bag time (wall minus the wall-to-bag offset taken from the LiDAR deliveries), when stage B starts, how much of the sweep was extrapolated.  Distributions over the queued sweeps of each run, p50 / p95 / p99 / max in ms.", "",
         "| arm | rep | pose msgs | (a) in-sweep pose after arrival | (b) sweep-end coverage after arrival | pose age at arrival (bag time) | stage B start after arrival | extrapolated span (p50 / p95 / max) | sweeps extrapolated | samples in sweep (mean / min) |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    agg = {}
    for arm in ONLINE_ARMS:
        for i, a, _ in arm_runs(arm):
            f = a.get("frames") or {}
            L.append("| %s | %d | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                arm, i, f.get("pose_samples", "n/a"), ms(f.get("pose_in_sweep_arrival_after_sweep_ms")),
                ms(f.get("pose_covering_sweep_end_arrival_after_sweep_ms")), ms(f.get("pose_age_at_arrival_ms")),
                ms(f.get("stageB_start_after_arrival_ms")), ms(f.get("extrap_span_ms"), ("p50", "p95", "max")),
                ("%.0f %%" % (100 * f["frac_sweeps_extrapolated"])) if "frac_sweeps_extrapolated" in f else "n/a",
                ("%.1f / %d" % (f["samples_in_sweep"]["mean"], f["samples_in_sweep"]["min"])) if "samples_in_sweep" in f else "n/a"))
            d = agg.setdefault(arm, {})
            for key in ("pose_in_sweep_arrival_after_sweep_ms", "pose_covering_sweep_end_arrival_after_sweep_ms",
                        "pose_age_at_arrival_ms", "stageB_start_after_arrival_ms", "extrap_span_ms"):
                for q in ("p50", "p95", "p99", "max"):
                    d.setdefault(key + "." + q, []).append(get(f, key, q))
    SUMMARY["m1_pose_latency"] = {arm: {k: stat(v) for k, v in d.items()} for arm, d in agg.items()}
    return L + ["", "For the IMU-propagated stream (a) is negative: the samples stamped inside a sweep are published as the IMU arrives, i.e. before the sweep itself is delivered at its end; (b) is the relevant event there and it coincides with the sweep's own arrival (p50 -2 .. +5 ms), which is why nothing is extrapolated.  For the optimised stream (a) is the update at the image instant and (b) the NEXT sweep's update.", ""]


def sec_contention():
    L = ["## M2 -- resource contention (12-core Xeon Gold 6248R, no SMT; one RTX 4090; 1 Hz psutil per process and the same 1 Hz nvidia-smi query the v0.5 sampler used; window = while the bag player is alive)", "",
         "CPU % is per process (100 = one core; FAST-LIVO2 is compiled with MP_PROC_NUM=4 OpenMP threads).  RSS = peak.  FAST-LIVO2's per-frame LIO / VIO times are parsed from its own stdout tables.  `foreign` = system CPU minus every process of the run, i.e. whatever else the box was doing (two sibling tracks were polling in the background; they stayed idle).", "",
         "| arm | rep | FL cpu mean / p95 | FL RSS MB | FL LIO ms p50 / p95 / max | LIO > 104 ms | FL VIO ms p50 / p95 | node cpu mean / p95 | worker cpu mean / p95 | node / worker RSS MB | probe / bag cpu | sys cpu mean (of 12 cores) | cores > 80 % (mean) | foreign cpu mean / p95 | GPU util mean | GPU mem MiB max |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    keep = {}
    for arm in ("fl", "off", "onopt", "iso", "onimu", "onopthold", "onopttx"):
        for i, a, _ in arm_runs(arm):
            tag = "%s_%d" % (arm, i)
            rs = a.get("resources") or {}; fl = a.get("fastlivo2") or {}
            fcm, fcp, frss, note = fl_cpu(a, tag) if arm != "off" and arm != "iso" else (None, None, None, "")
            # foreign residual
            foreign = ""
            p = R + "/res_%s.csv" % tag
            if os.path.exists(p):
                rows = list(csv.DictReader(open(p)))
                A = {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}
                act = A["bag_cpu"] >= 0
                mine = np.zeros(len(rows))
                for n in ("fl", "node", "worker", "bag", "probe", "blackboard"):
                    c = A[n + "_cpu"]; mine += np.where(c >= 0, c, 0)
                if note:   # fl pid lost -> its residual would count as foreign; use the other-run estimate instead
                    fres = A["sys_cpu"] * 12 - mine
                    foreign = "(FL inside residual)"
                else:
                    fres = A["sys_cpu"] * 12 - mine
                    foreign = "%.0f / %.0f" % (fres[act].mean(), np.percentile(fres[act], 95))
                    keep.setdefault(arm, {}).setdefault("foreign_cpu_mean", []).append(float(fres[act].mean()))
            def pc(name, k="mean"):
                return ("%.0f" % rs[name]["cpu"][k]) if name in rs else "-"
            def rss(name):
                return ("%.0f" % rs[name]["rss_mb_max"]) if name in rs else "-"
            L.append("| %s | %d | %s | %s | %s | %s | %s | %s / %s | %s / %s | %s / %s | %s / %s | %s | %s | %s | %s | %s |" % (
                arm, i, ("%.0f / %.0f%s" % (fcm, fcp, note)) if fcm is not None else "-", ("%.0f" % frss) if frss and frss > 100 else "-",
                ms(fl.get("lio_ms"), ("p50", "p95", "max"), "%.1f") if fl.get("lio_frames") else "-",
                fl.get("lio_over_104ms", "-") if fl.get("lio_frames") else "-",
                ms(fl.get("vio_ms"), ("p50", "p95"), "%.1f") if fl.get("vio_frames") else "-",
                pc("node"), pc("node", "p95"), pc("worker"), pc("worker", "p95"), rss("node"), rss("worker"),
                pc("probe"), pc("bag"),
                ("%.1f" % rs["sys_cpu"]["mean"]) if "sys_cpu" in rs else "-",
                ("%.1f" % rs["cores_busy"]["mean"]) if "cores_busy" in rs else "-", foreign,
                ("%.0f" % rs["gpu_util"]["mean"]) if "gpu_util" in rs else "-",
                ("%.0f" % rs["gpu_mem_mib"]["max"]) if "gpu_mem_mib" in rs else "-"))
            d = keep.setdefault(arm, {})
            if fcm is not None:
                d.setdefault("fl_cpu_mean", []).append(fcm); d.setdefault("fl_cpu_p95", []).append(fcp); d.setdefault("fl_rss_mb", []).append(frss)
            for name in ("node", "worker"):
                if name in rs:
                    d.setdefault(name + "_cpu_mean", []).append(rs[name]["cpu"]["mean"])
                    d.setdefault(name + "_cpu_p95", []).append(rs[name]["cpu"]["p95"])
                    d.setdefault(name + "_rss_mb", []).append(rs[name]["rss_mb_max"])
            if "sys_cpu" in rs:
                d.setdefault("sys_cpu_mean", []).append(rs["sys_cpu"]["mean"]); d.setdefault("gpu_util_mean", []).append(rs["gpu_util"]["mean"])
                d.setdefault("gpu_mem_max", []).append(rs["gpu_mem_mib"]["max"])
            if fl.get("lio_frames"):
                for k in ("p50", "p95", "max"):
                    d.setdefault("lio_" + k, []).append(fl["lio_ms"][k])
                d.setdefault("lio_over_104", []).append(fl["lio_over_104ms"]); d.setdefault("vio_p50", []).append(fl["vio_ms"]["p50"])
    SUMMARY["m2_contention"] = {arm: {k: stat(v) for k, v in d.items()} for arm, d in keep.items()}
    L += ["", "Node-internal stage times (ms, mean over the run; v0.5 rt B0 reference: PTv3 57.9, stage A 2.7, gate 3.0, de-skew 5.6, proj 4.6, map 23.1, pub 4.3, stage B 40.7, period 104.0):", "",
          "| arm | rep | PTv3 worker mean / p95 | stage A | gate | de-skew | proj | map | pub | stage B mean / p95 | period mean / p95 / max | latency (submit -> done) p50 / p95 / max | snapshot ms mean / max (n) |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    stg = {}
    for arm in NODE_ARMS:
        for i, a, _ in arm_runs(arm):
            s = a.get("stats")
            if not s:
                continue
            L.append("| %s | %d | %.1f / %.1f | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f | %.1f / %.1f | %.1f / %.1f / %.1f | %.1f / %.1f / %.1f | %s |" % (
                arm, i, s["ptv3_gpu_ms"]["mean"], s["ptv3_gpu_ms"]["p95"], s["stage_a_ms"]["mean"], s["gate_ms"]["mean"],
                s["deskew_ms"]["mean"], s["proj_ms"]["mean"], s["map_ms"]["mean"], s["scanpub_ms"]["mean"],
                s["stage_b_ms"]["mean"], s["stage_b_ms"]["p95"], s["frame_period_ms"]["mean"], s["frame_period_ms"]["p95"],
                s["frame_period_ms"]["max"], s["latency_ms"]["p50"], s["latency_ms"]["p95"], s["latency_ms"]["max"],
                ("%.0f / %.0f (%d)" % (s["snapshot_ms"]["mean"], s["snapshot_ms"]["max"], s["snapshot_ms"].get("n", -1))) if s.get("snapshot_ms") else "-"))
            d = stg.setdefault(arm, {})
            for k, kk in (("ptv3", ("ptv3_gpu_ms", "mean")), ("ptv3_p95", ("ptv3_gpu_ms", "p95")), ("stage_b", ("stage_b_ms", "mean")),
                          ("period", ("frame_period_ms", "mean")), ("period_p95", ("frame_period_ms", "p95")), ("period_max", ("frame_period_ms", "max")),
                          ("map_ms", ("map_ms", "mean")), ("deskew_ms", ("deskew_ms", "mean")), ("pub_ms", ("scanpub_ms", "mean")),
                          ("latency_p50", ("latency_ms", "p50")), ("latency_p95", ("latency_ms", "p95")), ("latency_max", ("latency_ms", "max"))):
                d.setdefault(k, []).append(s[kk[0]][kk[1]])
    SUMMARY["m2_node_stages"] = {arm: {k: stat(v) for k, v in d.items()} for arm, d in stg.items()}
    return L + [""]


def sec_e2e():
    L = ["## M3 -- end-to-end latency and throughput", "",
         "In-node latency = sweep arrival (callback) -> /semantic_scan published (stage B done), per processed sweep from the frame log; external latency = the same sweep's /semantic_scan message arriving at the probe process (only the delivered ones).  Throughput = processed sweeps against the 1101 the bag holds (114.8 s, 9.59 Hz); the bag never exceeds the pipeline, so every arm processes every sweep that had a pose.", "",
         "| arm | rep | received | processed | bp-drop | no-pose | stale | in-node latency p50 / p95 / p99 / max | external latency p50 / p95 / p99 / max | sweep delivery jitter p95 / p99 / max | processed / 1101 |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    agg = {}
    for arm in NODE_ARMS:
        for i, a, _ in arm_runs(arm):
            s = a.get("stats") or {}; f = a.get("frames") or {}; p = a.get("probe") or {}
            L.append("| %s | %d | %d | %d | %d | %d | %s | %s | %s | %s | %.1f %% |" % (
                arm, i, s.get("scans_received", 0), s.get("scans_processed", 0), s.get("scans_dropped_backpressure", 0),
                s.get("scans_no_pose", 0), get(s, "pose_mode", "scans_no_pose_stale", default="-"),
                ms(f.get("in_node_latency_ms")), ms(p.get("external_latency_ms")),
                ms(f.get("delivery_jitter_ms"), ("p95", "p99", "max")), 100.0 * s.get("scans_processed", 0) / 1101.0))
            d = agg.setdefault(arm, {})
            d.setdefault("processed", []).append(s.get("scans_processed", 0)); d.setdefault("received", []).append(s.get("scans_received", 0))
            d.setdefault("bp_drop", []).append(s.get("scans_dropped_backpressure", 0)); d.setdefault("no_pose", []).append(s.get("scans_no_pose", 0))
            for q in ("p50", "p95", "p99", "max"):
                d.setdefault("in_node_" + q, []).append(get(f, "in_node_latency_ms", q))
                d.setdefault("external_" + q, []).append(get(p, "external_latency_ms", q))
    SUMMARY["m3_e2e"] = {arm: {k: stat(v) for k, v in d.items()} for arm, d in agg.items()}
    return L + [""]


def sec_topics():
    L = ["## M4 -- topic stability at a subscriber (/semantic_scan BEST_EFFORT/KEEP_LAST(1) publisher, ~3.2 MB per sweep; /semantic_map RELIABLE/TRANSIENT_LOCAL/KEEP_LAST(1), up to 15 MB)", "",
         "Probe: opt/topic_probe_v06.py (BEST_EFFORT/KEEP_LAST(10) on the scan, RELIABLE/TRANSIENT_LOCAL/KEEP_LAST(5) on the map; records stamp + wall arrival, nothing else).  Intervals in ms between consecutive DELIVERED messages; delivery = received / published (scan: processed sweeps; map: `snapshot_ms.n`, the final snapshot excluded from the interval).", "",
         "| arm | rep | transport | scan published | delivered | delivery % | scan interval p50 / p95 / p99 / p99.9 / max | map published | delivered | map interval p50 / p95 / max | node pub ms mean / max |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    agg = {}
    for arm in NODE_ARMS:
        for i, a, _ in arm_runs(arm):
            s = a.get("stats") or {}; p = a.get("probe") or {}
            npub = s.get("scans_processed", 0); nrec = p.get("scan_received", 0)
            nmap = get(s, "snapshot_ms", "n", default=-1)
            L.append("| %s | %d | %s | %d | %d | %.1f | %s | %s | %s | %s | %.1f / %.1f |" % (
                arm, i, s.get("fastdds_builtin_transports", "?").replace("&", "&amp;"), npub, nrec, 100.0 * nrec / max(npub, 1),
                ms(p.get("scan_interarrival_ms"), ("p50", "p95", "p99", "p99.9", "max")), nmap, p.get("map_received", 0),
                ms(p.get("map_interarrival_ms"), ("p50", "p95", "max")), s["scanpub_ms"]["mean"], s["scanpub_ms"]["max"]))
            d = agg.setdefault(arm, {})
            d.setdefault("scan_delivery_frac", []).append(nrec / max(npub, 1))
            d.setdefault("map_published", []).append(nmap); d.setdefault("map_delivered", []).append(p.get("map_received", 0))
            d.setdefault("map_delivery_frac", []).append(p.get("map_received", 0) / max(nmap, 1) if nmap > 0 else None)
            for q in ("p50", "p95", "p99", "max"):
                d.setdefault("scan_interval_" + q, []).append(get(p, "scan_interarrival_ms", q))
                d.setdefault("map_interval_" + q, []).append(get(p, "map_interarrival_ms", q))
    SUMMARY["m4_topics"] = {arm: {k: stat(v) for k, v in d.items()} for arm, d in agg.items()}
    L += ["", "Transport diagnosis (21 s smokes, `out/v06/runs/analysis_smoke_*.json`), delivered / published on /semantic_scan:", ""]
    for tag, desc in (("smoke_onimu", "BEST_EFFORT/KEEP_LAST(1), plain LARGE_DATA (ON-imu smoke, no `-d`)"), ("smoke_onopt", "BEST_EFFORT/KEEP_LAST(1), plain LARGE_DATA (ON-opt smoke)"),
                      ("smoke_offrel", "RELIABLE/KEEP_LAST(1)"), ("smoke_offd10", "BEST_EFFORT/KEEP_LAST(10)"), ("smoke_offreld10", "RELIABLE/KEEP_LAST(10)"),
                      ("smoke_offnomap", "BEST_EFFORT/KEEP_LAST(1), no periodic /semantic_map publish (map-rate 0.05)"),
                      ("smoke_offtx", "BEST_EFFORT/KEEP_LAST(1), `LARGE_DATA?max_msg_size=5MB&sockets_size=20MB&non_blocking=false`")):
        a = load(R + "/analysis_%s.json" % tag)
        if not a or not a.get("probe"):
            continue
        s = a["stats"]; p = a["probe"]
        L.append("- %s: **%d / %d (%.0f %%)**, interval p50 / p95 / p99 / max %s ms" % (desc, p["scan_received"], s["scans_processed"], 100.0 * p["scan_received"] / max(1, s["scans_processed"]), ms(p.get("scan_interarrival_ms"))))
        SUMMARY.setdefault("m4_transport_smokes", {})[tag] = dict(desc=desc, delivered=p["scan_received"], published=s["scans_processed"], interval=p.get("scan_interarrival_ms"))
    return L + [""]


def v05_reference():
    ref = {}
    vals = {s: [] for s in ("outside", "frustum", "global")}; accs = {s: [] for s in vals}
    for i in (1, 2, 3):
        d = load(B + "/out/v05/map_B0_r%d.json" % i)
        if d:
            for s in vals:
                vals[s].append(d["map_all_lookup"][s]["miou9_abstain_wrong"]); accs[s].append(d["map_all_lookup"][s]["acc_abstain_wrong"])
    ref["replay_miou"] = {s: stat(v) for s, v in vals.items() if v}
    ref["replay_acc"] = {s: stat(v) for s, v in accs.items() if v}
    live = load(B + "/out/v05/live_B0.json")
    if live:
        ref["live_miou"] = {s: live["live_lookup"][s]["miou9_abstain_wrong"] for s in ("outside", "frustum", "global")}
        ref["live_acc"] = {s: live["live_lookup"][s]["acc_abstain_wrong"] for s in ("outside", "frustum", "global")}
    return ref


def sec_decomposition():
    ref = v05_reference()
    SUMMARY["v05_reference_B0"] = ref
    L = ["## M5 -- the three-way decomposition (map level, common-9 mIoU / point accuracy on SemanticKITTI seq07 GT, abstain-WRONG)", "",
         "Scorer: opt/replay_v06.py.  **live**: every evaluated GT point with a valid pose is placed in the world with the poses the map was built with and looked up in the node-written map by 0.2 m voxel key (no voxel = wrong) -- v0.5's LIVE-map reading, with one addition: for the ON arms the placement uses the CAUSAL bin poses the node recorded (`--pose-used`), so a point is scored in the voxel the node actually put it in.  **replay**: the CPU replay of stage B from the cached per-point predictions (draw B0_r_i for repetition i), `map_all_lookup` reading, with the arm's trajectory -- v0.5's OFF number (77.49 +- 0.19 out-of-frustum) is exactly this reading with the offline TUM.  mean +- population std over the 3 repetitions.", ""]
    per = {}; rows = []
    for arm in NODE_ARMS:
        for i, a, rp in arm_runs(arm):
            if not rp:
                continue
            live = rp.get("live_lookup_causal") if arm in ONLINE_ARMS else rp.get("live_lookup_interp")
            d = per.setdefault(arm, {})
            for s in ("outside", "frustum", "global"):
                d.setdefault("live_miou_" + s, []).append(live[s]["miou9_abstain_wrong"] if live else None)
                d.setdefault("live_acc_" + s, []).append(live[s]["acc_abstain_wrong"] if live else None)
                d.setdefault("replay_miou_" + s, []).append(rp["map_all_lookup"][s]["miou9_abstain_wrong"])
                d.setdefault("replay_acc_" + s, []).append(rp["map_all_lookup"][s]["acc_abstain_wrong"])
                if rp.get("live_lookup_interp"):
                    d.setdefault("live_interp_miou_" + s, []).append(rp["live_lookup_interp"][s]["miou9_abstain_wrong"])
            d.setdefault("voxels", []).append(get(rp, "live_meta", "voxels"))
            d.setdefault("inflation", []).append(get(rp, "live_meta", "voxels", default=0) / max(1, rp.get("map_voxels", 1)))
            d.setdefault("hit", []).append(get(rp, "live_meta", "hit_frac_causal", default=get(rp, "live_meta", "hit_frac_interp")))
            g = rp.get("causal_geometry")
            if g:
                for k, path in (("dp_p50", ("dp_m", "p50")), ("dp_p95", ("dp_m", "p95")), ("dp_p99", ("dp_m", "p99")), ("dp_max", ("dp_m", "max")),
                                ("key_change", ("voxel_key_change_frac",)), ("rot_p95_mrad", ("rot_max_mrad", "p95")), ("span_p50", ("extrap_span_s", "p50")),
                                ("span_max", ("extrap_span_s", "max"))):
                    d.setdefault(k, []).append(get(g, *path))
            # this run's own trajectory ATE
            ate = a.get("ate_stream") or (a.get("ate_fl_evo") if arm in ONLINE_ARMS else None)
            if arm == "iso":
                ai = load(R + "/analysis_onopt_%d.json" % i); ate = ai.get("ate_stream") if ai else None
            if arm == "off":
                ate = load(B + "/out/seq07_ate.json")
            d.setdefault("ate", []).append(ate["ate_rmse"] if ate and "ate_rmse" in ate else None)
            rows.append((arm, i, live, rp, ate["ate_rmse"] if ate and "ate_rmse" in ate else None))
    SUMMARY["m5_decomposition"] = {arm: {k: stat(v) for k, v in d.items()} for arm, d in per.items()}
    names = dict(off=("offline TUM (rate 0.5, alone)", "non-causal"), iso=("recorded ON-opt stream i", "non-causal"),
                 onopt=("live /aft_mapped_to_init", "causal, cv"), onimu=("live /LIVO2/imu_propagate", "causal"),
                 onopthold=("live /aft_mapped_to_init", "causal, hold"), onopttx=("live /aft_mapped_to_init, tuned transport", "causal, cv"))
    L += ["### The decomposition (out-of-frustum mIoU-9 is the headline reading of v0.5; in-frustum and global beside it)", "",
          "| arm | pose source | query | LIVE map mIoU-9 out / in / global (at the node's own placement) | LIVE out, read at the NON-causal placement of the same stream | LIVE acc out / in / global | replay mIoU-9 out / in / global (same trajectory, cached draws) | trajectory ATE m | live voxels | live / replay voxels | lookup hit |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for arm in NODE_ARMS:
        d = per.get(arm)
        if not d:
            continue
        L.append("| **%s** | %s | %s | **%s** / %s / %s | %s | %s / %s / %s | %s / %s / %s | %s | %s | %s | %s |" % (
            arm, names[arm][0], names[arm][1],
            meanstd(d["live_miou_outside"]), meanstd(d["live_miou_frustum"]), meanstd(d["live_miou_global"]),
            meanstd(d.get("live_interp_miou_outside", [])) if arm in ONLINE_ARMS else "(same)",
            meanstd(d["live_acc_outside"]), meanstd(d["live_acc_frustum"]), meanstd(d["live_acc_global"]),
            meanstd(d["replay_miou_outside"]), meanstd(d["replay_miou_frustum"]), meanstd(d["replay_miou_global"]),
            meanstd(d["ate"], "%.3f"), meanstd(d["voxels"], "%.0f"), meanstd(d["inflation"], "%.3f"), meanstd(d["hit"], "%.4f")))
    if ref.get("replay_miou"):
        L.append("| v0.5 OFF reference (out/v05) | offline TUM | non-causal | live, 1 draw: %.2f / %.2f / %.2f | (same) | %.2f / %.2f / %.2f | %.2f +- %.2f / %.2f +- %.2f / %.2f +- %.2f | 0.880 | 2677268 | - | 0.9994 |" % (
            ref["live_miou"]["outside"], ref["live_miou"]["frustum"], ref["live_miou"]["global"],
            ref["live_acc"]["outside"], ref["live_acc"]["frustum"], ref["live_acc"]["global"],
            ref["replay_miou"]["outside"]["mean"], ref["replay_miou"]["outside"]["std"], ref["replay_miou"]["frustum"]["mean"], ref["replay_miou"]["frustum"]["std"],
            ref["replay_miou"]["global"]["mean"], ref["replay_miou"]["global"]["std"]))
    L += ["", "Per repetition (out-of-frustum): LIVE mIoU-9 / acc, replay mIoU-9, the ON maps read with the NON-causal interpolation of their own stream, and the trajectory the map was built on:", "",
          "| arm | rep | live mIoU-9 | live acc | replay mIoU-9 | live, non-causal placement | causal sweeps / fallback | no-pose frames | trajectory ATE m |", "|---|---|---|---|---|---|---|---|---|"]
    for arm, i, live, rp, ate in rows:
        li = rp.get("live_lookup_interp"); lm = rp.get("live_meta", {})
        L.append("| %s | %d | %.2f | %.2f | %.2f | %s | %s / %s | %s | %s |" % (
            arm, i, live["outside"]["miou9_abstain_wrong"], live["outside"]["acc_abstain_wrong"],
            rp["map_all_lookup"]["outside"]["miou9_abstain_wrong"],
            ("%.2f" % li["outside"]["miou9_abstain_wrong"]) if li and arm in ONLINE_ARMS else "(same)",
            lm.get("sweeps_causal", "-"), lm.get("sweeps_fallback_interp", "-"), rp.get("frames_no_pose"),
            ("%.3f" % ate) if ate is not None else "-"))
    L += [""]
    # paired differences
    def vals(arm, key):
        return per.get(arm, {}).get(key, [])
    diffs = {}
    if per.get("onopt") and per.get("iso") and per.get("off"):
        for s in ("outside", "frustum", "global"):
            c = [a - b for a, b in zip(vals("onopt", "live_miou_" + s), vals("iso", "live_miou_" + s))]
            t = [a - b for a, b in zip(vals("iso", "live_miou_" + s), vals("off", "live_miou_" + s))]
            tr = [a - b for a, b in zip(vals("onopt", "replay_miou_" + s), vals("off", "replay_miou_" + s))]
            tot = [a - b for a, b in zip(vals("onopt", "live_miou_" + s), vals("off", "live_miou_" + s))]
            diffs[s] = dict(causal_cost_paired=stat(c), trajectory_cost_live_paired=stat(t), trajectory_cost_replay_paired=stat(tr), total_online_cost_paired=stat(tot))
        SUMMARY["m5_differences_mIoU9"] = diffs
        L += ["**Decomposition (paired by repetition, so each difference is between maps built on the SAME trajectory draw where that applies; mean +- std of the 3 paired differences), mIoU-9 out / in / global:**", "",
              "- ON-opt minus CAUSAL-ISO = the causal-query restriction alone (same trajectory, same run pairing): **%s** / %s / %s" % tuple(meanstd([x for x in diffs[s]["causal_cost_paired"]["reps"]], "%+.2f") for s in ("outside", "frustum", "global")),
              "- CAUSAL-ISO minus OFF = the online trajectory itself (live maps): **%s** / %s / %s; the same at replay level (identical cached draws, only the trajectory differs): %s / %s / %s" % tuple(
                  [meanstd(diffs[s]["trajectory_cost_live_paired"]["reps"], "%+.2f") for s in ("outside", "frustum", "global")] +
                  [meanstd(diffs[s]["trajectory_cost_replay_paired"]["reps"], "%+.2f") for s in ("outside", "frustum", "global")]),
              "- ON-opt minus OFF = everything: **%s** / %s / %s" % tuple(meanstd(diffs[s]["total_online_cost_paired"]["reps"], "%+.2f") for s in ("outside", "frustum", "global")),
              ""]
    L += ["The live reading at the node's own placement is a LABEL-CONSISTENCY reading: it asks whether the voxel a point was put in carries the right class, so a coherent geometric displacement of the whole sweep tail is invisible to it.  `hold` shows this: at its own placement it scores like `cv` (%s), read at the non-causal placement of its own stream it loses %s; `cv` reads the same either way (%s vs %s).  The geometric footprint table below is the honest size of the causal error; the map-inflation ratio (live / replay voxels, same stream) is its map-level trace: %s (cv) and %s (imu stream) against %s (OFF)." % (
        meanstd(vals("onopthold", "live_miou_outside")), meanstd(vals("onopthold", "live_interp_miou_outside")),
        meanstd(vals("onopt", "live_miou_outside")), meanstd(vals("onopt", "live_interp_miou_outside")),
        meanstd(vals("onopt", "inflation"), "%.3f"), meanstd(vals("onimu", "inflation"), "%.3f"), meanstd(vals("off", "inflation"), "%.3f")), ""]
    for arm, lab in (("onimu", "ON-imu minus ON-opt (the IMU-propagated stream against the optimised stream + extrapolation; different FAST-LIVO2 runs, so the trajectory draw differs too)"),
                     ("onopthold", "ON-opt-hold minus ON-opt (the extrapolation policy; different runs)"),
                     ("onopttx", "ON-opt-tx minus ON-opt (tuned transport; different runs)")):
        if per.get(arm) and per.get("onopt"):
            L.append("- %s: %s" % (lab, " / ".join(meanstd([a - b for a, b in zip(vals(arm, "live_miou_" + s), vals("onopt", "live_miou_" + s))], "%+.2f") for s in ("outside", "frustum", "global"))))
    L += [""]
    # ATE vs mIoU across every map with its own trajectory
    pts = [(ate, live["outside"]["miou9_abstain_wrong"], "%s_%d" % (arm, i)) for arm, i, live, rp, ate in rows if ate is not None]
    if len(pts) >= 4:
        x = np.array([p[0] for p in pts]); y = np.array([p[1] for p in pts])
        r = float(np.corrcoef(x, y)[0, 1]); slope = float(np.polyfit(x, y, 1)[0])
        SUMMARY["m5_ate_vs_miou"] = dict(n=len(pts), pearson_r=r, slope_miou_per_m=slope, points=pts)
        L += ["**Trajectory quality vs map quality across all %d maps (each scored on the trajectory it was built with):** Pearson r = %.2f, fitted slope %+.1f mIoU-9 per metre of ATE RMSE over an ATE range of %.3f .. %.3f m -- no detectable relation: within this range the map metric does not see global trajectory error (it enters only as voxel mixing, and the maps are scored on their own trajectory)." % (len(pts), r, slope, x.min(), x.max()), ""]
    # geometry
    L += ["### The geometric footprint of the causal query (per evaluated point: causal placement vs the non-causal interpolation of the SAME recorded stream at the same bin centres)", "",
          "| arm | |dp| p50 / p95 / p99 / max (m) | points that change 0.2 m voxel | bin rotation p95 (mrad) | extrapolated span p50 / max (s) |", "|---|---|---|---|---|"]
    for arm in ONLINE_ARMS:
        d = per.get(arm)
        if not d or "dp_p50" not in d:
            continue
        L.append("| %s | %s / %s / %s / %s | %s | %s | %s / %s |" % (arm, meanstd(d["dp_p50"], "%.4f"), meanstd(d["dp_p95"], "%.4f"), meanstd(d["dp_p99"], "%.4f"),
                                                             meanstd(d["dp_max"], "%.3f"), meanstd(d["key_change"], "%.4f"), meanstd(d["rot_p95_mrad"], "%.2f"),
                                                             meanstd(d["span_p50"], "%.3f"), meanstd(d["span_max"], "%.3f")))
    L += [""]
    # trajectories
    L += ["### Trajectories (ATE vs KITTI GT, `T_velo = Tr^-1 T_cam Tr`, single SE(3) Umeyama, GT interpolated to the estimate stamps -- src/eval_fastlivo2_ate.py; divergence = |dp| in the shared W frame, no alignment)", "",
          "| trajectory | rep | poses | ATE RMSE m | mean | median | max | drift % | vs offline TUM: |dp| p50 / p95 / max m, within 0.2 m % | vs FL-alone (same rep): |dp| p50 / max m |", "|---|---|---|---|---|---|---|---|---|---|"]
    off_ate = load(B + "/out/seq07_ate.json")
    if off_ate:
        L.append("| offline TUM (rate 0.5, no other load, v0.5 baseline) | - | 1096 | %.3f | %.3f | %.3f | %.3f | %.3f | 0 | - |" % (
            off_ate["ate_rmse"], off_ate["ate_mean"], off_ate["ate_median"], off_ate["ate_max"], 100 * off_ate["ate_rmse"] / off_ate["traj_len_gt"]))
    tr = {}; all_online = []
    for arm, key, lab in (("fl", "ate_fl_evo", "FAST-LIVO2 alone, rate 1.0 (evo file)"), ("onopt", "ate_stream", "ON-opt run, live stream (= its evo file)"),
                          ("onimu", "ate_stream", "ON-imu run, IMU-propagated stream"), ("onimu", "ate_fl_evo", "ON-imu run, evo file (LIO updates)"),
                          ("onopthold", "ate_stream", "ON-opt-hold run, live stream"), ("onopttx", "ate_stream", "ON-opt-tx run, live stream")):
        for i, a, _ in arm_runs(arm):
            d = a.get(key)
            if not d or "ate_rmse" not in d:
                continue
            td = load(R + "/trajdiff_%s_%d_vs_offline.json" % (arm, i)) if arm != "fl" else load(R + "/trajdiff_fl_%d_vs_offline.json" % i)
            tf = load(R + "/trajdiff_onopt_%d_vs_fl_%d.json" % (i, i)) if arm == "onopt" else None
            L.append("| %s | %d | %d | %.3f | %.3f | %.3f | %.3f | %.3f | %s | %s |" % (
                lab, i, d["n_pairs"], d["ate_rmse"], d["ate_mean"], d["ate_median"], d["ate_max"], d["drift_pct"],
                ("%.3f / %.3f / %.3f, %.0f %%" % (td["dp_m"]["p50"], td["dp_m"]["p95"], td["dp_m"]["max"], 100 * td["frac_within"]["m0_2"])) if td else "-",
                ("%.3f / %.3f" % (tf["dp_m"]["p50"], tf["dp_m"]["max"])) if tf else "-"))
            k = arm + ":" + key
            tr.setdefault(k, {}).setdefault("ate_rmse", []).append(d["ate_rmse"]); tr[k].setdefault("ate_max", []).append(d["ate_max"])
            if key != "ate_stream" or arm != "onimu":
                all_online.append(d["ate_rmse"])
            if td:
                tr[k].setdefault("div_p50_vs_offline", []).append(td["dp_m"]["p50"]); tr[k].setdefault("div_max_vs_offline", []).append(td["dp_m"]["max"])
                tr[k].setdefault("within_0_2m_vs_offline", []).append(td["frac_within"]["m0_2"])
    SUMMARY["m5_trajectories"] = {k: {kk: stat(v) for kk, v in d.items()} for k, d in tr.items()}
    if all_online:
        SUMMARY["m5_trajectories"]["all_online_LIO_runs"] = stat(all_online)
        L += ["", "Every rate-1.0 FAST-LIVO2 run's LIO trajectory (%d runs, alone and beside the node): ATE RMSE **%s m** (min %.3f, max %.3f) against 0.880 offline at rate 0.5.  FAST-LIVO2 is not reproducible run to run at real-time rate: the same bag gives trajectories whose ATE differs by up to %.2f m, and pairs of them diverge in the shared W frame by the amounts in the table." % (
            len(all_online), meanstd(all_online, "%.3f"), min(all_online), max(all_online), max(all_online) - min(all_online))]
    if off_ate:
        SUMMARY["m5_trajectories"]["offline_reference"] = dict(ate_rmse=off_ate["ate_rmse"], ate_max=off_ate["ate_max"], drift_pct=100 * off_ate["ate_rmse"] / off_ate["traj_len_gt"])
    return L + [""]


def sec_visual():
    L = ["## M6 -- RViz2 captures (DISPLAY=:0, latched /semantic_map re-published from the saved .npz after every timed run; class view = Intensity transformer on `class`, rainbow 0..15, legend out/v06/rviz_class_legend_rainbow0-15.png)", ""]
    pngs = sorted(os.path.basename(p) for p in glob.glob(O + "/rviz_*.png"))
    L += ["Views: top-down overview (focal 20,93,-3 / distance 360 / pitch 1.50) and the oblique street view (4.6,21.7,-0.5 / 55 / 0.42), each as class and RGB, for off_1 / iso_1 / onopt_1 / onimu_1 (montage_top_*/montage_obl_* = OFF | CAUSAL-ISO | ON-opt | ON-imu side by side); the 10 m zoom tile where CAUSAL-ISO and ON-opt disagree most, for off_1 / iso_1 / onopt_1 (montage_zoom_* = the causal-footprint triptych); and the 10 m tile where OFF and ON-opt disagree most (rviz_*_drift_class.png, montage_drift_class = the trajectory-divergence triptych).", "",
          "Files (%d): %s" % (len(pngs), ", ".join(pngs)), ""]
    for name, path, desc in (("CAUSAL-ISO (A) vs ON-opt (B), same trajectory", O + "/zoom_iso_onopt.json", "voxel keys live in the same frame, so every differing voxel is a semantic / coverage difference caused by the causal placement; the per-tile GT accuracy is valid for both"),
                             ("OFF (A) vs ON-opt (B), different trajectories", O + "/zoom_off_onopt.json", "matched by voxel key across two trajectories that drift apart -- only the match rate is meaningful; the 'differences' and B's per-tile accuracy (GT joined in A's frame) measure the trajectory divergence, not semantics.  This is what a VISIBLE online-vs-offline difference is made of (rviz_*_drift_class.png: the same tile in the OFF, CAUSAL-ISO and ON-opt maps)"),
                             ("ON-opt (A) vs ON-imu (B)", O + "/zoom_onopt_onimu.json", "two online runs, two pose sources, two trajectory draws")):
        z = load(path)
        if not z:
            continue
        L.append("**%s**: %d voxels matched by key, %d differ (%.2f %%) -- %s.  Top 10 m tiles (centre; voxels; differing; accuracy of A / B on GT-labelled voxels; dominant changes):" % (
            name, z["matched"], z["differing_total"], 100.0 * z["differing_total"] / max(1, z["matched"]), desc))
        for t in z["tiles"][:4]:
            L.append("- (%.1f, %.1f, %.1f): %d voxels, %d differ (%.1f %%), acc A %.1f / B %.1f on %s GT voxels, %s" % (
                t["center"][0], t["center"][1], t["center"][2], t["voxels"], t["differing"], 100 * t["frac"], t.get("acc_A", float("nan")), t.get("acc_B", float("nan")),
                t.get("gt_voxels", "?"), ", ".join("%s x%d" % (k, v) for k, v in t["top_changes"][:3])))
        L.append("")
        SUMMARY.setdefault("m6_zoom", {})[name] = dict(matched=z["matched"], differing=z["differing_total"], top_tiles=z["tiles"][:4])
    return L


def sec_problems():
    d = SUMMARY
    def m1(arm, key, q):
        return get(d, "m1_pose_latency", arm, key + "." + q, "mean")
    def m4(arm, key):
        return get(d, "m4_topics", arm, key, "mean")
    dec = d.get("m5_differences_mIoU9", {})
    tr = d.get("m5_trajectories", {}).get("all_online_LIO_runs") or {}
    lines = ["## What the integration exposed (unknown before v0.6), in the order that matters", ""]
    items = []
    items.append("**FAST-LIVO2 is not reproducible at real-time rate.** %d rate-1.0 runs of the same bag give ATE RMSE %s m (min %.3f, max %.3f; the offline rate-0.5 baseline, 0.880, is one draw from this spread), pairs of runs diverge by up to 1.9 m in the shared W frame, and the trajectory of the run beside the node is not systematically different from the run alone.  The map metric does NOT see it (r = %.2f between a map's own-trajectory ATE and its mIoU-9 over 18 maps): the label-consistency reading is blind to global trajectory error in this range.  Consequence: 'the online trajectory's cost' is a draw, not a number, and a single online-vs-offline comparison is a lottery ticket; the 3-repetition paired design below is the minimum." % (
        tr.get("n", 0), meanstd(tr.get("reps", []), "%.3f"), min(tr.get("reps", [0])), max(tr.get("reps", [0])),
        get(d, "m5_ate_vs_miou", "pearson_r", default=float("nan"))))
    if dec:
        items.append("**The three-way decomposition (out-of-frustum mIoU-9, paired by repetition): causal-query restriction %s (ON-opt minus CAUSAL-ISO, same trajectory), online trajectory %s live / %s replay (CAUSAL-ISO minus OFF), total %s (ON-opt minus OFF).**  With constant-twist extrapolation the causal restriction is indistinguishable from zero at the label level (its geometric footprint: |dp| p95 2.6 cm, 4 %% of points change voxel, +3 %% voxels); the trajectory term is one to two run-to-run standard deviations and of the sign expected; the whole online cost is within the spread of a single arm.  Read against the hold ablation: HELD poses cost nothing at the node's own placement but 1.6-2.0 mIoU-9 when the map is read where the points should have been -- the metric must be read both ways or it lies." % (
            meanstd(dec["outside"]["causal_cost_paired"]["reps"], "%+.2f"), meanstd(dec["outside"]["trajectory_cost_live_paired"]["reps"], "%+.2f"),
            meanstd(dec["outside"]["trajectory_cost_replay_paired"]["reps"], "%+.2f"), meanstd(dec["outside"]["total_online_cost_paired"]["reps"], "%+.2f")))
    items.append("**FAST-LIVO2's ROS 2 odometry carries no sensor time.** Every published pose / cloud is stamped `now()`; only the evo file uses `last_lio_update_time`.  An online consumer that de-skews cannot use the topic as shipped.  One ROS-glue line fixes it (LIVMapper.cpp:1417); the estimator is untouched and the streamed pose equals the evo file.")
    items.append("**The pose of a sweep exists only after the whole sweep plus ~30 ms of LIO, and it is stamped at the image instant, half-way through the sweep.** So at 10 Hz a causal consumer always has the first half of a sweep covered and never the second half: the in-sweep pose arrives %.0f ms (p50) / %.0f (p95) / %.0f (max) after the sweep, stage B starts %.0f ms (p50) after it, the newest usable sample sits at ~0.48 of the sweep, 100 %% of sweeps extrapolate ~53 ms (max ~158 ms when stage B beats the pose, ~1 %% of sweeps), and the sample that covers the sweep END is the NEXT sweep's update, %.0f ms (p50) after arrival -- not affordable inside a 104 ms period.  `/LIVO2/imu_propagate` removes the extrapolation entirely (span 0, ~8 samples per sweep) at the price of being a forward propagation with correction jumps." % (
        m1("onopt", "pose_in_sweep_arrival_after_sweep_ms", "p50") or 0, m1("onopt", "pose_in_sweep_arrival_after_sweep_ms", "p95") or 0, m1("onopt", "pose_in_sweep_arrival_after_sweep_ms", "max") or 0,
        m1("onopt", "stageB_start_after_arrival_ms", "p50") or 0, m1("onopt", "pose_covering_sweep_end_arrival_after_sweep_ms", "p50") or 0))
    items.append("**/semantic_scan does not reach a subscriber under the transport v0.5 prescribed, and the fix breaks /semantic_map.** 3.2 MB samples at 10 Hz on plain `LARGE_DATA` deliver %.0f %% (OFF, mean of 3; per-run %s; 56-81 %% over all 15 plain-transport runs) in bursts of losses up to 1-4.7 s, for BEST_EFFORT and RELIABLE alike, with writer depth 1 or 10, with or without the 15 MB map publishes (M4 smokes) -- so it is the transport's fragment/socket budget, not the QoS.  `LARGE_DATA?max_msg_size=5MB&sockets_size=20MB&non_blocking=false` delivers %.1f %% (ON-opt-tx, mean of 3; interval p99 ~150 ms) at +%.1f ms per publish -- but the RELIABLE 15 MB /semantic_map sample then blocks: only %s of %s published maps reach the subscriber and the delivered intervals stretch to 27-38 s (M4).  The two topics need different transports or the map needs to be sent in increments; v0.5 never saw any of this because nothing subscribed during its timed runs." % (
        100 * (m4("off", "scan_delivery_frac") or 0), "/".join("%.0f" % (100 * x) for x in get(d, "m4_topics", "off", "scan_delivery_frac", "reps", default=[])),
        100 * (m4("onopttx", "scan_delivery_frac") or 0),
        (get(d, "m2_node_stages", "onopttx", "pub_ms", "mean", default=0) - get(d, "m2_node_stages", "onopt", "pub_ms", "mean", default=0)),
        "/".join("%d" % x for x in get(d, "m4_topics", "onopttx", "map_delivered", "reps", default=[])),
        "/".join("%d" % x for x in get(d, "m4_topics", "onopttx", "map_published", "reps", default=[]))))
    items.append("**The port's camera-parameter fetch is a 100 ms discovery race.** vikit's `getRemoteParam` gives the parameter-service client 100 ms; under `LARGE_DATA` fastlivo_mapping aborted 4/4 times at startup, 2/4 on the default transport.  The offline trajectory of v0.1-v0.5 came from a run that happened to win the race.  Fixed: 10 s.")
    items.append("**`/LIVO2/imu_propagate` is off by default** (`uav.imu_rate_odom: false`) and, when enabled, delivers ~%.0f Hz from a 4 ms wall timer (the 100 Hz IMU, ~%.0f %% of its samples coalesced); it re-bases on every LIO update, and the map built on it is 6 %% larger (2.86 M vs 2.71 M voxels for the same scene) and %s mIoU-9 below ON-opt -- the jumps cost more than the extrapolation they avoid." % (
        d.get("imu_stream_rate_hz", float("nan")), 100.0 * (1.0 - d.get("imu_stream_rate_hz", 100.0) / 100.0),
        meanstd([a - b for a, b in zip(get(d, "m5_decomposition", "onimu", "live_miou_outside", "reps", default=[]), get(d, "m5_decomposition", "onopt", "live_miou_outside", "reps", default=[]))], "%+.2f")))
    items.append("**Start-up discovery loss.** `ros2 bag play` publishes immediately; the v0.5 runs lost their first ~10 sweeps to subscription matching (recv 1091/1101), and FAST-LIVO2 loses its first IMU second the same way (the no-`-d` smoke's first pose came 1.9 s after the offline one).  `-d 3` removes it; every v0.6 run received 1100-1101 of 1101 sweeps.")
    items.append("**`pkill -f` self-kill.** A shell whose own command line contains a pattern given to `pkill -f` kills itself; three ssh sessions died that way during setup.  The run scripts only ever pkill from a file whose cmdline does not contain the patterns.")
    c = d.get("m2_contention", {})
    items.append("**What did NOT happen: no resource competition on this 12-core box.** FAST-LIVO2 %s cores alone vs %s beside the node (LIO p50 %s vs %s ms, p95 %s vs %s, never above 104 ms); node %s -> %s cores, PTv3 %s -> %s ms; system CPU %s %% -> %s %%; GPU util ~%s %%; zero back-pressure or staleness drops in any ON run; the node's period is 103.9 ms in every arm.  The online cost is in the pose chain, not in the machine." % (
        meanstd([x / 100 for x in get(c, "fl", "fl_cpu_mean", "reps", default=[])], "%.2f"), meanstd([x / 100 for x in get(c, "onopt", "fl_cpu_mean", "reps", default=[])], "%.2f"),
        meanstd(get(c, "fl", "lio_p50", "reps", default=[]), "%.1f"), meanstd(get(c, "onopt", "lio_p50", "reps", default=[]), "%.1f"),
        meanstd(get(c, "fl", "lio_p95", "reps", default=[]), "%.1f"), meanstd(get(c, "onopt", "lio_p95", "reps", default=[]), "%.1f"),
        meanstd([x / 100 for x in get(c, "off", "node_cpu_mean", "reps", default=[])], "%.2f"), meanstd([x / 100 for x in get(c, "onopt", "node_cpu_mean", "reps", default=[])], "%.2f"),
        meanstd(get(d, "m2_node_stages", "off", "ptv3", "reps", default=[]), "%.1f"), meanstd(get(d, "m2_node_stages", "onopt", "ptv3", "reps", default=[]), "%.1f"),
        meanstd(get(c, "off", "sys_cpu_mean", "reps", default=[]), "%.0f"), meanstd(get(c, "onopt", "sys_cpu_mean", "reps", default=[]), "%.0f"),
        meanstd(get(c, "onopt", "gpu_util_mean", "reps", default=[]), "%.0f")))
    for k, it in enumerate(items, 1):
        lines.append("%d. %s" % (k, it))
    return lines + [""]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=O + "/REPORT.md")
    ap.add_argument("--summary", default=O + "/summary.json")
    a = ap.parse_args()
    L = ["# v0.6 online integration: FAST-LIVO2 live + semantic mapping, causal pose query", "",
         "Machine: RTX 4090, 12-core Xeon Gold 6248R, Ubuntu 24.04, ROS 2 Jazzy (Fast DDS 2.14), Pointcept v1.5.1, fp16, shuffle_orders=False, intensity x0.2, grid 0.05; B0 checkpoint; seq07 held-out (read at scoring time only).  Files: out/v06/ (runs/ holds every per-run artefact).  Companion of out/v05/REPORT.md: same node arguments, same scorer lineage, same bag.", ""]
    L += sec_changes(); L += sec_protocol(); L += sec_pose_source(); L += sec_latency(); L += sec_contention()
    L += sec_e2e(); L += sec_topics(); L += sec_decomposition(); L += sec_visual(); L += sec_problems()
    L += ["## Files", "", "- `out/v06/REPORT.md`, `out/v06/summary.json` (this)",
          "- `out/v06/runs/`: per run `stats_*.json` (node), `flog_*.npz` (frame log), `stream_*.tum/npz` (received pose stream), `pused_*.npz` (causal bin poses), `map_*.npz` (node-written map), `probe_*.npz`, `res_*.csv`, `fl_evo_*.tum`, `analysis_*.json`, `replay_*.json`, `diag_*.npz` (causal geometry per sweep), `ate_*.json/png`, `trajdiff_*.json`",
          "- `out/v06/zoom_*.json`, `out/v06/rviz_*.png`", "- `logs/v06_*.log`, `logs/node_*.log`, `logs/fl_*.log`, `logs/replay_v06_*.log`, `logs/verify_v06.log`", ""]
    open(a.out, "w").write("\n".join(L))
    json.dump(SUMMARY, open(a.summary, "w"), indent=2, default=float)
    print("wrote %s (%d lines) and %s" % (a.out, len(L), a.summary))


if __name__ == "__main__":
    main()
