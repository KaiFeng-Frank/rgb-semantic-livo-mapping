#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/v06_analyze.py -- per-run measurement extraction for v0.6.  Every number comes from a
file a run produced; nothing is typed in.

  stats_<tag>.json   node counters and in-node stage timings (semantic_map_node --stats-out)
  flog_<tag>.npz     per received sweep: header stamp, sweep end, wall arrival, status;
                     per processed sweep: stage-B wall start/end, newest pose at query,
                     extrapolated span, samples in sweep, pose wait  (--frame-log)
  stream_<tag>.npz   every pose sample the node received + wall arrival  (--pose-record-npz)
  probe_<tag>.npz    subscriber-side arrivals of /semantic_scan and /semantic_map
  res_<tag>.csv      1 Hz psutil / nvidia-smi samples (opt/res_sampler_v06.py)
  fl_<tag>.log       FAST-LIVO2 stdout: per-frame LIO / VIO time tables
  fl_evo_<tag>.tum   FAST-LIVO2's own evo file of the run (pose_output_en)

usage: python3 tools/v06_analyze.py --tag onopt_1 [--ate]   -> out/v06/runs/analysis_<tag>.json
"""
import argparse, csv, json, os, re, subprocess, sys
import numpy as np

B = "/data/livo_sem"
R = B + "/out/v06/runs"
L = B + "/logs"


def dist(v, scale=1.0, extra=()):
    v = np.asarray(v, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return dict(n=0)
    d = dict(n=int(v.size), p50=float(np.percentile(v, 50) * scale), p95=float(np.percentile(v, 95) * scale),
             p99=float(np.percentile(v, 99) * scale), max=float(v.max() * scale), mean=float(v.mean() * scale),
             min=float(v.min() * scale))
    for e in extra:
        d["p%g" % e] = float(np.percentile(v, e) * scale)
    return d


def join_by_stamp(a, b, tol=1e-4):
    """indices into a and b of entries whose stamps agree within tol (both sorted-unique on a)."""
    a = np.asarray(a); b = np.asarray(b)
    o = np.argsort(a); asrt = a[o]
    j = np.searchsorted(asrt, b)
    j = np.clip(j, 0, len(asrt) - 1)
    jm = np.clip(j - 1, 0, len(asrt) - 1)
    d1 = np.abs(asrt[j] - b); d0 = np.abs(asrt[jm] - b)
    pick = np.where(d0 < d1, jm, j)
    ok = np.minimum(d0, d1) < tol
    return o[pick[ok]], np.flatnonzero(ok)


def analyze_flog(tag, stream):
    p = R + "/flog_%s.npz" % tag
    if not os.path.exists(p):
        return None
    z = np.load(p)
    out = {}
    hdr, tend, wall, status = z["recv_hdr"], z["recv_tend"], z["recv_wall"], z["recv_status"]
    n = len(hdr)
    out["received"] = int(n)
    out["status_counts"] = {"queued": int((status == 0).sum()), "bp_drop": int((status == 1).sum()),
                            "no_pose": int((status == 2).sum())}
    out["stale_drops"] = int(len(z["stale_hdr"]))
    # bag delivery: the player publishes a sweep at its recording time = sweep end; the wall - bag
    # offset is constant at rate 1.0, so (wall - tend) minus its median is the delivery jitter
    if n > 2:
        off = wall - tend
        med = float(np.median(off))
        out["wall_minus_bagtime_median_s"] = med
        out["delivery_jitter_ms"] = dist(off - med, 1e3)
        out["sweep_interarrival_ms"] = dist(np.diff(wall), 1e3)
        out["sweeps_missing_vs_stamp_count"] = int(round((hdr[-1] - hdr[0]) / 0.1039)) + 1 - n
    # in-node end-to-end: sweep arrival -> stage-B done, per processed sweep
    ph, b0, b1 = z["proc_hdr"], z["proc_b0"], z["proc_b1"]
    ia, ib = join_by_stamp(hdr, ph)
    out["processed"] = int(len(ph))
    if len(ia):
        e2e = b1[ib] - wall[ia]
        out["in_node_latency_ms"] = dist(e2e, 1e3)
        out["stageB_start_after_arrival_ms"] = dist(b0[ib] - wall[ia], 1e3)
    span = z["proc_span"]; tl = z["proc_tlast"]; npose = z["proc_npose"]
    if np.isfinite(tl).any():
        out["extrap_span_ms"] = dist(span, 1e3)
        out["frac_sweeps_extrapolated"] = float(np.mean(span > 0))
        out["samples_in_sweep"] = dist(npose)
        out["pose_wait_ms"] = dist(z["proc_pwait"])
        # where in the sweep did the newest sample sit at query time?  (0 = sweep start, 1 = end)
        te_p = tend[ia]; hs = hdr[ia]
        frac = (tl[ib] - hs) / np.maximum(te_p - hs, 1e-6)
        out["newest_sample_position_in_sweep"] = dist(np.clip(frac, -1, 2))
    # pose arrival latency relative to the sweep's own arrival
    if stream is not None and n > 2:
        st, sw = stream["t"], stream["wall"]
        med = out.get("wall_minus_bagtime_median_s", 0.0)
        # (a) the LIO update INSIDE the sweep (stamped at its image instant): first sample with
        #     hdr < t <= tend  -> arrival(wall) - sweep arrival(wall)
        q = status == 0                      # sweeps the node actually queued (not the no-pose start-up)
        j = np.searchsorted(st, hdr, side="right")
        has = (j < len(st)) & (st[np.minimum(j, len(st) - 1)] <= tend) & q
        lat_in = sw[np.minimum(j, len(st) - 1)][has] - wall[has]
        out["pose_in_sweep_arrival_after_sweep_ms"] = dist(lat_in, 1e3)
        out["pose_in_sweep_missing_frac_of_queued"] = float(1.0 - has[q].mean()) if q.any() else None
        # (b) first sample covering the sweep END (what a wait-for-coverage policy waits for)
        k = np.searchsorted(st, tend, side="left")
        hk = (k < len(st)) & q
        lat_cov = sw[np.minimum(k, len(st) - 1)][hk] - wall[hk]
        out["pose_covering_sweep_end_arrival_after_sweep_ms"] = dist(lat_cov, 1e3)
        # (c) sample age at arrival, in bag time: (wall - offset) - stamp  (offset from the LiDAR
        #     deliveries; contains FAST-LIVO2's own processing + queueing + DDS)
        age = (sw - med) - st
        out["pose_age_at_arrival_ms"] = dist(age, 1e3)
        out["pose_stream_interarrival_ms"] = dist(np.diff(sw), 1e3)
        out["pose_stream_stamp_spacing_ms"] = dist(np.diff(st), 1e3)
        out["pose_samples"] = int(len(st))
    return out


def analyze_probe(tag, flog_tag=None):
    p = R + "/probe_%s.npz" % tag
    if not os.path.exists(p):
        return None
    z = np.load(p)
    sc, mp = z["scan"], z["map"]
    out = dict(scan_received=int(len(sc)), map_received=int(len(mp)))
    if len(sc) > 2:
        out["scan_interarrival_ms"] = dist(np.diff(sc[:, 1]), 1e3, extra=(99.9,))
        out["scan_points"] = dist(sc[:, 2])
        out["scan_bytes_mean"] = float(sc[:, 3].mean())
        out["scan_unique_stamps"] = int(len(np.unique(np.round(sc[:, 0], 4))))
    if len(mp) > 2:
        # the last map message is the final snapshot published by finish() after the idle
        # timeout; the interval before it measures the timeout, not the publisher
        out["map_interarrival_ms"] = dist(np.diff(mp[:-1, 1]), 1e3)
        out["map_final_after_previous_s"] = float(mp[-1, 1] - mp[-2, 1])
        out["map_points"] = dist(mp[:, 2])
        out["map_bytes_max"] = float(mp[:, 3].max())
    fp = R + "/flog_%s.npz" % (flog_tag or tag)
    if os.path.exists(fp) and len(sc):
        f = np.load(fp)
        ia, ib = join_by_stamp(f["recv_hdr"], sc[:, 0])
        if len(ia):
            out["external_latency_ms"] = dist(sc[ib, 1] - f["recv_wall"][ia], 1e3)
            out["scan_matched_to_received_sweeps"] = int(len(ia))
        ph = f["proc_hdr"]
        pa, pb = join_by_stamp(ph, sc[:, 0])
        out["scan_delivery_frac_of_processed"] = float(len(pa) / max(1, len(ph)))
        out["scan_msgs_lost_in_transport"] = int(len(ph) - len(pa))
    return out


def analyze_res(tag):
    p = R + "/res_%s.csv" % tag
    if not os.path.exists(p):
        return None
    rows = list(csv.DictReader(open(p)))
    if not rows:
        return None
    A = {k: np.array([float(r[k]) for r in rows]) for k in rows[0].keys()}
    active = A["bag_cpu"] >= 0            # the bag player is alive = data is flowing
    if active.sum() < 5:
        active = np.ones(len(rows), bool)
    out = dict(samples=int(len(rows)), active_samples=int(active.sum()))
    for k in ("sys_cpu", "cores_busy", "core_max", "gpu_util", "gpu_mem_mib"):
        out[k] = dist(A[k][active])
    for name in ("fl", "node", "worker", "bag", "probe", "blackboard"):
        c = A[name + "_cpu"]; m = A[name + "_rss_mb"]; t = A[name + "_thr"]
        pres = active & (c >= 0)
        if pres.sum() < 3:
            continue
        out[name] = dict(cpu=dist(c[pres]), rss_mb_max=float(m[pres].max()), threads_max=int(t[pres].max()),
                         present_samples=int(pres.sum()))
    return out


def parse_fl_log(tag):
    p = L + "/fl_%s.log" % tag
    if not os.path.exists(p):
        return None
    lio, vio = [], []
    cur = None
    ansi = re.compile(r"\x1b\[[0-9;]*m")
    n_throw = n_lost = 0
    for line in open(p, errors="replace"):
        s = ansi.sub("", line)
        if "LIO Mapping Time" in s:
            cur = lio
        elif "VIO Time" in s:
            cur = vio
        elif "Current Total Time" in s and cur is not None:
            m = re.search(r"\|\s*([0-9.]+)\s*\|", s.split("Current Total Time")[1])
            if m:
                cur.append(float(m.group(1)))
        elif "Throw one image" in s:
            n_throw += 1
        elif "lost" in s.lower():
            n_lost += 1
    return dict(lio_frames=len(lio), vio_frames=len(vio), lio_ms=dist(lio, 1e3), vio_ms=dist(vio, 1e3),
                lio_plus_vio_ms=dist(np.asarray(lio[:min(len(lio), len(vio))]) + np.asarray(vio[:min(len(lio), len(vio))]), 1e3)
                if lio and vio else None,
                images_thrown=n_throw, lost_lines=n_lost,
                lio_over_104ms=int(np.sum(np.asarray(lio) > 0.104)) if lio else 0)


def ate(tum, tag):
    if not os.path.exists(tum) or os.path.getsize(tum) == 0:
        return None
    r = subprocess.run(["/usr/bin/python3", B + "/src/eval_fastlivo2_ate.py", "07", tum,
                        B + "/data/raw/2011_09_30/2011_09_30_drive_0027_sync", "v06_" + tag],
                       capture_output=True, text=True)
    jp = B + "/out/v06_%s_ate.json" % tag
    if not os.path.exists(jp):
        return dict(error=r.stderr[-500:])
    d = json.load(open(jp))
    for ext in ("_ate.json", "_traj.png", "_aligned_est_velo.txt"):
        src = B + "/out/v06_%s%s" % (tag, ext)
        if os.path.exists(src):
            os.replace(src, R + "/ate_%s%s" % (tag, ext))
    d["drift_pct"] = 100.0 * d["ate_rmse"] / d["traj_len_gt"]
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--ate", action="store_true")
    a = ap.parse_args()
    tag = a.tag
    out = dict(tag=tag)
    sp = R + "/stats_%s.json" % tag
    if os.path.exists(sp):
        out["stats"] = json.load(open(sp))
    stream = None
    if os.path.exists(R + "/stream_%s.npz" % tag):
        z = np.load(R + "/stream_%s.npz" % tag)
        stream = dict(t=z["t"], wall=z["wall"])
    out["frames"] = analyze_flog(tag, stream)
    out["probe"] = analyze_probe(tag)
    out["resources"] = analyze_res(tag)
    out["fastlivo2"] = parse_fl_log(tag)
    if a.ate:
        if os.path.exists(R + "/fl_evo_%s.tum" % tag):
            out["ate_fl_evo"] = ate(R + "/fl_evo_%s.tum" % tag, tag + "_evo")
        if os.path.exists(R + "/stream_%s.tum" % tag):
            out["ate_stream"] = ate(R + "/stream_%s.tum" % tag, tag + "_stream")
    jp = R + "/analysis_%s.json" % tag
    json.dump(out, open(jp, "w"), indent=2, default=float)
    # one-line digest
    f = out.get("frames") or {}
    pr = out.get("probe") or {}
    rs = out.get("resources") or {}
    fl = out.get("fastlivo2") or {}
    msg = ["[%s]" % tag]
    if out.get("stats"):
        s = out["stats"]
        msg.append("recv %d proc %d bp %d nopose %d" % (s["scans_received"], s["scans_processed"],
                                                       s["scans_dropped_backpressure"], s["scans_no_pose"]))
    if "in_node_latency_ms" in f:
        d = f["in_node_latency_ms"]; msg.append("in-node lat p50/p95/p99/max %.0f/%.0f/%.0f/%.0f ms" % (d["p50"], d["p95"], d["p99"], d["max"]))
    if "pose_in_sweep_arrival_after_sweep_ms" in f:
        d = f["pose_in_sweep_arrival_after_sweep_ms"]; msg.append("pose(in-sweep) arrival p50/p95/p99/max %.0f/%.0f/%.0f/%.0f ms" % (d["p50"], d["p95"], d["p99"], d["max"]))
    if "extrap_span_ms" in f:
        d = f["extrap_span_ms"]; msg.append("extrap span p50/p95/max %.0f/%.0f/%.0f ms" % (d["p50"], d["p95"], d["max"]))
    if "scan_interarrival_ms" in pr:
        d = pr["scan_interarrival_ms"]; msg.append("scan interarrival p50/p95/p99/max %.0f/%.0f/%.0f/%.0f ms, delivered %s/%s" % (
            d["p50"], d["p95"], d["p99"], d["max"], pr.get("scan_received"), (out.get("stats") or {}).get("scans_processed")))
    if "map_interarrival_ms" in pr:
        d = pr["map_interarrival_ms"]; msg.append("map msgs %d interarrival p50/p95/max %.0f/%.0f/%.0f ms" % (pr["map_received"], d["p50"], d["p95"], d["max"]))
    for name in ("fl", "node", "worker", "probe", "bag"):
        if name in rs:
            msg.append("%s cpu mean/p95 %.0f/%.0f%% rss %.0f MB" % (name, rs[name]["cpu"]["mean"], rs[name]["cpu"]["p95"], rs[name]["rss_mb_max"]))
    if "sys_cpu" in rs:
        msg.append("sys cpu %.0f%% gpu %.0f%%" % (rs["sys_cpu"]["mean"], rs["gpu_util"]["mean"]))
    if fl.get("lio_frames"):
        msg.append("FL lio p50/p95/max %.1f/%.1f/%.1f ms (%d frames, %d > 104 ms) vio p50 %.1f" % (
            fl["lio_ms"]["p50"], fl["lio_ms"]["p95"], fl["lio_ms"]["max"], fl["lio_frames"], fl["lio_over_104ms"], fl["vio_ms"]["p50"]))
    for k in ("ate_fl_evo", "ate_stream"):
        if out.get(k) and "ate_rmse" in out[k]:
            msg.append("%s ATE %.3f m (%d pairs, drift %.3f%%)" % (k, out[k]["ate_rmse"], out[k]["n_pairs"], out[k]["drift_pct"]))
    print(" | ".join(msg))
    print("-> %s" % jp)


if __name__ == "__main__":
    main()
