#!/usr/bin/env python3
"""
opt/res_sampler_v06.py -- 1 Hz resource sampler for the v0.6 contention measurement.

Per named process (found by a cmdline pattern, re-scanned every 5 s so processes that
start later are picked up): CPU % over the last interval (psutil; > 100 means more than
one core), RSS in MB, thread count.  System: total CPU %, per-core CPU %.  GPU: the same
1 Hz `nvidia-smi --query-gpu` call opt/run.sh's sampler made in v0.5, so the perturbation
is the one the v0.5 numbers already contain.

CSV columns:
  t, sys_cpu, cores_busy(>80%), core_max, gpu_util, gpu_mem_mib,
  then for each of fl node worker bag probe blackboard: <name>_cpu, <name>_rss_mb, <name>_thr
(-1 when the process is not running).
"""
import argparse, csv, signal, subprocess, sys, time
import psutil

PATTERNS = [("fl", "fastlivo_mapping"), ("node", "semantic_map_node.py"),
            ("worker", "ptv3_worker.py"), ("bag", "bag play"),
            ("probe", "topic_probe_v06.py"), ("blackboard", "parameter_blackboard")]


def find(pattern, exclude_pid):
    """the match with the largest RSS: `ros2 run X` is a python wrapper whose cmdline also
    carries the binary name, and the wrapper is not the process we want."""
    best, best_rss = None, -1
    for p in psutil.process_iter(["pid", "cmdline", "memory_info"]):
        try:
            cl = " ".join(p.info["cmdline"] or [])
        except Exception:
            continue
        if pattern in cl and p.info["pid"] != exclude_pid and "res_sampler_v06" not in cl:
            rss = p.info["memory_info"].rss if p.info["memory_info"] else 0
            if rss > best_rss:
                best, best_rss = p.info["pid"], rss
    return best


def gpu():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True,
                             timeout=2.0).stdout.strip().split(",")
        return float(out[0]), float(out[1])
    except Exception:
        return -1.0, -1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--period", type=float, default=1.0)
    a = ap.parse_args()
    stop = [False]
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__(0, True))
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__(0, True))
    me = psutil.Process().pid
    procs = {}
    f = open(a.out, "w", newline="")
    w = csv.writer(f)
    w.writerow(["t", "sys_cpu", "cores_busy", "core_max", "gpu_util", "gpu_mem_mib"]
               + sum([[n + "_cpu", n + "_rss_mb", n + "_thr"] for n, _ in PATTERNS], []))
    psutil.cpu_percent(percpu=True)
    last_scan = 0.0
    while not stop[0]:
        t0 = time.time()
        if t0 - last_scan > 5.0:
            for name, pat in PATTERNS:
                p = procs.get(name)
                if p is None or not p.is_running():
                    pid = find(pat, me)
                    if pid is not None:
                        try:
                            pr = psutil.Process(pid)
                            pr.cpu_percent(None)
                            procs[name] = pr
                        except Exception:
                            procs.pop(name, None)
                    else:
                        procs.pop(name, None)
            last_scan = t0
        cores = psutil.cpu_percent(percpu=True)
        g = gpu()
        row = [round(t0, 3), round(sum(cores) / len(cores), 1), sum(c > 80 for c in cores),
               round(max(cores), 1), g[0], g[1]]
        for name, _ in PATTERNS:
            p = procs.get(name)
            if p is None:
                row += [-1, -1, -1]
                continue
            try:
                with p.oneshot():
                    row += [round(p.cpu_percent(None), 1), round(p.memory_info().rss / 1e6, 1),
                            p.num_threads()]
            except Exception:
                row += [-1, -1, -1]
                procs.pop(name, None)
        w.writerow(row)
        f.flush()
        dt = a.period - (time.time() - t0)
        if dt > 0:
            time.sleep(dt)
    f.close()


if __name__ == "__main__":
    main()
