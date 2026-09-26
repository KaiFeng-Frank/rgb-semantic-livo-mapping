#!/usr/bin/env python3
"""Score FAST-LIVO2 reproduction runs against GT and against the committed trajectories.

  usage: tools/score_fastlivo2_repro.py RUN_DIR OUT_JSON

RUN_DIR holds kitti_seq07_run*.txt / kitti_seq04_run*.txt (TUM, T_{W<-IMU}).
Each run and each committed results/kitti_seq0?_fastlivo2_tum.txt is scored with the
unchanged src/eval_fastlivo2_ate.py, so the GT is whatever sequences/<seq>/poses.txt
holds on this host. drift = ATE RMSE / GT path length, as in the handoff table.
The run-vs-committed comparison needs no GT: both trajectories are in the same frame
(W = IMU at initialisation) and keyed on the same sensor stamps.
"""
import glob, json, os, subprocess, sys
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = "/data/livo_sem/data/raw/2011_09_30"
OUT = "/data/livo_sem/out"
SEQS = {"07": "2011_09_30_drive_0027_sync", "04": "2011_09_30_drive_0016_sync"}
run_dir, out_json = sys.argv[1], sys.argv[2]


def score(seq, traj, tag):
    subprocess.run([sys.executable, f"{REPO}/src/eval_fastlivo2_ate.py", seq, traj,
                    f"{RAW}/{SEQS[seq]}", tag], check=True, stdout=subprocess.DEVNULL)
    s = json.load(open(f"{OUT}/{tag}_ate.json"))
    s["drift_pct"] = 100 * s["ate_rmse"] / s["traj_len_gt"]
    return s


def versus(ref, run):
    a, b = np.loadtxt(ref), np.loadtxt(run)
    same_stamps = len(a) == len(b) and bool(np.all(a[:, 0] == b[:, 0]))
    out = dict(n_ref=len(a), n_run=len(b), same_stamps=same_stamps)
    if same_stamps:
        d = np.linalg.norm(a[:, 1:4] - b[:, 1:4], axis=1)
        out.update(first_diverging_pose_1mm=int(np.argmax(d > 1e-3)) if (d > 1e-3).any() else None,
                   pos_diff_max=float(d.max()), pos_diff_final=float(d[-1]))
    return out


summary = {}
for seq in SEQS:
    committed = f"{REPO}/results/kitti_seq{seq}_fastlivo2_tum.txt"
    entry = {"committed": score(seq, committed, f"seq{seq}_committed"), "runs": {}}
    for run in sorted(glob.glob(f"{run_dir}/kitti_seq{seq}_run*.txt")):
        name = os.path.basename(run)[:-4]
        entry["runs"][name] = {"ate": score(seq, run, f"seq{seq}_{name}"),
                               "vs_committed": versus(committed, run)}
    r = [v["ate"]["ate_rmse"] for v in entry["runs"].values()]
    if r:
        entry["run_ate_rmse_mean"], entry["run_ate_rmse_sd"] = float(np.mean(r)), float(np.std(r, ddof=1)) if len(r) > 1 else 0.0
    summary[seq] = entry

json.dump(summary, open(out_json, "w"), indent=2)
for seq, e in summary.items():
    c = e["committed"]
    print(f"seq{seq}  committed: ATE {c['ate_rmse']:.4f} m  drift {c['drift_pct']:.3f} %  "
          f"GT {c['traj_len_gt']:.2f} m  EST {c['traj_len_est']:.2f} m")
    for n, v in e["runs"].items():
        a, d = v["ate"], v["vs_committed"]
        print(f"  {n:18s} ATE {a['ate_rmse']:.4f} m  drift {a['drift_pct']:.3f} %  EST {a['traj_len_est']:.2f} m  "
              f"poses {d['n_run']}  same stamps {d['same_stamps']}  "
              f"max |p_run - p_committed| {d.get('pos_diff_max', float('nan')):.3f} m")
