#!/usr/bin/env python3
"""
ATE of a FAST-LIVO2 TUM trajectory against SemanticKITTI/KITTI-odometry ground truth.

Frames (F7 of the brief):
  poses.txt is expressed in the RECTIFIED cam0 frame (first row = identity).
  T_velo(i) = Tr^-1 @ T_cam(i) @ Tr ,  Tr = sequences/<seq>/calib.txt 'Tr'  (= T_{rect0 <- velo})

The FAST-LIVO2 evo file is T_{W <- IMU}  (LIVMapper.cpp:472 writes _state.pos_end /
_state.rot_end; LIVMapper.cpp:727 shows p_W = R_end*(extR*p_L + extT) + p_end, so the
state is the IMU pose and extR/extT = T_{IMU<-LiDAR}).  We therefore right-multiply:
      T_{W<-L}(t) = T_{W<-I}(t) @ T_{I<-L}
A left-multiplied Umeyama alignment CANNOT absorb this right-hand constant, so skipping
it would corrupt the ATE.

Time association: the GT is defined at the KITTI frame times (times.txt, = the camera
trigger).  The estimate is keyed on last_lio_update_time, which lands mid-frame.  At
14 m/s one frame is 1.46 m, so nearest-frame association alone would inject a ~0.7 m
systematic error.  We interpolate the GT (slerp + linear) onto the estimate timestamps.
"""
import sys, os, json
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

SEQ   = sys.argv[1] if len(sys.argv) > 1 else "04"
EST   = sys.argv[2] if len(sys.argv) > 2 else "/data/livo_sem/out/kitti_seq04_fastlivo2_tum.txt"
DRIVE = sys.argv[3] if len(sys.argv) > 3 else "/data/livo_sem/data/raw/2011_09_30/2011_09_30_drive_0016_sync"
TAG   = sys.argv[4] if len(sys.argv) > 4 else "seq04"
ODO   = f"/data/livo_sem/data/odometry/dataset/sequences/{SEQ}"
OUT   = "/data/livo_sem/out"

# ---- T_{IMU <- LiDAR}, identical to the yaml's extrin_calib.extrinsic_R/_T
T_I_L = np.array([
    [ 0.999997685, -0.000785403,  0.002024406,  0.810543972],
    [ 0.000755307,  0.999889850,  0.014824544, -0.307054372],
    [-0.002035826, -0.014822976,  0.999888022,  0.802723995],
    [ 0.0, 0.0, 0.0, 1.0]])

def parse_ts_file(path):
    import calendar
    out = []
    for line in open(path):
        line = line.strip()
        if not line: continue
        date, clock = line.split(" ")
        hms, frac = (clock.split(".") + ["0"])[:2]
        y, mo, d = (int(v) for v in date.split("-"))
        h, mi, s = (int(v) for v in hms.split(":"))
        out.append(calendar.timegm((y, mo, d, h, mi, s, 0, 0, 0)) * 1e9 + int((frac+"000000000")[:9]))
    return np.array(out) / 1e9

# ---------------------------------------------------------------- ground truth
Tr = None
for line in open(f"{ODO}/calib.txt"):
    if line.startswith("Tr:"):
        Tr = np.eye(4); Tr[:3, :4] = np.array([float(x) for x in line.split()[1:]]).reshape(3, 4)
Tr_inv = np.linalg.inv(Tr)
P = np.loadtxt(f"{ODO}/poses.txt").reshape(-1, 3, 4)
T_cam = np.tile(np.eye(4), (len(P), 1, 1)); T_cam[:, :3, :4] = P
T_gt = Tr_inv @ T_cam @ Tr                                   # velodyne frame  (F7)

# GT node times: anchor times.txt onto the raw drive's own image clock
t_rel = np.loadtxt(f"{ODO}/times.txt")
t_img = parse_ts_file(os.path.join(DRIVE, "image_02", "timestamps.txt"))[:len(t_rel)]
drift = np.abs((t_rel - t_rel[0]) - (t_img - t_img[0])).max()
t_gt = t_img                                                 # absolute, same clock as the bag
print(f"[gt ] {len(T_gt)} poses  |  times.txt vs image timestamps: max |drift| = {drift*1e3:.3f} ms")

# ---------------------------------------------------------------- estimate
E = np.loadtxt(EST)
if E.ndim == 1: E = E[None, :]
t_e, p_e, q_e = E[:, 0], E[:, 1:4], E[:, 4:8]                # tx ty tz qx qy qz qw
T_e = np.tile(np.eye(4), (len(E), 1, 1))
T_e[:, :3, :3] = R.from_quat(q_e).as_matrix()
T_e[:, :3, 3]  = p_e
T_est_L = T_e @ T_I_L                                        # T_{W<-L}
print(f"[est] {len(E)} poses  t = [{t_e[0]:.6f}, {t_e[-1]:.6f}]  span {t_e[-1]-t_e[0]:.3f} s")

# ---------------------------------------------------------------- associate (interpolate GT)
lo, hi = t_gt[0], t_gt[-1]
keep = (t_e >= lo) & (t_e <= hi)
if keep.sum() < len(t_e):
    print(f"[ass] {len(t_e)-keep.sum()} estimate poses outside the GT time span -> dropped")
t_e, T_est_L = t_e[keep], T_est_L[keep]
slerp = Slerp(t_gt, R.from_matrix(T_gt[:, :3, :3]))
gt_R = slerp(t_e).as_matrix()
gt_t = np.stack([np.interp(t_e, t_gt, T_gt[:, i, 3]) for i in range(3)], axis=1)
T_gt_i = np.tile(np.eye(4), (len(t_e), 1, 1)); T_gt_i[:, :3, :3] = gt_R; T_gt_i[:, :3, 3] = gt_t
print(f"[ass] {len(t_e)} matched pairs")

def path_len(x): return float(np.linalg.norm(np.diff(x, axis=0), axis=1).sum())

def umeyama(src, dst):
    """global SE(3), no scale.  src,dst : N x 3"""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    H = (src - mu_s).T @ (dst - mu_d) / len(src)
    U, S, Vt = np.linalg.svd(H)
    D = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0: D[2, 2] = -1
    Rm = Vt.T @ D @ U.T
    return Rm, mu_d - Rm @ mu_s

est_p, gt_p = T_est_L[:, :3, 3], T_gt_i[:, :3, 3]
Rm, tm = umeyama(est_p, gt_p)
aln_p = (Rm @ est_p.T).T + tm
err = np.linalg.norm(aln_p - gt_p, axis=1)

stats = dict(
    seq=SEQ, n_pairs=int(len(err)),
    ate_rmse=float(np.sqrt((err**2).mean())), ate_mean=float(err.mean()),
    ate_median=float(np.median(err)), ate_max=float(err.max()), ate_min=float(err.min()),
    traj_len_gt=path_len(gt_p), traj_len_est=path_len(est_p),
    gt_duration=float(t_e[-1]-t_e[0]),
    mean_speed_gt=path_len(gt_p)/max(1e-9, t_e[-1]-t_e[0]),
    mean_speed_est=path_len(est_p)/max(1e-9, t_e[-1]-t_e[0]))
print("\n================ ATE (single global SE(3) Umeyama, no scale) ================")
for k in ("n_pairs","ate_rmse","ate_mean","ate_median","ate_max","ate_min"):
    print("  %-12s %s" % (k, ("%.4f m" % stats[k]) if k.startswith("ate") else stats[k]))
print("  trajectory length   GT  %8.2f m   EST %8.2f m   (ratio %.3f)"
      % (stats["traj_len_gt"], stats["traj_len_est"],
         stats["traj_len_est"]/max(1e-9, stats["traj_len_gt"])))
print("  mean speed          GT  %8.2f m/s EST %8.2f m/s" % (stats["mean_speed_gt"], stats["mean_speed_est"]))
print("  duration            %.2f s" % stats["gt_duration"])

json.dump(stats, open(f"{OUT}/{TAG}_ate.json", "w"), indent=2)
np.savetxt(f"{OUT}/{TAG}_aligned_est_velo.txt",
           np.column_stack([t_e, aln_p]), fmt="%.6f",
           header="t x y z  (estimate, LiDAR frame, Umeyama-aligned to GT)")

# ---------------------------------------------------------------- plot
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(15, 6))
ax[0].plot(gt_p[:, 0], gt_p[:, 1], "k-", lw=2.2, label="ground truth (velodyne frame)")
ax[0].plot(aln_p[:, 0], aln_p[:, 1], "r-", lw=1.6, label="FAST-LIVO2 (aligned)")
ax[0].scatter(gt_p[0, 0], gt_p[0, 1], c="g", s=70, zorder=5, label="start")
ax[0].set_xlabel("x [m]  (forward)"); ax[0].set_ylabel("y [m]  (left)")
ax[0].set_title("KITTI seq %s  top-down XY\nATE RMSE %.3f m   len GT %.1f m / EST %.1f m"
                % (SEQ, stats["ate_rmse"], stats["traj_len_gt"], stats["traj_len_est"]))
ax[0].axis("equal"); ax[0].grid(alpha=.3); ax[0].legend()
ax[1].plot(t_e - t_e[0], err, "r-", lw=1.4)
ax[1].set_xlabel("t [s]"); ax[1].set_ylabel("|position error| [m]")
ax[1].set_title("per-pose ATE after global alignment"); ax[1].grid(alpha=.3)
fig.tight_layout()
png = f"{OUT}/{TAG}_traj.png"
fig.savefig(png, dpi=130)
print("\n  plot -> %s" % png)
print("  json -> %s/%s_ate.json" % (OUT, TAG))
