#!/usr/bin/env python3
"""measure_overlap.py -- of the voxels a moving object contaminates, how many
also hold genuine static returns?  Those cannot be emptied; the rest are pure
ribbon and vanish cleanly when the moving points are never inserted.  CPU only."""
import calendar, json, os, sys
import numpy as np
sys.path.insert(0, "/data/livo_sem/src")
import sem_core as S
from kitti_scan import synth
RAW = "/data/livo_sem/data/raw/2011_09_30/2011_09_30_drive_0027_sync"
LB = "/data/livo_sem/data/odometry/dataset/sequences/07/labels"
VOX = 0.20


def parse_ts(p):
    o = []
    for line in open(p):
        line = line.strip()
        if not line:
            continue
        d_, c_ = line.split(" ")
        hms, frac = (c_.split(".") + ["0"])[:2]
        y, mo, d = (int(v) for v in d_.split("-"))
        h, mi, s = (int(v) for v in hms.split(":"))
        o.append(calendar.timegm((y, mo, d, h, mi, s, 0, 0, 0)) * 10**9 + int((frac + "000000000")[:9]))
    return np.array(o, dtype=np.int64)


ts, te = parse_ts(RAW + "/velodyne_points/timestamps_start.txt"), parse_ts(RAW + "/velodyne_points/timestamps_end.txt")
traj = S.TrajInterp("/data/livo_sem/out/kitti_seq07_fastlivo2_tum.txt")
R_IL, t_IL = S.T_I_L[:3, :3], S.T_I_L[:3, 3]
stat_keys, mv_keys = [], []
for f in range(1101):
    sp = RAW + "/velodyne_points/data/%010d.bin" % f
    lp = LB + "/%06d.label" % f
    if not (os.path.exists(sp) and os.path.exists(lp)):
        continue
    p = np.fromfile(sp, dtype=np.float32).reshape(-1, 4)
    lab = np.fromfile(lp, dtype=np.uint32)
    if len(p) != len(lab):
        continue
    sem = (lab & 0xFFFF).astype(np.int32)
    _, tsyn, order = synth(p[:, :3], sweep_duration=(te[f] - ts[f]) / 1e9)
    t_pt = np.empty(len(p))
    t_pt[order] = tsyn
    t_pt += ts[f] * 1e-9
    R, pp, ok = traj.query(t_pt)
    if ok.sum() < 100:
        continue
    pw = np.einsum("nij,nj->ni", R[ok], p[ok, :3].astype(np.float64) @ R_IL.T + t_IL) + pp[ok]
    k = S.voxel_key(pw, VOX)
    s = sem[ok]
    mv = (s >= 252) & (s <= 259)
    ign = (s == 0) | (s == 1) | (s == 99)
    stat_keys.append(np.unique(k[~mv & ~ign]))
    if mv.any():
        mv_keys.append(np.unique(k[mv]))
    if f % 200 == 0:
        print("  frame", f, flush=True)
SK = np.unique(np.concatenate(stat_keys))
MK = np.unique(np.concatenate(mv_keys))
both = np.isin(MK, SK, assume_unique=True)
res = dict(voxel=VOX, static_voxels=int(len(SK)), moving_voxels=int(len(MK)),
           moving_voxels_shared_with_static=int(both.sum()),
           moving_voxels_pure=int((~both).sum()),
           frac_moving_voxels_pure=float((~both).mean()),
           union_voxels=int(len(np.union1d(SK, MK))),
           frac_of_union_that_is_pure_ribbon=float((~both).sum() / len(np.union1d(SK, MK))))
json.dump(res, open("/data/livo_sem/out/dyn_overlap.json", "w"), indent=2)
print(json.dumps(res, indent=2))
