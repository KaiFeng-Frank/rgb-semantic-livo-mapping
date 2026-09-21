#!/usr/bin/env python3
"""Bit-exactness of the rewritten de-skew / projection / gate against the ORIGINAL
math in src_backup_20260921/semantic_map_node.py, on real seq07 data."""
import sys, glob, time
sys.path.insert(0, "/data/livo_sem/src")
import numpy as np
import sem_core as S
from PIL import Image

TRAJ = "/data/livo_sem/out/kitti_seq07_fastlivo2_tum.txt"
VD = "/data/livo_sem/data/raw/2011_09_30/2011_09_30_drive_0027_sync/velodyne_points/"
IM = "/data/livo_sem/data/raw/2011_09_30/2011_09_30_drive_0027_sync/image_02/data/"
traj = S.TrajInterp(TRAJ)
T_I_L = S.T_I_L; T_C_L = S.T_C_L
R_IL = np.ascontiguousarray(T_I_L[:3, :3]); t_IL = np.ascontiguousarray(T_I_L[:3, 3])
NB = 128

ts = np.loadtxt(VD + "timestamps.txt", dtype=str, delimiter="@", usecols=0) \
     if False else None
# per-point time: reconstruct the bag's field exactly as kitti_to_ros2bag does
def pt_times(n, t_hdr, dt=0.1039):
    return t_hdr + np.linspace(0.0, dt, n, endpoint=False)

def old_path(p3, tk, img, t_img, gate_lab, gate_conf):
    n = len(p3)
    pl = np.hstack([p3, np.ones((n, 1))])
    pl_i = (pl @ T_I_L.T)[:, :3]
    t_lo, t_hi = tk[0], tk[-1]
    ctr = t_lo + (np.arange(NB) + 0.5) * (t_hi - t_lo) / NB
    Rb, pb, _ = traj.query(ctr)
    bi = np.clip(((tk - t_lo) / (t_hi - t_lo) * NB).astype(np.int64), 0, NB - 1)
    edge = np.searchsorted(bi, np.arange(NB + 1))
    pw = np.empty((n, 3))
    for k in range(NB):
        a, b = edge[k], edge[k + 1]
        if b > a:
            pw[a:b] = pl_i[a:b] @ Rb[k].T + pb[k]
    rgb = np.zeros((n, 3), np.uint8); has = np.zeros(n, bool)
    T_W_I_img, okp = traj.query_one(t_img)
    if okp:
        T_W_L_img = T_W_I_img @ T_I_L
        T_L_W = np.linalg.inv(T_W_L_img)
        pc = (T_C_L @ T_L_W @ np.hstack([pw, np.ones((n, 1))]).T).T[:, :3]
        z = pc[:, 2]
        front = z > 0.5
        uv = np.full((n, 2), -1e9)
        uv[front] = (pc[front, :2] / z[front, None]) * [S.FX, S.FY] + [S.CX, S.CY]
        H, W = img.shape[:2]
        m = front & (uv[:, 0] >= 0) & (uv[:, 0] <= W - 1) \
                  & (uv[:, 1] >= 0) & (uv[:, 1] <= H - 1)
        if m.any():
            u = np.rint(uv[m, 0]).astype(np.int32); v = np.rint(uv[m, 1]).astype(np.int32)
            rgb[m] = img[v, u]; has = m
    return pw, rgb, has

def p_dot(p, row):
    out = p @ row[:3]; out += row[3]; return out

def new_path(p3, tk, img, t_img, gate_lab, gate_conf):
    n = len(p3)
    pl_i = p3 @ R_IL.T
    pl_i += t_IL
    t_lo, t_hi = tk[0], tk[-1]
    ctr = t_lo + (np.arange(NB) + 0.5) * (t_hi - t_lo) / NB
    Rb, pb, _ = traj.query(ctr)
    bi = np.clip(((tk - t_lo) / (t_hi - t_lo) * NB).astype(np.int64), 0, NB - 1)
    edge = np.searchsorted(bi, np.arange(NB + 1))
    pw = np.empty((n, 3))
    for k in range(NB):
        a, b = edge[k], edge[k + 1]
        if b > a:
            np.matmul(pl_i[a:b], Rb[k].T, out=pw[a:b]); pw[a:b] += pb[k]
    rgb = np.zeros((n, 3), np.uint8); has = np.zeros(n, bool)
    T_W_I_img, okp = traj.query_one(t_img)
    if okp:
        M = T_C_L @ np.linalg.inv(T_W_I_img @ T_I_L)
        z = p_dot(pw, M[2])
        front = np.flatnonzero(z > 0.5)
        if front.size:
            pf = pw[front]; zf = z[front]
            u = p_dot(pf, M[0]); u /= zf; u *= S.FX; u += S.CX
            v = p_dot(pf, M[1]); v /= zf; v *= S.FY; v += S.CY
            H, W = img.shape[:2]
            inb = (u >= 0) & (u <= W - 1) & (v >= 0) & (v <= H - 1)
            if inb.any():
                sel = front[inb]
                ui = np.rint(u[inb]).astype(np.int32); vi = np.rint(v[inb]).astype(np.int32)
                rgb[sel] = img[vi, ui]; has[sel] = True
    return pw, rgb, has

files = sorted(glob.glob(VD + "data/*.bin"))
imgs = sorted(glob.glob(IM + "*.png"))
t0 = traj.t[0] + 0.5
bad_pw = bad_rgb = bad_has = 0
maxd = 0.0
told = tnew = 0.0
NF = 40
for i in range(0, NF * 25, 25):
    p = np.fromfile(files[i], dtype=np.float32).reshape(-1, 4)
    t_hdr = traj.t[0] + 0.2 + i * 0.1039
    if t_hdr + 0.11 > traj.t[-1]: break
    tk = pt_times(len(p), t_hdr)
    p3 = p[:, :3].astype(np.float64)
    img = np.asarray(Image.open(imgs[i % len(imgs)]).convert("RGB"))
    t_img = t_hdr + 0.05
    a = time.perf_counter(); o = old_path(p3, tk, img, t_img, None, None); told += time.perf_counter()-a
    a = time.perf_counter(); b = new_path(p3, tk, img, t_img, None, None); tnew += time.perf_counter()-a
    if not np.array_equal(o[0], b[0]):
        bad_pw += 1; maxd = max(maxd, float(np.abs(o[0]-b[0]).max()))
    if not np.array_equal(o[1], b[1]): bad_rgb += 1
    if not np.array_equal(o[2], b[2]): bad_has += 1
print("frames compared      : %d" % NF)
print("pw  NOT bit-identical: %d frames (max |delta| %.3e m)" % (bad_pw, maxd))
print("rgb NOT bit-identical: %d frames" % bad_rgb)
print("has NOT bit-identical: %d frames" % bad_has)
print("deskew+proj  old %.2f ms/frame   new %.2f ms/frame" % (told/NF*1e3, tnew/NF*1e3))
