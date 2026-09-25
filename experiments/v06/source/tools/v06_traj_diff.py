#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/v06_traj_diff.py -- how far apart are two trajectories of the SAME sequence in the SAME
world frame (FAST-LIVO2's W = IMU frame at init, no alignment applied)?  Used to say whether an
online map and the offline map are voxel-comparable at all: the semantic lookup joins by 0.2 m
voxel key, which is meaningless where the two trajectories are further apart than that.

  --a A.tum --b B.tum   -> B interpolated at A's stamps; |dp| and rotation angle distributions,
                           the fraction of stamps with |dp| < 0.1 / 0.2 / 0.5 m, drift vs time.
"""
import argparse, json, sys
import numpy as np
sys.path.insert(0, "/data/livo_sem/src")
import sem_core as S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args()
    A = np.loadtxt(a.a); Bt = S.TrajInterp(a.b)
    t = A[:, 0]; ok = Bt.valid(t)
    t = t[ok]; pa = A[ok, 1:4]; qa = A[ok, 4:8]
    Rb, pb, _ = Bt.query(t)
    Ra = S.quat_to_R(qa / np.linalg.norm(qa, axis=1, keepdims=True))
    dp = np.linalg.norm(pa - pb, axis=1)
    c = (np.einsum("nii->n", np.einsum("nij,nkj->nik", Ra, Rb)) - 1.0) * 0.5
    ang = np.degrees(np.arccos(np.clip(c, -1, 1)))
    el = t - t[0]
    out = dict(a=a.a, b=a.b, n=int(len(t)),
               dp_m=dict(p50=float(np.percentile(dp, 50)), p95=float(np.percentile(dp, 95)),
                         max=float(dp.max()), mean=float(dp.mean()), final=float(dp[-1])),
               rot_deg=dict(p50=float(np.percentile(ang, 50)), p95=float(np.percentile(ang, 95)), max=float(ang.max())),
               frac_within=dict(m0_1=float(np.mean(dp < 0.1)), m0_2=float(np.mean(dp < 0.2)), m0_5=float(np.mean(dp < 0.5))),
               dp_by_quarter=[float(dp[(el >= q * el[-1] / 4) & (el < (q + 1) * el[-1] / 4 + 1e-6)].mean()) for q in range(4)])
    print("%s vs %s: %d stamps | |dp| p50 %.3f p95 %.3f max %.3f m (final %.3f) | rot p50 %.3f p95 %.3f deg | within 0.1/0.2/0.5 m: %.1f/%.1f/%.1f %% | by quarter %s"
          % (a.a.split("/")[-1], a.b.split("/")[-1], out["n"], out["dp_m"]["p50"], out["dp_m"]["p95"], out["dp_m"]["max"], out["dp_m"]["final"],
             out["rot_deg"]["p50"], out["rot_deg"]["p95"], 100 * out["frac_within"]["m0_1"], 100 * out["frac_within"]["m0_2"],
             100 * out["frac_within"]["m0_5"], ["%.2f" % v for v in out["dp_by_quarter"]]))
    if a.json_out:
        json.dump(out, open(a.json_out, "w"), indent=2)


if __name__ == "__main__":
    main()
