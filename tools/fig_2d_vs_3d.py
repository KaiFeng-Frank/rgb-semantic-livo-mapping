#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""STEP 8 -- the two figures.

  fig_arms_<f>.png      one scan, four panels: GT / 3D / 2D / hybrid, bird's-eye,
                        common-9 colours.  Wrong points are not hidden.
  fig_coverage_<f>.png  what the 2D arm cannot label, bird's-eye: the camera frustum
                        drawn over the scan, points coloured by why the 2D arm is
                        silent (outside frustum / occluded / answered).
"""
import os, sys, argparse
sys.path.insert(0, "/data/livo_sem/src")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import label_spaces as LS
import score_2d_vs_3d as SC

COARSE = list(LS.COARSE)
COL = {
    "car":           "#1f77b4",
    "large_vehicle": "#17becf",
    "two_wheeler":   "#e377c2",
    "person":        "#d62728",
    "road":          "#7f7f7f",
    "sidewalk":      "#bcbd22",
    "terrain":       "#2ca02c",
    "vegetation":    "#006d2c",
    "manmade":       "#ff7f0e",
}
CARR = np.array([matplotlib.colors.to_rgb(COL[c]) for c in COARSE])
GREY = np.array([0.82, 0.82, 0.82])


def bev(ax, xyz, rgb, s=0.6, xlim=(-30, 30), ylim=(-12, 62)):
    # KITTI velodyne: x forward, y left.  Plot y (left) horizontally, x (fwd) up.
    ax.scatter(-xyz[:, 1], xyz[:, 0], s=s, c=rgb, marker=".", linewidths=0)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#999999")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", type=int, default=380)
    ap.add_argument("--arm3d", required=True)
    ap.add_argument("--arm2d", required=True)
    ap.add_argument("--seq", default="07")
    ap.add_argument("--out", default="/data/livo_sem/out/vs2d")
    a = ap.parse_args()

    SC.set_sequence(a.seq)
    proj = SC.Projector()
    lut = LS.sk_lut()
    f = a.frame
    pts = SC.read_scan(f)
    xyz = pts[:, :3].astype(np.float64)
    gt = SC.read_gt_coarse(f, lut)
    u, v, z, inm = proj.project(xyz)
    ctx = SC.frame_context(f, proj, lut, bcache=None, need_boundary=False)

    a2 = SC.Arm2D(a.arm2d, "2d")
    a3 = SC.Arm3D(a.arm3d, "3d")
    hy = SC.ArmHybrid(a2, a3, "hybrid")
    p2 = a2.predict(f, ctx); p3 = a3.predict(f, ctx); ph = hy.predict(f, ctx)

    valid = gt != LS.EXCLUDED

    def rgb_of(lab):
        c = np.tile(GREY, (len(lab), 1))
        ok = lab >= 0
        c[ok] = CARR[lab[ok]]
        return c

    os.makedirs(a.out, exist_ok=True)

    # ---------------- figure 1: four arms ---------------------------------- #
    fig, axes = plt.subplots(1, 4, figsize=(19.5, 6.6))
    panels = [("Ground truth (SemanticKITTI)", gt),
              ("A. 3D  PTv3 on the point cloud", p3),
              ("B. 2D  Cityscapes projected", p2),
              ("C. Hybrid  2D in frustum, 3D outside", ph)]
    for ax, (t, lab) in zip(axes, panels):
        m = valid
        bev(ax, xyz[m], rgb_of(lab[m].astype(np.int64)))
        ax.set_title(t, fontsize=11)
    axes[0].plot([0], [0], marker="^", color="k", ms=9)
    hs = [Patch(facecolor=COL[c], label=c) for c in COARSE]
    hs.append(Patch(facecolor=GREY, label="no label / unnameable"))
    fig.legend(handles=hs, loc="lower center", ncol=10, frameon=False, fontsize=9)
    fig.suptitle("SemanticKITTI seq %s frame %06d -- one scan, three arms, "
                 "common-9 label space (bird's-eye, +/-30 m lateral)" % (a.seq, f),
                 fontsize=12)
    fig.tight_layout(rect=[0, 0.055, 1, 0.965])
    p1 = "%s/fig_arms_%06d.png" % (a.out, f)
    fig.savefig(p1, dpi=140); plt.close(fig)

    # ---------------- figure 2: the coverage gap --------------------------- #
    vis = np.zeros(len(xyz), bool)
    if inm.any():
        vis[inm] = SC.visible_mask(np.rint(u[inm]).astype(np.int64),
                                   np.rint(v[inm]).astype(np.int64), z[inm])
    cat = np.full(len(xyz), 0, np.int8)          # 0 outside frustum
    cat[inm & ~vis] = 1                          # occluded
    cat[inm & vis] = 2                           # answered
    ccol = np.array([[0.80, 0.80, 0.80], [0.85, 0.30, 0.10], [0.10, 0.45, 0.80]])
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.8))
    m = valid
    bev(axes[0], xyz[m], ccol[cat[m]], s=0.7)
    n = int(m.sum())
    axes[0].set_title("Where the 2D arm can speak at all\n"
                      "answered %.1f %%   occluded %.1f %%   outside the frustum %.1f %%"
                      % (100.0 * (cat[m] == 2).sum() / n,
                         100.0 * (cat[m] == 1).sum() / n,
                         100.0 * (cat[m] == 0).sum() / n), fontsize=10)
    fov = np.arctan2(SC.IMG_W / 2.0, 707.0912)
    for sgn in (-1, 1):
        axes[0].plot([0, sgn * 62 * np.tan(fov)], [0, 62], color="k", lw=1.0, ls="--")
    axes[0].plot([0], [0], marker="^", color="k", ms=9)

    err = np.zeros(len(xyz), np.int8)
    err[m & (p2 == gt)] = 2
    err[m & (p2 >= 0) & (p2 != gt)] = 1
    err[m & (p2 == LS.UNMAPPED)] = 1
    ecol = np.array([[0.80, 0.80, 0.80], [0.85, 0.10, 0.10], [0.15, 0.60, 0.20]])
    bev(axes[1], xyz[m], ecol[err[m]], s=0.7)
    lab_n = int((p2[m] != SC.ABSTAIN).sum())
    axes[1].set_title("2D arm, same scan: correct / wrong / silent\n"
                      "coverage %.1f %%   accuracy where it speaks %.1f %%"
                      % (100.0 * lab_n / n,
                         100.0 * (err[m] == 2).sum() / max(lab_n, 1)), fontsize=10)
    axes[1].plot([0], [0], marker="^", color="k", ms=9)
    hs = [Patch(facecolor="#cccccc", label="2D arm silent"),
          Patch(facecolor="#d91a1a", label="2D arm wrong"),
          Patch(facecolor="#2699 33".replace(" ", ""), label="2D arm correct"),
          Patch(facecolor="#d94d1a", label="occluded (left panel)"),
          Patch(facecolor="#1a73cc", label="camera can see (left panel)")]
    fig.legend(handles=hs, loc="lower center", ncol=5, frameon=False, fontsize=9)
    fig.suptitle("The coverage gap -- seq %s frame %06d (bird's-eye)" % (a.seq, f),
                 fontsize=12)
    fig.tight_layout(rect=[0, 0.07, 1, 0.95])
    p2f = "%s/fig_coverage_%06d.png" % (a.out, f)
    fig.savefig(p2f, dpi=140); plt.close(fig)
    print(p1); print(p2f)


if __name__ == "__main__":
    main()
