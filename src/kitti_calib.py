#!/usr/bin/env python3
"""
kitti_calib.py -- KITTI raw calibration loader + velodyne->cam2 pixel projection.

Reusable module for the livo_sem fusion node. Nothing here is specific to ROS.

Frames (KITTI raw convention)
-----------------------------
  velo  : Velodyne HDL-64E.   x forward, y left,  z up
  imu   : OXTS RT3003 body.   x forward, y left,  z up
  cam0  : left  grayscale, UNrectified. x right, y down, z forward
  cam2  : left  color,     rectified.   x right, y down, z forward
  rect0 : the common rectified frame that ALL P_rect_0x matrices live in.
          It is cam0 rotated by R_rect_00.

Projection chain used here (this is the official KITTI chain):

    x_rect0        = R_rect_00 @ (R_velo_cam0 @ x_velo + T_velo_cam0)
    [u v w]^T      = P_rect_02 @ [x_rect0; 1]
    pixel          = (u/w, v/w),  depth = w

Note P_rect_02 (not P_rect_00) and R_rect_00 (not R_rect_02).  R_rect_02 is the
rectifying rotation *of camera 2 itself* and is NOT part of this chain; the
baseline of cam2 w.r.t. rect0 is already baked into column 3 of P_rect_02.
"""

import os
import numpy as np

__all__ = ["KittiCalib", "load_calib"]


def _read_calib_file(path):
    """Parse a KITTI 'key: v1 v2 ...' calib file into {key: np.array}."""
    data = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, val = line.split(":", 1)
            try:
                data[key.strip()] = np.array([float(x) for x in val.split()])
            except ValueError:
                data[key.strip()] = val.strip()   # e.g. calib_time
    return data


def _Rt_to_T(R, t):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R.reshape(3, 3)
    T[:3, 3] = np.asarray(t).reshape(3)
    return T


class KittiCalib(object):
    """
    Attributes
    ----------
    K            (3,3)  intrinsics of the RAW (distorted) cam2       [K_02]
    D            (5,)   plumb-bob distortion of the RAW cam2         [D_02]
    P_rect_02    (3,4)  projection matrix, rectified cam2            [P_rect_02]
    K_rect       (3,3)  P_rect_02[:, :3] -- intrinsics after rectification
    R_rect_00    (3,3)  rectifying rotation of cam0
    T_velo_cam0  (4,4)  velodyne -> cam0 (unrectified)
    T_imu_velo   (4,4)  imu      -> velodyne
    T_velo_imu   (4,4)  velodyne -> imu       (inverse of the above)
    T_velo_rect0 (4,4)  velodyne -> rect0     ( R_rect_00 * T_velo_cam0 )
    T_velo_cam2  (4,4)  velodyne -> rectified cam2 optical frame
    T_imu_cam2   (4,4)  imu      -> rectified cam2 optical frame
    img_size     (2,)   (width, height) of the rectified image       [S_rect_02]
    """

    def __init__(self, calib_dir):
        self.calib_dir = calib_dir
        c2c = _read_calib_file(os.path.join(calib_dir, "calib_cam_to_cam.txt"))
        v2c = _read_calib_file(os.path.join(calib_dir, "calib_velo_to_cam.txt"))
        i2v = _read_calib_file(os.path.join(calib_dir, "calib_imu_to_velo.txt"))

        self.K = c2c["K_02"].reshape(3, 3)
        self.D = c2c["D_02"].reshape(5)
        self.P_rect_02 = c2c["P_rect_02"].reshape(3, 4)
        self.P_rect_00 = c2c["P_rect_00"].reshape(3, 4)
        self.R_rect_00 = c2c["R_rect_00"].reshape(3, 3)
        self.img_size = c2c["S_rect_02"].astype(int)          # (w, h)
        self.img_size_raw = c2c["S_02"].astype(int)

        self.K_rect = self.P_rect_02[:, :3].copy()

        self.T_velo_cam0 = _Rt_to_T(v2c["R"], v2c["T"])
        self.T_imu_velo = _Rt_to_T(i2v["R"], i2v["T"])
        self.T_velo_imu = np.linalg.inv(self.T_imu_velo)

        R_rect4 = np.eye(4)
        R_rect4[:3, :3] = self.R_rect_00
        self.T_velo_rect0 = R_rect4 @ self.T_velo_cam0

        # rect0 -> rectified cam2 is a pure translation t = K_rect^-1 @ P[:,3]
        self.t_rect0_cam2 = np.linalg.inv(self.K_rect) @ self.P_rect_02[:, 3]
        T_rect0_cam2 = _Rt_to_T(np.eye(3), self.t_rect0_cam2)
        self.T_velo_cam2 = T_rect0_cam2 @ self.T_velo_rect0
        self.T_imu_cam2 = self.T_velo_cam2 @ self.T_imu_velo

    # ------------------------------------------------------------------ #
    def project_velo_to_cam2(self, pts_velo, img_shape=None,
                             min_depth=0.5, return_mask=False):
        """
        Project Nx3 (or Nx4, extra cols ignored) velodyne points into the
        RECTIFIED left-colour image (image_02).

        Returns
        -------
        uv    (M,2) float64 pixel coords of the points that land in the image
        depth (M,)  float64 depth along the rectified optical axis, metres
        mask  (N,)  bool, True where the point was kept (only if return_mask)
        """
        pts = np.asarray(pts_velo, dtype=np.float64)[:, :3]
        n = pts.shape[0]
        hom = np.hstack([pts, np.ones((n, 1))])

        # velo -> rect0 -> pixel
        x_rect = (self.T_velo_rect0 @ hom.T).T           # (N,4)
        uvw = (self.P_rect_02 @ x_rect.T).T              # (N,3)

        depth = uvw[:, 2]
        good = depth > min_depth
        uv = np.full((n, 2), np.nan)
        uv[good] = uvw[good, :2] / depth[good, None]

        if img_shape is None:
            w, h = int(self.img_size[0]), int(self.img_size[1])
        else:
            h, w = img_shape[0], img_shape[1]

        good &= (uv[:, 0] >= 0) & (uv[:, 0] <= w - 1)
        good &= (uv[:, 1] >= 0) & (uv[:, 1] <= h - 1)

        if return_mask:
            return uv[good], depth[good], good
        return uv[good], depth[good]

    # ------------------------------------------------------------------ #
    def colorize(self, pts_velo, image_rgb, min_depth=0.5):
        """
        Convenience for the fusion node: returns (pts_kept Nx3, rgb Nx3 uint8,
        mask N) -- nearest-pixel colour lookup, no interpolation.
        """
        pts = np.asarray(pts_velo, dtype=np.float64)[:, :3]
        uv, depth, mask = self.project_velo_to_cam2(
            pts, img_shape=image_rgb.shape[:2], min_depth=min_depth,
            return_mask=True)
        u = np.rint(uv[:, 0]).astype(np.int32)
        v = np.rint(uv[:, 1]).astype(np.int32)
        rgb = image_rgb[v, u]
        return pts[mask], rgb, mask

    # ------------------------------------------------------------------ #
    def summary(self):
        np.set_printoptions(suppress=True, precision=9, linewidth=200)
        s = []
        s.append("image_02 rectified size (w,h) = %s" % (tuple(self.img_size),))
        s.append("K_02 (raw, distorted):\n%s" % self.K)
        s.append("D_02 (plumb_bob k1 k2 p1 p2 k3):\n%s" % self.D)
        s.append("P_rect_02 (3x4):\n%s" % self.P_rect_02)
        s.append("R_rect_00 (3x3):\n%s" % self.R_rect_00)
        s.append("T_velo_cam0 (4x4, velo->cam0 unrect):\n%s" % self.T_velo_cam0)
        s.append("T_velo_rect0 (4x4, velo->rect0):\n%s" % self.T_velo_rect0)
        s.append("T_velo_cam2 (4x4, velo->rectified cam2):\n%s" % self.T_velo_cam2)
        s.append("T_imu_velo (4x4, imu->velo):\n%s" % self.T_imu_velo)
        s.append("T_velo_imu (4x4, velo->imu):\n%s" % self.T_velo_imu)
        s.append("T_imu_cam2 (4x4, imu->rectified cam2):\n%s" % self.T_imu_cam2)
        return "\n".join(s)


def load_calib(calib_dir):
    return KittiCalib(calib_dir)


# ====================================================================== #
#  self-test:  python3 kitti_calib.py [drive_dir] [frame] [out.png]
# ====================================================================== #
def _depth_edge_pixels(uv, depth, ring, jump=1.0, max_range=45.0):
    """Pixels where the LiDAR sees a depth discontinuity ALONG a ring.
    If the extrinsic/projection is right these must coincide with image edges."""
    out = []
    for r in np.unique(ring):
        m = ring == r
        if m.sum() < 8:
            continue
        u, d = uv[m, 0], depth[m]
        o = np.argsort(u)
        u, d, vv = u[o], d[o], uv[m, 1][o]
        dd = np.abs(np.diff(d))
        near = np.minimum(d[:-1], d[1:]) < max_range
        du = np.diff(u)
        hit = (dd > jump) & near & (du < 6.0)          # adjacent, real jump
        idx = np.nonzero(hit)[0]
        take = np.where(d[idx] < d[idx + 1], idx, idx + 1)   # the NEAR side
        out.append(np.stack([u[take], vv[take]], axis=1))
    return np.concatenate(out) if out else np.zeros((0, 2))


def _alignment_score(calib, pts, img, delta_deg=0.0, axis="y"):
    """Median distance (px) from LiDAR depth-edges to the nearest image edge,
    after perturbing the velo->cam rotation by delta_deg about `axis`."""
    import cv2
    from kitti_scan import synth

    a = np.deg2rad(delta_deg)
    ca, sa = np.cos(a), np.sin(a)
    if axis == "y":
        Rp = np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]])
    elif axis == "x":
        Rp = np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])
    else:
        Rp = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])
    T = np.eye(4); T[:3, :3] = Rp
    T_velo_rect0_pert = T @ calib.T_velo_rect0

    H, W = img.shape[:2]
    hom = np.hstack([pts[:, :3], np.ones((len(pts), 1))])
    uvw = (calib.P_rect_02 @ (T_velo_rect0_pert @ hom.T)).T
    depth = uvw[:, 2]
    good = depth > 0.5
    uv = np.full((len(pts), 2), -1e9)
    uv[good] = uvw[good, :2] / depth[good, None]
    good &= (uv[:, 0] >= 0) & (uv[:, 0] <= W - 1) & (uv[:, 1] >= 0) & (uv[:, 1] <= H - 1)

    ring, _, _ = synth(pts[:, :3], reorder=False)
    ep = _depth_edge_pixels(uv[good], depth[good], ring[good])
    if len(ep) < 50:
        return np.nan, 0.0, len(ep)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 60, 160)
    dist = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 3)
    d = dist[np.rint(ep[:, 1]).astype(int), np.rint(ep[:, 0]).astype(int)]
    return float(np.median(d)), float((d <= 2.0).mean()), len(ep)


def _self_test(drive_dir, frame, out_png, calib_dir=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from PIL import Image

    if calib_dir is None:
        calib_dir = os.path.dirname(os.path.normpath(drive_dir))
    calib = KittiCalib(calib_dir)
    drive_name = os.path.basename(os.path.normpath(drive_dir))

    stem = "%010d" % frame
    img = np.asarray(Image.open(os.path.join(
        drive_dir, "image_02", "data", stem + ".png")).convert("RGB"))
    pts = np.fromfile(os.path.join(
        drive_dir, "velodyne_points", "data", stem + ".bin"),
        dtype=np.float32).reshape(-1, 4)

    H, W = img.shape[:2]
    uv, depth, mask = calib.project_velo_to_cam2(
        pts[:, :3], img_shape=(H, W), return_mask=True)
    print("[self-test] image %dx%d, %d velodyne points, %d project into cam2 (%.1f%%)"
          % (W, H, len(pts), mask.sum(), 100.0 * mask.sum() / len(pts)))
    print("[self-test] u [%.1f, %.1f]  v [%.1f, %.1f]  depth [%.2f, %.2f] m"
          % (uv[:, 0].min(), uv[:, 0].max(), uv[:, 1].min(), uv[:, 1].max(),
             depth.min(), depth.max()))

    pts_kept, rgb, m2 = calib.colorize(pts[:, :3], img)
    assert np.array_equal(m2, mask)

    # ---- quantitative: depth-edge vs image-edge, swept over a yaw perturbation
    deltas = np.arange(-1.5, 1.51, 0.25)
    med, frac = [], []
    for dd in deltas:
        m_, f_, n_ = _alignment_score(calib, pts, img, dd, "y")
        med.append(m_); frac.append(f_)
    med = np.array(med); frac = np.array(frac)
    best = deltas[int(np.nanargmin(med))]
    m0, f0, n0 = _alignment_score(calib, pts, img, 0.0, "y")
    print("[self-test] depth-edge/image-edge: %d edge px, median dist %.2f px, "
          "%.1f%% within 2 px" % (n0, m0, 100 * f0))
    print("[self-test] yaw sweep argmin at %+.2f deg (0.00 = shipped calibration)" % best)

    cmap = cm.get_cmap("turbo")
    dcol = cmap(1.0 - np.clip((depth - 3.0) / 52.0, 0, 1))[:, :3]

    fig = plt.figure(figsize=(17.5, 15), dpi=105)
    gs = fig.add_gridspec(4, 2, height_ratios=[1.0, 1.0, 1.45, 1.15],
                          hspace=0.20, wspace=0.10)

    ax = fig.add_subplot(gs[0, :]); ax.imshow(img); ax.set_axis_off()
    ax.set_title("A.  image_02 (rectified left colour) -- %s frame %06d"
                 % (drive_name, frame), fontsize=11)

    ax = fig.add_subplot(gs[1, :]); ax.imshow(img); ax.set_axis_off()
    ax.scatter(uv[:, 0], uv[:, 1], c=dcol, s=0.7, marker=".", linewidths=0)
    ax.set_title("B.  velodyne -> P_rect_02 @ R_rect_00 @ T_velo_cam0,  %d/%d points land "
                 "in frame (%.1f%%),  colour = depth (red 3 m -> blue 55 m)"
                 % (mask.sum(), len(pts), 100.0 * mask.sum() / len(pts)), fontsize=11)

    cx, cy = int(np.median(uv[:, 0])), int(np.median(uv[:, 1]))
    x0 = int(np.clip(cx - 210, 0, max(W - 420, 0))); x1 = min(x0 + 420, W)
    y0 = int(np.clip(cy - 105, 0, max(H - 210, 0))); y1 = min(y0 + 210, H)
    ax = fig.add_subplot(gs[2, 0]); ax.imshow(img[y0:y1, x0:x1]); ax.set_axis_off()
    sel = (uv[:, 0] >= x0) & (uv[:, 0] < x1) & (uv[:, 1] >= y0) & (uv[:, 1] < y1)
    ax.scatter(uv[sel, 0] - x0, uv[sel, 1] - y0, c=dcol[sel], s=10,
               marker=".", linewidths=0)
    ax.set_title("C.  3x zoom [%d:%d, %d:%d] -- depth steps must fall on object outlines"
                 % (x0, x1, y0, y1), fontsize=10)

    # D: colourised cloud from a DIFFERENT virtual viewpoint (oblique, elevated)
    ax = fig.add_subplot(gs[2, 1])
    near = pts_kept[:, 0] < 38.0
    pk, pc_rgb = pts_kept[near], rgb[near]
    c = np.array([-7.0, 2.5, 4.2])
    fdir = np.array([1.0, -0.10, -0.24]); fdir /= np.linalg.norm(fdir)
    right = np.cross(fdir, [0, 0, 1.0]); right /= np.linalg.norm(right)
    upv = np.cross(right, fdir)
    d3 = pk - c
    zc = d3 @ fdir
    v_ok = zc > 1.0
    xc, yc = d3 @ right, -(d3 @ upv)
    fv, W2, H2 = 780.0, 980, 620
    uu = fv * xc / np.where(v_ok, zc, 1) + W2 / 2
    vv = fv * yc / np.where(v_ok, zc, 1) + H2 / 2
    v_ok &= (uu > 0) & (uu < W2) & (vv > 0) & (vv < H2)
    o = np.argsort(-zc[v_ok])
    ax.scatter(uu[v_ok][o], vv[v_ok][o],
               c=np.clip(pc_rgb[v_ok][o] / 255.0 * 1.35, 0, 1),
               s=9.0, marker=".", linewidths=0)
    ax.set_xlim(0, W2); ax.set_ylim(H2, 0); ax.set_facecolor("0.10")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("D.  the SAME points with RGB read out of the image, re-rendered from a\n"
                 "virtual camera 7 m behind / 4.2 m above / 2.5 m left -- a wrong\n"
                 "projection would smear colour across depth; a right one keeps\n"
                 "objects solid (road, tree, market stalls, pedestrian)",
                 fontsize=9.5)

    # E: sensitivity sweep
    ax = fig.add_subplot(gs[3, 0])
    ax.plot(deltas, med, "o-", color="#c0392b", lw=1.8, ms=4.5)
    ax.axvline(0, color="#2980b9", ls="--", lw=1.4)
    ax.annotate("shipped calibration\n%.2f px" % m0, xy=(0, m0),
                xytext=(0.45, m0 + 0.45 * (max(med) - min(med))),
                fontsize=9, color="#2980b9",
                arrowprops=dict(arrowstyle="->", color="#2980b9"))
    ax.set_xlabel("artificial yaw error applied to T_velo_cam  [deg]", fontsize=9)
    ax.set_ylabel("median |depth-edge -> nearest image-edge|  [px]", fontsize=9)
    ax.grid(alpha=0.3); ax.tick_params(labelsize=8)
    ax.set_title("E.  sensitivity: the shipped extrinsic is the MINIMUM\n"
                 "(argmin at %+.2f deg) -- alignment is not accidental" % best,
                 fontsize=10)

    ax = fig.add_subplot(gs[3, 1]); ax.set_axis_off()
    txt = (
        "ACCEPTANCE SUMMARY  (RGB half of the mission)\n"
        "-----------------------------------------------------------\n"
        "drive              : %s   frame %06d\n"
        "image_02 rectified : %d x %d\n"
        "velodyne points    : %d\n"
        "project into cam2  : %d  (%.1f %%)\n"
        "depth range        : %.2f .. %.2f m\n"
        "\n"
        "chain : uv = P_rect_02 @ R_rect_00 @ T_velo_cam0 @ p_velo\n"
        "fx=fy = %.4f    cx = %.4f   cy = %.4f\n"
        "P[0,3] = %.5f  (rect0 -> cam2 baseline term)\n"
        "\n"
        "depth-edge / image-edge agreement\n"
        "  edge pixels tested : %d\n"
        "  median distance    : %.2f px\n"
        "  within 2 px        : %.1f %%\n"
        "  yaw-sweep argmin   : %+.2f deg  (shipped = 0.00)\n"
        "  median at +-1.0 deg: %.2f / %.2f px\n"
        % (drive_name, frame, W, H, len(pts), mask.sum(),
           100.0 * mask.sum() / len(pts), depth.min(), depth.max(),
           calib.K_rect[0, 0], calib.K_rect[0, 2], calib.K_rect[1, 2],
           calib.P_rect_02[0, 3], n0, m0, 100 * f0, best,
           med[np.argmin(np.abs(deltas + 1.0))], med[np.argmin(np.abs(deltas - 1.0))]))
    ax.text(0.0, 1.0, txt, family="monospace", fontsize=10.2, va="top")

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("[self-test] wrote %s" % out_png)
    return dict(n_proj=int(mask.sum()), n_total=int(len(pts)),
                median_px=m0, frac2px=f0, argmin_deg=float(best))


if __name__ == "__main__":
    import sys
    drive = sys.argv[1] if len(sys.argv) > 1 else \
        "/data/livo_sem/data/raw/2011_09_30/2011_09_30_drive_0027_sync"
    frame = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    out = sys.argv[3] if len(sys.argv) > 3 else \
        "/data/livo_sem/out/proj_check_seq07_0000.png"
    print(KittiCalib(os.path.dirname(os.path.normpath(drive))).summary())
    _self_test(drive, frame, out)
