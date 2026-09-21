#!/usr/bin/env python3
"""
sem_core.py -- pure numpy/scipy core of the RGB-semantic LIVO map.

Deliberately ROS-free and torch-free so that it imports identically under the
ROS 2 Jazzy system python (3.12) and under the ptv3 conda env (3.10).

Contents
--------
  T_I_L                 the IMU<-LiDAR extrinsic (the contract)
  TrajInterp            TUM trajectory, slerp + linear interpolation
  VoxelHash             vectorised open-addressing int64 -> row hash table
  SemanticVoxelMap      the global map: confidence-weighted class vote + RGB mean
  CLASS_COLORS          16 nuScenes class colours
"""

import numpy as np

# --------------------------------------------------------------------------- #
#  THE CONTRACT.  FAST-LIVO2's state is the IMU pose:  T_W_L = T_W_I @ T_I_L
#  T_I_L = inv(calib_imu_to_velo)     (KITTI 2011_09_30)
# --------------------------------------------------------------------------- #
T_I_L = np.array([
    [ 0.999997685, -0.000785403,  0.002024406,  0.810543972],
    [ 0.000755307,  0.999889850,  0.014824544, -0.307054372],
    [-0.002035826, -0.014822976,  0.999888022,  0.802723995],
    [ 0.0,          0.0,          0.0,          1.0]], dtype=np.float64)

# T_{rectified cam2 <- velodyne}
T_C_L = np.array([
    [-0.001857739, -0.999965951, -0.008039975,  0.056246554],
    [-0.006481466,  0.008051860, -0.999946608, -0.074814016],
    [ 0.999977310, -0.001805529, -0.006496204, -0.327793583],
    [ 0.0,          0.0,          0.0,          1.0]], dtype=np.float64)

# rectified image_02 intrinsics (P_rect_02[:, :3]); ZERO distortion
FX = FY = 707.0912
CX, CY = 601.8873, 183.1104

NUSCENES_CLASSES = [
    "barrier", "bicycle", "bus", "car", "construction_vehicle", "motorcycle",
    "pedestrian", "traffic_cone", "trailer", "truck", "driveable_surface",
    "other_flat", "sidewalk", "terrain", "manmade", "vegetation",
]
NUM_CLASSES = 16

# nuScenes-devkit-ish palette, index == class id
CLASS_COLORS = np.array([
    [255, 120,  50],   # 0  barrier
    [255, 192, 203],   # 1  bicycle
    [255, 255,   0],   # 2  bus
    [  0, 150, 245],   # 3  car
    [  0, 255, 255],   # 4  construction_vehicle
    [255,  61,  99],   # 5  motorcycle
    [  0,   0, 255],   # 6  pedestrian
    [255, 240, 150],   # 7  traffic_cone
    [135,  60,   0],   # 8  trailer
    [160,  32, 240],   # 9  truck
    [255,   0, 255],   # 10 driveable_surface
    [139, 137, 137],   # 11 other_flat
    [ 75,   0,  75],   # 12 sidewalk
    [150, 240,  80],   # 13 terrain
    [230, 230, 250],   # 14 manmade
    [  0, 175,   0],   # 15 vegetation
], dtype=np.uint8)

NO_RGB_COLOR = np.array([48, 48, 48], dtype=np.uint8)   # explicit sentinel grey


# --------------------------------------------------------------------------- #
class TrajInterp(object):
    """TUM trajectory -> T_{W<-IMU}(t), slerp on rotation, linear on translation.

    Queries outside [t0, t1] are marked invalid rather than extrapolated.
    """

    def __init__(self, tum_path, t_min=None):
        d = np.loadtxt(tum_path)
        if t_min is not None:
            d = d[d[:, 0] >= t_min]
        order = np.argsort(d[:, 0])
        d = d[order]
        self.t = d[:, 0].astype(np.float64)
        self.p = d[:, 1:4].astype(np.float64)
        self.q = d[:, 4:8].astype(np.float64)          # x y z w
        self.q /= np.linalg.norm(self.q, axis=1, keepdims=True)
        # enforce hemisphere continuity so slerp takes the short way
        for i in range(1, len(self.q)):
            if np.dot(self.q[i], self.q[i - 1]) < 0:
                self.q[i] *= -1.0
        self.t0, self.t1 = float(self.t[0]), float(self.t[-1])

    def valid(self, ts):
        ts = np.asarray(ts, dtype=np.float64)
        return (ts >= self.t0) & (ts <= self.t1)

    def query(self, ts):
        """ts (N,) -> R (N,3,3), p (N,3), ok (N,) bool."""
        ts = np.atleast_1d(np.asarray(ts, dtype=np.float64))
        ok = self.valid(ts)
        tc = np.clip(ts, self.t0, self.t1)
        j = np.clip(np.searchsorted(self.t, tc, side="right"), 1, len(self.t) - 1)
        i = j - 1
        dt = self.t[j] - self.t[i]
        u = np.where(dt > 0, (tc - self.t[i]) / np.where(dt > 0, dt, 1.0), 0.0)

        p = self.p[i] + (self.p[j] - self.p[i]) * u[:, None]

        q0, q1 = self.q[i], self.q[j]
        dot = np.sum(q0 * q1, axis=1)
        q1 = np.where(dot[:, None] < 0, -q1, q1)
        dot = np.abs(dot).clip(-1.0, 1.0)
        theta = np.arccos(dot)
        st = np.sin(theta)
        lin = st < 1e-7                       # fall back to nlerp when parallel
        s0 = np.where(lin, 1.0 - u, np.sin((1.0 - u) * theta) / np.where(lin, 1.0, st))
        s1 = np.where(lin, u, np.sin(u * theta) / np.where(lin, 1.0, st))
        q = q0 * s0[:, None] + q1 * s1[:, None]
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        return quat_to_R(q), p, ok

    def query_one(self, t):
        R, p, ok = self.query(np.array([t]))
        T = np.eye(4)
        T[:3, :3] = R[0]
        T[:3, 3] = p[0]
        return T, bool(ok[0])


def quat_to_R(q):
    """q (N,4) x y z w -> (N,3,3)."""
    q = np.atleast_2d(np.asarray(q, dtype=np.float64))
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((len(q), 3, 3), dtype=np.float64)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - z * w)
    R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w)
    R[:, 2, 1] = 2 * (y * z + x * w)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


# --------------------------------------------------------------------------- #
_EMPTY = np.int64(-(2 ** 63))
_M1 = np.uint64(0xff51afd7ed558ccd)
_M2 = np.uint64(0xc4ceb9fe1a85ec53)
_S33 = np.uint64(33)


class VoxelHash(object):
    """Open addressing, linear probing, fully vectorised.  int64 key -> int32 row."""

    def __init__(self, cap=1 << 22):
        self.cap = int(cap)
        self.key = np.full(self.cap, _EMPTY, dtype=np.int64)
        self.row = np.full(self.cap, -1, dtype=np.int32)
        self.n = 0

    @staticmethod
    def _mix(k, cap):
        h = k.astype(np.uint64, copy=True)
        h ^= h >> _S33
        h *= _M1
        h ^= h >> _S33
        h *= _M2
        h ^= h >> _S33
        return (h & np.uint64(cap - 1)).astype(np.int64)

    def _grow(self):
        old_k, old_r = self.key, self.row
        occ = old_k != _EMPTY
        self.cap *= 2
        self.key = np.full(self.cap, _EMPTY, dtype=np.int64)
        self.row = np.full(self.cap, -1, dtype=np.int32)
        k, r = old_k[occ], old_r[occ]
        slot = self._mix(k, self.cap)
        pend = np.arange(len(k))
        while pend.size:
            s = slot[pend]
            free = self.key[s] == _EMPTY
            fi = pend[free]
            if fi.size:
                us, first = np.unique(slot[fi], return_index=True)
                take = fi[first]
                self.key[slot[take]] = k[take]
                self.row[slot[take]] = r[take]
                done = np.zeros(len(k), dtype=bool)
                done[take] = True
                pend = pend[~done[pend]]
            slot[pend] = (slot[pend] + 1) & (self.cap - 1)

    def get_or_add(self, keys):
        """keys: UNIQUE int64 (n,).  Returns rows (n,) int32 and n_new."""
        keys = np.asarray(keys, dtype=np.int64)
        n = len(keys)
        if n == 0:
            return np.zeros(0, dtype=np.int32), 0
        while (self.n + n) * 2 > self.cap:
            self._grow()

        out = np.full(n, -1, dtype=np.int64)
        slot = self._mix(keys, self.cap)
        pend = np.arange(n)
        n_new = 0
        guard = 0
        while pend.size:
            guard += 1
            if guard > 10000:
                raise RuntimeError("VoxelHash probing did not converge")
            s = slot[pend]
            ks = self.key[s]
            hit = ks == keys[pend]
            if hit.any():
                hi = pend[hit]
                out[hi] = self.row[slot[hi]]
            rest = pend[~hit]
            if rest.size:
                s2 = slot[rest]
                free = self.key[s2] == _EMPTY
                fi = rest[free]
                if fi.size:
                    us, first = np.unique(slot[fi], return_index=True)
                    take = fi[first]
                    m = len(take)
                    new_rows = np.arange(self.n, self.n + m, dtype=np.int32)
                    self.key[slot[take]] = keys[take]
                    self.row[slot[take]] = new_rows
                    out[take] = new_rows
                    self.n += m
                    n_new += m
            pend = np.nonzero(out < 0)[0]
            slot[pend] = (slot[pend] + 1) & (self.cap - 1)
        return out.astype(np.int32), n_new


# --------------------------------------------------------------------------- #
_BITS = 21
_MASK = (1 << _BITS) - 1
_HALF = 1 << (_BITS - 1)


def voxel_key(coord, voxel):
    """world xyz (N,3) float -> int64 key, collision-free for |i| < 2^20."""
    g = np.floor(np.asarray(coord, dtype=np.float64) / voxel).astype(np.int64)
    gx = (g[:, 0] + _HALF) & _MASK
    gy = (g[:, 1] + _HALF) & _MASK
    gz = (g[:, 2] + _HALF) & _MASK
    return (gx << (2 * _BITS)) | (gy << _BITS) | gz


class SemanticVoxelMap(object):
    """Global RGB-semantic voxel map.

    FUSION RULE
    -----------
    class      : confidence-weighted vote.  every observation of class c with
                 softmax confidence p adds p to score[voxel, c];
                 final class = argmax_c score.
    confidence : score[argmax] / sum(score)  -- the vote share won by the winner,
                 in (0, 1].  1.0 means every observation agreed.
    rgb        : arithmetic mean over ONLY the observations that actually had a
                 camera measurement (n_rgb).  Voxels with n_rgb == 0 are flagged
                 has_rgb = 0 and published with the explicit sentinel grey
                 NO_RGB_COLOR -- never silently black.
    position   : centroid of every world point that fell in the voxel.
    """

    def __init__(self, voxel=0.20, cap0=1 << 21):
        self.voxel = float(voxel)
        self.h = VoxelHash(cap=1 << 22)
        self._cap = int(cap0)
        self.xyz = np.zeros((self._cap, 3), dtype=np.float64)
        self.score = np.zeros((self._cap, NUM_CLASSES), dtype=np.float32)
        self.rgb = np.zeros((self._cap, 3), dtype=np.float32)
        self.n_rgb = np.zeros(self._cap, dtype=np.uint32)
        self.n_obs = np.zeros(self._cap, dtype=np.uint32)
        self.n = 0
        self.n_points_inserted = 0
        self._auto_f = 1         # cached integer coarsening factor (only grows)
        self.key = np.zeros(self._cap, dtype=np.int64)

    def _reserve(self, need):
        if need <= self._cap:
            return
        cap = self._cap
        while cap < need:
            cap *= 2
        self.xyz = np.resize(self.xyz, (cap, 3)); self.xyz[self._cap:] = 0
        s = np.zeros((cap, NUM_CLASSES), dtype=np.float32); s[:self._cap] = self.score
        self.score = s
        r = np.zeros((cap, 3), dtype=np.float32); r[:self._cap] = self.rgb
        self.rgb = r
        self.n_rgb = np.resize(self.n_rgb, cap); self.n_rgb[self._cap:] = 0
        self.n_obs = np.resize(self.n_obs, cap); self.n_obs[self._cap:] = 0
        self.key = np.resize(self.key, cap); self.key[self._cap:] = 0
        self._cap = cap

    def insert(self, pts_w, cls, conf, rgb=None, has_rgb=None):
        """pts_w (N,3) world, cls (N,) uint8/16, conf (N,) float, rgb (N,3) uint8."""
        n = len(pts_w)
        if n == 0:
            return 0
        key = voxel_key(pts_w, self.voxel)
        uk, inv = np.unique(key, return_inverse=True)
        rows, n_new = self.h.get_or_add(uk)
        self._reserve(self.h.n)
        m = len(uk)

        self.key[rows] = uk
        cnt = np.bincount(inv, minlength=m)
        sx = np.bincount(inv, weights=pts_w[:, 0], minlength=m)
        sy = np.bincount(inv, weights=pts_w[:, 1], minlength=m)
        sz = np.bincount(inv, weights=pts_w[:, 2], minlength=m)
        self.xyz[rows, 0] += sx
        self.xyz[rows, 1] += sy
        self.xyz[rows, 2] += sz
        self.n_obs[rows] += cnt.astype(np.uint32)

        ci = inv * NUM_CLASSES + cls.astype(np.int64)
        sc = np.bincount(ci, weights=conf.astype(np.float64),
                         minlength=m * NUM_CLASSES).reshape(m, NUM_CLASSES)
        self.score[rows] += sc.astype(np.float32)

        if rgb is not None and has_rgb is not None and has_rgb.any():
            hm = has_rgb
            iv = inv[hm]
            rr = np.bincount(iv, weights=rgb[hm, 0].astype(np.float64), minlength=m)
            gg = np.bincount(iv, weights=rgb[hm, 1].astype(np.float64), minlength=m)
            bb = np.bincount(iv, weights=rgb[hm, 2].astype(np.float64), minlength=m)
            nn = np.bincount(iv, minlength=m)
            self.rgb[rows, 0] += rr.astype(np.float32)
            self.rgb[rows, 1] += gg.astype(np.float32)
            self.rgb[rows, 2] += bb.astype(np.float32)
            self.n_rgb[rows] += nn.astype(np.uint32)

        self.n = self.h.n
        self.n_points_inserted += n
        return n_new

    # ------------------------------------------------------------------ #
    def _decode(self, keys):
        gx = ((keys >> (2 * _BITS)) & _MASK) - _HALF
        gy = ((keys >> _BITS) & _MASK) - _HALF
        gz = (keys & _MASK) - _HALF
        return gx, gy, gz

    def _reduce(self, rows):
        """class / confidence / rgb / centroid for the given map rows."""
        sc = self.score[rows]
        tot = sc.sum(axis=1)
        cls = np.argmax(sc, axis=1)
        conf = (sc[np.arange(len(rows)), cls] / np.maximum(tot, 1e-9)).astype(np.float32)
        nob = np.maximum(self.n_obs[rows], 1)[:, None].astype(np.float64)
        xyz = (self.xyz[rows] / nob).astype(np.float32)
        nrgb = self.n_rgb[rows]
        has = nrgb > 0
        rgb = np.tile(NO_RGB_COLOR, (len(rows), 1)).astype(np.float32)
        if has.any():
            rgb[has] = self.rgb[rows][has] / nrgb[has][:, None].astype(np.float32)
        return xyz, np.clip(rgb, 0, 255).astype(np.uint8), cls.astype(np.uint16), \
            conf, has.astype(np.uint8)

    def snapshot(self, stride_voxel=None, max_points=3_000_000):
        """(xyz f4 (M,3), rgb u8 (M,3), class u16, confidence f4, has_rgb u1).

        If the map holds more voxels than max_points it is downsampled onto a
        grid f times coarser (f integer, so the coarse cell is an exact union of
        fine cells and the whole thing is integer arithmetic on the stored voxel
        keys -- no (N,3) float pass).  One fine voxel represents each coarse
        cell, and the composite sort key puts the voxels that actually carry a
        camera colour first, so a coarse cell keeps its RGB whenever any of its
        1-2 fine members had one.  The chosen f is cached: it only ever grows.
        """
        n = self.n
        if n == 0:
            z = np.zeros((0, 3), dtype=np.float32)
            return (z, z.astype(np.uint8), np.zeros(0, np.uint16),
                    np.zeros(0, np.float32), np.zeros(0, np.uint8))

        f = self._auto_f
        if stride_voxel is not None:
            f = max(1, int(round(stride_voxel / self.voxel)))
        if f <= 1 and n <= max_points:
            self._auto_f = 1
            return self._reduce(np.arange(n))

        keys = self.key[:n]
        gx, gy, gz = self._decode(keys)
        has_rgb_flag = (self.n_rgb[:n] == 0).astype(np.int64)   # 0 sorts first
        f = max(f, 2)
        while True:
            cx = np.floor_divide(gx, f) + _HALF
            cy = np.floor_divide(gy, f) + _HALF
            cz = np.floor_divide(gz, f) + _HALF
            ck = ((cx & _MASK) << (2 * _BITS)) | ((cy & _MASK) << _BITS) | (cz & _MASK)
            ck = ck * 2 + has_rgb_flag                # prefer a coloured member
            uk, rep = np.unique(ck // 2, return_index=True)
            self._auto_f = f
            if len(uk) <= max_points or f > 64:
                return self._reduce(rep)
            f += 1

# --------------------------------------------------------------------------- #
def pack_rgb_float(rgb_u8):
    """(N,3) uint8 -> (N,) float32 whose bits are 0x00RRGGBB (PCL convention)."""
    rgb_u8 = np.asarray(rgb_u8, dtype=np.uint8)
    packed = (rgb_u8[:, 0].astype(np.uint32) << 16 |
              rgb_u8[:, 1].astype(np.uint32) << 8 |
              rgb_u8[:, 2].astype(np.uint32))
    return packed.view(np.float32) if packed.dtype == np.uint32 else packed.astype(np.uint32).view(np.float32)


# published PointCloud2 record --------------------------------------------- #
#  x f4 @0 | y f4 @4 | z f4 @8 | rgb f4 @12 | class u2 @16 | (pad 2) |
#  confidence f4 @20 | has_rgb u1 @24 | (pad 3)      point_step = 28
CLOUD_DTYPE = np.dtype({
    "names":   ["x", "y", "z", "rgb", "class", "confidence", "has_rgb"],
    "formats": ["<f4", "<f4", "<f4", "<f4", "<u2", "<f4", "u1"],
    "offsets": [0, 4, 8, 12, 16, 20, 24],
    "itemsize": 28,
})


def build_record(xyz, rgb_u8, cls, conf, has_rgb):
    rec = np.zeros(len(xyz), dtype=CLOUD_DTYPE)
    rec["x"] = xyz[:, 0]
    rec["y"] = xyz[:, 1]
    rec["z"] = xyz[:, 2]
    rec["rgb"] = pack_rgb_float(rgb_u8)
    rec["class"] = cls
    rec["confidence"] = conf
    rec["has_rgb"] = has_rgb
    return rec


# --------------------------------------------------------------------------- #
#  Coarse label space shared by nuScenes-16 and SemanticKITTI, for GT checks.
#  (copied verbatim from ptv3_infer.py, which cannot be imported without torch)
# --------------------------------------------------------------------------- #
COARSE = ["car", "bicycle", "motorcycle", "truck", "bus", "other_vehicle", "person",
          "road", "sidewalk", "other_flat", "terrain", "vegetation", "manmade"]
_C = {n: i for i, n in enumerate(COARSE)}
IGNORE = -1

NUSC16_TO_COARSE = np.array([
    _C["manmade"], _C["bicycle"], _C["bus"], _C["car"], _C["other_vehicle"],
    _C["motorcycle"], _C["person"], _C["manmade"], _C["other_vehicle"],
    _C["truck"], _C["road"], _C["other_flat"], _C["sidewalk"], _C["terrain"],
    _C["manmade"], _C["vegetation"]], dtype=np.int32)

SK_TO_COARSE = {
    0: IGNORE, 1: IGNORE,
    10: _C["car"], 252: _C["car"],
    11: _C["bicycle"], 31: _C["bicycle"], 253: _C["bicycle"],
    13: _C["bus"], 257: _C["bus"],
    15: _C["motorcycle"], 32: _C["motorcycle"], 255: _C["motorcycle"],
    16: _C["other_vehicle"], 256: _C["other_vehicle"],
    18: _C["truck"], 258: _C["truck"],
    20: _C["other_vehicle"], 259: _C["other_vehicle"],
    30: _C["person"], 254: _C["person"],
    40: _C["road"], 44: _C["road"], 60: _C["road"],
    48: _C["sidewalk"], 49: _C["other_flat"],
    50: _C["manmade"], 51: _C["manmade"], 52: _C["manmade"],
    70: _C["vegetation"], 71: _C["vegetation"], 72: _C["terrain"],
    80: _C["manmade"], 81: _C["manmade"], 99: IGNORE,
}


def sk_lut():
    lut = np.full(300, IGNORE, dtype=np.int32)
    for k, v in SK_TO_COARSE.items():
        lut[k] = v
    return lut
