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
#  v0.3 -- POTENTIALLY-MOVABLE nuScenes classes.
#  These select WHERE geometric effort is spent.  They are NEVER the verdict:
#  nuScenes has no moving/parked distinction and seq07 holds 383 stationary car
#  clusters against 51 moving ones, so a class-only policy deletes every parked
#  car (MEASURED control arm: 9.81 % static false-kill, 211 whole objects).
# --------------------------------------------------------------------------- #
MOVABLE_NUSC = np.zeros(NUM_CLASSES, dtype=bool)
MOVABLE_NUSC[[1, 2, 3, 4, 5, 6, 8, 9]] = True   # bicycle bus car constr motorcycle
                                                # pedestrian trailer truck


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

    def __init__(self, voxel=0.20, cap0=1 << 21, hash_cap=None, dyn=False):
        """cap0 / hash_cap PRESIZE the map.  MEASURED: with the stock cap0 = 1<<21
        rows and hash cap = 1<<22 slots, seq07 (2.70 M voxels) trips BOTH growth
        paths in the SAME frame at ~2.07 M voxels -- _reserve copies ~490 MB while
        VoxelHash._grow rehashes 2.1 M keys through a vectorised probe loop.  That
        one frame is the 415.7 ms max in stats_seq07.json; every other frame is
        under 30 ms.  Presizing costs ~1 GB of RSS up front on a 62 GB box and the
        reallocation never fires.  The growth path stays as the fallback."""
        self.voxel = float(voxel)
        self.h = VoxelHash(cap=int(hash_cap) if hash_cap else (1 << 22))
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
        self._sbuf = np.empty((1 << 17, NUM_CLASSES), dtype=np.float32)

        # ---- v0.3 per-voxel dynamic state.  OFF by default: with dyn=False not
        # one array is allocated and not one branch is taken inside insert(), so
        # the v0.2 fusion arithmetic is bit-identical (opt/verify_v03.py V1).
        self.dyn = bool(dyn)
        if self.dyn:
            self.n_free = np.zeros(self._cap, dtype=np.uint16)   # free-space votes
            self.n_seen = np.zeros(self._cap, dtype=np.uint16)   # re-occupied votes
            self.dom = np.zeros(self._cap, dtype=np.uint8)       # cached argmax
            self.in_cand = np.zeros(self._cap, dtype=bool)
            self._cand_cap = 1 << 16
            self._cand_rows = np.empty(self._cand_cap, dtype=np.int32)
            self._cand_xyz = np.empty((self._cand_cap, 3), dtype=np.float32)
            self.n_cand = 0
        self._last_point_rows = None

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
        if self.dyn:
            self.n_free = np.resize(self.n_free, cap); self.n_free[self._cap:] = 0
            self.n_seen = np.resize(self.n_seen, cap); self.n_seen[self._cap:] = 0
            self.dom = np.resize(self.dom, cap); self.dom[self._cap:] = 0
            self.in_cand = np.resize(self.in_cand, cap); self.in_cand[self._cap:] = False
        self._cap = cap

    def insert(self, pts_w, cls, conf, rgb=None, has_rgb=None,
               want_point_rows=False):
        """pts_w (N,3) world, cls (N,) uint8/16, conf (N,) float, rgb (N,3) uint8.

        want_point_rows stashes rows[inv] (the map row every POINT landed in) in
        self._last_point_rows.  Used only by the offline mask dump; it is a single
        int32 gather and is skipped entirely when False.
        """
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
        # take -> add -> scatter instead of a fancy-index read-modify-write.
        # MEASURED 4.22 -> 3.42 ms at the 2.7 M-voxel steady state.  BIT-EXACT only
        # because sc is cast to float32 BEFORE the add: casting after would add a
        # float64 intermediate and drift the confidences by ~3e-4.
        sc32 = sc.astype(np.float32)
        if len(self._sbuf) < m:
            self._sbuf = np.empty((max(m, 2 * len(self._sbuf)), NUM_CLASSES), np.float32)
        buf = self._sbuf[:m]
        np.take(self.score, rows, axis=0, out=buf)
        buf += sc32
        self.score[rows] = buf

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

        # ---- v0.3.  Reuses the (uk, inv, rows) triple and `buf`, the updated
        # score block, that the fusion above already computed: np.unique is the
        # most expensive item in this method (5.31 ms) and costs nothing extra.
        if self.dyn:
            dom = np.argmax(buf, axis=1).astype(np.uint8)
            self.dom[rows] = dom
            want = MOVABLE_NUSC[dom] & ~self.in_cand[rows]
            if want.any():
                nr = rows[want]
                self.in_cand[nr] = True
                self._cand_append(nr, uk[want])
        self._last_point_rows = rows[inv] if want_point_rows else None

        self.n = self.h.n
        self.n_points_inserted += n
        return n_new

    # ------------------------------------------------------------------ #
    def _cand_append(self, new_rows, new_keys):
        """Append rows that have just become movable-dominant to the candidate
        index, together with their voxel CENTRES.

        The centre, not the running centroid: the centre is a pure function of
        the key and can therefore be cached once and never goes stale, whereas
        self.xyz keeps moving as observations accumulate.  The two differ by at
        most half a voxel (0.1 m), well inside the 0.6 m visibility margin.
        """
        k = len(new_rows)
        need = self.n_cand + k
        if need > self._cand_cap:
            cap = self._cand_cap
            while cap < need:
                cap *= 2
            self._cand_rows = np.resize(self._cand_rows, cap)
            x = np.empty((cap, 3), dtype=np.float32)
            x[:self.n_cand] = self._cand_xyz[:self.n_cand]
            self._cand_xyz = x
            self._cand_cap = cap
        gx, gy, gz = self._decode(new_keys)
        a, b = self.n_cand, self.n_cand + k
        self._cand_rows[a:b] = new_rows
        self._cand_xyz[a:b, 0] = (gx + 0.5) * self.voxel
        self._cand_xyz[a:b, 1] = (gy + 0.5) * self.voxel
        self._cand_xyz[a:b, 2] = (gz + 0.5) * self.voxel
        self.n_cand = b

    def dyn_mask(self, n, k_free):
        """Per-row boolean: enough free-space evidence to withhold from publish."""
        if not self.dyn or k_free <= 0:
            return None
        return self.n_free[:n] >= k_free

    # ------------------------------------------------------------------ #
    def _decode(self, keys):
        gx = ((keys >> (2 * _BITS)) & _MASK) - _HALF
        gy = ((keys >> _BITS) & _MASK) - _HALF
        gz = (keys & _MASK) - _HALF
        return gx, gy, gz

    def _reduce(self, rows, chunk=1 << 18):
        """class / confidence / rgb / centroid for the given map rows.

        When rows is the whole map (the common case -- snapshot with f == 1) the
        six fancy-index gathers are pure overhead: self.xyz[rows] alone measured
        43.8 ms at 2.7 M voxels, and np.tile of the sentinel grey another 11.0 ms.
        Slicing instead is EXACT and takes the full-map snapshot 480 -> 290 ms.

        CHUNKED (v0.3).  This is the branch the LIVE /semantic_map feed actually
        takes -- seq07 ends at 2.70 M voxels against --max-pub-points 1000000, so
        snapshot() always coarsens and always lands here -- and un-chunked it held
        the GIL for its full 155-540 ms on the publisher thread.  _reduce_all was
        given chunk + time.sleep(0) in v0.2 precisely because a 276 ms snapshot
        stalled the fuse thread for up to 6 frames and caused all 7 backpressure
        drops; _reduce never got that treatment.  The arithmetic is unchanged --
        identical per element, just batched -- and opt/verify_v03.py V5 asserts
        bit-identity against the one-shot form.
        """
        if isinstance(rows, slice):
            return self._reduce_all(rows.stop)
        import time as _t
        n = len(rows)
        xyz = np.empty((n, 3), dtype=np.float32)
        rgb = np.empty((n, 3), dtype=np.uint8)
        cls = np.empty(n, dtype=np.uint16)
        conf = np.empty(n, dtype=np.float32)
        has = np.empty(n, dtype=np.uint8)
        ar = np.arange(chunk)
        for a in range(0, n, chunk):
            b = min(a + chunk, n)
            m = b - a
            rr = rows[a:b]
            sc = self.score[rr]
            tot = sc.sum(axis=1)
            c = np.argmax(sc, axis=1)
            conf[a:b] = (sc[ar[:m], c] / np.maximum(tot, 1e-9)).astype(np.float32)
            cls[a:b] = c
            nob = np.maximum(self.n_obs[rr], 1)[:, None].astype(np.float64)
            xyz[a:b] = (self.xyz[rr] / nob).astype(np.float32)
            nrgb = self.n_rgb[rr]
            h = nrgb > 0
            r = np.empty((m, 3), dtype=np.float32)
            r[:] = NO_RGB_COLOR
            if h.any():
                r[h] = self.rgb[rr][h] / nrgb[h][:, None].astype(np.float32)
            rgb[a:b] = np.clip(r, 0, 255).astype(np.uint8)
            has[a:b] = h
            _t.sleep(0)
        return xyz, rgb, cls, conf, has

    def _reduce_all(self, n, chunk=1 << 18, keep=None):
        """The rows == arange(n) case, in chunks, yielding the GIL between them.

        Two problems with the one-shot form.  (a) The six fancy-index gathers are
        pure overhead when rows is the whole map: self.xyz[rows] alone measured
        43.8 ms at 2.7 M voxels and np.tile of the sentinel grey another 11.0 ms.
        (b) A single 276 ms mean / 599 ms max snapshot on the publisher thread
        stalls the fuse thread for up to 6 frames at 10 Hz -- MEASURED as the
        cause of all 7 backpressure drops in the seq07 rate-1.0 run, which
        occurred at frames 696/725/744/821/879/975/1061, i.e. only once the map
        was large enough for the snapshot to be expensive.
        Chunking is EXACT -- identical arithmetic per element, just batched -- and
        time.sleep(0) between chunks gives the fuse thread the GIL back.

        keep (v0.3): optional boolean over [0, n).  Rows that are False are
        WITHHELD from the output.  The withholding happens inside the chunk loop,
        so the slice gathers stay slices; converting this branch to the fancy-index
        path instead would cost +177.8 ms (MEASURED).  keep=None is bit-identical
        to v0.2.
        """
        import time as _t
        if keep is None:
            out_n = n
            offs = None
        else:
            cnt = np.array([int(keep[a:min(a + chunk, n)].sum())
                            for a in range(0, n, chunk)], dtype=np.int64)
            offs = np.concatenate(([0], np.cumsum(cnt)))
            out_n = int(offs[-1])
        xyz = np.empty((out_n, 3), dtype=np.float32)
        rgb = np.empty((out_n, 3), dtype=np.uint8)
        cls = np.empty(out_n, dtype=np.uint16)
        conf = np.empty(out_n, dtype=np.float32)
        has = np.empty(out_n, dtype=np.uint8)
        ar = np.arange(chunk)
        for ci, a in enumerate(range(0, n, chunk)):
            b = min(a + chunk, n)
            m = b - a
            sc = self.score[a:b]
            tot = sc.sum(axis=1)
            c = np.argmax(sc, axis=1)
            cf = (sc[ar[:m], c] / np.maximum(tot, 1e-9)).astype(np.float32)
            nob = np.maximum(self.n_obs[a:b], 1)[:, None].astype(np.float64)
            xy = (self.xyz[a:b] / nob).astype(np.float32)
            nrgb = self.n_rgb[a:b]
            h = nrgb > 0
            r = np.empty((m, 3), dtype=np.float32)
            r[:] = NO_RGB_COLOR
            if h.any():
                np.divide(self.rgb[a:b], nrgb[:, None].astype(np.float32),
                          out=r, where=h[:, None])
            rg = np.clip(r, 0, 255).astype(np.uint8)
            if keep is None:
                conf[a:b] = cf
                cls[a:b] = c
                xyz[a:b] = xy
                rgb[a:b] = rg
                has[a:b] = h
            else:
                k = keep[a:b]
                u, v = int(offs[ci]), int(offs[ci + 1])
                conf[u:v] = cf[k]
                cls[u:v] = c[k]
                xyz[u:v] = xy[k]
                rgb[u:v] = rg[k]
                has[u:v] = h[k]
            _t.sleep(0)
        return xyz, rgb, cls, conf, has

    def snapshot(self, stride_voxel=None, max_points=3_000_000, k_free=0):
        """(xyz f4 (M,3), rgb u8 (M,3), class u16, confidence f4, has_rgb u1).

        If the map holds more voxels than max_points it is downsampled onto a
        grid f times coarser (f integer, so the coarse cell is an exact union of
        fine cells and the whole thing is integer arithmetic on the stored voxel
        keys -- no (N,3) float pass).  One fine voxel represents each coarse
        cell, and the composite sort key puts the voxels that actually carry a
        camera colour first, so a coarse cell keeps its RGB whenever any of its
        1-2 fine members had one.  The chosen f is cached: it only ever grows.

        k_free > 0 (v0.3) WITHHOLDS every voxel carrying at least that many
        free-space votes.  This is publish-time filtering, not deletion: the row
        keeps its score / rgb / centroid and reappears the moment the evidence is
        reset by a re-observation, so the policy is a runtime flag rather than a
        destructive act.  The filter is applied in BOTH branches -- the coarsening
        one the live feed takes and the slice one finish() writes the .npz from --
        because otherwise the saved map and RViz would disagree.  In the coarsening
        branch the withheld voxels are removed BEFORE the representative is chosen,
        so a coarse cell that also holds a static fine voxel keeps it.
        """
        n = self.n
        if n == 0:
            z = np.zeros((0, 3), dtype=np.float32)
            return (z, z.astype(np.uint8), np.zeros(0, np.uint16),
                    np.zeros(0, np.float32), np.zeros(0, np.uint8))

        dyn = self.dyn_mask(n, k_free)
        f = self._auto_f
        if stride_voxel is not None:
            f = max(1, int(round(stride_voxel / self.voxel)))
        if f <= 1 and n <= max_points:
            self._auto_f = 1
            if dyn is None:
                return self._reduce(slice(0, n))
            return self._reduce_all(n, keep=~dyn)

        keys = self.key[:n]
        gx, gy, gz = self._decode(keys)
        has_rgb_flag = (self.n_rgb[:n] == 0).astype(np.int64)   # 0 sorts first
        alive = np.flatnonzero(~dyn) if dyn is not None else None
        f = max(f, 2)
        while True:
            cx = np.floor_divide(gx, f) + _HALF
            cy = np.floor_divide(gy, f) + _HALF
            cz = np.floor_divide(gz, f) + _HALF
            ck = ((cx & _MASK) << (2 * _BITS)) | ((cy & _MASK) << _BITS) | (cz & _MASK)
            ck = ck * 2 + has_rgb_flag                # prefer a coloured member
            if alive is None:
                uk, rep = np.unique(ck // 2, return_index=True)
            else:
                uk, rp = np.unique(ck[alive] // 2, return_index=True)
                rep = alive[rp]
            self._auto_f = f
            if len(uk) <= max_points or f > 64:
                return self._reduce(rep)
            f += 1


# --------------------------------------------------------------------------- #
#  v0.3 -- FREE-SPACE EVIDENCE by range-image visibility.
# --------------------------------------------------------------------------- #
class FreeSpaceCarver(object):
    """Accumulate per-voxel evidence that a voxel's contents have LEFT.

    THE TEST.  A sweep gives, for every (azimuth, elevation) bin it looked into,
    the range of the NEAREST return in that direction.  A map voxel that falls in
    such a bin at range r_v is behind that surface if r_img > r_v + margin -- the
    beam went straight through where the voxel says something is, and terminated
    farther away.  That is one vote of free-space EVIDENCE, never a verdict: a
    single-sweep test mislabels 12-30 % of PARKED cars (prior-art measurement),
    so the verdict needs k_free independent votes and is reversed by any
    re-observation.

    WHY A RANGE IMAGE AND NOT RAY CASTING.  Explicit volumetric traversal of the
    same criterion costs 6.0-8.0 M voxel crossings per sweep at 0.2 m -- 377-503 ms
    of hash probing, 11-15x the entire wall-clock slack, and the two independent
    2025/2026 C++ systems that do it honestly (FreeDOM 62 ms/scan on an i9-13900HX,
    Raymoval 93.8 ms with 92.9 % of it in the ray-cast cache) confirm the price.
    Projection and table lookup are cheap; explicit traversal is not.  The criterion
    is identical; only its evaluation changes from O(rays x steps) to O(N + K).

    THE FOUR GUARDS, each against a MEASURED failure mode.

    G1 range gate (r_max).  An azimuth bin subtends a fixed ANGLE, so its footprint
       grows with range: at 450 columns (0.8 deg) it is 0.35 m at 25 m -- the voxel
       size -- and 0.84 m at 60 m, far wider than a voxel, at which point the
       "nearest return in this direction" stops being a statement about this voxel.
       Candidates outside [r_min, r_max] get NO verdict.
    G2 semantic candidate gate.  Only voxels whose current dominant class is
       potentially-movable are ever tested.  Road / terrain / sidewalk therefore
       CANNOT be carved by the grazing near-horizontal beams that are the classic
       free-space failure, and a wrong verdict can only ever cost a vehicle-shaped
       surface.  This is the one place the semantics earns its keep; it selects
       where to spend effort and never decides.
    G3 per-bin de-skew.  The ego moves up to 1.27 m (p90 1.06 m) DURING one 104 ms
       sweep -- 1.8-2.1x the 0.6 m margin, and directed along travel, which is
       exactly where the ghosts are.  A single sweep pose would therefore bias the
       test in the direction that makes it look like it works.  Candidates are
       transformed with the pose of the time bin their own azimuth falls in (two
       passes: the first picks the bin, the second is the measurement).
    G4 occlusion-boundary conservatism.  The image stores the MINIMUM range per
       bin, so a bin straddling a near object and far background keeps the near
       value and cannot carve.  Coarser azimuth is therefore SAFER, not worse.
       dil_el / dil_az extend that to a neighbourhood min (Dynablox's 26-neighbour
       rule and DUFOMap's d_p, projected), which also absorbs pose error.

    Evidence is reset, not decayed, by a re-observation when reset_on_seen: the
    voxel must be free on k_free CONSECUTIVE informative sweeps.  That makes "I no
    longer believe there is something here" a reversible soft state.
    """

    # Rows are uniform in SIN(elevation), not in the angle.  Over the HDL-64E
    # vertical FOV that is 0.44-0.49 deg per row at n_el=64 -- as uniform in angle
    # as it needs to be -- and it costs one divide (z / r) instead of an arcsin on
    # every point of every sweep.  The SAME transform is applied to the candidates,
    # so the correspondence is exact by construction rather than by calibration.
    SIN_MIN = float(np.sin(np.radians(-25.0)))
    SIN_MAX = float(np.sin(np.radians(4.0)))
    INV_2PI = float(1.0 / (2.0 * np.pi))

    def __init__(self, n_az=450, n_el=64, margin=0.6, r_max=25.0, r_min=3.0,
                 k_free=5, reset_on_seen=1, dil_el=0, dil_az=0,
                 sin_min=None, sin_max=None):
        self.n_az = int(n_az)
        self.n_el = int(n_el)
        self.margin = float(margin)
        self.r_max = float(r_max)
        self.r_min = float(r_min)
        self.k_free = int(k_free)
        self.reset_on_seen = int(reset_on_seen)
        self.dil_el = int(dil_el)
        self.dil_az = int(dil_az)
        self.s0 = float(sin_min if sin_min is not None else self.SIN_MIN)
        self.s1 = float(sin_max if sin_max is not None else self.SIN_MAX)
        self._ks = self.n_el / (self.s1 - self.s0)
        self._ka = self.n_az * self.INV_2PI
        self._buf = np.empty((self.n_el + 2) * self.n_az, dtype=np.float32)
        self._img = self._buf[self.n_az:(self.n_el + 1) * self.n_az].reshape(
            self.n_el, self.n_az)
        self.stats = dict(cand=0, tested=0, free=0, seen=0, frames=0, box=0)
        self.probe = None          # optional dict row -> diagnostics, see carve()

    # -------------------------------------------------------------- image
    def build_image(self, p_L):
        """p_L (N,3) float32 RAW sensor-frame sweep -> (n_el, n_az) min range.

        Empty bins hold +inf, which (a) makes them fail the finite check so they
        can never produce a verdict and (b) is the identity for the neighbourhood
        minimum, so dilation ignores them instead of being poisoned by them.
        """
        # The three columns are copied out CONTIGUOUSLY first.  Working on the
        # stride-12 views of an (N,3) array costs 1.0 ms of the 1.5 ms this whole
        # method takes: the copies pay for themselves three times over.
        x = np.ascontiguousarray(p_L[:, 0], dtype=np.float32)
        y = np.ascontiguousarray(p_L[:, 1], dtype=np.float32)
        z = np.ascontiguousarray(p_L[:, 2], dtype=np.float32)
        r = np.sqrt(x * x + y * y + z * z, dtype=np.float32)
        rs = np.maximum(r, 1e-3)
        col = ((np.arctan2(y, x) + np.pi) * self._ka).astype(np.int32)
        np.clip(col, 0, self.n_az - 1, out=col)
        # Out-of-FOV rows are folded onto two GUARD rows rather than masked out:
        # that keeps the scatter a single flat np.minimum.at over the whole sweep
        # and removes three 116 k fancy-index gathers.  The guard rows are never
        # read back.
        row = ((z / rs - self.s0) * self._ks).astype(np.int32)
        row += 1
        np.clip(row, 0, self.n_el + 1, out=row)
        buf = self._buf
        buf.fill(np.inf)
        np.minimum.at(buf, row * self.n_az + col, r)
        return self._img

    def dilate(self, img):
        """Neighbourhood minimum: wrap in azimuth, clamp in elevation."""
        if self.dil_el == 0 and self.dil_az == 0:
            return img
        out = img
        for d in range(1, self.dil_az + 1):
            out = np.minimum(out, np.minimum(np.roll(img, d, axis=1),
                                             np.roll(img, -d, axis=1)))
        base = out
        for d in range(1, self.dil_el + 1):
            up = np.empty_like(base); up[:-d] = base[d:]; up[-d:] = base[-1]
            dn = np.empty_like(base); dn[d:] = base[:-d]; dn[:d] = base[0]
            out = np.minimum(out, np.minimum(up, dn))
        return out

    # -------------------------------------------------------------- carve
    def carve(self, vmap, p_raw, R_WL, o_W, az0, span):
        """One sweep of evidence.  MUTATES vmap.n_free / vmap.n_seen only.

        p_raw  (N,3) float32  the RAW body-frame sweep (NOT de-skewed -- the image
                              is indexed by the sensor's own angles, so de-skewing
                              the sweep would be wrong, not merely wasteful)
        R_WL   (NB,3,3)       world <- LiDAR rotation per de-skew time bin
        o_W    (NB,3)         LiDAR origin in world per de-skew time bin.  NOT one
                              origin: the sweep origin travels 0.48-0.85 m on seq07,
                              2-4 voxels at 0.2 m.
        az0, span             azimuth of the first gated point and the swept
                              azimuth, used only to map a candidate's azimuth to
                              its own TIME bin.
        """
        self.stats["frames"] += 1
        k = vmap.n_cand
        if k == 0 or not vmap.dyn:
            return 0, 0, 0
        nb = len(o_W)
        mid = nb // 2
        R32 = np.ascontiguousarray(R_WL, dtype=np.float32)
        O32 = np.ascontiguousarray(o_W, dtype=np.float32)
        C = vmap._cand_xyz
        cx = C[:k, 0]; cy = C[:k, 1]; cz = C[:k, 2]
        c = O32[mid]
        R = self.r_max + vmap.voxel

        # --- G1 as a cheap box, narrowing one axis at a time.  MEASURED 0.85 ms at
        # 270 k candidates; the (K,3) .all(1) form is 7.6 ms.
        i = np.flatnonzero(np.abs(cx - c[0]) < R)
        i = i[np.abs(cy[i] - c[1]) < R]
        i = i[np.abs(cz[i] - c[2]) < R]
        self.stats["box"] += len(i)
        if i.size == 0:
            return 0, 0, 0
        # --- G2: the CURRENT dominant class, not the one it had when first indexed
        i = i[MOVABLE_NUSC[vmap.dom[vmap._cand_rows[i]]]]
        if i.size == 0:
            return 0, 0, 0
        rows = vmap._cand_rows[i]
        P = C[i]                      # ONE (k,3) gather, then contiguous columns
        px = np.ascontiguousarray(P[:, 0]); py = np.ascontiguousarray(P[:, 1])
        pz = np.ascontiguousarray(P[:, 2])
        self.stats["cand"] += len(i)

        # --- G3 pass 1: mid-sweep pose only, to pick each candidate's TIME bin
        Rm = R32[mid]
        vx = px - c[0]; vy = py - c[1]; vz = pz - c[2]
        lx = Rm[0, 0] * vx + Rm[1, 0] * vy + Rm[2, 0] * vz
        ly = Rm[0, 1] * vx + Rm[1, 1] * vy + Rm[2, 1] * vz
        ph = np.mod(np.arctan2(ly, lx) - az0, 2.0 * np.pi)
        b = (ph * (nb / span)).astype(np.int32)
        np.clip(b, 0, nb - 1, out=b)

        # --- G3 pass 2: the measurement, with that bin's own pose.  Nine scalar
        # gathers of length-nb float32 lanes, NOT a (k,3,3) matrix gather: same
        # arithmetic, ~5x less memory traffic.
        Rf = R32.reshape(nb, 9)
        vx = px - O32[b, 0]; vy = py - O32[b, 1]; vz = pz - O32[b, 2]
        lx = Rf[b, 0] * vx + Rf[b, 3] * vy + Rf[b, 6] * vz
        ly = Rf[b, 1] * vx + Rf[b, 4] * vy + Rf[b, 7] * vz
        lz = Rf[b, 2] * vx + Rf[b, 5] * vy + Rf[b, 8] * vz
        r = np.sqrt(lx * lx + ly * ly + lz * lz)
        row = ((lz / np.maximum(r, 1e-3) - self.s0) * self._ks).astype(np.int32)
        m = (r >= self.r_min) & (r <= self.r_max) & (row >= 0) & (row < self.n_el)
        if not m.any():
            return len(i), 0, 0
        rows = rows[m]; r = r[m]; row = row[m]
        col = ((np.arctan2(ly[m], lx[m]) + np.pi) * self._ka).astype(np.int32)
        np.clip(col, 0, self.n_az - 1, out=col)

        img = self.build_image(p_raw)
        seen_img = img[row, col]                       # centre cell: did we LOOK?
        looked = np.isfinite(seen_img)
        near = self.dilate(img)[row, col] if (self.dil_el or self.dil_az) else seen_img
        free = looked & (near > r + self.margin)       # G4
        seen = looked & (np.abs(seen_img - r) <= self.margin)

        self.stats["tested"] += int(looked.sum())
        nf = int(free.sum()); ns = int(seen.sum())
        self.stats["free"] += nf
        self.stats["seen"] += ns
        if self.probe is not None:
            p = self.probe
            tr = p["rows"]
            hit = np.isin(rows, tr)
            if hit.any():
                p["looked"] += int(looked[hit].sum())
                p["free"] += int(free[hit].sum())
                p["seen"] += int(seen[hit].sum())
                p["inbox"] += int(hit.sum())
        if nf:
            rf = rows[free]
            cur = vmap.n_free[rf]
            vmap.n_free[rf] = np.where(cur < 65535, cur + 1, cur)
        if ns:
            rsn = rows[seen]
            cur = vmap.n_seen[rsn]
            vmap.n_seen[rsn] = np.where(cur < 65535, cur + 1, cur)
            if self.reset_on_seen:
                vmap.n_free[rsn] = 0
        return len(i), nf, ns


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
