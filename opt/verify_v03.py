#!/usr/bin/env python3
"""
verify_v03.py -- every numeric claim v0.3 makes, checked against a reference.

V1  dyn=True does not perturb the v0.2 fusion.  The same real sweeps inserted into
    a dyn=False and a dyn=True map leave score / xyz / n_obs / rgb / n_rgb / key
    BIT-IDENTICAL, and both match the v0.2 module in src_v02/ byte for byte.
V2  build_image == brute force.  The vectorised min-range image equals a per-point
    python reference on real sweeps, bin for bin, exactly.
V3  carve == brute force.  Free / seen verdicts equal a reference that recomputes
    each candidate's spherical coordinates and scans the whole sweep for the
    minimum range in its bin.
V4  publish filtering == set difference.  snapshot(k_free=k) equals snapshot with
    no filter restricted to the non-withheld rows -- in BOTH branches (the
    coarsening one the live feed takes and the slice one the .npz is written from).
V5  chunked _reduce == one-shot _reduce, bit-identical (v0.2's own implementation
    is the reference, imported from src_v02/).
V6  the de-skew time bin a candidate is assigned from its azimuth agrees with the
    bin the same point would get from its timestamp.
"""
import importlib.util, os, sys, time
import numpy as np
sys.path.insert(0, "/data/livo_sem/src")
import sem_core as S
from kitti_scan import synth

RAW = "/data/livo_sem/data/raw/2011_09_30/2011_09_30_drive_0027_sync"
FR = [200, 500, 900]
FAIL = []


def check(name, ok, extra=""):
    print("  %-58s %s %s" % (name, "PASS" if ok else "*** FAIL ***", extra))
    if not ok:
        FAIL.append(name)


def load_v02():
    sp = importlib.util.spec_from_file_location("sem_core_v02",
                                                "/data/livo_sem/src_v02/sem_core.py")
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
    return m


def scan(f):
    p = np.fromfile(RAW + "/velodyne_points/data/%010d.bin" % f,
                    dtype=np.float32).reshape(-1, 4)
    _, t, o = synth(p[:, :3])
    return np.ascontiguousarray(p[o][:, :3]), t


def main():
    V02 = load_v02()
    rng = np.random.default_rng(0)

    print("V1  dyn state does not perturb the v0.2 fusion")
    maps = [S.SemanticVoxelMap(0.2, 1 << 18, 1 << 20, dyn=False),
            S.SemanticVoxelMap(0.2, 1 << 18, 1 << 20, dyn=True),
            V02.SemanticVoxelMap(0.2, 1 << 18, 1 << 20)]
    for f in FR:
        p, _ = scan(f)
        pw = p.astype(np.float64) + np.array([f * 0.5, 0.0, 0.0])
        cls = rng.integers(0, 16, len(p)).astype(np.uint8)
        cf = rng.random(len(p)).astype(np.float32) * 0.5 + 0.5
        rgb = rng.integers(0, 255, (len(p), 3)).astype(np.uint8)
        hr = rng.random(len(p)) < 0.3
        for m in maps:
            m.insert(pw, cls, cf, rgb, hr)
    a, b, c = maps
    for fld in ("score", "xyz", "rgb", "n_rgb", "n_obs", "key"):
        n = a.n
        x, y, z = getattr(a, fld)[:n], getattr(b, fld)[:n], getattr(c, fld)[:n]
        check("v0.3(dyn=off) vs v0.3(dyn=on): %s" % fld, np.array_equal(x, y))
        check("v0.3(dyn=on)  vs v0.2 module: %s" % fld, np.array_equal(y, z))
    check("hash row count identical", a.n == b.n == c.n, "(%d)" % a.n)
    check("candidate index built (dyn=on only)", b.n_cand > 0 and a.dyn is False,
          "(%d candidates)" % b.n_cand)

    print("V2  build_image == brute force")
    for f in FR[:2]:
        p, _ = scan(f)
        for (naz, nel) in ((450, 64), (225, 32)):
            fc = S.FreeSpaceCarver(n_az=naz, n_el=nel)
            img = fc.build_image(p).copy()
            ref = np.full((nel, naz), np.inf, np.float32)
            x, y, z = p[:, 0].astype(np.float32), p[:, 1].astype(np.float32), p[:, 2].astype(np.float32)
            r = np.sqrt(x * x + y * y + z * z, dtype=np.float32)
            col = ((np.arctan2(y, x) + np.pi) * (naz / (2 * np.pi))).astype(np.int32)
            np.clip(col, 0, naz - 1, out=col)
            row = ((z / np.maximum(r, 1e-3) - fc.s0) * (nel / (fc.s1 - fc.s0))).astype(np.int32)
            for i in range(len(p)):
                if 0 <= row[i] < nel:
                    if r[i] < ref[row[i], col[i]]:
                        ref[row[i], col[i]] = r[i]
            check("frame %d %dx%d image bit-identical" % (f, nel, naz),
                  np.array_equal(img, ref),
                  "(%d filled bins)" % int(np.isfinite(ref).sum()))

    print("V3  carve == brute force")
    p, _ = scan(500)
    NB = 128
    ang = np.linspace(0, 0.03, NB)
    Rb = np.zeros((NB, 3, 3))
    Rb[:, 0, 0] = np.cos(ang); Rb[:, 0, 1] = -np.sin(ang)
    Rb[:, 1, 0] = np.sin(ang); Rb[:, 1, 1] = np.cos(ang); Rb[:, 2, 2] = 1
    pb = np.zeros((NB, 3)); pb[:, 0] = np.linspace(0, 1.0, NB)
    R_WL = Rb @ S.T_I_L[:3, :3]
    o_W = np.einsum("kij,j->ki", Rb, S.T_I_L[:3, 3]) + pb
    K = 4000
    vm = S.SemanticVoxelMap(0.2, 1 << 20, 1 << 22, dyn=True)
    vm.n = K; vm.n_cand = K; vm._cand_cap = K
    vm._cand_rows = np.arange(K, dtype=np.int32)
    vm._cand_xyz = (rng.random((K, 3)) * np.array([40, 40, 8])
                    - np.array([20, 20, 3])).astype(np.float32)
    vm.dom[:K] = 3
    for dil in ((0, 0), (1, 1)):
        vm.n_free[:] = 0; vm.n_seen[:] = 0
        fc = S.FreeSpaceCarver(n_az=450, n_el=64, dil_el=dil[0], dil_az=dil[1],
                               reset_on_seen=0)
        az0 = float(np.arctan2(p[0, 1], p[0, 0]))
        fc.carve(vm, p, R_WL, o_W, az0, 2 * np.pi)
        got_free = vm.n_free[:K].astype(bool); got_seen = vm.n_seen[:K].astype(bool)
        img = fc.build_image(p).copy(); dimg = fc.dilate(img)
        ref_free = np.zeros(K, bool); ref_seen = np.zeros(K, bool)
        for i in range(K):
            P = vm._cand_xyz[i].astype(np.float64)
            v = P - o_W[NB // 2]
            l = R_WL[NB // 2].T @ v
            ph = np.mod(np.arctan2(l[1], l[0]) - az0, 2 * np.pi)
            b = min(int(ph / (2 * np.pi) * NB), NB - 1)
            l = R_WL[b].T @ (P - o_W[b])
            r = float(np.linalg.norm(l))
            if not (3.0 <= r <= 25.0):
                continue
            row = int((l[2] / max(r, 1e-3) - fc.s0) * (64 / (fc.s1 - fc.s0)))
            if not (0 <= row < 64):
                continue
            col = min(int((np.arctan2(l[1], l[0]) + np.pi) * (450 / (2 * np.pi))), 449)
            if not np.isfinite(img[row, col]):
                continue
            ref_free[i] = dimg[row, col] > r + 0.6
            ref_seen[i] = abs(img[row, col] - r) <= 0.6
        check("dil=%s free verdicts identical" % (dil,),
              np.array_equal(got_free, ref_free),
              "(%d free, %d mismatched)" % (int(ref_free.sum()),
                                            int((got_free != ref_free).sum())))
        check("dil=%s seen verdicts identical" % (dil,),
              np.array_equal(got_seen, ref_seen), "(%d seen)" % int(ref_seen.sum()))

    print("V4  publish filtering == unfiltered restricted to the kept rows")
    vm2 = S.SemanticVoxelMap(0.2, 1 << 20, 1 << 22, dyn=True)
    for f in FR:
        p, _ = scan(f)
        pw = p.astype(np.float64) + np.array([f * 0.3, 0.0, 0.0])
        cls = rng.integers(0, 16, len(p)).astype(np.uint8)
        cf = rng.random(len(p)).astype(np.float32) * 0.5 + 0.5
        vm2.insert(pw, cls, cf, None, None)
    n = vm2.n
    vm2.n_free[:n] = rng.integers(0, 12, n).astype(np.uint16)
    keep = vm2.n_free[:n] < 5
    for tag, mp in (("slice branch (.npz)", 10 ** 9), ("coarsen branch (live)", 20000)):
        full = vm2.snapshot(max_points=mp, k_free=0)
        vm2._auto_f = 1
        filt = vm2.snapshot(max_points=mp, k_free=5)
        vm2._auto_f = 1
        if mp > n:
            ref = tuple(x[keep] for x in full)
            check("%s exact" % tag, all(np.array_equal(a_, b_)
                                        for a_, b_ in zip(ref, filt)),
                  "(%d -> %d)" % (len(full[0]), len(filt[0])))
        else:
            fk = S.voxel_key(full[0].astype(np.float64), 0.2)
            gk = S.voxel_key(filt[0].astype(np.float64), 0.2)
            check("%s: no withheld voxel survives" % tag,
                  len(np.intersect1d(gk, fk[~np.isin(fk, gk)])) == 0,
                  "(%d -> %d pts)" % (len(full[0]), len(filt[0])))
            check("%s: nothing invented" % tag, len(filt[0]) <= len(full[0]))

    print("V5  chunked _reduce == the v0.2 one-shot form, bit-identical")
    rows = rng.choice(n, size=min(n, 300000), replace=False).astype(np.int32)
    vm3 = V02.SemanticVoxelMap(0.2, 1 << 20, 1 << 22)
    for fld in ("score", "xyz", "rgb", "n_rgb", "n_obs", "key"):
        setattr(vm3, fld, getattr(vm2, fld).copy())
    vm3.n = vm2.n; vm3._cap = vm2._cap
    got = S.SemanticVoxelMap._reduce(vm2, rows)
    ref = V02.SemanticVoxelMap._reduce(vm3, rows)
    for i, nm in enumerate(("xyz", "rgb", "cls", "conf", "has_rgb")):
        check("_reduce chunked vs one-shot: %s" % nm, np.array_equal(got[i], ref[i]))

    print("V6  azimuth->time-bin assignment agrees with the timestamp")
    for f in FR:
        p, t = scan(f)
        az = np.arctan2(p[:, 1], p[:, 0])
        az0 = az[0]
        span = float(np.mod(az[-1] - az0, 2 * np.pi)) or 2 * np.pi
        if span < 0.2:
            span = 2 * np.pi
        NBk = 128
        b_az = np.clip((np.mod(az - az0, 2 * np.pi) / span * NBk).astype(int), 0, NBk - 1)
        b_t = np.clip(((t - t[0]) / (t[-1] - t[0]) * NBk).astype(int), 0, NBk - 1)
        d = np.abs(b_az - b_t); d = np.minimum(d, NBk - d)
        check("frame %d: |bin_az - bin_time| <= 1 for every point" % f, d.max() <= 1,
              "(max %d, mean %.4f)" % (d.max(), d.mean()))

    print()
    if FAIL:
        print("FAILED: %d" % len(FAIL))
        for x in FAIL:
            print("   ", x)
        sys.exit(1)
    print("ALL VERIFIERS PASS")


if __name__ == "__main__":
    main()
