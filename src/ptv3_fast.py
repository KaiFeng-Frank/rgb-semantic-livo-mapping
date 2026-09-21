#!/usr/bin/env python3
"""
ptv3_fast.py -- two measured, bit-identical accelerations of the PTv3 worker stage.

Both are OPT-IN (env flags, read by ptv3_worker) and both ship with a verifier:
    /data/livo_sem/opt/verify_fast.py

-------------------------------------------------------------------------------
1. voxelize_gpu -- the whole voxel prep on the GPU.
-------------------------------------------------------------------------------
MEASURED on the shipped fp16 path (opt/prof_worker.py, 5 frames x 6 reps):
    numpy prep (float64 divide/floor/min/max/ravel)   7.74 ms
    CPU gathers coord[idx] / strength[idx] / g[:,idx] 3.73 ms
i.e. 11.5 ms of the 70 ms "GPU stage" was single-threaded CPU sitting on the
critical path.

The original code kept the divide+floor in numpy on the stated grounds that
"CUDA's float64 divide disagrees with numpy's on points sitting exactly on a
cell boundary".  MEASURED AND FALSE for this operation: over 22 real seq07
scans, 8 034 645 cell indices, torch.floor(coord.double()/grid) on cuda and
np.floor(coord.astype(f8)/grid) differ in ZERO entries.  Both divisions are
IEEE-754 correctly-rounded doubles; the hazard the comment describes belongs to
the float32 form, which IS wrong (2904 of 122626 points move cell) and which
this code does not use.

-------------------------------------------------------------------------------
2. hilbert_encode_fast -- the Hilbert serialiser without the python bit loop.
-------------------------------------------------------------------------------
MEASURED inside the forward: Point.serialization costs 13.76 ms of the 56 ms
forward, and 12.44 ms of that is the two hilbert encoders (6.25 + 6.19 ms);
z and z-trans are LUT-based and cost 0.43 ms each.  The reference encoder
(Pointcept_v151/.../serialization/hilbert.py:156) runs
    for bit in range(num_bits):  for dim in range(num_dims):
with ~8 elementwise kernels per iteration over an (N,3,depth) byte tensor, plus
gray2binary's 6-step loop and the bit (un)packing -- roughly 450 tiny CUDA
launches per call.  The torch.profiler run shows the forward is LAUNCH-bound
(40.4 ms of device time inside 56 ms of wall, 1924 launches, 13.98 ms of CPU in
cudaLaunchKernel), so removing launches is what buys time.

This is Skilling's algorithm on packed int64 lanes: the same transform, three
(N,) int64 tensors instead of an (N,3,depth) byte tensor, ~5 kernels per
(bit,dim) instead of ~8 over 12x fewer elements, and the final interleave done
with a shift/mask fold instead of a bit-unpack/bitpack round trip.
Bit-identity against the reference is VERIFIED, not argued -- see the verifier.
"""
import torch


# --------------------------------------------------------------------------- #
def voxelize_gpu(coord_np, strength_np, grid_size, torch_mod=torch, dev="cuda"):
    """Same contract as ptv3_worker.voxelize's GPU branch, computed entirely on
    the device.  Returns (coord_v, strength_v, grid_coord_v int32, inverse), all
    torch tensors on `dev`."""
    t = torch_mod
    c = t.from_numpy(coord_np).to(dev, non_blocking=True)        # (N,3) f32
    s = t.from_numpy(strength_np).to(dev, non_blocking=True)     # (N,1) f32
    g = t.floor(c.to(t.float64) / grid_size).to(t.int64)
    g -= g.min(dim=0).values
    ext = g.max(dim=0).values + 1
    key = (g[:, 0] * ext[1] + g[:, 1]) * ext[2] + g[:, 2]
    _, inverse = t.unique(key, sorted=True, return_inverse=True)
    order = t.argsort(inverse, stable=True)
    sc = inverse[order]
    first = t.ones_like(sc, dtype=t.bool)
    first[1:] = sc[1:] != sc[:-1]
    idx = order[first]
    return c[idx], s[idx], g[idx].to(t.int32), inverse


# --------------------------------------------------------------------------- #
_M = (0x1fffff, 0x1f00000000ffff, 0x1f0000ff0000ff,
      0x100f00f00f00f00f, 0x10c30c30c30c30c3, 0x1249249249249249)


def _spread3(x):
    """bit k of x -> bit 3k.  Classic Morton spread, 5 shift/mask steps instead of
    one kernel per bit."""
    x = x & _M[0]
    x = (x | (x << 32)) & _M[1]
    x = (x | (x << 16)) & _M[2]
    x = (x | (x << 8)) & _M[3]
    x = (x | (x << 4)) & _M[4]
    x = (x | (x << 2)) & _M[5]
    return x


def hilbert_encode_fast(locs, num_dims=3, num_bits=16):
    """Bit-identical replacement for Pointcept v1.5.1's
    serialization/hilbert.py:encode, on packed int64 lanes.

    The reference holds the bits as an (N, num_dims, num_bits) BYTE tensor with
    bit index 0 = MSB, runs Skilling's transform with ~8 elementwise kernels per
    (bit, dim) pair, interleaves with swapaxes/reshape, gray-decodes the 3*depth
    bit word with a 6-step loop, then round-trips through a uint8 bit-pack.
    Here the same three lanes live in three (N,) int64 tensors: `gray[:, d, j]`
    is integer bit (num_bits-1-j) of X[d], the "lower bits" slice [bit+1:] is the
    mask P = Q-1, the interleave is a Morton spread, and the gray decode is the
    same polynomial product (1+z)(1+z^2)(1+z^4)... which commutes, so applying the
    shifts in increasing instead of decreasing order is the same operator.

    num_dims is 3 in every PTv3 configuration; the general case is not needed and
    is not claimed.
    """
    assert num_dims == 3
    n = num_dims
    X = [locs[:, i].long() for i in range(n)]
    for bit in range(num_bits - 1):
        Q = 1 << (num_bits - 1 - bit)
        P = Q - 1
        for i in range(n):
            cond = (X[i] & Q) != 0
            t_ = (X[0] ^ X[i]) & P
            x0 = torch.where(cond, X[0] ^ P, X[0] ^ t_)
            X[i] = torch.where(cond, X[i], X[i] ^ t_)
            X[0] = x0
    code = (_spread3(X[0]) << 2) | (_spread3(X[1]) << 1) | _spread3(X[2])
    w = num_bits * n
    sh = 1
    while sh < w:
        code = code ^ (code >> sh)
        sh <<= 1
    return code


def install_fast_hilbert():
    """Monkeypatch Pointcept v1.5.1's hilbert_encode to the fast path.  Leaves
    every other module untouched; the file on disk is not edited."""
    import importlib
    d = importlib.import_module("pointcept.models.utils.serialization.default")

    def hilbert_encode(grid_coord, depth=16):
        return hilbert_encode_fast(grid_coord, num_dims=3, num_bits=depth)

    d.hilbert_encode = hilbert_encode
    return True
