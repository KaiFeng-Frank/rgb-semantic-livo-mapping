#!/usr/bin/env python3
"""
ptv3_worker.py -- PTv3 semantic segmentation as a stdio co-process.

WHY A SEPARATE PROCESS
----------------------
ROS 2 Jazzy's rclpy is built against the system CPython 3.12; the verified PTv3
stack (torch 2.5.1+cu124, spconv-cu124, flash_attn) lives in the conda env
`ptv3` on CPython 3.10.  The two ABIs cannot share one interpreter, and
rebuilding either side would invalidate the already-verified numbers.  So the
rclpy fusion node spawns this file under
    /opt/miniconda3/envs/ptv3/bin/python
and talks to it over two pipes (~2 MB out, ~0.7 MB back per 122k-point scan).

MODEL LOAD PATH -- NON-NEGOTIABLE
--------------------------------
Uses ptv3_loader_verified.build_ptv3, i.e. **Pointcept v1.5.1** (72a7993), the
version the released nuScenes checkpoint was trained against: 488/488 tensors,
strict=True, no key renaming.
DO NOT use src/ptv3_infer.py.  It points at Pointcept HEAD and papers over the
architecture change by renaming cls_mode -> enc_mode.  The keys then line up but
the forward pass does not: measured over 20 seq07 frames against SemanticKITTI
GT it gives 25.66 % point accuracy and predicts `car` on 5 points out of 2.4 M,
versus 86.34 % / car IoU 91.22 % on this path.

MEASURED CONSTANTS (CRITICAL_CONSTRAINTS.md -- do not change blindly)
  intensity_scale = 0.2   86.3 % point acc / 61.3 % coarse mIoU on v1.5.1
                          (x1.0 -> 57.5 % / 32.1 %, road IoU collapses 84 -> 8)
  grid_size       = 0.05  one representative point per voxel, scatter back via
                          the inverse index
  TTA             = OFF   the config ships a 10-way multi-scale+flip TTA which
                          would multiply latency by 10

PROTOCOL (little-endian, binary, on a private dup of fd 1)
  worker -> parent, once:  b"RDY0" + uint32 num_classes
  parent -> worker:  b"S" + uint32 N + float32[N,4]  (x, y, z, intensity RAW)
  worker -> parent:  b"R" + uint32 N + uint16[N] label + float32[N] conf
                         + float64 infer_ms + uint32 n_voxels
  parent -> worker:  b"Q"  -> clean exit
"""
import os
import sys
import struct
import time

sys.path.insert(0, "/data/livo_sem/src")
import numpy as np

INTENSITY_SCALE = float(os.environ.get("PTV3_INTENSITY_SCALE", "0.2"))
GRID_SIZE = float(os.environ.get("PTV3_GRID_SIZE", "0.05"))


# --------------------------------------------------------------------------- #
def voxelize(coord, strength, grid_size=GRID_SIZE, torch_mod=None):
    """Same contract as Pointcept GridSample(mode='train', return_inverse=True,
    return_grid_coord=True): one representative point per grid cell plus the
    inverse index from every original point to its cell.

    Two benign differences from the reference implementation:
      * exact 3-D raveling instead of the FNV hash -> zero collisions;
      * the representative is the cell's first point in input order rather than
        a uniformly random one (GridSample's choice is arbitrary anyway).
    The feature `coord` handed to the network is the ORIGINAL, unshifted sensor
    coordinate, exactly as GridSample leaves data_dict['coord'].
    """
    scaled = coord.astype(np.float64) / grid_size
    g = np.floor(scaled).astype(np.int64)
    g -= g.min(0)
    ext = g.max(0) + 1
    key = (g[:, 0] * ext[1] + g[:, 1]) * ext[2] + g[:, 2]
    if torch_mod is not None:
        # The expensive part (sort/unique over ~122k int64 keys) on the GPU; the
        # floor/divide stays in numpy because CUDA's float64 divide disagrees with
        # numpy's on points sitting exactly on a cell boundary.
        kt = torch_mod.from_numpy(key).cuda(non_blocking=True)
        _, inverse = torch_mod.unique(kt, sorted=True, return_inverse=True)
        order = torch_mod.argsort(inverse, stable=True)
        sc = inverse[order]
        first = torch_mod.ones_like(sc, dtype=torch_mod.bool)
        first[1:] = sc[1:] != sc[:-1]
        idx_first = order[first].cpu().numpy()
        return (coord[idx_first], strength[idx_first], g[idx_first].astype(np.int32),
                inverse)
    uk, idx_first, inverse = np.unique(key, return_index=True, return_inverse=True)
    return (coord[idx_first], strength[idx_first], g[idx_first].astype(np.int32),
            inverse.astype(np.int64))


class Segmenter(object):
    def __init__(self, device="cuda", intensity_scale=INTENSITY_SCALE):
        import torch
        from ptv3_loader_verified import build_ptv3, NUSCENES_CLASSES
        self.torch = torch
        self.model, self.cfg = build_ptv3(device=device)
        self.device = torch.device(device)
        self.classes = NUSCENES_CLASSES
        self.num_classes = int(self.cfg.model.num_classes)
        self.intensity_scale = float(intensity_scale)
        self.grid_size = GRID_SIZE
        self.last_voxels = 0

    def segment(self, pts_n4):
        torch = self.torch
        coord = np.ascontiguousarray(pts_n4[:, :3], dtype=np.float32)
        strength = np.ascontiguousarray(
            pts_n4[:, 3:4] * self.intensity_scale, dtype=np.float32)
        cv, sv, gv, inv = voxelize(coord, strength, self.grid_size, torch)
        self.last_voxels = len(cv)
        dev = self.device
        with torch.inference_mode():
            c = torch.from_numpy(cv).to(dev)
            s = torch.from_numpy(sv).to(dev)
            gc = torch.from_numpy(gv).to(dev)
            feat = torch.cat([c, s], dim=1)                 # feat_keys=('coord','strength')
            inp = dict(coord=c, grid_coord=gc, feat=feat,
                       offset=torch.tensor([c.shape[0]], device=dev, dtype=torch.long))
            logits = self.model(inp)["seg_logits"]
            prob = torch.softmax(logits.float(), dim=-1)
            conf_v, lab_v = prob.max(dim=-1)
            iv = inv if isinstance(inv, torch.Tensor) else torch.from_numpy(inv).to(dev)
            lab = lab_v[iv].to(torch.int32).cpu().numpy().astype(np.uint16)
            conf = conf_v[iv].cpu().numpy().astype(np.float32)
        return lab, conf

    def warmup(self, n=122626, iters=3):
        rng = np.random.default_rng(0)
        fake = np.empty((n, 4), np.float32)
        fake[:, :3] = rng.uniform(-40, 40, (n, 3))
        fake[:, 2] = rng.uniform(-2.0, 2.0, n)
        fake[:, 3] = rng.uniform(0, 1, n)
        for _ in range(iters):
            self.segment(fake)
        self.torch.cuda.synchronize()


# --------------------------------------------------------------------------- #
def _readn(f, n):
    buf = bytearray()
    while len(buf) < n:
        c = f.read(n - len(buf))
        if not c:
            return None
        buf += c
    return bytes(buf)


def main():
    # Pointcept / spconv print banners on import and on first forward.  Steal
    # fd 1 for the binary protocol BEFORE anything else can write to it and
    # point normal stdout at stderr so banners land in the log, not the pipe.
    proto_fd = os.dup(1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    out = os.fdopen(proto_fd, "wb", buffering=0)

    seg = Segmenter()
    sys.stderr.write("[ptv3_worker] Pointcept v1.5.1 loader, strict=True, "
                     "intensity_scale=%.2f grid=%.3f classes=%d\n"
                     % (seg.intensity_scale, seg.grid_size, seg.num_classes))
    seg.warmup()
    sys.stderr.write("[ptv3_worker] warm\n")
    sys.stderr.flush()

    inp = sys.stdin.buffer
    out.write(b"RDY0" + struct.pack("<I", seg.num_classes))

    while True:
        tag = _readn(inp, 1)
        if tag is None or tag == b"Q":
            break
        if tag != b"S":
            sys.stderr.write("[ptv3_worker] bad tag %r\n" % tag)
            break
        n = struct.unpack("<I", _readn(inp, 4))[0]
        raw = _readn(inp, n * 16)
        if raw is None:
            break
        pts = np.frombuffer(raw, dtype=np.float32).reshape(n, 4)
        t0 = time.perf_counter()
        lab, conf = seg.segment(pts)
        dt = (time.perf_counter() - t0) * 1000.0
        out.write(b"R" + struct.pack("<I", n))
        out.write(np.ascontiguousarray(lab, dtype=np.uint16).tobytes())
        out.write(np.ascontiguousarray(conf, dtype=np.float32).tobytes())
        out.write(struct.pack("<dI", dt, int(seg.last_voxels)))


if __name__ == "__main__":
    main()
