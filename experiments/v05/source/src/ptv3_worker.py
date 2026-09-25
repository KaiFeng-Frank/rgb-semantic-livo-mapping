#!/usr/bin/env python3
"""
ptv3_worker.py -- PTv3 semantic segmentation as a co-process.

WHY A SEPARATE PROCESS
----------------------
ROS 2 Jazzy's rclpy is built against the system CPython 3.12; the verified PTv3
stack (torch 2.5.1+cu124, spconv-cu124, flash_attn) lives in the conda env
`ptv3` on CPython 3.10.  The two ABIs cannot share one interpreter, and
rebuilding either side would invalidate the already-verified numbers.  So the
rclpy fusion node spawns this file under
    /data/miniconda3/envs/ptv3/bin/python
and talks to it over two pipes plus (optionally) a /dev/shm payload ring.

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

RUNTIME KNOBS (env), all re-scored with src/eval_report.py -- see CRITICAL_CONSTRAINTS.md (R1-R8)
  PTV3_SHUFFLE      shuffle_orders.  The frozen config ships 1 and v1.5.1 applies
                    it inside forward() ungated by self.training, so .eval() does
                    NOT disable it: every forward draws a fresh torch.randperm and
                    the node's output is non-deterministic run to run.  Default 0.
  PTV3_HALF         fp16 weights + fp16 feats.  The checkpoint was trained under
                    fp16 autocast (config.py: enable_amp=True) and flash-attention
                    already casts qkv to fp16 inside every block, so fp32 mode pays
                    a round trip for nothing.  NOT torch.autocast: spconv's
                    implicit_gemm rejects fp16 activations with fp32 weights.
  PTV3_FAST_VOXEL   int32 + transposed-reduction voxel prep.  Bit-identical to the
                    int64 form (verified with np.array_equal on grid and key).
  PTV3_TF32         TF32 for nn.Linear and spconv.  Only meaningful in fp32 mode.
  PTV3_GPU_VOXEL    run the whole voxel prep on the device.  Removes 11.5 ms of
                    single-threaded numpy (float64 divide/floor/min/max/ravel plus
                    three CPU fancy-gathers) from the critical path.  Bit-identical
                    on all four outputs over 22 real scans -- see opt/verify_fast.py.
  PTV3_FAST_HILBERT replace v1.5.1's pure-python Hilbert bit loop with the packed
                    int64 form.  MEASURED 12.44 of the 13.76 ms that
                    Point.serialization costs inside the 56 ms forward.
                    Bit-identical over 2.2 M codes (random depths 10-13, both
                    axis orders, and real scans).

PROTOCOL (little-endian, binary, on a private dup of fd 1)
  worker -> parent, once:  b"RDY0" + uint32 num_classes
  PIPE MODE (legacy, PTV3_SHM=0)
    parent -> worker:  b"S" + uint32 N + float32[N,4]  (x, y, z, intensity RAW)
    worker -> parent:  b"R" + uint32 N + uint16[N] label + float32[N] conf
                           + float64 infer_ms + uint32 n_voxels
  SHM MODE (default): the payload rides /dev/shm and the pipe carries control only
    parent -> worker:  b"H" + uint32 seq + uint8 slot + uint32 N          (10 B)
    worker -> parent:  b"r" + uint32 seq + uint8 slot + uint32 N
                           + float64 infer_ms + uint32 n_voxels           (22 B)
  parent -> worker:  b"Q"  -> clean exit

  WHY SHM IS A PREREQUISITE, NOT A 4 ms OPTIMISATION: a POSIX pipe holds 64 KiB.
  The 1.96 MB request write therefore BLOCKS until the worker drains it, and the
  worker only drains after finishing the previous frame.  Over stdio, submit/collect
  cannot be split -- a naive split deadlocks (both sides blocked on pipe buffers).
"""
import os
import sys
import struct
import time

sys.path.insert(0, "/data/livo_sem/src")
import numpy as np

INTENSITY_SCALE = float(os.environ.get("PTV3_INTENSITY_SCALE", "0.2"))
GRID_SIZE = float(os.environ.get("PTV3_GRID_SIZE", "0.05"))
HALF = int(os.environ.get("PTV3_HALF", "0")) != 0
SHUFFLE = int(os.environ.get("PTV3_SHUFFLE", "0")) != 0
FAST_VOXEL = int(os.environ.get("PTV3_FAST_VOXEL", "1")) != 0
TF32 = int(os.environ.get("PTV3_TF32", "0")) != 0
GPU_VOXEL = int(os.environ.get("PTV3_GPU_VOXEL", "1")) != 0
FAST_HILBERT = int(os.environ.get("PTV3_FAST_HILBERT", "1")) != 0
CKPT = os.environ.get("PTV3_CKPT", "").strip() or None   # v0.5: None = released
USE_SHM = int(os.environ.get("PTV3_SHM", "1")) != 0
SHM_NAME = os.environ.get("PTV3_SHM_NAME", "")
SHM_NSLOT = int(os.environ.get("PTV3_SHM_NSLOT", "4"))
SHM_MAXPTS = int(os.environ.get("PTV3_SHM_MAXPTS", "200000"))

REAL_SCAN_DIR = ("/data/livo_sem/data/raw/2011_09_30/"
                 "2011_09_30_drive_0027_sync/velodyne_points/data/")
WARM_FRAMES = [0, 200, 400, 600, 800, 1000]


# --------------------------------------------------------------------------- #
def voxelize(coord, strength, grid_size=GRID_SIZE, torch_mod=None, fast=FAST_VOXEL):
    """Same contract as Pointcept GridSample(mode='train', return_inverse=True,
    return_grid_coord=True): one representative point per grid cell plus the
    inverse index from every original point to its cell.

    Two benign differences from the reference implementation:
      * exact 3-D raveling instead of the FNV hash -> zero collisions;
      * the representative is the cell's first point in input order rather than
        a uniformly random one (GridSample's choice is arbitrary anyway).
    The feature `coord` handed to the network is the ORIGINAL, unshifted sensor
    coordinate, exactly as GridSample leaves data_dict['coord'].

    `fast` selects the int32 + transposed-reduction form.  MEASURED bit-identical
    (np.array_equal on both the grid array and the key array, real seq07 scans):
    6.84 ms -> 1.69 ms on one core.  The 5.3 ms it removes was a single line,
    `g -= g.min(0); ext = g.max(0)+1`, whose axis-0 reduction over a C-contiguous
    (122626,3) int64 array is a stride-3 walk.
    DO NOT 'simplify' the float64 divide to float32: measured 2904 of 122626 points
    then land in a different cell, far worse than the ~14 of 82k the FNV-vs-ravel
    comment anticipates.
    """
    if fast:
        scaled = coord.astype(np.float64)
        scaled /= grid_size
        np.floor(scaled, out=scaled)
        gT = scaled.T.astype(np.int32)              # (3,N), C-contiguous
        gT -= gT.min(axis=1)[:, None]
        ext = (gT.max(axis=1) + 1).astype(np.int64)
        key = gT[0].astype(np.int64) * ext[1]
        key += gT[1]
        key *= ext[2]
        key += gT[2]
        g_src = gT
    else:
        scaled = coord.astype(np.float64) / grid_size
        g = np.floor(scaled).astype(np.int64)
        g -= g.min(0)
        ext = g.max(0) + 1
        key = (g[:, 0] * ext[1] + g[:, 1]) * ext[2] + g[:, 2]
        g_src = g

    def _gsel(idx):
        if fast:
            return np.ascontiguousarray(g_src[:, idx].T)
        return g_src[idx].astype(np.int32)

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
        return coord[idx_first], strength[idx_first], _gsel(idx_first), inverse
    uk, idx_first, inverse = np.unique(key, return_index=True, return_inverse=True)
    return coord[idx_first], strength[idx_first], _gsel(idx_first), inverse.astype(np.int64)


class Segmenter(object):
    def __init__(self, device="cuda", intensity_scale=INTENSITY_SCALE,
                 grid_size=GRID_SIZE, half=HALF, shuffle=SHUFFLE, tf32=TF32,
                 fast_voxel=FAST_VOXEL, gpu_voxel=GPU_VOXEL,
                 fast_hilbert=FAST_HILBERT, ckpt=CKPT):
        import torch
        from ptv3_loader_verified import build_ptv3, NUSCENES_CLASSES
        self.torch = torch
        if tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            try:
                import spconv as _sp
                _sp.constants.SPCONV_ALLOW_TF32 = True
            except Exception:
                pass
        self.model, self.cfg = build_ptv3(device=device, shuffle_orders=bool(shuffle),
                                          half=bool(half), ckpt_path=ckpt)
        self.ckpt = getattr(self.model, "_ckpt_path", None)
        self.ckpt_tensors = int(getattr(self.model, "_n_ckpt_tensors", 0))
        try:
            import hashlib
            with open(self.ckpt, "rb") as _f:
                self.ckpt_sha256 = hashlib.sha256(_f.read()).hexdigest()
        except Exception:
            self.ckpt_sha256 = "?"
        self.device = torch.device(device)
        self.classes = NUSCENES_CLASSES
        self.num_classes = int(self.cfg.model.num_classes)
        self.intensity_scale = float(intensity_scale)
        self.grid_size = float(grid_size)
        self.half = bool(half)
        self.shuffle = bool(shuffle)
        self.tf32 = bool(tf32)
        self.fast_voxel = bool(fast_voxel)
        self.gpu_voxel = bool(gpu_voxel)
        self.fast_hilbert = bool(fast_hilbert)
        if self.fast_hilbert:
            import ptv3_fast
            ptv3_fast.install_fast_hilbert()
        self.feat_dtype = torch.float16 if half else torch.float32
        self.last_voxels = 0

    def segment(self, pts_n4, out_lab=None, out_conf=None):
        """pts_n4 (N,4) float32.  Returns (label uint16 [N], conf float32 [N]).

        out_lab / out_conf, when given, are pre-allocated CPU uint16/float32 views
        (normally /dev/shm slots) that the device-to-host copy writes straight into,
        which deletes the `.to(int32).cpu().numpy().astype(uint16)` cast chain.
        """
        torch = self.torch
        coord = np.ascontiguousarray(pts_n4[:, :3], dtype=np.float32)
        strength = np.ascontiguousarray(
            pts_n4[:, 3:4] * self.intensity_scale, dtype=np.float32)
        dev = self.device
        fd = self.feat_dtype
        if self.gpu_voxel:
            import ptv3_fast
            with torch.inference_mode():
                c, s, gc, inv = ptv3_fast.voxelize_gpu(coord, strength,
                                                       self.grid_size, torch, dev)
            self.last_voxels = int(c.shape[0])
        else:
            cv, sv, gv, inv = voxelize(coord, strength, self.grid_size, torch,
                                       fast=self.fast_voxel)
            self.last_voxels = len(cv)
            c = s = gc = None
        with torch.inference_mode():
            if not self.gpu_voxel:
                c = torch.from_numpy(cv).to(dev)
                s = torch.from_numpy(sv).to(dev)
                gc = torch.from_numpy(gv).to(dev)
            feat = torch.cat([c.to(fd), s.to(fd)], dim=1)   # feat_keys=('coord','strength')
            inp = dict(coord=c.to(fd), grid_coord=gc, feat=feat,
                       offset=torch.tensor([c.shape[0]], device=dev, dtype=torch.long))
            logits = self.model(inp)["seg_logits"]
            prob = torch.softmax(logits.float(), dim=-1)
            conf_v, lab_v = prob.max(dim=-1)
            iv = inv if isinstance(inv, torch.Tensor) else torch.from_numpy(inv).to(dev)
            lab_g = lab_v[iv].to(torch.int16)
            conf_g = conf_v[iv]
            if out_lab is not None:
                out_lab.copy_(lab_g)
                out_conf.copy_(conf_g)
                return None, None
            lab = lab_g.cpu().numpy().view(np.uint16)
            conf = conf_g.cpu().numpy()
        return lab, conf

    def warmup(self, iters=2):
        """Warm on SEVERAL REAL scans, not on uniform noise.

        The old warm-up drew rng.uniform over an 80x80x4 box: nearly every point
        lands in its own voxel (~122k voxels vs 82.7k real) and the grid extent and
        sparse_shape differ from any scan the node will ever see, so spconv's
        per-shape algorithm tuning cache was primed on shapes that never recur and
        the first real frames paid the autotune again.  MEASURED: warming on ONE
        real scan still leaves a start-up transient -- the node dropped its first
        5 scans inside the first ~40 frames while spconv re-tuned for each new
        voxel count.  Warming across six scans spanning the sequence (78k-93k
        voxels) primes the tuner over the range the node will actually see.
        """
        scans = []
        try:
            for k in WARM_FRAMES:
                fp = REAL_SCAN_DIR + "%010d.bin" % k
                if os.path.exists(fp):
                    scans.append(np.fromfile(fp, dtype=np.float32).reshape(-1, 4))
        except Exception:
            scans = []
        if not scans:
            rng = np.random.default_rng(0)
            pts = np.empty((122626, 4), np.float32)
            pts[:, :3] = rng.uniform(-40, 40, (122626, 3))
            pts[:, 2] = rng.uniform(-2.0, 2.0, 122626)
            pts[:, 3] = rng.uniform(0, 1, 122626)
            scans = [pts]
        for _ in range(iters):
            for pts in scans:
                self.segment(pts)
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
                     "intensity_scale=%.2f grid=%.3f classes=%d | "
                     "shuffle_orders=%s weight_dtype=%s tf32=%s fast_voxel=%s shm=%s\n"
                     % (seg.intensity_scale, seg.grid_size, seg.num_classes,
                        seg.shuffle, "fp16" if seg.half else "fp32", seg.tf32,
                        seg.fast_voxel, USE_SHM))
    sys.stderr.write("[ptv3_worker] gpu_voxel=%s fast_hilbert=%s\n"
                     % (seg.gpu_voxel, seg.fast_hilbert))
    sys.stderr.write("[ptv3_worker] checkpoint=%s tensors=%d sha256=%s\n"
                     % (seg.ckpt, seg.ckpt_tensors, seg.ckpt_sha256))
    sys.stderr.flush()

    shm = None
    in_np = out_lab_t = out_conf_t = None
    if USE_SHM and SHM_NAME:
        import torch
        from multiprocessing import shared_memory
        shm = shared_memory.SharedMemory(name=SHM_NAME)
        # the parent created and owns this segment; keep the child's resource
        # tracker from trying to unlink it again at shutdown
        try:
            from multiprocessing import resource_tracker
            resource_tracker.unregister(shm._name, "shared_memory")
        except Exception:
            pass
        in_bytes = SHM_NSLOT * SHM_MAXPTS * 16
        buf = shm.buf
        in_np = np.frombuffer(buf, dtype=np.float32, count=SHM_NSLOT * SHM_MAXPTS * 4,
                              offset=0).reshape(SHM_NSLOT, SHM_MAXPTS, 4)
        lab_np = np.frombuffer(buf, dtype=np.int16, count=SHM_NSLOT * SHM_MAXPTS,
                               offset=in_bytes).reshape(SHM_NSLOT, SHM_MAXPTS)
        conf_np = np.frombuffer(buf, dtype=np.float32, count=SHM_NSLOT * SHM_MAXPTS,
                                offset=in_bytes + SHM_NSLOT * SHM_MAXPTS * 2
                                ).reshape(SHM_NSLOT, SHM_MAXPTS)
        out_lab_t = torch.from_numpy(lab_np)
        out_conf_t = torch.from_numpy(conf_np)
        sys.stderr.write("[ptv3_worker] shm attached %s (%d slots x %d pts)\n"
                         % (SHM_NAME, SHM_NSLOT, SHM_MAXPTS))
        sys.stderr.flush()

    seg.warmup()
    sys.stderr.write("[ptv3_worker] warm\n")
    sys.stderr.flush()

    inp = sys.stdin.buffer
    out.write(b"RDY0" + struct.pack("<I", seg.num_classes))

    while True:
        tag = _readn(inp, 1)
        if tag is None or tag == b"Q":
            break
        if tag == b"H":                     # shm control request
            seq, slot, n = struct.unpack("<IBI", _readn(inp, 9))
            pts = in_np[slot, :n]
            t0 = time.perf_counter()
            seg.segment(pts, out_lab=out_lab_t[slot, :n], out_conf=out_conf_t[slot, :n])
            dt = (time.perf_counter() - t0) * 1000.0
            out.write(b"r" + struct.pack("<IBIdI", seq, slot, n, dt,
                                         int(seg.last_voxels)))
            continue
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

    if shm is not None:
        try:
            shm.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
