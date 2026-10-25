#!/usr/bin/env python3
"""Parent-side handle for ptv3_worker.py (importable from the rclpy node).

TWO TRANSPORTS
--------------
PIPE (legacy): the request is the full float32[N,4] payload written to the worker's
stdin.  A POSIX pipe holds 64 KiB; the request is 1.96 MB.  The write therefore
BLOCKS until the worker drains it, and the worker only drains after it has finished
the previous frame.  Over stdio submit/collect CANNOT be split -- a naive split
deadlocks with both sides blocked on pipe buffers.  Kept only as a fallback.

SHM (default): the payload rides a /dev/shm ring of NSLOT input and output slots and
the pipe carries a 10-byte request and a 22-byte response.  Both fit in the 64 KiB
pipe buffer, so submit() never blocks and the node can overlap its own CPU stage with
the worker's GPU stage.

SLOT LIFETIME
-------------
in_flight is capped at NSLOT-2 (2 with the default NSLOT=4) so seq and seq-NSLOT can
never alias: by the time the parent writes input slot s for seq+4, the collector has
already taken seq+2, which means the worker finished seq and the fuse stage finished
processing seq's output slot.
"""
import os
import math
import select
import struct
import subprocess
import sys
import time
import numpy as np

PTV3_PY = os.environ.get("PTV3_PYTHON", "/opt/miniconda3/envs/ptv3/bin/python")
WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ptv3_worker.py")

REQ = struct.Struct("<IBI")        # seq, slot, n            -> 9 bytes after the tag
RESP = struct.Struct("<IBIdI")     # seq, slot, n, ms, nvox  -> 21 bytes after the tag
MEMORY_CONTEXT = struct.Struct('<QdI16d')  # frame, sensor time, coordinate epoch, T_memory_sensor


class Backpressure(Exception):
    pass


class PTv3Client(object):
    _cancel_event = None

    def __init__(self, log=None, intensity_scale=0.2, stderr_path=None,
                 use_shm=True, nslot=4, max_pts=200000, grid_size=None,
                 half=None, shuffle=None, fast_voxel=None, tf32=None,
                 memory_checkpoint=None, memory_shadow=False,
                 startup_timeout_s=120., response_timeout_s=10., cancel_event=None):
        if any(not math.isfinite(v) or v <= 0 for v in (startup_timeout_s, response_timeout_s)):
            raise ValueError('Worker timeouts must be finite and positive')
        self._cancel_event = cancel_event
        if cancel_event is not None and cancel_event.is_set():
            raise InterruptedError('PTv3 worker cancelled')
        self._read_timeout_s = startup_timeout_s
        env = dict(os.environ)
        # Memory is opt-in at the client, never inherited accidentally by legacy callers.
        env.pop('PTV3_MEMORY_CHECKPOINT', None)
        env.pop('PTV3_MEMORY_SHADOW', None)
        self.memory_enabled = bool(memory_checkpoint or memory_shadow)
        if self.memory_enabled and not use_shm:
            raise ValueError('Temporal memory requires shared-memory transport')
        if memory_checkpoint:
            env['PTV3_MEMORY_CHECKPOINT'] = os.path.abspath(memory_checkpoint)
        if memory_shadow:
            env['PTV3_MEMORY_SHADOW'] = '1'
        env["PTV3_INTENSITY_SCALE"] = "%.4f" % intensity_scale
        if grid_size is not None:
            env["PTV3_GRID_SIZE"] = "%.6f" % grid_size
        for k, v in (("PTV3_HALF", half), ("PTV3_SHUFFLE", shuffle),
                     ("PTV3_FAST_VOXEL", fast_voxel), ("PTV3_TF32", tf32)):
            if v is not None:
                env[k] = str(int(v))
        # keep the conda env's own libs first; strip ROS python paths
        env.pop("PYTHONPATH", None)

        self.use_shm = bool(use_shm)
        self.nslot = int(nslot)
        self.max_pts = int(max_pts)
        self.shm = None
        if self.use_shm:
            from multiprocessing import shared_memory
            in_bytes = self.nslot * self.max_pts * 16
            lab_bytes = self.nslot * self.max_pts * 2
            conf_bytes = self.nslot * self.max_pts * 4
            self.shm = shared_memory.SharedMemory(
                create=True, size=in_bytes + lab_bytes + conf_bytes)
            b = self.shm.buf
            self.in_np = np.frombuffer(b, dtype=np.float32,
                                       count=self.nslot * self.max_pts * 4,
                                       offset=0).reshape(self.nslot, self.max_pts, 4)
            self.lab_np = np.frombuffer(b, dtype=np.int16,
                                        count=self.nslot * self.max_pts,
                                        offset=in_bytes).reshape(self.nslot, self.max_pts)
            self.conf_np = np.frombuffer(b, dtype=np.float32,
                                         count=self.nslot * self.max_pts,
                                         offset=in_bytes + lab_bytes
                                         ).reshape(self.nslot, self.max_pts)
            env["PTV3_SHM"] = "1"
            env["PTV3_SHM_NAME"] = self.shm.name
            env["PTV3_SHM_NSLOT"] = str(self.nslot)
            env["PTV3_SHM_MAXPTS"] = str(self.max_pts)
        else:
            env["PTV3_SHM"] = "0"

        errf = self._errf = None
        try:
            errf = open(stderr_path, "wb") if stderr_path else None
            self._errf = errf
            self.p = subprocess.Popen(
                [PTV3_PY, "-u", WORKER],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=errf if errf else None, env=env, bufsize=0)
            hdr = self._readn(4)
            if hdr != b"RDY0":
                raise RuntimeError("ptv3 worker failed to start (got %r); see %s"
                                   % (hdr, stderr_path))
            self.num_classes = struct.unpack("<I", self._readn(4))[0]
        except Exception:
            if hasattr(self, 'p'):
                self.p.kill()
                self.p.wait(timeout=5)
                self.p.stdin.close()
                self.p.stdout.close()
            self._unlink_shm()
            if errf:
                errf.close()
            raise
        self._read_timeout_s = response_timeout_s
        self.last_infer_ms = 0.0
        self.last_voxels = 0
        self.seq = 0
        self.pending = []                     # FIFO of (seq, slot, n)
        self.max_inflight = max(1, self.nslot - 2) if self.use_shm else 1

    # ----------------------------------------------------------------- io
    def _readn(self, n):
        """Read with the original deadline; optional cancellation stays responsive.

        The constructing/calling thread owns teardown. Startup exceptions are
        cleaned here by __init__; runtime callers must close in their finally.
        """
        buf = bytearray()
        deadline = time.monotonic() + self._read_timeout_s
        while len(buf) < n:
            if self._cancel_event is not None and self._cancel_event.is_set():
                raise InterruptedError('PTv3 worker cancelled')
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.p.kill()
                raise TimeoutError('PTv3 worker response exceeded %.3f s' % self._read_timeout_s)
            wait = min(remaining, .1) if self._cancel_event is not None else remaining
            ready = select.select([self.p.stdout], [], [], wait)[0]
            if self._cancel_event is not None and self._cancel_event.is_set():
                raise InterruptedError('PTv3 worker cancelled')
            if not ready:
                if self._cancel_event is None:
                    self.p.kill()
                    raise TimeoutError('PTv3 worker response exceeded %.3f s' % self._read_timeout_s)
                continue
            c = os.read(self.p.stdout.fileno(), n - len(buf))
            if not c:
                raise RuntimeError("ptv3 worker died")
            buf += c
        return bytes(buf)

    # ----------------------------------------------------------------- async
    def can_submit(self):
        return len(self.pending) < self.max_inflight

    def submit(self, pts_n4, *, memory_context=None):
        """Non-blocking.  Copies the scan into a shm slot and writes 10 bytes."""
        if not self.use_shm:
            raise RuntimeError("submit() requires the shm transport")
        if len(self.pending) >= self.max_inflight:
            raise Backpressure()
        if self.memory_enabled != (memory_context is not None):
            raise ValueError('Temporal client requests require an explicit memory context')
        meta = b''
        if memory_context is not None:
            transform = np.asarray(memory_context['transform'], dtype=np.float64)
            stamp, frame, epoch = memory_context['stamp'], memory_context['frame'], memory_context.get('epoch', 0)
            if transform.shape != (4, 4) or not np.isfinite(transform).all() or not np.isfinite(stamp):
                raise ValueError('Invalid temporal context')
            if type(frame) is not int or frame < 0 or type(epoch) is not int or epoch < 0:
                raise ValueError('Invalid frame/epoch')
            meta = MEMORY_CONTEXT.pack(frame, stamp, epoch, *transform.ravel())
        n = len(pts_n4)
        if n > self.max_pts:
            raise RuntimeError("scan of %d points exceeds shm slot of %d" % (n, self.max_pts))
        slot = self.seq % self.nslot
        self.in_np[slot, :n] = pts_n4[:, :4]
        self.p.stdin.write((b'T' if self.memory_enabled else b'H') + REQ.pack(self.seq, slot, n) + meta)
        self.p.stdin.flush()
        h = (self.seq, slot, n)
        self.pending.append(h)
        self.seq += 1
        return h

    def collect(self):
        """Blocks until the oldest outstanding scan comes back.

        Returns (lab uint16 view, conf float32 view) into the shm output slot.
        Valid until the slot is reused, which the in-flight cap makes impossible
        before the caller has finished with it.
        """
        if not self.pending:
            raise RuntimeError("collect() with nothing pending")
        tag = self._readn(1)
        assert tag == b"r", tag
        seq, slot, n, ms, nvox = RESP.unpack(self._readn(RESP.size))
        exp = self.pending.pop(0)
        assert (seq, slot, n) == exp, ("out-of-order worker response", (seq, slot, n), exp)
        self.last_infer_ms, self.last_voxels = ms, nvox
        return self.lab_np[slot, :n].view(np.uint16), self.conf_np[slot, :n]

    # ----------------------------------------------------------------- sync
    def segment(self, pts_n4, *, memory_context=None):
        """Synchronous convenience wrapper (the legacy contract)."""
        if self.pending:
            raise RuntimeError('Collect pending asynchronous requests before segment()')
        if self.use_shm:
            self.submit(np.ascontiguousarray(pts_n4[:, :4], dtype=np.float32), memory_context=memory_context)
            lab, conf = self.collect()
            return lab.copy(), conf.copy()
        pts = np.ascontiguousarray(pts_n4[:, :4], dtype=np.float32)
        n = len(pts)
        self.p.stdin.write(b"S" + struct.pack("<I", n))
        self.p.stdin.write(pts.tobytes())
        self.p.stdin.flush()
        tag = self._readn(1)
        assert tag == b"R", tag
        m = struct.unpack("<I", self._readn(4))[0]
        lab = np.frombuffer(self._readn(m * 2), dtype=np.uint16).copy()
        conf = np.frombuffer(self._readn(m * 4), dtype=np.float32).copy()
        self.last_infer_ms, self.last_voxels = struct.unpack("<dI", self._readn(12))
        return lab, conf

    # ----------------------------------------------------------------- teardown
    def _unlink_shm(self):
        if self.shm is not None:
            shared = self.shm
            self.in_np = self.lab_np = self.conf_np = None
            try:
                shared.close()
            except BufferError:
                # A caller may still own a collect() view. Unlink regardless;
                # the mapping lives until its last exported view is released.
                pass
            finally:
                try:
                    shared.unlink()
                except FileNotFoundError:
                    pass
                self.shm = None

    def close(self):
        try:
            if self._cancel_event is not None and self._cancel_event.is_set():
                # A cancelled inference may not read Q until GPU work completes.
                self.p.kill()
            else:
                self.p.stdin.write(b"Q")
                self.p.stdin.flush()
            self.p.wait(timeout=5)
        except Exception:
            try:
                self.p.kill()
                self.p.wait(timeout=5)
            except Exception:
                pass
        self.p.stdin.close()
        self.p.stdout.close()
        self._unlink_shm()
        if self._errf:
            self._errf.close()
