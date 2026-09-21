#!/usr/bin/env python3
"""Parent-side handle for ptv3_worker.py (importable from the rclpy node)."""
import os
import struct
import subprocess
import sys
import numpy as np

PTV3_PY = os.environ.get("PTV3_PYTHON", "/opt/miniconda3/envs/ptv3/bin/python")
WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ptv3_worker.py")


class PTv3Client(object):
    def __init__(self, log=None, intensity_scale=0.2, stderr_path=None):
        env = dict(os.environ)
        env["PTV3_INTENSITY_SCALE"] = "%.4f" % intensity_scale
        # keep the conda env's own libs first; strip ROS python paths
        env.pop("PYTHONPATH", None)
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
        self.last_infer_ms = 0.0
        self.last_voxels = 0

    def _readn(self, n):
        buf = bytearray()
        while len(buf) < n:
            c = self.p.stdout.read(n - len(buf))
            if not c:
                raise RuntimeError("ptv3 worker died")
            buf += c
        return bytes(buf)

    def segment(self, pts_n4):
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

    def close(self):
        try:
            self.p.stdin.write(b"Q")
            self.p.stdin.flush()
            self.p.wait(timeout=5)
        except Exception:
            self.p.kill()
        if self._errf:
            self._errf.close()
