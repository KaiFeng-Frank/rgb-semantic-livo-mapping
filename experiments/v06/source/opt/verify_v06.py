#!/usr/bin/env python3
"""
opt/verify_v06.py -- the v0.6 node WITHOUT --pose-topic is the v0.5 node, bit for bit.

Both modules (src/_pre_v06_backup/semantic_map_node.py and src/semantic_map_node.py) are
loaded in one process with the SAME argument line the v0.5 `rt` runs used, the PTv3
client is replaced by a stub that hands both nodes the SAME cached per-point predictions
(out/v04/arms/B0_r1, bag order), and the same real sweeps + images are pushed through
cb_image / cb_lidar.  Then every map array, every counter and the written .npz are
compared with np.array_equal (dtype included).  PTv3's run-to-run non-determinism (R2)
is taken out by the stub, so any difference would be the code change -- and the check
is the SAME real code path (stage A -> queue -> fuse thread -> stage B -> insert).

usage: source /opt/ros/jazzy/setup.bash; python3 opt/verify_v06.py [--frames 40] [--f0 100]
"""
import argparse, importlib.util, os, re, sys, time, types, collections
import numpy as np

B = "/data/livo_sem"
RAW = B + "/data/raw/2011_09_30/2011_09_30_drive_0027_sync"
PRED = B + "/out/v04/arms/B0_r1"
sys.path.insert(0, B + "/src")
from kitti_scan import synth
import rclpy
from sensor_msgs.msg import PointCloud2, PointField, Image
from std_msgs.msg import Header
from builtin_interfaces.msg import Time as TimeMsg

FAIL = []


def check(name, ok, extra=""):
    print("  %-60s %s %s" % (name, "PASS" if ok else "*** FAIL ***", extra), flush=True)
    if not ok:
        FAIL.append(name)


def parse_ts(path):
    import calendar
    out = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        d, c = line.split(" ")
        hms, frac = (c.split(".") + ["0"])[:2]
        y, mo, dd = (int(v) for v in d.split("-"))
        h, mi, s = (int(v) for v in hms.split(":"))
        out.append(calendar.timegm((y, mo, dd, h, mi, s, 0, 0, 0)) * 10**9 + int((frac + "000000000")[:9]))
    return np.array(out, dtype=np.int64)


def load_module(path, name):
    sp = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    return m


def build_args(module_path, argv):
    """exec the argparse block of main() (self-contained) with a given argv."""
    src = open(module_path).read()
    m = re.search(r"def main\(\):\n(.*?)    args, _ = ap\.parse_known_args\(\)\n", src, re.S)
    body = m.group(1) + "    args, _ = ap.parse_known_args()\n"
    body = "\n".join(l[4:] if l.startswith("    ") else l for l in body.split("\n"))
    ns = dict(argparse=__import__("argparse"))
    old = sys.argv
    sys.argv = ["x"] + argv
    try:
        exec(body, ns)
    finally:
        sys.argv = old
    return ns["args"]


class StubPTv3(object):
    """deterministic stand-in for PTv3Client: returns cached predictions in submission order."""
    num_classes = 16
    last_infer_ms = 0.0

    def __init__(self, cache, **kw):
        self.cache = cache
        self.k = 0
        self.pending = collections.deque()

    def can_submit(self):
        return len(self.pending) < 2

    def submit(self, xyzi):
        lab, conf = self.cache[self.k]
        assert len(lab) == len(xyzi), (self.k, len(lab), len(xyzi))
        self.pending.append(self.k)
        self.k += 1

    def collect(self):
        k = self.pending.popleft()
        lab, conf = self.cache[k]
        return lab.view(np.uint16), conf

    def segment(self, xyzi):
        self.submit(xyzi)
        lab, conf = self.collect()
        return lab.copy(), conf.copy()

    def close(self):
        pass


def make_pc(stamp_ns, arr):
    m = PointCloud2()
    m.header = Header(stamp=TimeMsg(sec=int(stamp_ns // 10**9), nanosec=int(stamp_ns % 10**9)),
                      frame_id="velodyne")
    m.height = 1
    m.width = len(arr)
    m.fields = [PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
                PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
                PointField(name="time", offset=16, datatype=PointField.FLOAT32, count=1),
                PointField(name="ring", offset=20, datatype=PointField.UINT16, count=1)]
    m.is_bigendian = False
    m.point_step = arr.dtype.itemsize
    m.row_step = m.point_step * m.width
    m.is_dense = True
    m.data = arr.tobytes()
    return m


def make_img(stamp_ns, rgb):
    m = Image()
    m.header = Header(stamp=TimeMsg(sec=int(stamp_ns // 10**9), nanosec=int(stamp_ns % 10**9)),
                      frame_id="cam2")
    m.height, m.width = rgb.shape[:2]
    m.encoding = "rgb8"
    m.is_bigendian = False
    m.step = m.width * 3
    m.data = rgb.tobytes()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--f0", type=int, default=100)
    a = ap.parse_args()
    PC_DTYPE = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("intensity", "<f4"),
                         ("time", "<f4"), ("ring", "<u2")])
    ts = parse_ts(RAW + "/velodyne_points/timestamps_start.txt")
    te = parse_ts(RAW + "/velodyne_points/timestamps_end.txt")
    ti = parse_ts(RAW + "/image_02/timestamps.txt")
    try:
        from PIL import Image as PILImage
        have_pil = True
    except Exception:
        have_pil = False
    frames = list(range(a.f0, a.f0 + a.frames))
    # ---- the sweeps exactly as kitti_to_ros2bag writes them (+ cached predictions in bag order)
    msgs, imgs, cache = [], [], []
    for f in frames:
        pts = np.fromfile(RAW + "/velodyne_points/data/%010d.bin" % f, dtype=np.float32).reshape(-1, 4)
        ring, tt, order = synth(pts[:, :3], sweep_duration=(te[f] - ts[f]) / 1e9)
        q = pts[order]
        arr = np.empty(len(q), dtype=PC_DTYPE)
        arr["x"], arr["y"], arr["z"], arr["intensity"] = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        arr["time"] = (tt * 1e6).astype(np.float32)
        arr["ring"] = ring[order] if len(ring) == len(q) else ring
        msgs.append(make_pc(int(ts[f]), arr))
        with np.load(os.path.join(PRED, "f%06d.npz" % f)) as z:
            lab = z["lab"].astype(np.int16)[order]
            conf = z["conf"].astype(np.float32)[order]
        cache.append((np.ascontiguousarray(lab), np.ascontiguousarray(conf)))
        if have_pil:
            rgb = np.asarray(PILImage.open(RAW + "/image_02/data/%010d.png" % f).convert("RGB"))
            imgs.append(make_img(int(ti[f]), np.ascontiguousarray(rgb)))
        else:
            imgs.append(None)
    print("prepared %d sweeps (%s images)" % (len(msgs), "with" if have_pil else "WITHOUT"))

    argv = ["--reliable", "--conf-gate", "0.5", "--expect-voxels", "4000000",
            "--ptv3-ckpt", B + "/weights/v05/B0_student.pth", "--bag-rate", "1.0"]
    rclpy.init()
    nodes = {}
    for tag, path in (("old", B + "/src/_pre_v06_backup/semantic_map_node.py"),
                      ("new", B + "/src/semantic_map_node.py")):
        mod = load_module(path, "smn_" + tag)
        c = [(l.copy(), cf.copy()) for l, cf in cache]
        mod.PTv3Client = lambda **kw: StubPTv3(c, **kw)
        args = build_args(path, argv + ["--stats-out", B + "/out/v06/tmp/verify_stats_%s.json" % tag,
                                        "--npz-out", B + "/out/v06/tmp/verify_map_%s.npz" % tag])
        nodes[tag] = mod.SemanticMapNode(args)
        print("%s node up: online=%s" % (tag, getattr(nodes[tag], "online", "n/a")))
    # ---- drive both nodes through the identical message sequence
    for i, (pc, im) in enumerate(zip(msgs, imgs)):
        for tag, n in nodes.items():
            if im is not None:
                n.cb_image(im)
            before = n.stats["processed"] + n.stats["dropped_backpressure"] + n.stats["no_pose"]
            n.cb_lidar(pc)
            t0 = time.time()
            while n.stats["processed"] + n.stats["dropped_backpressure"] + n.stats["no_pose"] <= before:
                time.sleep(0.001)
                if time.time() - t0 > 20:
                    raise RuntimeError("%s node stalled at sweep %d" % (tag, i))
    for n in nodes.values():
        n._stop_fuse.set(); n._fuse_thread.join(timeout=30)
        n._stop_pub.set(); n._pub_thread.join(timeout=30)
    o, nw = nodes["old"], nodes["new"]
    print("processed old %d new %d, bp-drop %d/%d, no-pose %d/%d" % (
        o.stats["processed"], nw.stats["processed"], o.stats["dropped_backpressure"],
        nw.stats["dropped_backpressure"], o.stats["no_pose"], nw.stats["no_pose"]))
    check("same sweeps processed / dropped / no-pose",
          all(o.stats[k] == nw.stats[k] for k in ("recv", "processed", "dropped_backpressure",
                                                  "no_pose", "pts_in", "pts_mapped", "pts_rgb")))
    check("map voxel count", o.map.n == nw.map.n, "%d vs %d" % (o.map.n, nw.map.n))
    n = o.map.n
    names = [k for k in ("key", "score", "xyz", "n_obs", "rgb", "n_rgb", "n_seen", "n_free", "dom")
             if hasattr(o.map, k) and isinstance(getattr(o.map, k), np.ndarray)]
    for k in names:
        A, Bm = getattr(o.map, k), getattr(nw.map, k)
        ok = A.dtype == Bm.dtype and A.shape == Bm.shape and np.array_equal(A[:n], Bm[:n])
        check("map.%s identical (dtype %s, %s)" % (k, A.dtype, A.shape), ok)
    # hash table (VoxelHash: key int64 slots, row int32 slots)
    check("hash.key identical (%d slots)" % o.map.h.cap, np.array_equal(o.map.h.key, nw.map.h.key))
    check("hash.row identical", np.array_equal(o.map.h.row, nw.map.h.row))
    check("map.n_points_inserted identical", o.map.n_points_inserted == nw.map.n_points_inserted)
    # written outputs
    for nd in (o, nw):
        nd.finish()
    zo = np.load(B + "/out/v06/tmp/verify_map_old.npz"); zn = np.load(B + "/out/v06/tmp/verify_map_new.npz")
    check("written .npz keys", set(zo.files) == set(zn.files))
    for k in zo.files:
        check("npz[%s] identical" % k, zo[k].dtype == zn[k].dtype and np.array_equal(zo[k], zn[k]))
    import json
    so = json.load(open(B + "/out/v06/tmp/verify_stats_old.json"))
    sn = json.load(open(B + "/out/v06/tmp/verify_stats_new.json"))
    static = [k for k in so if not k.endswith("_ms") and k not in ("wall_clock_s", "qdepth", "peak_rss_gb", "snapshot_ms")]
    diff = [k for k in static if so.get(k) != sn.get(k)]
    check("stats.json non-timing fields identical (%d fields)" % len(static), not diff, str(diff))
    check("new stats has no pose_mode block without --pose-topic", "pose_mode" not in sn)
    print("\n%s" % ("ALL PASS" if not FAIL else "FAILED: %s" % FAIL))
    for nd in nodes.values():
        try:
            nd.destroy_node()
        except Exception:
            pass
    rclpy.shutdown()
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
