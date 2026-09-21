#!/usr/bin/env python3
"""
semantic_map_node.py -- RGB-semantic LIVO global mapper (rclpy, ROS 2 Jazzy).

Joins three already-verified upstream tracks:
  * FAST-LIVO2 trajectory  (TUM, T_{W<-IMU}, keyed on true sensor time)
  * PTv3 nuScenes semantics on the raw body-frame sweep  (via ptv3_worker.py)
  * KITTI rectified cam2 projection for per-point RGB

Per /velodyne_points sweep
  1. per-point absolute time  t_i = header.stamp + time[i] * 1e-6   (the bag's
     `time` field is MICROSECONDS -- the `_us` bags)
  2. T_{W<-I}(t_i) by slerp/linear INTERPOLATION of the TUM trajectory
     (nearest-frame association would inject ~0.7 m at 14 m/s)
  3. T_W_L = T_W_I @ T_I_L       <-- the contract.  FAST-LIVO2's state is the
     IMU pose; skipping this right-multiplication yields a map that looks almost
     right and is globally wrong.
  4. PTv3 on the RAW body-frame scan (intensity * 0.2) -> class + confidence
  5. RGB: the point is motion-compensated to the matched image's instant and
     projected with P_rect_02 @ R_rect_00 @ T_velo_cam0.  Outside the frustum or
     behind the camera -> has_rgb = 0, never a silent black.
  6. world points -> voxel-hashed global map

TWO-STAGE PIPELINE (2026-09-21)
-------------------------------
The scan callback used to be strictly serial: it blocked in PTv3Client.segment for
the whole 88 ms GPU stage while the CPU sat idle, then spent ~50 ms of CPU while the
GPU sat idle -- 144 ms/frame = 6.9 Hz against a 10 Hz sensor.  It is now split:

  executor thread (stage A) : header parse, per-point time, pose-validity gate,
                              copy the sweep into a /dev/shm slot, non-blocking
                              submit, hand a descriptor to a depth-2 queue.
  fuse thread     (stage B) : blocking collect, confidence gate, de-skew, camera
                              projection, map insert, /semantic_scan publish.

rclpy.spin / SingleThreadedExecutor is KEPT.  MultiThreadedExecutor with a
ReentrantCallbackGroup is the reflex and it is wrong here: two concurrent cb_lidar
instances would race on the worker pipe (whose protocol has no request id) and on
self.map, and forcing MutuallyExclusive just restores today's serialisation.  The
stage boundary we need is INSIDE one message's processing, which callback groups
cannot express.  A plain threading.Thread can.

WHY DEFERRING STAGE B IS EXACT, NOT APPROXIMATE
  * TrajInterp is immutable after __init__ and t_pt is derived purely from
    msg.header.stamp + the per-point `time` field, so the pose lookup is a pure
    function of the message.
  * Image association picks argmin |t_img - t_c| over a buffer that deferral can
    only make a SUPERSET of; the nearest element of a superset that already
    contains the true nearest is the same element.  (The 30-frame / ~3 s buffer
    dwarfs the <= 2-frame pipeline lag.)
  * One worker, one FIFO pipe, one submitter, one collector => responses arrive in
    submission order (asserted, not assumed).
  * Every accumulator in SemanticVoxelMap.insert is an additive reduction, so the
    fusion is a commutative monoid.
  => the pipelined map is bit-identical to the serial one.

Publishes
  /semantic_scan  the per-sweep incremental cloud   (every processed sweep)
  /semantic_map   a downsampled snapshot of the global map  (1 Hz by default)
Both with fields
  x f4 @0 | y f4 @4 | z f4 @8 | rgb f4 @12 | class u2 @16 | confidence f4 @20
  | has_rgb u1 @24          (point_step 28)

QoS: the LiDAR subscription is BEST_EFFORT + KEEP_LAST(1) so the node always
works on the newest sweep and drops the rest deterministically.  rosbag2's
player publishes RELIABLE/KEEP_LAST(10); a RELIABLE subscriber would queue and
drift unboundedly behind.
"""
import os
import sys
import time
import json
import queue
import argparse
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from rclpy.clock import Clock, ClockType
from sensor_msgs.msg import PointCloud2, PointField, Image
from std_msgs.msg import Header

import sem_core as S
from ptv3_client import PTv3Client, Backpressure

FIELDS = [
    PointField(name="x",          offset=0,  datatype=PointField.FLOAT32, count=1),
    PointField(name="y",          offset=4,  datatype=PointField.FLOAT32, count=1),
    PointField(name="z",          offset=8,  datatype=PointField.FLOAT32, count=1),
    PointField(name="rgb",        offset=12, datatype=PointField.FLOAT32, count=1),
    PointField(name="class",      offset=16, datatype=PointField.UINT16,  count=1),
    PointField(name="confidence", offset=20, datatype=PointField.FLOAT32, count=1),
    PointField(name="has_rgb",    offset=24, datatype=PointField.UINT8,   count=1),
]


def make_cloud(rec, frame_id, stamp):
    m = PointCloud2()
    m.header = Header()
    m.header.frame_id = frame_id
    m.header.stamp = stamp
    m.height = 1
    m.width = int(len(rec))
    m.fields = FIELDS
    m.is_bigendian = False
    m.point_step = S.CLOUD_DTYPE.itemsize
    m.row_step = m.point_step * m.width
    m.is_dense = True
    m.data = rec.tobytes()
    return m


class SemanticMapNode(Node):
    def __init__(self, args):
        super().__init__("semantic_map_node")
        self.args = args
        self.traj = S.TrajInterp(args.traj, t_min=args.t_min)
        self.get_logger().info(
            "trajectory %s: %d poses, t in [%.6f, %.6f]"
            % (os.path.basename(args.traj), len(self.traj.t), self.traj.t0, self.traj.t1))

        self.T_I_L = S.T_I_L if args.apply_T_I_L else np.eye(4)
        if not args.apply_T_I_L:
            self.get_logger().warn("T_I_L DISABLED (A3 control arm) -- map will be wrong")
        self.R_IL = np.ascontiguousarray(self.T_I_L[:3, :3])
        self.t_IL = np.ascontiguousarray(self.T_I_L[:3, 3])

        # Presize the map: see SemanticVoxelMap.__init__.  seq07 ends at 2.70 M
        # voxels and the stock caps trip BOTH growth paths in one frame (415.7 ms).
        rows = 1
        while rows < max(args.expect_voxels, 1):
            rows <<= 1
        self.map = S.SemanticVoxelMap(voxel=args.map_voxel, cap0=rows,
                                      hash_cap=rows << 2)
        self.get_logger().info("map presized: %d rows, %d hash slots (expect %d voxels)"
                               % (rows, rows << 2, args.expect_voxels))
        self.T_C_L = S.T_C_L
        self.K = np.array([[S.FX, 0, S.CX], [0, S.FY, S.CY], [0, 0, 1]])

        self.images = []          # (t, HxWx3 uint8)
        self.img_lock = threading.Lock()
        self.stats = dict(recv=0, processed=0, dropped_backpressure=0, no_pose=0,
                          pts_in=0, pts_mapped=0, pts_rgb=0,
                          t_ptv3=[], t_ptv3_gpu=[], t_total=[], t_map=[], t_proj=[],
                          t_stage_a=[], t_wait=[], t_deskew=[], t_gate=[], t_pub=[],
                          t_stage_b=[], t_period=[], qdepth=[])
        self.first_stamp = None
        self.last_hdr = None
        self.sweep_dt = 0.1039          # KITTI HDL-64E sweep period in this bag
        self.t_wall0 = None
        self.finished = False
        self.times_sorted = True
        self._last_done = None
        self._recbuf = None

        self.ptv3 = PTv3Client(intensity_scale=args.intensity_scale,
                               stderr_path=args.ptv3_log,
                               use_shm=not args.no_shm,
                               nslot=args.queue_depth + 2,
                               grid_size=args.grid_size,
                               half=args.half, shuffle=args.shuffle,
                               fast_voxel=args.fast_voxel, tf32=args.tf32)
        self.get_logger().info("PTv3 worker ready (%d classes), shm=%s pipeline=%s"
                               % (self.ptv3.num_classes, not args.no_shm,
                                  not args.no_pipeline))

        if args.reliable:
            qos_be = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                                history=HistoryPolicy.KEEP_LAST, depth=10)
            self.get_logger().warn("LiDAR subscription is RELIABLE/KEEP_LAST(10)")
        else:
            qos_be = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                                history=HistoryPolicy.KEEP_LAST, depth=args.lidar_depth)
        qos_img = QoSProfile(
            reliability=(ReliabilityPolicy.RELIABLE if args.reliable
                         else ReliabilityPolicy.BEST_EFFORT),
            history=HistoryPolicy.KEEP_LAST, depth=10)
        qos_pub = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                             history=HistoryPolicy.KEEP_LAST, depth=1,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)

        self.pub_map = self.create_publisher(PointCloud2, "/semantic_map", qos_pub)
        self.pub_scan = self.create_publisher(PointCloud2, "/semantic_scan",
                                              QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                                                         history=HistoryPolicy.KEEP_LAST, depth=1))
        self.create_subscription(Image, args.image_topic, self.cb_image, qos_img)
        self.create_subscription(PointCloud2, args.lidar_topic, self.cb_lidar, qos_be)

        # stage B
        self.q = queue.Queue(maxsize=args.queue_depth)
        self._stop_fuse = threading.Event()
        self._fuse_thread = threading.Thread(target=self.fuse_loop, daemon=True)
        self._fuse_thread.start()

        self._stop_pub = threading.Event()
        self._pub_thread = threading.Thread(target=self.map_pub_loop, daemon=True)
        self._pub_thread.start()
        self._steady = Clock(clock_type=ClockType.STEADY_TIME)
        self.create_timer(1.0, self.tick, clock=self._steady)

    # ------------------------------------------------------------------ #
    def cb_image(self, msg):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if msg.encoding not in ("rgb8", "bgr8"):
            self.get_logger().error("unexpected image encoding %s" % msg.encoding)
            return
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        if msg.encoding == "bgr8":
            img = img[:, :, ::-1]
        with self.img_lock:
            self.images.append((t, img))
            if len(self.images) > 30:
                self.images.pop(0)

    def match_image(self, t_lo, t_hi):
        with self.img_lock:
            if not self.images:
                return None, None
            ts = np.array([x[0] for x in self.images])
            tc = 0.5 * (t_lo + t_hi)
            k = int(np.argmin(np.abs(ts - tc)))
            if ts[k] < t_lo - 0.06 or ts[k] > t_hi + 0.06:
                return None, None
            return self.images[k]

    # ------------------------------------------------------------------ #
    #  STAGE A -- executor thread.  Must not touch `msg` after it returns.
    # ------------------------------------------------------------------ #
    def cb_lidar(self, msg):
        self.stats["recv"] += 1
        if self.t_wall0 is None:
            self.t_wall0 = time.time()
        t0 = time.perf_counter()
        try:
            self.stage_a(msg)
        except Exception as e:  # noqa
            import traceback
            self.get_logger().error("stage A failed: %s\n%s" % (e, traceback.format_exc()))
        self.stats["t_stage_a"].append((time.perf_counter() - t0) * 1000.0)

    def stage_a(self, msg):
        t_hdr = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(-1, msg.point_step)
        xyzi = raw[:, 0:16].copy().view(np.float32).reshape(-1, 4)
        t_off = raw[:, 16:20].copy().view(np.float32).ravel().astype(np.float64)
        t_pt = t_hdr + t_off * 1e-6          # MICROSECONDS -- the `_us` bags
        n_in = len(xyzi)
        self.stats["pts_in"] += n_in

        if self.stats["recv"] == 1 and not np.all(np.diff(t_off) >= 0):
            self.get_logger().error("per-point `time` is not sorted; bin de-skew invalid")
            self.args.deskew_bins = 1
            self.times_sorted = False

        # Pose-validity gate.  The per-point times are sorted (asserted above), so
        # the valid set is a contiguous RANGE and two binary searches replace a
        # 122 k-element boolean mask plus four fancy-index gathers.  Falls back to
        # the boolean form when the cloud is not sorted, or the range would select
        # the wrong points silently.
        if self.times_sorted:
            lo = int(np.searchsorted(t_pt, self.traj.t0, side="left"))
            hi = int(np.searchsorted(t_pt, self.traj.t1, side="right"))
            nok = hi - lo
            okmask = None
        else:
            okmask = self.traj.valid(t_pt)
            nok = int(okmask.sum())
            lo = hi = 0
        if nok < 100:
            self.stats["no_pose"] += 1
            return

        if self.args.no_pipeline:
            t_sub = time.perf_counter()
            lab, conf = self.ptv3.segment(xyzi)
            self.stats["t_wait"].append((time.perf_counter() - t_sub) * 1000.0)
            self.stats["t_ptv3_gpu"].append(float(self.ptv3.last_infer_ms))
            self.stage_b((xyzi, t_hdr, msg.header.stamp, t_pt, lo, hi, okmask,
                          t_sub), lab, conf)
            return

        if not self.ptv3.can_submit() or self.q.full():
            self.stats["dropped_backpressure"] += 1
            return
        try:
            self.ptv3.submit(xyzi)
        except Backpressure:
            self.stats["dropped_backpressure"] += 1
            return
        self.q.put_nowait((xyzi, t_hdr, msg.header.stamp, t_pt, lo, hi, okmask,
                           time.perf_counter()))

    # ------------------------------------------------------------------ #
    #  STAGE B -- fuse thread.  The ONLY writer of self.map.
    # ------------------------------------------------------------------ #
    def fuse_loop(self):
        while True:
            try:
                item = self.q.get(timeout=0.2)
            except queue.Empty:
                if self._stop_fuse.is_set():
                    break
                continue
            if item is None:
                break
            try:
                t0 = time.perf_counter()
                lab, conf = self.ptv3.collect()
                self.stats["t_wait"].append((time.perf_counter() - t0) * 1000.0)
                self.stats["t_ptv3_gpu"].append(float(self.ptv3.last_infer_ms))
                self.stage_b(item, lab, conf)
            except Exception as e:  # noqa
                import traceback
                self.get_logger().error("stage B failed: %s\n%s"
                                        % (e, traceback.format_exc()))

    def stage_b(self, item, lab, conf):
        xyzi, t_hdr, stamp, t_pt, lo, hi, okmask, t_sub = item
        t_b0 = time.perf_counter()
        self.stats["qdepth"].append(self.q.qsize())
        if self.first_stamp is None:
            self.first_stamp = t_hdr
        self.last_hdr = t_hdr

        # ---- confidence gate, fused with the pose gate into ONE index pass
        t0 = time.perf_counter()
        gate = self.args.conf_gate
        if okmask is None:
            if gate > 0:
                idx = np.flatnonzero(conf[lo:hi] >= gate)
                idx += lo
                p3 = xyzi[idx, :3].astype(np.float64)
                cls = lab[idx]
                cf = conf[idx]
                tk = t_pt[idx]
            else:
                p3 = xyzi[lo:hi, :3].astype(np.float64)
                cls = lab[lo:hi]
                cf = conf[lo:hi]
                tk = t_pt[lo:hi]
        else:
            m = okmask
            if gate > 0:
                m = m & (conf >= gate)
            idx = np.flatnonzero(m)
            p3 = xyzi[idx, :3].astype(np.float64)
            cls = lab[idx]
            cf = conf[idx]
            tk = t_pt[idx]
        n = len(p3)
        self.stats["t_gate"].append((time.perf_counter() - t0) * 1000.0)
        if n == 0:
            return

        # ---- 1+2. per-point pose, de-skewed:  p_W = T_W_I(t_i) @ T_I_L @ p_L
        # The interpolation is evaluated at NB time bins across the 104 ms sweep
        # rather than once per point -- measured max deviation from the exact
        # per-point solve is 4 mm at NB=64 (2 mm at 128) against a 200 mm voxel.
        # The hstack to homogeneous coordinates is gone: it allocated a 3.9 MB
        # (n,4) float64 and left an (n,3) NON-CONTIGUOUS view that handed BLAS 128
        # strided slices.  p3 @ R.T + t is the same arithmetic on contiguous data.
        t0 = time.perf_counter()
        pl_i = p3 @ self.R_IL.T
        pl_i += self.t_IL
        t_lo, t_hi = tk[0], tk[-1]
        NB = self.args.deskew_bins
        if NB <= 1 or t_hi <= t_lo:
            R1, p1, _ = self.traj.query(np.array([0.5 * (t_lo + t_hi)]))
            pw = pl_i @ R1[0].T + p1[0]
        else:
            ctr = t_lo + (np.arange(NB) + 0.5) * (t_hi - t_lo) / NB
            Rb, pb, _ = self.traj.query(ctr)
            bi = np.clip(((tk - t_lo) / (t_hi - t_lo) * NB).astype(np.int64), 0, NB - 1)
            edge = np.searchsorted(bi, np.arange(NB + 1))
            pw = np.empty((n, 3))
            for k in range(NB):
                a, b = edge[k], edge[k + 1]
                if b > a:
                    np.matmul(pl_i[a:b], Rb[k].T, out=pw[a:b])
                    pw[a:b] += pb[k]
        self.stats["t_deskew"].append((time.perf_counter() - t0) * 1000.0)

        # ---- 4. RGB by projecting into the time-matched image
        t0 = time.perf_counter()
        rgb = np.zeros((n, 3), dtype=np.uint8)
        has_rgb = np.zeros(n, dtype=bool)
        t_img, img = self.match_image(tk.min(), tk.max())
        if img is not None:
            T_W_I_img, okp = self.traj.query_one(t_img)
            if okp:
                # world -> LiDAR frame AT THE IMAGE INSTANT -> rectified cam2.
                # Using the de-skewed world point makes this exact even though the
                # sweep spans 104 ms while the shutter is one instant.
                M = self.T_C_L @ np.linalg.inv(T_W_I_img @ self.T_I_L)
                # Only z is needed for all n points; u and v are needed only for the
                # ~48 % in front of the camera.  The old form built a (n,4) hstack,
                # a (4,4)@(4,n) product whose .T[:, :3] was a stride-n view, a full
                # (n,2) array filled with -1e9, and four n-sized boolean temporaries.
                z = p_dot(pw, M[2])
                front = np.flatnonzero(z > 0.5)
                if front.size:
                    pf = pw[front]
                    zf = z[front]
                    u = p_dot(pf, M[0]); u /= zf; u *= S.FX; u += S.CX
                    v = p_dot(pf, M[1]); v /= zf; v *= S.FY; v += S.CY
                    H, W = img.shape[:2]
                    inb = (u >= 0) & (u <= W - 1) & (v >= 0) & (v <= H - 1)
                    if inb.any():
                        sel = front[inb]
                        ui = np.rint(u[inb]).astype(np.int32)
                        vi = np.rint(v[inb]).astype(np.int32)
                        rgb[sel] = img[vi, ui]
                        has_rgb[sel] = True
        self.stats["t_proj"].append((time.perf_counter() - t0) * 1000.0)
        self.stats["pts_rgb"] += int(has_rgb.sum())

        # ---- 5. insert into the global voxel map
        t0 = time.perf_counter()
        self.map.insert(pw, cls, cf, rgb, has_rgb)
        self.stats["t_map"].append((time.perf_counter() - t0) * 1000.0)
        self.stats["pts_mapped"] += n
        self.stats["processed"] += 1

        # ---- per-sweep incremental cloud
        t0 = time.perf_counter()
        if not self.args.no_scan_pub:
            rec = S.build_record(pw.astype(np.float32), rgb, cls.astype(np.uint16),
                                 cf, has_rgb.astype(np.uint8))
            self.pub_scan.publish(make_cloud(rec, self.args.world_frame, stamp))
        self.stats["t_pub"].append((time.perf_counter() - t0) * 1000.0)
        self.last_stamp = stamp
        now = time.perf_counter()
        self.stats["t_stage_b"].append((now - t_b0) * 1000.0)
        self.stats["t_total"].append((now - t_sub) * 1000.0)
        if self._last_done is not None:
            self.stats["t_period"].append((now - self._last_done) * 1000.0)
        self._last_done = now

    # ------------------------------------------------------------------ #
    def map_pub_loop(self):
        """Snapshot + publish the global map on its own thread.

        At ~8 M voxels a snapshot costs seconds (argmax over an 8 M x 16 score
        table plus a re-voxelisation), so doing it inside the LiDAR callback
        would stall the pipeline and make the drop statistics meaningless.
        The map arrays are append/accumulate-only, so a snapshot taken while a
        scan is being inserted is at worst one scan stale -- never corrupt.
        """
        period = 1.0 / max(self.args.map_rate, 1e-3)
        last_n = -1
        wait = period
        while not self._stop_pub.wait(wait):
            if self.map.n == last_n or self.map.n == 0:
                wait = period
                continue
            last_n = self.map.n
            try:
                self.publish_map(getattr(self, "last_stamp", None))
            except Exception as e:
                self.get_logger().warn("map publish failed: %s" % e)
            # Adaptive throttle: snapshotting an N-voxel map costs O(N); at 2.7 M
            # voxels that is hundreds of ms.  Republishing on a fixed 1 Hz tick
            # would steal the wall clock from the scan path and the drop statistics
            # would be measuring the visualiser, not the mapper.
            wait = max(period, 4.0 * getattr(self, "last_snap_ms", 0.0) / 1000.0)

    def publish_map(self, stamp):
        if stamp is None:
            from builtin_interfaces.msg import Time as TimeMsg
            stamp = TimeMsg()
        t0 = time.perf_counter()
        xyz, rgb, cls, conf, has = self.map.snapshot(
            stride_voxel=self.args.pub_voxel, max_points=self.args.max_pub_points)
        rec = S.build_record(xyz, rgb, cls, conf, has)
        self.pub_map.publish(make_cloud(rec, self.args.world_frame, stamp))
        self.last_pub_count = len(rec)
        self.last_snap_ms = (time.perf_counter() - t0) * 1000.0
        self.stats.setdefault("t_snapshot", []).append(self.last_snap_ms)

    def tick(self):
        s = self.stats
        if s["processed"] == 0:
            return
        el = time.time() - (self.t_wall0 or time.time())
        self.get_logger().info(
            "recv %d | processed %d | bp-drop %d | voxels %.2fM | pts %.1fM | "
            "rgb %.1f%% | gpu %.0f ms | stageB %.0f ms | period %.0f ms | %.1f Hz"
            % (s["recv"], s["processed"], s["dropped_backpressure"], self.map.n / 1e6,
               s["pts_mapped"] / 1e6,
               100.0 * s["pts_rgb"] / max(s["pts_mapped"], 1),
               np.mean(s["t_ptv3_gpu"][-20:]) if s["t_ptv3_gpu"] else 0.0,
               np.mean(s["t_stage_b"][-20:]),
               np.mean(s["t_period"][-20:]) if s["t_period"] else 0.0,
               s["processed"] / max(el, 1e-3)))

    # ------------------------------------------------------------------ #
    def finish(self):
        if self.finished:
            return
        self.finished = True
        # drain the pipeline BEFORE the stats: otherwise the in-flight frames are
        # silently lost from both the map and the counters.
        self._stop_fuse.set()
        self._fuse_thread.join(timeout=60)
        self._stop_pub.set()
        self._pub_thread.join(timeout=60)
        stamp = getattr(self, "last_stamp", None)
        if stamp is None:
            from builtin_interfaces.msg import Time as TimeMsg
            stamp = TimeMsg()
        self.publish_map(stamp)
        s = self.stats
        el = time.time() - (self.t_wall0 or time.time())
        # SemanticVoxelMap caches the coarsening factor and only ever GROWS it, so
        # after the live feed has been downsampled for --max-pub-points the cached
        # f would silently coarsen the saved map too.  Reset it for this snapshot.
        self.map._auto_f = 1
        xyz, rgb, cls, conf, has = self.map.snapshot(
            stride_voxel=self.args.pub_voxel, max_points=self.args.npz_max_points)
        hist = np.bincount(cls.astype(np.int64), minlength=16)

        def st(key, extra=False):
            v = s.get(key) or [0.0]
            d = dict(mean=float(np.mean(v)), p50=float(np.percentile(v, 50)),
                     p95=float(np.percentile(v, 95)), max=float(np.max(v)))
            return d

        out = dict(
            traj=self.args.traj, bag_rate=self.args.bag_rate,
            apply_T_I_L=bool(self.args.apply_T_I_L),
            pipeline=not self.args.no_pipeline, shm=not self.args.no_shm,
            ptv3_half=self.args.half, ptv3_shuffle=self.args.shuffle,
            ptv3_grid=self.args.grid_size, ptv3_fast_voxel=self.args.fast_voxel,
            scan_pub=not self.args.no_scan_pub,
            queue_depth=self.args.queue_depth, expect_voxels=self.args.expect_voxels,
            lidar_qos=("RELIABLE/KEEP_LAST(10)" if self.args.reliable
                       else "BEST_EFFORT/KEEP_LAST(%d)" % self.args.lidar_depth),
            fastdds_builtin_transports=os.environ.get("FASTDDS_BUILTIN_TRANSPORTS", "default"),
            map_voxel=self.map.voxel, pub_voxel=self.args.pub_voxel,
            conf_gate=self.args.conf_gate,
            scans_received=s["recv"], scans_processed=s["processed"],
            scans_dropped_backpressure=s["dropped_backpressure"],
            scans_no_pose=s["no_pose"],
            scans_published_est=int(round((self.last_hdr - self.first_stamp) /
                                          self.sweep_dt)) + 1
            if (self.last_hdr is not None and self.first_stamp is not None) else 0,
            points_in=int(s["pts_in"]), points_mapped=int(s["pts_mapped"]),
            points_with_rgb=int(s["pts_rgb"]),
            frac_points_with_rgb=float(s["pts_rgb"] / max(s["pts_mapped"], 1)),
            map_voxels=int(self.map.n),
            published_points=int(len(xyz)),
            published_frac_has_rgb=float(has.mean()) if len(has) else 0.0,
            wall_clock_s=el,
            ptv3_gpu_ms=st("t_ptv3_gpu"),
            stage_a_ms=st("t_stage_a"), wait_ms=st("t_wait"),
            gate_ms=st("t_gate"), deskew_ms=st("t_deskew"),
            proj_ms=st("t_proj"), map_ms=st("t_map"), scanpub_ms=st("t_pub"),
            stage_b_ms=st("t_stage_b"),
            frame_period_ms=st("t_period"),
            latency_ms=st("t_total"),
            qdepth=dict(mean=float(np.mean(s["qdepth"])) if s["qdepth"] else 0.0,
                        max=int(np.max(s["qdepth"])) if s["qdepth"] else 0),
            published_voxel_m=float(self.map.voxel * self.map._auto_f),
            snapshot_ms=dict(mean=float(np.mean(s["t_snapshot"])),
                             max=float(np.max(s["t_snapshot"])))
            if s.get("t_snapshot") else None,
            class_histogram={S.NUSCENES_CLASSES[i]: int(hist[i]) for i in range(16)},
        )
        try:
            import resource
            out["peak_rss_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
        except Exception:
            pass
        with open(self.args.stats_out, "w") as f:
            json.dump(out, f, indent=2)
        self.get_logger().info("stats -> %s" % self.args.stats_out)
        if self.args.npz_out:
            np.savez_compressed(self.args.npz_out, xyz=xyz, rgb=rgb, cls=cls,
                                conf=conf, has_rgb=has)
            self.get_logger().info("cloud -> %s" % self.args.npz_out)
        self.ptv3.close()


def p_dot(p, row):
    """p (n,3) float64 . row[:3] + row[3] -- one contiguous gemv, no hstack."""
    out = p @ row[:3]
    out += row[3]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", default="/data/livo_sem/out/kitti_seq07_fastlivo2_tum.txt")
    ap.add_argument("--t-min", type=float, default=None)
    ap.add_argument("--lidar-topic", default="/velodyne_points")
    ap.add_argument("--image-topic", default="/camera/image_raw")
    ap.add_argument("--world-frame", default="map")
    ap.add_argument("--map-voxel", type=float, default=0.20)
    ap.add_argument("--pub-voxel", type=float, default=None)
    ap.add_argument("--max-pub-points", type=int, default=1_000_000,
                    help="cap on the LIVE /semantic_map feed only.  MEASURED: "
                         "publishing the full 2.70 M-voxel map (snapshot 282 ms "
                         "mean / 665 ms max including build_record + the 75 MB DDS "
                         "publish) back-pressures the fuse thread and costs 4 of "
                         "1096 scans on seq07; at 1 M it costs none.  The saved "
                         ".npz is unaffected -- see --npz-max-points.")
    ap.add_argument("--npz-max-points", type=int, default=3_000_000,
                    help="resolution of the map written by --npz-out, independent "
                         "of the live visualiser feed")
    ap.add_argument("--map-rate", type=float, default=1.0)
    ap.add_argument("--intensity-scale", type=float, default=0.2)
    ap.add_argument("--conf-gate", type=float, default=0.0)
    ap.add_argument("--deskew-bins", type=int, default=128)
    ap.add_argument("--expect-voxels", type=int, default=4_000_000,
                    help="presize the map arrays and the hash from this bound; "
                         "seq07 ends at 2.70 M")
    ap.add_argument("--queue-depth", type=int, default=2,
                    help="submitted-but-not-fused scans.  2 = one on the GPU plus "
                         "one queued behind it.  1 idles the GPU for stage B; 3+ "
                         "adds latency with no throughput gain.")
    ap.add_argument("--no-pipeline", action="store_true",
                    help="serial arm: block in the callback exactly as before")
    ap.add_argument("--no-shm", action="store_true",
                    help="legacy pipe transport (forces --no-pipeline semantics)")
    ap.add_argument("--no-scan-pub", action="store_true",
                    help="skip the per-sweep /semantic_scan publish (timing runs)")
    ap.add_argument("--lidar-depth", type=int, default=1)
    ap.add_argument("--half", type=int, default=1, help="fp16 PTv3 weights")
    ap.add_argument("--shuffle", type=int, default=0, help="PTv3 shuffle_orders")
    ap.add_argument("--fast-voxel", type=int, default=1)
    ap.add_argument("--tf32", type=int, default=0)
    ap.add_argument("--grid-size", type=float, default=0.05)
    ap.add_argument("--reliable", action="store_true",
                    help="RELIABLE/KEEP_LAST(10) LiDAR subscription instead of the "
                         "specified BEST_EFFORT/KEEP_LAST(1); see the note in __init__")
    ap.add_argument("--no-T-I-L", dest="apply_T_I_L", action="store_false")
    ap.add_argument("--bag-rate", type=float, default=1.0)
    ap.add_argument("--idle-finish", type=float, default=12.0,
                    help="seconds with no new scan before writing stats and exiting")
    ap.add_argument("--stats-out", default="/data/livo_sem/out/semantic_map_stats.json")
    ap.add_argument("--ptv3-log", default="/data/livo_sem/logs/ptv3_worker.log")
    ap.add_argument("--npz-out", default=None)
    ap.add_argument("--hold", action="store_true",
                    help="stay alive after the bag ends so RViz can be driven")
    args, _ = ap.parse_known_args()
    if args.no_shm:
        args.no_pipeline = True

    rclpy.init()
    node = SemanticMapNode(args)
    last_seen = [0, time.time()]

    def watchdog():
        s = node.stats
        if s["recv"] != last_seen[0]:
            last_seen[0] = s["recv"]
            last_seen[1] = time.time()
            return
        if s["recv"] > 0 and time.time() - last_seen[1] > args.idle_finish and not node.finished:
            node.finish()
            node.get_logger().info("bag finished.")
            if not args.hold:
                raise SystemExit
    node.create_timer(2.0, watchdog, clock=node._steady)

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        if not node.finished:
            try:
                node.finish()
            except Exception as e:
                node.get_logger().error("finish failed: %s" % e)
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
