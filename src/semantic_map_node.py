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

Publishes
  /semantic_scan  the per-sweep incremental cloud   (every processed sweep)
  /semantic_map   a downsampled snapshot of the global map  (1 Hz by default)
Both with fields
  x f4 @0 | y f4 @4 | z f4 @8 | rgb f4 @12 | class u2 @16 | confidence f4 @20
  | has_rgb u1 @24          (point_step 28)
The first six are exactly the layout the brief specifies; has_rgb is appended so
that "no camera measurement" is explicit rather than encoded as black.

QoS: the LiDAR subscription is BEST_EFFORT + KEEP_LAST(1) so the node always
works on the newest sweep and drops the rest deterministically.  rosbag2's
player publishes RELIABLE/KEEP_LAST(10); a RELIABLE subscriber would queue and
drift unboundedly behind.
"""
import os
import sys
import time
import json
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
from ptv3_client import PTv3Client

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

        self.map = S.SemanticVoxelMap(voxel=args.map_voxel)
        self.T_C_L = S.T_C_L
        self.K = np.array([[S.FX, 0, S.CX], [0, S.FY, S.CY], [0, 0, 1]])

        self.images = []          # (t, HxWx3 uint8)
        self.img_lock = threading.Lock()
        self.stats = dict(recv=0, processed=0, dropped_qos=0, no_pose=0,
                          pts_in=0, pts_mapped=0, pts_rgb=0,
                          t_ptv3=[], t_ptv3_gpu=[], t_total=[], t_map=[], t_proj=[])
        self.first_stamp = None
        self.last_hdr = None
        self.sweep_dt = 0.1039          # KITTI HDL-64E sweep period in this bag
        self.t_wall0 = None
        self.last_pub = 0.0
        self.busy = False
        self.finished = False

        self.ptv3 = PTv3Client(intensity_scale=args.intensity_scale,
                               stderr_path=args.ptv3_log)
        self.get_logger().info("PTv3 worker ready (%d classes)" % self.ptv3.num_classes)

        # Default, as specified: BEST_EFFORT + KEEP_LAST(1) -- always work on the
        # newest sweep, drop the rest deterministically.  MEASURED CAVEAT on this
        # machine (ROS 2 Jazzy, rmw_fastrtps_cpp): a 2.7 MB PointCloud2 sent
        # BEST_EFFORT loses ~22 % of samples in the transport even when the
        # subscriber is completely idle, and ~63 % with the stock UDP builtin
        # transports (FASTDDS_BUILTIN_TRANSPORTS=LARGE_DATA recovers most of it).
        # The same publisher into a RELIABLE/KEEP_LAST(10) reader delivers 100 %.
        # So --reliable exists for the runs that must see every sweep; it is safe
        # here only because the node (~6 Hz) is faster than the replay rate used,
        # which is what keeps a reliable reader from drifting behind.
        if args.reliable:
            qos_be = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                                history=HistoryPolicy.KEEP_LAST, depth=10)
            self.get_logger().warn("LiDAR subscription is RELIABLE/KEEP_LAST(10)")
        else:
            qos_be = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                                history=HistoryPolicy.KEEP_LAST, depth=1)
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
    def cb_lidar(self, msg):
        self.stats["recv"] += 1
        if self.busy:
            self.stats["dropped_qos"] += 1
            return
        self.busy = True
        try:
            self.process(msg)
        except Exception as e:  # noqa
            import traceback
            self.get_logger().error("scan failed: %s\n%s" % (e, traceback.format_exc()))
        finally:
            self.busy = False

    def process(self, msg):
        t_all0 = time.perf_counter()
        if self.t_wall0 is None:
            self.t_wall0 = time.time()
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
        ok = self.traj.valid(t_pt)
        if ok.sum() < 100:
            self.stats["no_pose"] += 1
            return

        # ---- 3. PTv3 on the RAW body-frame scan (all points, before any culling)
        t0 = time.perf_counter()
        lab, conf = self.ptv3.segment(xyzi)
        self.stats["t_ptv3"].append((time.perf_counter() - t0) * 1000.0)
        self.stats["t_ptv3_gpu"].append(float(self.ptv3.last_infer_ms))
        if self.first_stamp is None:
            self.first_stamp = t_hdr
        self.last_hdr = t_hdr

        xyzi = xyzi[ok]; lab = lab[ok]; conf = conf[ok]; t_pt = t_pt[ok]
        if self.args.conf_gate > 0:
            g = conf >= self.args.conf_gate
            xyzi = xyzi[g]; lab = lab[g]; conf = conf[g]; t_pt = t_pt[g]
        n = len(xyzi)
        if n == 0:
            return

        # ---- 1+2. per-point pose, de-skewed:  p_W = T_W_I(t_i) @ T_I_L @ p_L
        # The trajectory is interpolated (slerp + linear), never nearest-frame:
        # at 14 m/s one sweep is 1.46 m, so nearest-frame would cost ~0.7 m.
        # The interpolation is evaluated at NB time bins across the 104 ms sweep
        # rather than once per point -- measured max deviation from the exact
        # per-point solve is 4 mm at NB=64 (2 mm at 128) against a 200 mm voxel,
        # for 3.4 ms instead of 38.9 ms.
        pl = np.hstack([xyzi[:, :3].astype(np.float64), np.ones((n, 1))])
        pl_i = (pl @ self.T_I_L.T)[:, :3]                         # LiDAR -> IMU body
        t_lo, t_hi = t_pt[0], t_pt[-1]
        NB = self.args.deskew_bins
        if NB <= 1 or t_hi <= t_lo:
            R1, p1, _ = self.traj.query(np.array([0.5 * (t_lo + t_hi)]))
            pw = pl_i @ R1[0].T + p1[0]
        else:
            ctr = t_lo + (np.arange(NB) + 0.5) * (t_hi - t_lo) / NB
            Rb, pb, _ = self.traj.query(ctr)
            bi = np.clip(((t_pt - t_lo) / (t_hi - t_lo) * NB).astype(np.int64), 0, NB - 1)
            edge = np.searchsorted(bi, np.arange(NB + 1))
            pw = np.empty((n, 3))
            for k in range(NB):
                a, b = edge[k], edge[k + 1]
                if b > a:
                    pw[a:b] = pl_i[a:b] @ Rb[k].T + pb[k]

        # ---- 4. RGB by projecting into the time-matched image
        t0 = time.perf_counter()
        rgb = np.zeros((n, 3), dtype=np.uint8)
        has_rgb = np.zeros(n, dtype=bool)
        t_img, img = self.match_image(t_pt.min(), t_pt.max())
        if img is not None:
            T_W_I_img, okp = self.traj.query_one(t_img)
            if okp:
                # world -> LiDAR frame AT THE IMAGE INSTANT -> rectified cam2.
                # Using the de-skewed world point makes this exact even though the
                # sweep spans 104 ms while the shutter is one instant.
                T_W_L_img = T_W_I_img @ self.T_I_L
                T_L_W = np.linalg.inv(T_W_L_img)
                pc = (self.T_C_L @ T_L_W @ np.hstack([pw, np.ones((n, 1))]).T).T[:, :3]
                z = pc[:, 2]
                front = z > 0.5
                uv = np.full((n, 2), -1e9)
                uv[front] = (pc[front, :2] / z[front, None]) * [S.FX, S.FY] + [S.CX, S.CY]
                H, W = img.shape[:2]
                m = front & (uv[:, 0] >= 0) & (uv[:, 0] <= W - 1) \
                          & (uv[:, 1] >= 0) & (uv[:, 1] <= H - 1)
                if m.any():
                    u = np.rint(uv[m, 0]).astype(np.int32)
                    v = np.rint(uv[m, 1]).astype(np.int32)
                    rgb[m] = img[v, u]
                    has_rgb = m
        self.stats["t_proj"].append((time.perf_counter() - t0) * 1000.0)
        self.stats["pts_rgb"] += int(has_rgb.sum())

        # ---- 5. insert into the global voxel map
        t0 = time.perf_counter()
        self.map.insert(pw, lab, conf, rgb, has_rgb)
        self.stats["t_map"].append((time.perf_counter() - t0) * 1000.0)
        self.stats["pts_mapped"] += n
        self.stats["processed"] += 1

        # ---- per-sweep incremental cloud
        rec = S.build_record(pw.astype(np.float32), rgb, lab.astype(np.uint16),
                             conf, has_rgb.astype(np.uint8))
        self.pub_scan.publish(make_cloud(rec, self.args.world_frame, msg.header.stamp))
        self.last_stamp = msg.header.stamp
        self.stats["t_total"].append((time.perf_counter() - t_all0) * 1000.0)


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
            # Adaptive throttle, stated plainly: snapshotting an N-voxel map costs
            # O(N) and at 8 M voxels that is hundreds of ms.  Republishing on a
            # fixed 1 Hz tick would then steal most of the wall clock from the
            # LiDAR callback and the drop statistics would be measuring the
            # visualiser, not the mapper.  So the republish period is the larger
            # of the requested one and 4x the last snapshot cost.
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
            "recv %d | processed %d | dropped %d | voxels %.2fM | pts %.1fM | "
            "rgb %.1f%% | ptv3 %.0f ms | total %.0f ms | %.1f Hz"
            % (s["recv"], s["processed"], s["dropped_qos"], self.map.n / 1e6,
               s["pts_mapped"] / 1e6,
               100.0 * s["pts_rgb"] / max(s["pts_mapped"], 1),
               np.mean(s["t_ptv3"][-20:]), np.mean(s["t_total"][-20:]),
               s["processed"] / max(el, 1e-3)))

    # ------------------------------------------------------------------ #
    def finish(self):
        if self.finished:
            return
        self.finished = True
        self._stop_pub.set()
        self._pub_thread.join(timeout=60)
        stamp = getattr(self, "last_stamp", None)
        if stamp is None:
            from builtin_interfaces.msg import Time as TimeMsg
            stamp = TimeMsg()
        self.publish_map(stamp)
        s = self.stats
        el = time.time() - (self.t_wall0 or time.time())
        xyz, rgb, cls, conf, has = self.map.snapshot(
            stride_voxel=self.args.pub_voxel, max_points=self.args.max_pub_points)
        hist = np.bincount(cls.astype(np.int64), minlength=16)
        out = dict(
            traj=self.args.traj, bag_rate=self.args.bag_rate,
            apply_T_I_L=bool(self.args.apply_T_I_L),
            lidar_qos=("RELIABLE/KEEP_LAST(10)" if self.args.reliable
                       else "BEST_EFFORT/KEEP_LAST(1)"),
            fastdds_builtin_transports=os.environ.get("FASTDDS_BUILTIN_TRANSPORTS", "default"),
            map_voxel=self.map.voxel, pub_voxel=self.args.pub_voxel,
            conf_gate=self.args.conf_gate,
            scans_received=s["recv"], scans_processed=s["processed"],
            scans_dropped_busy=s["dropped_qos"], scans_no_pose=s["no_pose"],
            # what the publisher actually put on the wire over the span this node
            # saw, inferred from the constant sweep cadence.  The difference from
            # `scans_received` is what the BEST_EFFORT/KEEP_LAST(1) queue dropped
            # inside DDS before the callback ever ran.
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
            ptv3_gpu_ms=dict(mean=float(np.mean(s["t_ptv3_gpu"])),
                             p50=float(np.percentile(s["t_ptv3_gpu"], 50)),
                             p95=float(np.percentile(s["t_ptv3_gpu"], 95))),
            ptv3_ms=dict(mean=float(np.mean(s["t_ptv3"])), p50=float(np.percentile(s["t_ptv3"], 50)),
                         p95=float(np.percentile(s["t_ptv3"], 95)), max=float(np.max(s["t_ptv3"]))),
            proj_ms=dict(mean=float(np.mean(s["t_proj"])), p95=float(np.percentile(s["t_proj"], 95))),
            map_ms=dict(mean=float(np.mean(s["t_map"])), p95=float(np.percentile(s["t_map"], 95))),
            total_ms=dict(mean=float(np.mean(s["t_total"])), p50=float(np.percentile(s["t_total"], 50)),
                          p95=float(np.percentile(s["t_total"], 95)), max=float(np.max(s["t_total"]))),
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", default="/data/livo_sem/out/kitti_seq07_fastlivo2_tum.txt")
    ap.add_argument("--t-min", type=float, default=None)
    ap.add_argument("--lidar-topic", default="/velodyne_points")
    ap.add_argument("--image-topic", default="/camera/image_raw")
    ap.add_argument("--world-frame", default="map")
    ap.add_argument("--map-voxel", type=float, default=0.20)
    ap.add_argument("--pub-voxel", type=float, default=None)
    ap.add_argument("--max-pub-points", type=int, default=3_000_000)
    ap.add_argument("--map-rate", type=float, default=1.0)
    ap.add_argument("--intensity-scale", type=float, default=0.2)
    ap.add_argument("--conf-gate", type=float, default=0.0)
    ap.add_argument("--deskew-bins", type=int, default=128)
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
