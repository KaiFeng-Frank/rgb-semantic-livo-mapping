#!/usr/bin/env python3
"""
kitti_to_ros2bag.py -- KITTI raw (_sync drive) -> ROS 2 bag for livo_sem.

MUST be run with /usr/bin/python3 after `source /opt/ros/jazzy/setup.bash`.
Do NOT use conda python: rclpy is built against the system interpreter.

Topics written
--------------
  /velodyne_points     sensor_msgs/PointCloud2   ~9.6 Hz   frame_id velodyne
  /camera/image_raw    sensor_msgs/Image         ~9.6 Hz   frame_id camera
  /camera/camera_info  sensor_msgs/CameraInfo    ~9.6 Hz   frame_id camera
  /imu                 sensor_msgs/Imu           100 Hz    frame_id imu
  /tf_static           tf2_msgs/TFMessage        once      (only with --tf-static)

/clock is NOT recorded; `ros2 bag play --clock` generates it.

See HANDOFF_kitti_bag.md for the full contract.
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kitti_calib import KittiCalib                      # noqa: E402
from kitti_scan import synth                            # noqa: E402

import rclpy.serialization                              # noqa: E402
import rosbag2_py                                       # noqa: E402
from builtin_interfaces.msg import Time as TimeMsg      # noqa: E402
from std_msgs.msg import Header                         # noqa: E402
from sensor_msgs.msg import PointCloud2, PointField, Image, CameraInfo, Imu  # noqa: E402
from geometry_msgs.msg import Quaternion, Vector3       # noqa: E402

# ---------------------------------------------------------------- oxts fields
# index -> name, straight out of oxts/dataformat.txt shipped with the drive.
OX = dict(lat=0, lon=1, alt=2, roll=3, pitch=4, yaw=5,
          vn=6, ve=7, vf=8, vl=9, vu=10,
          ax=11, ay=12, az=13,          # vehicle body frame  (x fwd, y left, z up)
          af=14, al=15, au=16,          # roll/pitch-levelled frame
          wx=17, wy=18, wz=19,          # body-frame angular rates
          wf=20, wl=21, wu=22,          # levelled angular rates
          pos_accuracy=23, vel_accuracy=24, navstat=25, numsats=26)

PC_DTYPE = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                     ("intensity", "<f4"), ("time", "<f4"), ("ring", "<u2")])
assert PC_DTYPE.itemsize == 22, PC_DTYPE.itemsize

TIME_SCALE = {"s": 1.0, "ms": 1.0e3, "us": 1.0e6}


# ------------------------------------------------------------------ utilities
def parse_ts_file(path):
    """KITTI 'YYYY-MM-DD HH:MM:SS.fffffffff' -> integer nanoseconds since the
    unix epoch, interpreted as UTC so the result does not depend on the host
    timezone.  (The absolute offset is irrelevant -- only consistency across
    the sensors and across machines matters.)"""
    import calendar
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            date, clock = line.split(" ")
            hms, frac = (clock.split(".") + ["0"])[:2]
            y, mo, d = (int(v) for v in date.split("-"))
            h, mi, s = (int(v) for v in hms.split(":"))
            secs = calendar.timegm((y, mo, d, h, mi, s, 0, 0, 0))
            out.append(secs * 1_000_000_000 + int((frac + "000000000")[:9]))
    return np.array(out, dtype=np.int64)


def read_oxts_row(path):
    """~40x faster than np.loadtxt for these 1-line, 30-column files."""
    with open(path, "rb") as f:
        return np.array(f.read().split(), dtype=np.float64)


def ns_to_timemsg(ns):
    return TimeMsg(sec=int(ns // 1_000_000_000), nanosec=int(ns % 1_000_000_000))


def euler_to_quat(roll, pitch, yaw):
    cr, sr = np.cos(roll * .5), np.sin(roll * .5)
    cp, sp = np.cos(pitch * .5), np.sin(pitch * .5)
    cy, sy = np.cos(yaw * .5), np.sin(yaw * .5)
    return (sr * cp * cy - cr * sp * sy,     # x
            cr * sp * cy + sr * cp * sy,     # y
            cr * cp * sy - sr * sp * cy,     # z
            cr * cp * cy + sr * sp * sy)     # w


# ------------------------------------------------------------------- messages
def make_pointcloud2(stamp_ns, arr, frame_id):
    m = PointCloud2()
    m.header = Header(stamp=ns_to_timemsg(stamp_ns), frame_id=frame_id)
    m.height = 1
    m.width = int(arr.shape[0])
    m.fields = [
        PointField(name="x",         offset=0,  datatype=PointField.FLOAT32, count=1),
        PointField(name="y",         offset=4,  datatype=PointField.FLOAT32, count=1),
        PointField(name="z",         offset=8,  datatype=PointField.FLOAT32, count=1),
        PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
        PointField(name="time",      offset=16, datatype=PointField.FLOAT32, count=1),
        PointField(name="ring",      offset=20, datatype=PointField.UINT16,  count=1),
    ]
    m.is_bigendian = False
    m.point_step = PC_DTYPE.itemsize
    m.row_step = m.point_step * m.width
    m.data = arr.tobytes()
    m.is_dense = True
    return m


def make_image(stamp_ns, rgb, frame_id):
    m = Image()
    m.header = Header(stamp=ns_to_timemsg(stamp_ns), frame_id=frame_id)
    m.height, m.width = int(rgb.shape[0]), int(rgb.shape[1])
    m.encoding = "rgb8"
    m.is_bigendian = 0
    m.step = m.width * 3
    m.data = np.ascontiguousarray(rgb).tobytes()
    return m


def make_caminfo(stamp_ns, calib, frame_id):
    m = CameraInfo()
    m.header = Header(stamp=ns_to_timemsg(stamp_ns), frame_id=frame_id)
    m.width = int(calib.img_size[0])
    m.height = int(calib.img_size[1])
    m.distortion_model = "plumb_bob"
    # image_02 of a _sync drive is ALREADY rectified and cropped, so the
    # published image has no distortion and needs no rectifying rotation.
    m.d = [0.0] * 5
    m.k = calib.K_rect.reshape(9).tolist()
    m.r = np.eye(3).reshape(9).tolist()
    m.p = calib.P_rect_02.reshape(12).tolist()
    return m


def make_imu(stamp_ns, row, frame_id, leveled=False):
    m = Imu()
    m.header = Header(stamp=ns_to_timemsg(stamp_ns), frame_id=frame_id)
    if leveled:
        a = (row[OX["af"]], row[OX["al"]], row[OX["au"]])
        w = (row[OX["wf"]], row[OX["wl"]], row[OX["wu"]])
    else:
        a = (row[OX["ax"]], row[OX["ay"]], row[OX["az"]])
        w = (row[OX["wx"]], row[OX["wy"]], row[OX["wz"]])
    m.linear_acceleration = Vector3(x=float(a[0]), y=float(a[1]), z=float(a[2]))
    m.angular_velocity = Vector3(x=float(w[0]), y=float(w[1]), z=float(w[2]))
    qx, qy, qz, qw = euler_to_quat(row[OX["roll"]], row[OX["pitch"]], row[OX["yaw"]])
    m.orientation = Quaternion(x=float(qx), y=float(qy), z=float(qz), w=float(qw))
    # RTK-fused attitude; FAST-LIVO2 ignores it, kept for reference / eval.
    m.orientation_covariance = [1e-4, 0., 0., 0., 1e-4, 0., 0., 0., 1e-3]
    m.angular_velocity_covariance = [4e-6, 0., 0., 0., 4e-6, 0., 0., 0., 4e-6]
    m.linear_acceleration_covariance = [4e-4, 0., 0., 0., 4e-4, 0., 0., 0., 4e-4]
    return m


def make_tf_static(stamp_ns, calib):
    from tf2_msgs.msg import TFMessage
    from geometry_msgs.msg import TransformStamped, Transform, Vector3 as V3

    def mat_to_tf(T, parent, child):
        R = T[:3, :3]
        qw = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
        qx = (R[2, 1] - R[1, 2]) / (4 * qw)
        qy = (R[0, 2] - R[2, 0]) / (4 * qw)
        qz = (R[1, 0] - R[0, 1]) / (4 * qw)
        ts = TransformStamped()
        ts.header = Header(stamp=ns_to_timemsg(stamp_ns), frame_id=parent)
        ts.child_frame_id = child
        ts.transform = Transform(
            translation=V3(x=float(T[0, 3]), y=float(T[1, 3]), z=float(T[2, 3])),
            rotation=Quaternion(x=float(qx), y=float(qy), z=float(qz), w=float(qw)))
        return ts

    # T_velo_imu maps a point in velo into imu  ->  it IS the imu->velo frame tf
    return TFMessage(transforms=[
        mat_to_tf(calib.T_velo_imu, "imu", "velodyne"),
        mat_to_tf(np.linalg.inv(calib.T_velo_cam2), "velodyne", "camera"),
    ])


# ----------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive", required=True, help="..._sync drive directory")
    ap.add_argument("--calib", default=None, help="dir with calib_*.txt (default: drive's parent)")
    ap.add_argument("--out", required=True, help="output bag directory")
    ap.add_argument("--oxts100", default=None,
                    help="oxts dir of the matching _extract drive (true 100 Hz IMU)")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=-1, help="exclusive; -1 = all")
    ap.add_argument("--time-unit", choices=["s", "us", "ms"], default="s",
                    help="unit of the PointCloud2 `time` field (default s, the "
                         "Velodyne driver convention). FAST-LIVO2 as shipped "
                         "wants us -- see HANDOFF_kitti_bag.md")
    ap.add_argument("--imu-frame", choices=["body", "leveled"], default="body",
                    help="body = ax,ay,az/wx,wy,wz (consistent with "
                         "calib_imu_to_velo). leveled = af,al,au/wf,wl,wu")
    ap.add_argument("--cloud-bag-time", choices=["start", "end"], default="end",
                    help="bag receive-time of /velodyne_points; header.stamp is "
                         "ALWAYS the sweep start")
    ap.add_argument("--storage", default="mcap", choices=["mcap", "sqlite3"])
    ap.add_argument("--tf-static", action="store_true")
    ap.add_argument("--no-imu-dedup", action="store_true",
                    help="keep oxts samples exactly as the files order them, "
                         "including KITTI's own non-monotonic entries")
    args = ap.parse_args()

    drive = os.path.normpath(args.drive)
    calib_dir = args.calib or os.path.dirname(drive)
    calib = KittiCalib(calib_dir)

    velo_dir = os.path.join(drive, "velodyne_points")
    img_dir = os.path.join(drive, "image_02")

    t_start = parse_ts_file(os.path.join(velo_dir, "timestamps_start.txt"))
    t_end = parse_ts_file(os.path.join(velo_dir, "timestamps_end.txt"))
    t_img = parse_ts_file(os.path.join(img_dir, "timestamps.txt"))

    n_all = len(t_start)
    lo = args.start
    hi = n_all if args.end < 0 else min(args.end, n_all)
    frames = range(lo, hi)
    print("[bag] drive      : %s" % drive)
    print("[bag] frames     : %d..%d of %d" % (lo, hi - 1, n_all))
    print("[bag] time unit  : %s   imu frame: %s" % (args.time_unit, args.imu_frame))

    # ---- IMU source
    if args.oxts100 and os.path.isdir(args.oxts100):
        oxts_dir, oxts_tag = args.oxts100, "extract (100 Hz)"
    else:
        oxts_dir, oxts_tag = os.path.join(drive, "oxts"), "sync (10 Hz)"
    t_oxts = parse_ts_file(os.path.join(oxts_dir, "timestamps.txt"))
    oxts_files = sorted(os.listdir(os.path.join(oxts_dir, "data")))
    assert len(oxts_files) == len(t_oxts), (len(oxts_files), len(t_oxts))
    win_lo, win_hi = t_start[lo] - 200_000_000, t_end[hi - 1] + 200_000_000
    ok = (t_oxts >= win_lo) & (t_oxts <= win_hi)
    imu_idx = np.nonzero(ok)[0]

    # KITTI defect: 2011_09_30_drive_0027_extract/oxts/timestamps.txt has one
    # misordered, duplicated line (raw index 7664 repeats the timestamp that
    # also appears at 7670, so the file order steps -50 ms then +70 ms).
    # Sort by time and drop any sample that is not strictly newer, so the bag
    # carries a strictly increasing IMU stream.
    if not args.no_imu_dedup:
        imu_idx = imu_idx[np.argsort(t_oxts[imu_idx], kind="stable")]
        keep = np.ones(len(imu_idx), dtype=bool)
        last = np.int64(-1)
        for k, j in enumerate(imu_idx):
            if t_oxts[j] <= last:
                keep[k] = False
            else:
                last = t_oxts[j]
        n_drop = int((~keep).sum())
        if n_drop:
            print("[bag] imu dedup : dropped %d non-increasing sample(s) at raw "
                  "index %s" % (n_drop, imu_idx[~keep].tolist()))
        imu_idx = imu_idx[keep]
    print("[bag] imu source : %s -> %s  (%d samples in window, %.2f Hz)"
          % (oxts_tag, oxts_dir, len(imu_idx),
             (len(imu_idx) - 1) * 1e9 / max(1, (t_oxts[imu_idx[-1]] - t_oxts[imu_idx[0]]))))

    # ---- writer
    if os.path.exists(args.out):
        raise SystemExit("output %s already exists" % args.out)
    writer = rosbag2_py.SequentialWriter()
    writer.open(rosbag2_py.StorageOptions(uri=args.out, storage_id=args.storage),
                rosbag2_py.ConverterOptions("", ""))
    topics = [("/velodyne_points", "sensor_msgs/msg/PointCloud2"),
              ("/camera/image_raw", "sensor_msgs/msg/Image"),
              ("/camera/camera_info", "sensor_msgs/msg/CameraInfo"),
              ("/imu", "sensor_msgs/msg/Imu")]
    if args.tf_static:
        topics.append(("/tf_static", "tf2_msgs/msg/TFMessage"))
    for name, typ in topics:
        writer.create_topic(rosbag2_py.TopicMetadata(
            id=0, name=name, type=typ, serialization_format="cdr"))

    if args.tf_static:
        writer.write("/tf_static",
                     rclpy.serialization.serialize_message(
                         make_tf_static(int(t_start[lo]), calib)),
                     int(t_start[lo]))

    tscale = TIME_SCALE[args.time_unit]
    from PIL import Image as PILImage

    ip = 0                      # imu cursor
    n_imu = 0
    t0 = time.time()
    stats = dict(npts=[], tmin=[], tmax=[], rings=set())

    for k, i in enumerate(frames):
        sweep = (t_end[i] - t_start[i]) / 1e9

        pts = np.fromfile(os.path.join(velo_dir, "data", "%010d.bin" % i),
                          dtype=np.float32).reshape(-1, 4)
        ring, tt, order = synth(pts[:, :3], sweep_duration=sweep)
        q = pts[order]
        arr = np.empty(q.shape[0], dtype=PC_DTYPE)
        arr["x"], arr["y"], arr["z"] = q[:, 0], q[:, 1], q[:, 2]
        arr["intensity"] = q[:, 3]
        arr["time"] = (tt * tscale).astype(np.float32)
        arr["ring"] = ring
        if arr["time"][-1] <= 0:
            raise RuntimeError("frame %d: last point time <= 0, FAST-LIVO2 would "
                               "fall back to yaw estimation" % i)

        stats["npts"].append(len(arr))
        stats["tmin"].append(float(arr["time"][0]))
        stats["tmax"].append(float(arr["time"][-1]))
        stats["rings"].add((int(ring.min()), int(ring.max())))

        # IMU up to the end of this sweep, in time order
        while ip < len(imu_idx) and t_oxts[imu_idx[ip]] <= t_end[i]:
            j = imu_idx[ip]
            row = read_oxts_row(os.path.join(oxts_dir, "data", oxts_files[j]))
            msg = make_imu(int(t_oxts[j]), row, "imu",
                           leveled=(args.imu_frame == "leveled"))
            writer.write("/imu", rclpy.serialization.serialize_message(msg),
                         int(t_oxts[j]))
            n_imu += 1
            ip += 1

        rgb = np.asarray(PILImage.open(
            os.path.join(img_dir, "data", "%010d.png" % i)).convert("RGB"))
        writer.write("/camera/image_raw",
                     rclpy.serialization.serialize_message(
                         make_image(int(t_img[i]), rgb, "camera")), int(t_img[i]))
        writer.write("/camera/camera_info",
                     rclpy.serialization.serialize_message(
                         make_caminfo(int(t_img[i]), calib, "camera")), int(t_img[i]))

        bag_ns = int(t_end[i] if args.cloud_bag_time == "end" else t_start[i])
        writer.write("/velodyne_points",
                     rclpy.serialization.serialize_message(
                         make_pointcloud2(int(t_start[i]), arr, "velodyne")), bag_ns)

        if k % 50 == 0 or i == hi - 1:
            el = time.time() - t0
            print("  frame %5d/%d  %6d pts  t[%.6f, %.6f] %s  imu=%d  %.1fs"
                  % (i, hi - 1, len(arr), arr["time"][0], arr["time"][-1],
                     args.time_unit, n_imu, el), flush=True)

    # drain remaining imu
    while ip < len(imu_idx):
        j = imu_idx[ip]
        row = read_oxts_row(os.path.join(oxts_dir, "data", oxts_files[j]))
        writer.write("/imu", rclpy.serialization.serialize_message(
            make_imu(int(t_oxts[j]), row, "imu",
                     leveled=(args.imu_frame == "leveled"))), int(t_oxts[j]))
        n_imu += 1
        ip += 1

    del writer
    print("[bag] DONE %s" % args.out)
    print("[bag]   clouds=%d  images=%d  caminfo=%d  imu=%d"
          % (hi - lo, hi - lo, hi - lo, n_imu))
    print("[bag]   points/cloud min=%d max=%d" % (min(stats["npts"]), max(stats["npts"])))
    print("[bag]   time field  min=%.9g  max=%.9g  (%s)"
          % (min(stats["tmin"]), max(stats["tmax"]), args.time_unit))
    print("[bag]   ring ranges seen: %s" % sorted(stats["rings"]))
    print("[bag]   wall time %.1f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
