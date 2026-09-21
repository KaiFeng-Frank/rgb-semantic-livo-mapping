#!/usr/bin/env python3
"""Verify a ROS 1 .bag produced by rosbags-convert still satisfies the
FAST-LIVO2 velodyne_ros::Point contract.  Run with the venv python:
  /data/livo_sem/venv_rosbags/bin/python verify_ros1bag.py <bag> [unit]"""
import sys
from pathlib import Path
import numpy as np
from rosbags.highlevel import AnyReader

DT = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
               ("intensity", "<f4"), ("time", "<f4"), ("ring", "<u2")])
EXPECT = [("x", 0, 7), ("y", 4, 7), ("z", 8, 7),
          ("intensity", 12, 7), ("time", 16, 7), ("ring", 20, 4)]
FAILED = []
def chk(c, w):
    print(("  PASS  " if c else "  FAIL  ") + w)
    if not c: FAILED.append(w)

bag = Path(sys.argv[1])
unit = sys.argv[2] if len(sys.argv) > 2 else "s"
scale = {"s": 1.0, "ms": 1e3, "us": 1e6}[unit]

with AnyReader([bag]) as r:
    print("connections:")
    for c in r.connections:
        print("   %-22s %-32s count=%d" % (c.topic, c.msgtype, c.msgcount))
    tset = {c.topic: c.msgtype for c in r.connections}
    chk(tset.get("/velodyne_points") == "sensor_msgs/msg/PointCloud2",
        "/velodyne_points is sensor_msgs/PointCloud2")
    chk(tset.get("/camera/image_raw") == "sensor_msgs/msg/Image", "/camera/image_raw is Image")
    chk(tset.get("/camera/camera_info") == "sensor_msgs/msg/CameraInfo", "/camera/camera_info is CameraInfo")
    chk(tset.get("/imu") == "sensor_msgs/msg/Imu", "/imu is Imu")

    n = 0
    imu_t = []
    pc_t = []
    for conn, ts, raw in r.messages():
        if conn.topic == "/velodyne_points":
            pc_t.append(ts)
            if n < 3:
                m = r.deserialize(raw, conn.msgtype)
                got = [(f.name, int(f.offset), int(f.datatype)) for f in m.fields]
                a = np.frombuffer(m.data.tobytes(), dtype=DT)
                print("  cloud %d: frame_id=%r N=%d point_step=%d is_dense=%s  "
                      "ring %d..%d (%d)  time [%.6f, %.6f] %s  monotonic=%s"
                      % (n, m.header.frame_id, m.width, m.point_step, m.is_dense,
                         a["ring"].min(), a["ring"].max(), len(np.unique(a["ring"])),
                         a["time"].min(), a["time"].max(), unit,
                         bool(np.all(np.diff(a["time"]) >= 0))))
                if n == 0:
                    chk(got == EXPECT, "field layout survived ROS2->ROS1 conversion")
                    chk(m.point_step == 22, "point_step == 22")
                    chk(m.header.frame_id == "velodyne", "frame_id == velodyne")
                    chk(len(np.unique(a["ring"])) == 64 and a["ring"].max() == 63,
                        "ring 0..63, 64 distinct")
                    chk(a["time"].max() < 0.11 * scale and a["time"][-1] > 0,
                        "time in [0, 0.11) s and last point > 0")
                    chk(bool(np.all(np.diff(a["time"]) >= 0)), "time-ascending order")
                n += 1
        elif conn.topic == "/imu":
            imu_t.append(ts)

pc_t = np.array(pc_t); imu_t = np.array(imu_t)
print("  /velodyne_points n=%d  rate=%.3f Hz" % (len(pc_t), (len(pc_t)-1)/((pc_t[-1]-pc_t[0])/1e9)))
print("  /imu             n=%d  rate=%.3f Hz" % (len(imu_t), (len(imu_t)-1)/((imu_t[-1]-imu_t[0])/1e9)))
chk(bool(np.all(np.diff(imu_t) > 0)), "ROS1 imu timestamps strictly increasing")
chk(9.0 < (len(pc_t)-1)/((pc_t[-1]-pc_t[0])/1e9) < 10.5, "velodyne ~9.6 Hz")
print("\n%s (%d failed)" % ("ALL CHECKS PASSED" if not FAILED else "FAILURES:", len(FAILED)))
for f in FAILED: print("   - " + f)
sys.exit(1 if FAILED else 0)
