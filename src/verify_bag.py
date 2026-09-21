#!/usr/bin/env python3
"""verify_bag.py -- deserialise a produced bag and assert the whole contract."""
import sys, os
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

DT = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
               ("intensity", "<f4"), ("time", "<f4"), ("ring", "<u2")])
EXPECT_FIELDS = [("x", 0, 7), ("y", 4, 7), ("z", 8, 7),
                 ("intensity", 12, 7), ("time", 16, 7), ("ring", 20, 4)]
DTNAME = {1: "INT8", 2: "UINT8", 3: "INT16", 4: "UINT16",
          5: "INT32", 6: "UINT32", 7: "FLOAT32", 8: "FLOAT64"}

FAILED = []
def chk(cond, what):
    print(("  PASS  " if cond else "  FAIL  ") + what)
    if not cond:
        FAILED.append(what)

def main(uri, storage="mcap", n_clouds=5, unit="s"):
    scale = {"s": 1.0, "ms": 1e3, "us": 1e6}[unit]
    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=uri, storage_id=storage),
           rosbag2_py.ConverterOptions("", ""))
    types = {t.name: t.type for t in r.get_all_topics_and_types()}
    print("topics:", types)

    seen, first, bagts = {}, {}, {}
    cloud_pairs = []
    imu_hdr = []
    ncloud = 0
    while r.has_next():
        topic, data, ts = r.read_next()
        seen[topic] = seen.get(topic, 0) + 1
        bagts.setdefault(topic, []).append(ts)
        if topic == "/velodyne_points":
            if ncloud < n_clouds:
                m = deserialize_message(data, get_message(types[topic]))
                first.setdefault(topic, []).append(m)
                cloud_pairs.append(
                    (m.header.stamp.sec * 1_000_000_000 + m.header.stamp.nanosec, ts))
                ncloud += 1
        elif topic == "/imu":
            m = deserialize_message(data, get_message(types[topic]))
            imu_hdr.append(m.header.stamp.sec * 1_000_000_000 + m.header.stamp.nanosec)
            first.setdefault(topic, [m])
        elif topic not in first:
            first[topic] = [deserialize_message(data, get_message(types[topic]))]
    print("counts:", seen)

    print("\n--- PointCloud2 layout ---")
    pc = first["/velodyne_points"][0]
    print("  frame_id=%r  height=%d width=%d point_step=%d row_step=%d is_dense=%s is_bigendian=%s"
          % (pc.header.frame_id, pc.height, pc.width, pc.point_step,
             pc.row_step, pc.is_dense, pc.is_bigendian))
    got = [(f.name, f.offset, f.datatype) for f in pc.fields]
    for f in pc.fields:
        print("    field %-10s offset=%2d datatype=%-8s count=%d"
              % (f.name, f.offset, DTNAME.get(f.datatype, f.datatype), f.count))
    chk(got == EXPECT_FIELDS, "field layout == velodyne_ros::Point "
        "(x0,y4,z8,intensity12,time16 FLOAT32; ring20 UINT16)")
    chk(pc.point_step == 22, "point_step == 22")
    chk(pc.row_step == pc.point_step * pc.width, "row_step == point_step*width")
    chk(pc.height == 1, "height == 1 (unorganised)")
    chk(pc.header.frame_id == "velodyne", "cloud frame_id == 'velodyne'")
    chk(len(pc.data) == pc.point_step * pc.width, "len(data) == point_step*width")

    print("\n--- PointCloud2 content (%d clouds) ---" % len(first["/velodyne_points"]))
    for i, pc in enumerate(first["/velodyne_points"]):
        a = np.frombuffer(bytes(pc.data), dtype=DT)
        rings = np.unique(a["ring"])
        cnt = np.bincount(a["ring"], minlength=64)
        print("  cloud %d: N=%d  ring %d..%d (%d distinct)  pts/ring min=%d max=%d  "
              "time [%.6f, %.6f] %s  monotonic=%s  xyz|max|=%.1f  intens[%.2f,%.2f]"
              % (i, len(a), rings.min(), rings.max(), len(rings),
                 cnt.min(), cnt.max(), a["time"].min(), a["time"].max(), unit,
                 bool(np.all(np.diff(a["time"]) >= 0)),
                 np.abs(np.stack([a["x"], a["y"], a["z"]])).max(),
                 a["intensity"].min(), a["intensity"].max()))
        if i == 0:
            chk(len(rings) == 64 and rings.min() == 0 and rings.max() == 63,
                "ring spans exactly 0..63, 64 distinct values")
            chk(0.0 <= a["time"].min() and a["time"].max() < 0.11 * scale,
                "time in [0, 0.11) s == [0, %g) %s  (one 0.1041 s sweep)"
                % (0.11 * scale, unit))
            chk(a["time"][-1] > 0,
                "LAST point time > 0  (else FAST-LIVO2 given_offset_time=false)")
            chk(bool(np.all(np.diff(a["time"]) >= 0)),
                "points sorted time-ascending (UndistortPcl walks backwards)")
            chk(cnt.min() > 500, "every ring has >500 points")

    print("\n--- Image ---")
    im = first["/camera/image_raw"][0]
    print("  %dx%d encoding=%r step=%d frame_id=%r len(data)=%d"
          % (im.width, im.height, im.encoding, im.step, im.header.frame_id, len(im.data)))
    chk(im.encoding == "rgb8", "encoding == rgb8")
    chk(im.step == im.width * 3 and len(im.data) == im.step * im.height,
        "step/data size consistent")
    chk(im.header.frame_id == "camera", "image frame_id == 'camera'")
    chk((im.width, im.height) == (1226, 370), "image is 1226x370 (rectified cam2)")

    print("\n--- CameraInfo ---")
    ci = first["/camera/camera_info"][0]
    print("  %dx%d model=%r\n  K=%s\n  D=%s\n  P=%s"
          % (ci.width, ci.height, ci.distortion_model,
             np.round(np.array(ci.k), 4).tolist(),
             np.array(ci.d).tolist(), np.round(np.array(ci.p), 4).tolist()))
    chk(abs(ci.k[0] - 707.0912) < 1e-3 and abs(ci.k[2] - 601.8873) < 1e-3,
        "K == P_rect_02[:, :3]  (fx=707.0912, cx=601.8873)")
    chk(abs(ci.p[3] - 46.88783) < 1e-3, "P[0,3] == 46.88783 (cam2 baseline term)")

    print("\n--- Imu ---")
    iu = first["/imu"][0]
    print("  frame_id=%r\n  acc =(%.6f, %.6f, %.6f)\n  gyro=(%.6f, %.6f, %.6f)\n"
          "  quat=(%.6f, %.6f, %.6f, %.6f)"
          % (iu.header.frame_id,
             iu.linear_acceleration.x, iu.linear_acceleration.y, iu.linear_acceleration.z,
             iu.angular_velocity.x, iu.angular_velocity.y, iu.angular_velocity.z,
             iu.orientation.x, iu.orientation.y, iu.orientation.z, iu.orientation.w))
    chk(iu.header.frame_id == "imu", "imu frame_id == 'imu'")
    chk(8.5 < iu.linear_acceleration.z < 11.0,
        "acc.z ~ +g  (x fwd, y left, z up; gravity reads positive on z)")
    chk(abs(iu.linear_acceleration.x) < 5 and abs(iu.linear_acceleration.y) < 5,
        "acc.x/.y small (vehicle roughly level)")

    print("\n--- rates / stamps ---")
    for t in ["/velodyne_points", "/camera/image_raw", "/camera/camera_info", "/imu"]:
        v = np.array(bagts[t], dtype=np.int64)
        d = np.diff(v) / 1e9
        print("  %-20s n=%6d  span=%.3f s  rate=%.3f Hz  dt mean=%.6f min=%.6f max=%.6f"
              % (t, len(v), (v[-1] - v[0]) / 1e9,
                 (len(v) - 1) / ((v[-1] - v[0]) / 1e9),
                 d.mean(), d.min(), d.max()))
        chk(bool(np.all(d >= 0)), "%s bag timestamps non-decreasing" % t)
    vr = (len(bagts["/velodyne_points"]) - 1) / (
        (bagts["/velodyne_points"][-1] - bagts["/velodyne_points"][0]) / 1e9)
    ir = (len(bagts["/imu"]) - 1) / ((bagts["/imu"][-1] - bagts["/imu"][0]) / 1e9)
    chk(9.0 < vr < 10.5, "velodyne rate ~9.6 Hz (HDL-64E actual spin rate)")
    chk(95 < ir < 105, "imu rate ~100 Hz")

    ih = np.array(imu_hdr, dtype=np.int64)
    chk(bool(np.all(np.diff(ih) > 0)),
        "IMU header stamps STRICTLY increasing (no dt==0, no loop-back)")
    print("  imu header dt: min=%.6f max=%.6f s" % (np.diff(ih).min()/1e9, np.diff(ih).max()/1e9))

    print("\n--- cloud header.stamp (sweep START) vs bag ts (sweep END) ---")
    deltas = []
    for i, (hs, bt) in enumerate(cloud_pairs):
        deltas.append((bt - hs) / 1e9)
        print("  cloud %d: header.stamp=%d  bag_ts=%d  bag-hdr=%+0.6f s"
              % (i, hs, bt, deltas[-1]))
    chk(all(0.100 < d < 0.108 for d in deltas),
        "bag_ts - header.stamp == one sweep (~0.1041 s) for every cloud")
    hdr_dt = np.diff([p[0] for p in cloud_pairs]) / 1e9
    chk(bool(np.all((hdr_dt > 0.100) & (hdr_dt < 0.108))),
        "consecutive cloud header.stamps differ by one sweep")

    print("\n%s  (%d checks failed)" % ("ALL CHECKS PASSED" if not FAILED else "FAILURES:", len(FAILED)))
    for f in FAILED:
        print("   - " + f)
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "mcap",
                  unit=sys.argv[3] if len(sys.argv) > 3 else "s"))
