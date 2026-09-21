# HANDOFF -- KITTI raw -> ROS 2 bag  (track: bag conversion)

Produced by `/data/livo_sem/src/kitti_to_ros2bag.py`.
Everything below was measured on the produced bags, not assumed.

## 1. What exists

| path | what |
|---|---|
| `/data/livo_sem/bags/kitti_seq04`    | SMOKE. drive_0016 frames 0..270 (271), 28.6 s, 1.0 GiB |
| `/data/livo_sem/bags/kitti_seq07`    | MAIN.  drive_0027 frames 0..1100 (1101), 114.8 s, 4.1 GiB |
| `/data/livo_sem/bags/kitti_seq04_us` | identical but `time` field in MICROSECONDS (see S4) |
| `/data/livo_sem/bags/kitti_seq07_us` | identical but `time` field in MICROSECONDS (see S4) |
| `/data/livo_sem/src/kitti_to_ros2bag.py` | converter |
| `/data/livo_sem/src/kitti_calib.py`      | calibration + velo->cam2 projection (import this) |
| `/data/livo_sem/src/kitti_scan.py`       | ring/time synthesis + bag ordering (import this) |
| `/data/livo_sem/bags_ros1/kitti_seq04.bag`    | ROS 1 v2.0 bag, same content (1.0 GiB) |
| `/data/livo_sem/bags_ros1/kitti_seq04_us.bag` | ROS 1, `time` in us |
| `/data/livo_sem/bags_ros1/kitti_seq07.bag`    | ROS 1, MAIN (4.1 GiB) |
| `/data/livo_sem/bags_ros1/kitti_seq07_us.bag` | ROS 1, MAIN, `time` in us -- **this is the one to feed unpatched FAST-LIVO2** |
| `/data/livo_sem/src/verify_bag.py`       | re-reads a ROS 2 bag and asserts this whole contract |
| `/data/livo_sem/src/verify_ros1bag.py`   | same, for the ROS 1 bags |
| `/data/livo_sem/out/proj_check_seq07_0000.png` | RGB-projection acceptance figure |

Storage is **mcap**. Run everything with `/usr/bin/python3` after
`source /opt/ros/jazzy/setup.bash`. Conda python has no rclpy.

## 2. Topics

| topic | type | frame_id | rate (measured) | count seq07 |
|---|---|---|---|---|
| `/velodyne_points`    | `sensor_msgs/msg/PointCloud2` | `velodyne` | 9.621 Hz  | 1101 |
| `/camera/image_raw`   | `sensor_msgs/msg/Image`       | `camera`   | 9.621 Hz  | 1101 |
| `/camera/camera_info` | `sensor_msgs/msg/CameraInfo`  | `camera`   | 9.621 Hz  | 1101 |
| `/imu`                | `sensor_msgs/msg/Imu`         | `imu`      | 100.002 Hz| 11483 |

`/clock` is NOT in the bag; `ros2 bag play --clock` generates it (verified).
`/tf_static` is only written with the `--tf-static` flag (off by default so the
topic set matches the spec exactly).

**The LiDAR is 9.621 Hz, not 10 Hz.** The HDL-64E sweep measured from KITTI's
own `timestamps_start/end.txt` is 0.1041 s. Do not hard-code 0.1.

## 3. PointCloud2 field layout  (EXACT)

`point_step = 22`, `height = 1`, `is_dense = true`, `is_bigendian = false`,
`row_step = 22 * width`. Tightly packed, no padding.

| offset | name | datatype | enum | meaning |
|---|---|---|---|---|
| 0  | `x`         | FLOAT32 | 7 | metres, velodyne frame, x forward |
| 4  | `y`         | FLOAT32 | 7 | metres, y left |
| 8  | `z`         | FLOAT32 | 7 | metres, z up |
| 12 | `intensity` | FLOAT32 | 7 | KITTI reflectance, **[0, 0.99], 99 distinct values** (NOT 0-255) |
| 16 | `time`      | FLOAT32 | 7 | offset from sweep start -- UNIT depends on the bag, see S4 |
| 20 | `ring`      | UINT16  | 4 | 0..63 |

numpy view:

```python
DT = np.dtype([("x","<f4"),("y","<f4"),("z","<f4"),
               ("intensity","<f4"),("time","<f4"),("ring","<u2")])   # itemsize 22
a = np.frombuffer(bytes(msg.data), dtype=DT)
```

This is byte-for-byte what FAST-LIVO2's `velodyne_ros::Point`
(`include/preprocess.h:69-79`) expects, so `pcl::fromROSMsg` binds all six
fields with zero missing.

**Points are ordered by `time` ascending** (azimuth-major, all 64 lasers
interleaved) -- NOT laser-major as the raw `.bin` stores them. This is
mandatory: `ImuProcess::UndistortPcl` walks the cloud backwards
(`src/IMU_Processing.cpp:514`) and its `sort(... time_list)` at
`src/IMU_Processing.cpp:286` is **commented out**. A laser-major cloud would
undistort silently wrong.

## 4. `time` UNITS -- READ THIS BEFORE RUNNING FAST-LIVO2

FAST-LIVO2 `src/preprocess.cpp:402`:

```cpp
added_pt.curvature = pl_orig.points[i].time / 1000.0;   // units: ms
```

and `curvature` is consumed as milliseconds everywhere downstream, e.g.
`src/IMU_Processing.cpp:157` `pcl_beg_time + pcl_out.points.back().curvature / double(1000)`
and the fallback `added_pt.curvature = (yaw_fp - yaw)/omega_l` with
`omega_l = 3.61` deg/ms. So **curvature is ms, therefore the shipped code
expects `point.time` in MICROSECONDS.** FAST-LIVO2 has no `time_unit`
parameter (FAST-LIO2 does; this fork dropped it).

Two ways to be consistent, pick ONE:

* **A. use `kitti_seq07_us` / `kitti_seq04_us`** -- `time` is in us, the
  shipped FAST-LIVO2 needs **no patch**. Peak value ~104105 us.
* **B. use `kitti_seq07` / `kitti_seq04`** (`time` in SECONDS, the Velodyne
  ROS driver convention, peak ~0.1041) and patch one line:
  `added_pt.curvature = pl_orig.points[i].time * 1000.0;`
  at `src/preprocess.cpp:402` **and** `:465` (the second, feature-branch copy of the same line).

Getting this wrong by 1e6 does not crash: the undistortion just does nothing
(or explodes), and the map still looks roughly right at first glance.

Guard already satisfied: `preprocess.cpp:364` does
`if (pl_orig.points[plsize-1].time > 0) given_offset_time = true;`
-- the last point's time is ~0.1039 s / 103921 us, positive in both bags, so
FAST-LIVO2 will use our per-point times and NOT fall back to its own
yaw-based estimation.

## 5. ring / time synthesis

KITTI `.bin` has only (x, y, z, intensity) float32. Both `ring` and `time`
are reconstructed by `kitti_scan.synth()`:

* KITTI stores a scan **laser-major**: one full revolution of laser 0, then
  laser 1, ... Measured on drive_0016 frame 0: `atan2(y,x)` is monotonically
  **increasing** in file order and the total unwrapped span is 63.90 rev.
* `ring = floor(cumulative_unwrapped_azimuth / 2pi)`, clipped to 0..63.
  Result: exactly 64 rings, mean elevation strictly decreasing from
  **+2.35 deg to -23.66 deg** (HDL-64E FOV is +2..-24.8), per-ring counts
  1124..2171.
  Elevation binning was rejected: HDL-64E spacing is non-uniform
  (~1/3 deg upper block, ~1/2 deg lower block).
* `phase = cum - ring*2pi` in [0, 2pi);
  `time = phase/(2pi) * (timestamps_end[i] - timestamps_start[i])`.
  **Azimuth increases**, so the sign is `(az - az_start)`, not
  `(az_start - az)`.
* A few samples per scan step backwards by <0.001 deg (encoder
  quantisation); those diffs are clamped to 0 so the ring index cannot
  flicker at a 2pi boundary.

Cross-check against KITTI's own files: `timestamps_end[i] == timestamps_start[i+1]`
to 1.4 us, and `timestamps.txt` is exactly the arithmetic midpoint of start
and end (so it carries no azimuth information and is NOT used).

### mapping bag order back to SemanticKITTI `.label` order
`.label` files index the ORIGINAL `.bin` order. `synth()` returns the
permutation:

```python
from kitti_scan import synth
ring, t, order = synth(pts_xyz)          # pts_xyz in raw .bin order
labels_in_bag_order = labels_from_file[order]
```

## 6. Timestamps

* `header.stamp` of `/velodyne_points` = **sweep START** (`timestamps_start.txt`).
  That is the base FAST-LIVO2 adds `time` to.
* bag receive-time of `/velodyne_points` = **sweep END** (`timestamps_end.txt`),
  i.e. one sweep later, so on `--clock` playback the IMU covering the sweep has
  already arrived. Verified: `bag_ts - header.stamp = +0.1039 s` on every cloud.
  Use `--cloud-bag-time start` to change this.
* image / camera_info / imu: `header.stamp` == bag time == the sensor's own
  `timestamps.txt`.
* KITTI datetimes are parsed **as UTC** (`calendar.timegm`) so the epoch does
  not depend on host timezone. seq04 first sweep start = `1317383440.297223942`,
  seq07 = `1317386425.505853454`.

### IMU stream defect fixed
`2011_09_30_drive_0027_extract/oxts/timestamps.txt` has one misordered,
duplicated line (raw index 7664 repeats the timestamp that also occurs at
7670, so the file order steps -50 ms then +70 ms). The converter sorts the
oxts samples by time and drops any that are not strictly newer -- **1 sample
dropped in seq07 (raw index 7670, the later of the two copies), 0 in seq04**.
The resulting `/imu` stream is strictly increasing, with a real 19.96 ms gap
where the duplicate used to be (measured `/imu` header dt: seq07 min 8.216 ms,
max 19.960 ms; seq04 min 9.658 ms, max 10.342 ms), so FAST-LIVO2's
`if (timestamp < last_timestamp_imu) { ROS_ERROR("imu loop back, clear buffer"); }`
never fires. `--no-imu-dedup` disables this. All `_sync` sensor timestamp
files were audited and are strictly increasing.

## 7. IMU: where the 100 Hz comes from, and the axis convention

**A `_sync` drive only has 10 Hz oxts** (one file per image frame). The true
100 Hz oxts lives only in the `_extract` drive. Those were downloaded on the
remote and unpacked to
`/data/livo_sem/data/extract/2011_09_30/2011_09_30_drive_00{16,27}_extract/oxts/`
(2967 and 11556 samples). The converter takes clouds+images from `_sync` and
IMU from `_extract`, matched purely on absolute timestamps. Without
`--oxts100` it silently falls back to the 10 Hz `_sync` oxts.

Fields used, 0-based, from `oxts/dataformat.txt` shipped in the drive:

| index | name | used as |
|---|---|---|
| 3,4,5    | roll, pitch, yaw | `orientation` (ENU: yaw 0 = east, CCW) |
| 11,12,13 | `ax, ay, az` | `linear_acceleration.x/y/z` |
| 17,18,19 | `wx, wy, wz` | `angular_velocity.x/y/z` |

**Body frame: x forward, y left, z up**; gravity reads as **+9.6 to +9.8 on z**.

`ax,ay,az` (body), NOT `af,al,au` (levelled), because the body frame is the one
`calib_imu_to_velo` is expressed in. Verified numerically on drive_0016 frame 0
(roll 0.042744, pitch 0.011528 rad):
`ax + pitch*az = 0.1998 + 0.0115*9.5722 = 0.310` vs `af = 0.3041`, and
`ay - roll*az = 0.4202 - 0.0427*9.5722 = 0.0115` vs `al = 0.01457`
-- i.e. `af/al/au` are `ax/ay/az` with roll/pitch removed. Feeding the levelled
ones to a filter that estimates gravity itself double-counts the attitude.
`--imu-frame leveled` switches to `af,al,au` / `wf,wl,wu` if ever needed.

`orientation` is filled from the RTK-fused roll/pitch/yaw. FAST-LIVO2 ignores
it; it is there for evaluation.

## 8. Calibration -- plain numbers, ready to paste

Source: `/data/livo_sem/data/raw/2011_09_30/calib_{cam_to_cam,velo_to_cam,imu_to_velo}.txt`

Rectified image_02 is **1226 x 370**.

`K_02` (RAW distorted cam2 -- for reference only, the bag's images are already rectified):
  +959.197700000  +0.000000000  +694.438300000
  +0.000000000  +952.932400000  +241.679300000
  +0.000000000  +0.000000000  +1.000000000

`D_02` (plumb_bob k1 k2 p1 p2 k3, RAW cam2):
    -0.372563700  +0.197980300  +0.000179997  +0.001250593  -0.066084810

`P_rect_02` (3x4) -- **this is what `/camera/camera_info.p` carries**:
  +707.091200000  +0.000000000  +601.887300000  +46.887830000
  +0.000000000  +707.091200000  +183.110400000  +0.117860100
  +0.000000000  +0.000000000  +1.000000000  +0.006203223

`R_rect_00` (3x3) -- rectifying rotation of cam0; all `P_rect_0x` live in this frame:
  +0.999928000  +0.008085985  -0.008866797
  -0.008123205  +0.999958300  -0.004169750
  +0.008832711  +0.004241477  +0.999952000

`T_velo_cam0` (4x4, velodyne -> UNrectified cam0), from `calib_velo_to_cam`:
  +0.007027555  -0.999975300  +0.000025996  -0.007137748
  -0.002254837  -0.000041843  -0.999997500  -0.074826560
  +0.999972800  +0.007027479  -0.002255075  -0.333632400
  +0.000000000  +0.000000000  +0.000000000  +1.000000000

`T_velo_rect0` = `R_rect_00 * T_velo_cam0` (velodyne -> rectified cam0 frame):
  -0.001857739  -0.999965951  -0.008039975  -0.004784030
  -0.006481466  +0.008051860  -0.999946608  -0.073374295
  +0.999977310  -0.001805529  -0.006496204  -0.333996806
  +0.000000000  +0.000000000  +0.000000000  +1.000000000

`T_velo_cam2` (4x4, velodyne -> RECTIFIED cam2 optical frame):
  -0.001857739  -0.999965951  -0.008039975  +0.056246554
  -0.006481466  +0.008051860  -0.999946608  -0.074814016
  +0.999977310  -0.001805529  -0.006496204  -0.327793583
  +0.000000000  +0.000000000  +0.000000000  +1.000000000

`T_imu_velo` (4x4, imu -> velodyne), from `calib_imu_to_velo`:
  +0.999997600  +0.000755307  -0.002035826  -0.808675900
  -0.000785403  +0.999889800  -0.014822980  +0.319555900
  +0.002024406  +0.014824540  +0.999888100  -0.799723100
  +0.000000000  +0.000000000  +0.000000000  +1.000000000

`T_velo_imu` (4x4, velodyne -> imu) = inverse of the above:
  +0.999997685  -0.000785403  +0.002024406  +0.810543972
  +0.000755307  +0.999889850  +0.014824544  -0.307054372
  -0.002035826  -0.014822976  +0.999888022  +0.802723995
  +0.000000000  +0.000000000  +0.000000000  +1.000000000

`T_imu_cam2` (4x4, imu -> rectified cam2):
  -0.001088635  -0.999976347  +0.006787182  -0.255366402
  -0.008512072  -0.006777671  -0.999940872  +0.732680810
  +0.999963177  -0.001146343  -0.008504493  -1.131832938
  +0.000000000  +0.000000000  +0.000000000  +1.000000000

### projection chain (do not re-derive -- import `kitti_calib`)

```
uv_hom = P_rect_02 @ R_rect_00_4x4 @ T_velo_cam0 @ [x_velo; 1]
u = uv_hom[0]/uv_hom[2] ;  v = uv_hom[1]/uv_hom[2] ;  depth = uv_hom[2]
```

Use `R_rect_00`, NOT `R_rect_02`. The rect0->cam2 baseline is already in
column 3 of `P_rect_02` (equivalently a pure translation
`t = K_rect^-1 @ P_rect_02[:,3] = [0.061030584, -0.001439722, 0.006203223]`).

```python
import sys; sys.path.insert(0, "/data/livo_sem/src")
from kitti_calib import load_calib
calib = load_calib("/data/livo_sem/data/raw/2011_09_30")
uv, depth, mask = calib.project_velo_to_cam2(pts_xyz, img_shape=(370,1226), return_mask=True)
pts_kept, rgb, mask = calib.colorize(pts_xyz, image_rgb)     # the fusion-node call
```

## 9. FAST-LIVO2 config values

```yaml
common:
  lid_topic: "/velodyne_points"
  imu_topic: "/imu"
  img_topic: "/camera/image_raw"
  img_en: 1
  lidar_en: 1

preprocess:
  lidar_type: 2        # VELO16, i.e. velodyne_handler (common_lib.h:41)
  scan_line: 64        # MUST be 64, the handler drops points with ring >= N_SCANS
  point_filter_num: 3  # ~125k points/scan is a lot
  blind: 2.0

extrin_calib:
  # lidar -> imu  (== T_velo_imu above)
  extrinsic_T: [0.810543972, -0.307054372, 0.802723995]
  extrinsic_R: [0.999997685, -0.000785403, 0.002024406,
           0.000755307, 0.999889850, 0.014824544,
           -0.002035826, -0.014822976, 0.999888022]
  # camera <- lidar  (== T_velo_cam2 above); Rcl row-major 3x3, Pcl 3x1
  Rcl: [-0.001857739, -0.999965951, -0.008039975,
           -0.006481466, 0.008051860, -0.999946608,
           0.999977310, -0.001805529, -0.006496204]
  Pcl: [0.056246554, -0.074814016, -0.327793583]

time_offset:
  lidar_time_offset: 0.0   # header.stamp is already the sweep START
  imu_time_offset: 0.0
  img_time_offset: 0.0
```

`evo: pose_output_en: true` writes the TUM trajectory keyed on
`LidarMeasures.last_lio_update_time` (`src/LIVMapper.cpp:385-405`) -- that is
the true LiDAR time, which is the D3 interface. The ROS topics are stamped
`ros::Time::now()` (`LIVMapper.cpp:1188`, `:1340`) and are NOT usable for
timing without the 2-line patch.

**FAST-LIVO2 is ROS 1, and there is no `ros1_bridge` for Jazzy.** Already
handled: ROS 1 v2.0 bags of all four variants are in
`/data/livo_sem/bags_ros1/`, produced with `rosbags` 0.11.5 in
`/data/livo_sem/venv_rosbags` (no ROS 1 install needed) and re-verified
field-by-field by `src/verify_ros1bag.py`: the 22-byte
x/y/z/intensity/time/ring layout, `frame_id`, 64 rings, time range, time
ordering and IMU monotonicity all survive the conversion intact.

So the FAST-LIVO2 track does NOT need a ROS 2 port. Use
`/data/livo_sem/bags_ros1/kitti_seq07_us.bag` with stock FAST-LIVO2
(no source patch), or `kitti_seq07.bag` with the one-line `* 1000.0` patch
from S4.

## 10. Frame correspondence (agrees with CRITICAL_CONSTRAINTS.md C5)

| bag | raw drive | raw frames used | SemanticKITTI seq | labels |
|---|---|---|---|---|
| `kitti_seq04` | `2011_09_30_drive_0016_sync` | 0..270 of 279 | 04 | 271 |
| `kitti_seq07` | `2011_09_30_drive_0027_sync` | 0..1100 of 1106 | 07 | 1101 |

Bag message index == raw frame index == SemanticKITTI frame index, from 0.
The unlabelled raw frames are at the tail and are excluded.
Labels: `/data/livo_sem/data/odometry/dataset/sequences/{04,07}/labels/%06d.label`.

## 11. Facts measured for the other tracks

For CRITICAL_CONSTRAINTS.md C2 (ground-plane z) -- RANSAC plane fit on road
points, velodyne frame:

| drive | frames | ground plane z | sensor height |
|---|---|---|---|
| drive_0027 (seq07) | 0, 300, 700, 1000 | -1.699, -1.833, -1.836, -1.737 | **1.776 m** |
| drive_0016 (seq04) | 0, 150, 270       | -1.751, -1.746, -1.733          | **1.743 m** |

Plane normals are (~0, ~0, +0.9999), i.e. the velodyne frame really is z-up and
gravity-aligned to <2 deg. nuScenes implies ~1.84 m, so KITTI sits ~0.06-0.10 m
lower.

For C3 (intensity) -- drive_0027 frame 0: float32, min 0.0000, max 0.9900,
mean 0.2927, percentiles [1,25,50,75,99] = [0.0, 0.23, 0.31, 0.37, 0.66],
**99 distinct values, quantised to k/100 (not k/255)**. The bag carries this
unchanged.

## 12. Reproduce / verify

```bash
source /opt/ros/jazzy/setup.bash

# rebuild every bag
/data/livo_sem/build_bags.sh

# one bag
/usr/bin/python3 /data/livo_sem/src/kitti_to_ros2bag.py \
  --drive   /data/livo_sem/data/raw/2011_09_30/2011_09_30_drive_0027_sync \
  --oxts100 /data/livo_sem/data/extract/2011_09_30/2011_09_30_drive_0027_extract/oxts \
  --out     /data/livo_sem/bags/kitti_seq07 --end 1101

# assert the whole contract by re-reading the bag
/usr/bin/python3 /data/livo_sem/src/verify_bag.py \
  /data/livo_sem/bags/kitti_seq07 mcap s        # "us" for the _us bags

# projection acceptance figure
/usr/bin/python3 /data/livo_sem/src/kitti_calib.py

ros2 bag info /data/livo_sem/bags/kitti_seq07
ros2 bag play /data/livo_sem/bags/kitti_seq07 --clock
```

ROS 1 bags (no ROS 1 install required):

```bash
V=/data/livo_sem/venv_rosbags/bin
$V/rosbags-convert --src /data/livo_sem/bags/kitti_seq07_us     --dst /data/livo_sem/bags_ros1/kitti_seq07_us.bag --dst-typestore ros1_noetic
$V/python /data/livo_sem/src/verify_ros1bag.py     /data/livo_sem/bags_ros1/kitti_seq07_us.bag us
```

## 13. Measured playback throughput

A plain rclpy subscriber (queue 200, RELIABLE) on `ros2 bag play kitti_seq07 --clock`,
30 s wall clock: **289 clouds (9.630 Hz), 289 images (9.630 Hz), 3002 imu
(100.04 Hz), 38.5 MB/s, zero drops.** The bag plays in real time.

Do NOT trust `ros2 topic hz` here -- on 2.7 MB messages it reports ~6.6 Hz
because the meter itself cannot keep up. Count messages instead.
