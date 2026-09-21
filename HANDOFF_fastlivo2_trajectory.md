# HANDOFF -- FAST-LIVO2 on ROS 2 Jazzy, KITTI trajectories  (track: FAST-LIVO2)

FAST-LIVO2 is used as a **POSE SOURCE ONLY**. Its point clouds are never consumed.
The deliverable is a TUM trajectory keyed on true sensor time.

## 1. Deliverables

| path | what |
|---|---|
| `/data/livo_sem/ros2_ws/` | colcon workspace, built, ROS 2 Jazzy |
| `/data/livo_sem/config/kitti_velodyne64.yaml` | FAST-LIVO2 config, KITTI 2011_09_30 |
| `/data/livo_sem/config/camera_kitti_cam2.yaml` | rectified cam2 intrinsics for `parameter_blackboard` |
| `/data/livo_sem/run_fastlivo2_kitti.sh` | one-shot runner (node + blackboard + bag play) |
| `/data/livo_sem/src/fastlivo2_extrinsics.py` | derives every extrinsic from KITTI calib, prints the arithmetic |
| `/data/livo_sem/src/eval_fastlivo2_ate.py` | GT frame conversion + Umeyama ATE + plot |
| **`/data/livo_sem/out/kitti_seq04_fastlivo2_tum.txt`** | **SMOKE trajectory, 266 poses** |
| **`/data/livo_sem/out/kitti_seq07_fastlivo2_tum.txt`** | **MAIN trajectory, 1096 poses** |
| `/data/livo_sem/out/seq0{4,7}_traj.png` | estimated vs GT top-down XY + error curve |
| `/data/livo_sem/out/seq0{4,7}_ate.json` | ATE numbers, machine readable |

## 2. Trajectory format -- what the fusion node consumes

Plain text, one line per LIO update, space separated, **TUM**:

```
timestamp  tx ty tz  qx qy qz qw
```

* `timestamp` is `LidarMeasures.last_lio_update_time` (`LIVMapper.cpp:472`), i.e.
  **TRUE LiDAR sensor time in the bag's absolute clock**, NOT `ros::Time::now()`.
  Enabled by `evo: pose_output_en: true` -- **zero code changes**. The ROS topics
  (`/cloud_registered`, `/aft_mapped_to_init`) are still stamped `now()` and must not
  be used for timing.
* The pose is **T_{W <- IMU}**, not T_{W <- LiDAR}. `LIVMapper.cpp:472` writes
  `_state.pos_end / rot_end`, and `LIVMapper.cpp:727` shows
  `p_W = R_end * (extR * p_L + extT) + p_end`, so the state is the IMU body pose and
  `extR/extT = T_{IMU<-LiDAR}`.
  **The fusion node must right-multiply:**

  ```python
  T_I_L = np.array([[ 0.999997685,-0.000785403, 0.002024406, 0.810543972],
                    [ 0.000755307, 0.999889850, 0.014824544,-0.307054372],
                    [-0.002035826,-0.014822976, 0.999888022, 0.802723995],
                    [ 0, 0, 0, 1]])
  T_W_L = T_W_I @ T_I_L        # now p_W = T_W_L @ p_velodyne
  ```

  A left-multiplied global alignment cannot absorb this right-hand constant.
* The world frame W = the IMU frame at initialisation (`rot_end = I`), gravity only
  enters through `state.gravity`. W is NOT gravity-aligned and NOT the GT frame.
* One pose per LiDAR sweep after IMU init; the first ~5 sweeps have no pose
  (they are consumed by `ImuProcess` initialisation). Sample spacing 0.1041 s.

## 3. Route taken: ROUTE A, the Robotic-Developer-Road ROS 2 port

Three candidates were cloned to `/data/livo_sem/third_party/ports/` and diffed
line-by-line against upstream `hku-mars/FAST-LIVO2 @ 0d2c034`
(`/data/livo_sem/third_party/FAST-LIVO2`).

| file (upstream LOC) | **rdr** `humble` @ 837b7bb | freshleesh @ 320deb2 |
|---|---|---|
| `src/vio.cpp` (1875) | **27** | 2243 |
| `src/voxel_map.cpp` (970) | **72** | 1483 |
| `src/IMU_Processing.cpp` (587) | **48** | 420 |
| `src/preprocess.cpp` (1125) | **26** | 826 |
| `src/visual_point.cpp` (126) | **0** | 0 |
| `src/frame.cpp` (65) | **2** | 2 |
| `include/utils/so3_math.h` (89) | **0** | 370 |
| `include/common_lib.h` (243) | **13** | 693 |
| `src/LIVMapper.cpp` (1370, ROS glue) | 445 | 3795 |

`integralrobotics/FAST-LIVO2` cloned EMPTY (`master` has no commits) -- not viable.

**rdr's estimator core is 175 changed lines out of 4748** (3.7%), and reading every one
of them shows they are purely mechanical:
`ros::Publisher` -> `rclcpp::Publisher`, `stamp.toSec()` -> `stamp2Sec(stamp)`,
`ROS_INFO` -> `RCLCPP_INFO`, `nh.param` -> `declare_parameter`/`get_parameter`,
`sensor_msgs::X` -> `sensor_msgs::msg::X`, and the templated-Sophus rename
`SE3` -> `SE3<double>` / `rotation_matrix()` -> `rotationMatrix()`.
The ONLY numerical change in the whole core is `vio.cpp`:
`new_frame_->T_f_w_ = SE3(Eigen::Quaterniond(Rcw).normalized().toRotationMatrix(), Pcw)`
(re-orthonormalisation before constructing SE3 -- the templated Sophus asserts on a
non-orthogonal R where the old one did not). `velodyne_handler` is byte-identical.

freshleesh is a research fork (ground constraints, wheel odometry, ZUPT, multi-session,
SC-relocalisation) -- rejected.

**No estimator source file was modified by this track.** The only edits are three
CMakeLists lines (see S6).

## 4. Extrinsics -- the arithmetic

FAST-LIVO2's conventions, read from the code, not assumed:

* `extrin_calib.extrinsic_R/_T` == **T_{IMU <- LiDAR}**  (`LIVMapper.cpp:727`)
* `extrin_calib.Rcl/Pcl`        == **T_{camera <- LiDAR}**
  (`vio.cpp:32` `Rli = extR^T, Pli = -extR^T extT` gives T_{L<-I};
   `vio.cpp:58` `Rci = Rcl*Rli`, i.e. T_{C<-I} = T_{C<-L} T_{L<-I})

### 4.1 Rcl / Pcl = T_{rect cam2 <- velodyne}

```
T_{cam0u<-velo}  (calib_velo_to_cam.txt, verbatim)
  [ 0.007027555 -0.999975300  0.000025996 | -0.007137748 ]
  [-0.002254837 -0.000041843 -0.999997500 | -0.074826560 ]
  [ 0.999972800  0.007027479 -0.002255075 | -0.333632400 ]

K2 = P_rect_02[:, :3] = [[707.0912, 0, 601.8873],[0, 707.0912, 183.1104],[0,0,1]]
P_rect_02[:,3]       = [46.887830, 0.11786010, 0.006203223]
t2 = K2^-1 P_rect_02[:,3] = [0.061030584, -0.001439722, 0.006203223]   (|t2| = 0.061362 m)

T_{rect2<-cam0u} = (R_rect_00, t2)
  [ 0.999928000  0.008085985 -0.008866797 |  0.061030584 ]
  [-0.008123205  0.999958300 -0.004169750 | -0.001439722 ]
  [ 0.008832711  0.004241477  0.999952000 |  0.006203223 ]

Rcl|Pcl = T_{rect2<-cam0u} @ T_{cam0u<-velo} =
  [-0.001857739 -0.999965951 -0.008039975 |  0.056246554 ]
  [-0.006481466  0.008051860 -0.999946608 | -0.074814016 ]
  [ 0.999977310 -0.001805529 -0.006496204 | -0.327793583 ]
```

### 4.2 extrinsic_R / extrinsic_T = T_{IMU <- velodyne}

`calib_imu_to_velo.txt` stores T_{velo<-imu}, so this is its INVERSE:

```
T_{velo<-imu}                              T_{imu<-velo}  ( = extrinsic_R | extrinsic_T )
 [ 0.999997600  0.000755307 -0.002035826 |-0.808675900]   [ 0.999997685 -0.000785403  0.002024406 | 0.810543972]
 [-0.000785403  0.999889800 -0.014822980 | 0.319555900]   [ 0.000755307  0.999889850  0.014824544 |-0.307054372]
 [ 0.002024406  0.014824540  0.999888100 |-0.799723100]   [-0.002035826 -0.014822976  0.999888022 | 0.802723995]
```

### 4.3 Numerical sanity check (the brief's test)

```
camera origin in the VELODYNE frame (x fwd, y left, z up):
    t_{L<-C} = inv(T_{cam2<-velo})[:3,3] = [+0.3274, +0.0563, -0.0765] m
expected                                   [~+0.27  , ~+0.06  , ~-0.08 ] m     PASS
orthonormality:  max|R R^T - I| = 8.1e-08 (Rcl), 8.5e-08 (extR);  det = 1.000000
```

These numbers were derived independently of the bag track and agree with
`HANDOFF_kitti_bag.md` S8 digit for digit.

### 4.4 Images: RECTIFIED, not raw

The bag publishes `image_02` of a `_sync` drive, which KITTI ships already rectified
and cropped to `S_rect_02 = 1226 x 370`. The camera yaml therefore uses
`P_rect_02[:, :3]` (fx=fy=707.0912, cx=601.8873, cy=183.1104) and **zero distortion**,
NOT `K_02`/`D_02`. `Rcl/Pcl` above are consistently in the same rectified cam2 frame.

## 5. Results

`ros2 bag play --rate 0.5`; single global SE(3) Umeyama (no scale); GT converted to the
velodyne frame with `T_velo(i) = Tr^-1 T_cam(i) Tr`, `Tr` from `sequences/<seq>/calib.txt`.
GT is interpolated (slerp + linear) onto the estimate timestamps -- at 14 m/s one frame
is 1.46 m, so nearest-frame association alone would inject a ~0.7 m systematic error.

| | **seq07 (MAIN, drive_0027)** | **seq04 (SMOKE, drive_0016)** | seq04 CONTROL, `time` in SECONDS |
|---|---|---|---|
| pairs | 1096 | 266 | 265 |
| ATE RMSE | **0.880 m** | **2.556 m** | 50.646 m |
| ATE mean | 0.812 m | 1.220 m | 40.728 m |
| ATE median | 0.727 m | 0.771 m | 35.245 m |
| ATE max | 1.844 m | 16.953 m | 133.892 m |
| ATE min | 0.259 m | 0.122 m | 0.470 m |
| traj length GT / EST | 692.68 / 694.35 m (1.002) | 387.75 / 371.52 m (0.958) | 386.12 / 290.86 m (0.753) |
| mean speed GT / EST | 6.09 / 6.10 m/s | 14.05 / 13.47 m/s | 14.05 / 10.58 m/s |
| drift | **0.127 %** | 0.659 % | 13.1 % |

### 5.1 seq04's 2.56 m is ONE cold-start transient, not drift

KITTI drive_0016 opens with the car **already at 12.7 m/s**. `ImuProcess::IMU_init`
sets `vel_end = 0`, and an accelerometer cannot observe constant velocity, so the
filter must recover the speed through the LiDAR residual alone. Measured per-frame
displacement vs GT:

```
 k= 0  est 0.123 m  gt 1.323 m  ratio 0.09   err 16.95 m
 k= 4  est 0.315 m  gt 1.376 m  ratio 0.23   err 11.83 m
 k=11  est 0.590 m  gt 1.365 m  ratio 0.43   err  5.35 m
 t+3.13 s          ratio ~1.03  err  1.23 m      <- caught up
```

**ATE over seq04 excluding the first 3 s: RMSE 0.801 m, mean 0.712, median 0.753,
max 1.411 m over 360 m (0.22 %)** -- the same regime as seq07.
seq07 starts from ~1.2 m/s and shows no transient at all (frame-to-frame ratio is
1.03 by k=2), which is why its full-sequence number is already 0.88 m.

Consumers should treat the first ~3 s of seq04 as invalid, or start the map at
`t >= 1317383443.9`. seq07 needs no such cut.

### 5.2 The `time`-unit control proves de-skew is live and correctly scaled

`preprocess.cpp:496` is `curvature = point.time / 1000.0   // units: ms`, so the
shipped code wants `point.time` in **MICROSECONDS**. Running the identical config on
`bags/kitti_seq04` (`time` in seconds, i.e. de-skew effectively switched off) gives
**ATE RMSE 50.6 m and a 25 % short trajectory** versus 2.56 m on `bags/kitti_seq04_us`.
Use the `_us` bags. This was measured, not argued.

## 6. Workspace: what was built and the three build-only patches

```
/data/livo_sem/ros2_ws/src/
  FAST-LIVO2/          <- copy of third_party/ports/rdr (branch humble @ 837b7bb)
  rpg_vikit/           <- Robotic-Developer-Road/rpg_vikit (vikit_common, vikit_ros, vikit_py)
  livox_ros_driver2/   <- msg-only stand-in written here (CustomMsg + CustomPoint)
```

apt: `ros-jazzy-sophus` (1.22.9102, the **templated** Sophus -- the rdr port uses
`SE3<double>` / `rotationMatrix()`, so this is the right one, NOT strasdat a621ff2),
`libgoogle-glog-dev`. `pcl-ros`, `pcl-conversions`, `cv-bridge`, `image-transport`,
`libeigen3-dev`, `libpcl-dev`, `libopencv-dev`, `demo-nodes-cpp` were already present.

Patches applied (**build system only -- no estimator code touched**):

1. `ros-jazzy-sophus` exports only the `Sophus::Sophus` target, never
   `Sophus_INCLUDE_DIRS`, which all three CMakeLists dereference. Added a
   `get_target_property(... INTERFACE_INCLUDE_DIRECTORIES)` fallback after every
   `find_package(Sophus REQUIRED)` and linked `Sophus::Sophus` explicitly
   (`vikit_common`, `vikit_ros`, `FAST-LIVO2`).
2. `vikit_common/CMakeLists.txt` forced `-std=c++0x`; templated Sophus needs C++17.
3. `livox_ros_driver2` was vendored as an interface-only package. Upstream's real
   driver needs Livox-SDK2 and is only required for the AVIA path, which KITTI never
   enters. The two `.msg` files are wire-compatible with the official ones.

`lio/max_iterations` was renamed to `lio.min_iterations` by the port
(`voxel_map.cpp:37`, still assigned to `max_iterations_`). The yaml sets both.

## 7. Reproduce

```bash
source /opt/ros/jazzy/setup.bash
source /data/livo_sem/ros2_ws/install/setup.bash

# rebuild from scratch
cd /data/livo_sem/ros2_ws
colcon build --packages-select livox_ros_driver2 vikit_common vikit_ros fast_livo \
             --cmake-args -DCMAKE_BUILD_TYPE=Release

# re-derive and re-check every extrinsic
/usr/bin/python3 /data/livo_sem/src/fastlivo2_extrinsics.py

# run  (seq_name, bag, bag-play rate)
/data/livo_sem/run_fastlivo2_kitti.sh kitti_seq07 /data/livo_sem/bags/kitti_seq07_us 0.5
/data/livo_sem/run_fastlivo2_kitti.sh kitti_seq04 /data/livo_sem/bags/kitti_seq04_us 0.5

# evaluate   (seq, tum file, raw drive, output tag)
/usr/bin/python3 /data/livo_sem/src/eval_fastlivo2_ate.py 07 \
  /data/livo_sem/out/kitti_seq07_fastlivo2_tum.txt \
  /data/livo_sem/data/raw/2011_09_30/2011_09_30_drive_0027_sync seq07
```

Runtime on this box: LIO 0.038 s + VIO 0.008 s per 0.104 s sweep, i.e. ~2.3x real time
headroom. `--rate 0.5` was used for margin; nothing is dropped either way (the
subscriptions are depth 200000, `LIVMapper.cpp:262-267`).

Trajectories also land at `ros2_ws/src/FAST-LIVO2/Log/result/<seq_name>.txt`
(`ROOT_DIR` is the SOURCE dir, not the install dir) and are copied to `out/`.

## 8. Known limits

* `map_sliding_en: false`. seq07 is 693 m and total system memory in use stayed at 5 GB of 62 GB (per-process RSS was not isolated). A much
  longer sequence should set `local_map.map_sliding_en: true`.
* The cold-start transient of S5.1 is inherent to `IMU_init` setting `vel_end = 0`.
  Not patched, because the brief's rule is to keep the estimator core untouched.
* `img_en: 1` (full LIVO, the camera does participate). The VIO reports a healthy
  ~82k-point sparse map and ~0.008 s/frame on KITTI.
* The 0.88 m / 2.56 m figures are pure odometry -- no loop closure exists in
  FAST-LIVO2 and none was added.
