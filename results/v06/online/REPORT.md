# v0.6 online integration: FAST-LIVO2 live + semantic mapping, causal pose query

Machine: RTX 4090, 12-core Xeon Gold 6248R, Ubuntu 24.04, ROS 2 Jazzy (Fast DDS 2.14), Pointcept v1.5.1, fp16, shuffle_orders=False, intensity x0.2, grid 0.05; B0 checkpoint; seq07 held-out (read at scoring time only).  Files: out/v06/ (runs/ holds every per-run artefact).  Companion of out/v05/REPORT.md: same node arguments, same scorer lineage, same bag.

## Code changes and why

| file | change | why |
|---|---|---|
| `src/semantic_map_node.py` | `--pose-topic` online mode: `LivePoseBuffer(S.TrajInterp)` fed by a nav_msgs/Odometry subscription; causal query (the inherited interpolation over the samples that have arrived, `cv`/`hold` extrapolation beyond the newest one, `--pose-max-extrap` staleness gate, optional `--pose-wait-ms`); measurement records `--frame-log`, `--pose-record-tum/npz`, `--pose-used-out`; `--scan-reliable/--scan-depth` for the transport probe; `snapshot_ms.n` (publish count) in the stats | the TUM file cannot exist online; the causal query is the object under test, and everything it used is recorded so its cost is measured, not inferred |
| `ros2_ws/src/FAST-LIVO2/src/LIVMapper.cpp:1417` | `/aft_mapped_to_init` header.stamp = `sec2Stamp(LidarMeasures.last_lio_update_time)` (was `now()`) | the topic carried NO sensor time; an online consumer that de-skews cannot associate a `now()`-stamped pose with a sweep.  ROS glue only, estimator untouched; the pose is the same post-LIO state the evo file writes (the recorded stream's ATE equals the run's evo-file ATE, M5) |
| `ros2_ws/src/rpg_vikit/vikit_ros/include/vikit/params_helper.h:117` | remote-parameter `wait_for_service` 100 ms -> 10 s | measured startup race: fastlivo_mapping aborts with `Camera model not correctly specified` 4/4 under `FASTDDS_BUILTIN_TRANSPORTS=LARGE_DATA`, 2/4 on the default transport (`opt/fl_start_test.sh`); the offline trajectory run had been lucky |
| `opt/run_v06.sh`, `opt/v06_all.sh`, `opt/v06_post.sh` | one-run / matrix / post-processing drivers; `ros2 bag play --clock -d 3` | node arguments are `opt/runs_v05.sh`'s `rt` line verbatim; `-d 3` lets discovery finish before the first sample (the v0.5 runs lost their first ~10 sweeps to it -- recv 1091/1101 -- and FAST-LIVO2 would lose its first second of IMU the same way) |
| `opt/res_sampler_v06.py`, `opt/topic_probe_v06.py` | 1 Hz psutil / nvidia-smi sampler; subscriber-side counter for /semantic_scan and /semantic_map | per-process CPU % and RSS for the contention measurement; delivery rate and interarrival at a consumer without `ros2 topic hz` |
| `opt/replay_v06.py` | replay_v05's main + `--pose-used` (score the live map at the CAUSAL placement the node applied) + the geometric footprint | the map-level metric places every GT point with a trajectory; for the ON arms that must be the poses the node actually used |
| `opt/verify_v06.py` | old and new module in one process, same cached predictions, same sweeps and images | byte identity of the default path |
| `tools/v06_analyze.py`, `tools/v06_traj_diff.py`, `tools/v06_online_report.py` | measurement extraction / trajectory divergence / this report | |

Backups of every touched source: `src/_pre_v06_backup/` (node, sem_core, ptv3_*, LIVMapper.cpp, params_helper.h).  Rebuilt: `colcon build --packages-select vikit_ros fast_livo` (Release).

**Default-path byte identity (`opt/verify_v06.py`, logs/verify_v06.log): 19 checks PASS, 0 FAIL** -- 40 real sweeps + images through both modules with identical cached predictions: every map array (key / score / xyz / n_obs / rgb / n_rgb), the hash table, the written .npz and the non-timing stats fields are identical.

## Arms and protocol

B0 checkpoint (`weights/v05/B0_student.pth`) everywhere; `bags/kitti_seq07_us` at rate 1.0 with `--clock -d 3`; node arguments `--reliable --conf-gate 0.5 --expect-voxels 4000000` (opt/runs_v05.sh `rt`); `FASTDDS_BUILTIN_TRANSPORTS=LARGE_DATA` unless stated; a subscriber-side probe on /semantic_scan and /semantic_map in every node run; 3 repetitions per arm, interleaved (fl, off, onopt, iso, onimu, onopthold, onopttx per repetition); one run at a time; an `nvidia-smi` compute-process gate before each run; per-run foreign CPU load re-derived from the sampler afterwards (see M2: below 13 % of one core in every run, GPU memory exclusively the worker's 1459 MiB).

| arm | processes | pose source | query |
|---|---|---|---|
| fl | parameter_blackboard + fastlivo_mapping | -- | -- (contention baseline; its evo file is the online-alone trajectory) |
| off | node + PTv3 worker + probe | `out/kitti_seq07_fastlivo2_tum.txt` (offline, rate 0.5) | non-causal interpolation (v0.5) |
| onopt | fastlivo_mapping + node + worker + probe | `/aft_mapped_to_init` live | causal; tail beyond the newest sample: constant twist from the two newest samples (`cv`) |
| iso | node + worker + probe | `stream_onopt_i.tum` = every sample the ON-opt run i received | non-causal interpolation |
| onimu | fastlivo_mapping (`uav.imu_rate_odom:=true`) + node + worker + probe | `/LIVO2/imu_propagate` live | causal (the stream reaches beyond the sweep end) |
| onopthold | as onopt | `/aft_mapped_to_init` | causal; tail HELD at the newest pose |
| onopttx | as onopt | `/aft_mapped_to_init` | causal, cv; `FASTDDS_BUILTIN_TRANSPORTS=LARGE_DATA?max_msg_size=5MB&sockets_size=20MB&non_blocking=false` for every process |

## M0 -- what FAST-LIVO2 actually publishes on this ROS 2 port (measured)

- `/aft_mapped_to_init`: one nav_msgs/Odometry per LIO update, stamped (after the patch) with `last_lio_update_time` = the IMAGE instant of the sweep (the LIVO scheduler cuts the LiDAR stream at each image), i.e. 0.49 of the way through the sweep (position of the newest sample at query time, p50 of run 1); stamp spacing p50 103.9 ms; 1096 samples for 1101 sweeps (the first 6 sweeps precede IMU initialisation).
- `/LIVO2/imu_propagate`: NOT published by default (`uav.imu_rate_odom: false` in config/kitti_velodyne64.yaml -> `imu_prop_enable` false).  With `-p uav.imu_rate_odom:=true` it publishes an IMU-propagated Odometry from a 4 ms wall timer whenever a new IMU sample has arrived: 8144 samples over 114.0 s = 71 Hz (stamp spacing p50 10.0 ms, i.e. the 100 Hz IMU with ~29 % of the samples coalesced by the timer), 7.4 samples inside a 104 ms sweep (min 3), stamped with the IMU sample time.  Between LIO updates it is a pure forward propagation; at each update it re-bases on the new state (a correction jump inside the stream).
- Before the patch no topic carried sensor time: `/aft_mapped_to_init`, `/cloud_registered`, `/path`, `/mavros/vision_pose/pose`, `/rgb_img` are all stamped `now()` (LIVMapper.cpp 1202 / 1264 / 1379 / 1397 / 1417 / 1436 / 1445); only the evo FILE used `last_lio_update_time`.
- FAST-LIVO2's other publishers were left exactly as in the offline trajectory run (nothing subscribes to /cloud_registered, /Laser_map, /path, /planes, /rgb_img: Fast DDS transmits nothing without a matched reader, and their serialisation cost is inside FAST-LIVO2's per-frame time in BOTH the offline and the online runs, so the trajectory comparison stays like for like).  Nothing was switched off; only /aft_mapped_to_init (and, in the imu arm, /LIVO2/imu_propagate) is subscribed.

## M1 -- pose arrival latency (wall clock, relative to the sweep's own arrival at the node)

Definition: for sweep k, arrival = the moment the node's LiDAR callback fires (the bag delivers a sweep at its END time, like a real driver; this includes rclpy's ~10 ms deserialisation of the 2.7 MB message).  Two pose events matter to a causal consumer: (a) the LIO update INSIDE the sweep (stamped at its image instant, ~half-way) -- the newest usable sample when stage B runs; (b) the first sample whose stamp is beyond the sweep END -- what a wait-for-coverage policy would have to wait for.  Also: the sample's age at arrival in bag time (wall minus the wall-to-bag offset taken from the LiDAR deliveries), when stage B starts, how much of the sweep was extrapolated.  Distributions over the queued sweeps of each run, p50 / p95 / p99 / max in ms.

| arm | rep | pose msgs | (a) in-sweep pose after arrival | (b) sweep-end coverage after arrival | pose age at arrival (bag time) | stage B start after arrival | extrapolated span (p50 / p95 / max) | sweeps extrapolated | samples in sweep (mean / min) |
|---|---|---|---|---|---|---|---|---|---|
| onopt | 1 | 1096 | 44 / 63 / 69 / 80 | 149 / 169 / 177 / 183 | 93 / 114 / 120 / 125 | 62 / 70 / 75 / 104 | 53 / 53 / 158 | 100 % | 1.0 / 0 |
| onopt | 2 | 1096 | 43 / 61 / 77 / 90 | 147 / 166 / 180 / 197 | 100 / 118 / 132 / 145 | 65 / 75 / 87 / 117 | 53 / 53 / 157 | 100 % | 1.0 / 0 |
| onopt | 3 | 1096 | 45 / 65 / 79 / 96 | 149 / 171 / 183 / 204 | 96 / 115 / 131 / 151 | 64 / 69 / 84 / 114 | 53 / 53 / 158 | 100 % | 1.0 / 0 |
| onimu | 1 | 8144 | -101 / -54 / -40 / -8 | -2 / 54 / 70 / 106 | -10 / 36 / 49 / 90 | 63 / 73 / 81 / 110 | 0 / 0 / 9 | 1 % | 7.4 / 3 |
| onimu | 2 | 8021 | -100 / -49 / -38 / -25 | 3 / 59 / 70 / 83 | -9 / 38 / 52 / 79 | 65 / 73 / 86 / 108 | 0 / 0 / 9 | 1 % | 7.3 / 3 |
| onimu | 3 | 7983 | -98 / -46 / -38 / -9 | 5 / 61 / 70 / 94 | -0 / 47 / 62 / 139 | 66 / 73 / 84 / 122 | 0 / 0 / 80 | 1 % | 7.3 / 2 |
| onopthold | 1 | 1096 | 42 / 63 / 68 / 86 | 147 / 168 / 178 / 191 | 98 / 122 / 127 / 141 | 65 / 72 / 91 / 162 | 53 / 53 / 157 | 100 % | 1.0 / 0 |
| onopthold | 2 | 1096 | 44 / 64 / 81 / 92 | 148 / 171 / 184 / 201 | 94 / 115 / 127 / 145 | 64 / 72 / 87 / 125 | 53 / 53 / 158 | 100 % | 1.0 / 0 |
| onopthold | 3 | 1096 | 44 / 68 / 80 / 93 | 148 / 173 / 186 / 199 | 101 / 126 / 139 / 152 | 66 / 74 / 90 / 115 | 53 / 54 / 158 | 100 % | 1.0 / 0 |
| onopttx | 1 | 1095 | 42 / 63 / 73 / 110 | 146 / 168 / 177 / 267 | 96 / 118 / 126 / 196 | 66 / 78 / 92 / 131 | 53 / 53 / 157 | 100 % | 1.0 / 0 |
| onopttx | 2 | 1095 | 43 / 64 / 80 / 87 | 146 / 168 / 185 / 260 | 96 / 117 / 133 / 189 | 64 / 76 / 95 / 117 | 53 / 53 / 157 | 100 % | 1.0 / 0 |
| onopttx | 3 | 1095 | 41 / 62 / 68 / 79 | 145 / 165 / 173 / 262 | 95 / 115 / 123 / 197 | 67 / 76 / 86 / 110 | 53 / 53 / 157 | 100 % | 1.0 / 0 |

For the IMU-propagated stream (a) is negative: the samples stamped inside a sweep are published as the IMU arrives, i.e. before the sweep itself is delivered at its end; (b) is the relevant event there and it coincides with the sweep's own arrival (p50 -2 .. +5 ms), which is why nothing is extrapolated.  For the optimised stream (a) is the update at the image instant and (b) the NEXT sweep's update.

## M2 -- resource contention (12-core Xeon Gold 6248R, no SMT; one RTX 4090; 1 Hz psutil per process and the same 1 Hz nvidia-smi query the v0.5 sampler used; window = while the bag player is alive)

CPU % is per process (100 = one core; FAST-LIVO2 is compiled with MP_PROC_NUM=4 OpenMP threads).  RSS = peak.  FAST-LIVO2's per-frame LIO / VIO times are parsed from its own stdout tables.  `foreign` = system CPU minus every process of the run, i.e. whatever else the box was doing (two sibling tracks were polling in the background; they stayed idle).

| arm | rep | FL cpu mean / p95 | FL RSS MB | FL LIO ms p50 / p95 / max | LIO > 104 ms | FL VIO ms p50 / p95 | node cpu mean / p95 | worker cpu mean / p95 | node / worker RSS MB | probe / bag cpu | sys cpu mean (of 12 cores) | cores > 80 % (mean) | foreign cpu mean / p95 | GPU util mean | GPU mem MiB max |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fl | 1 | 137 / 162 | 3588 | 28.6 / 42.2 / 60.4 | 0 | 8.6 / 12.3 | - / - | - / - | - / - | - / 8 | 12.8 | 0.0 | 8 / 20 | 0 | 15 |
| fl | 2 | 134 / 165 | 3563 | 27.5 / 40.9 / 59.3 | 0 | 8.9 / 12.7 | - / - | - / - | - / - | - / 7 | 12.5 | 0.0 | 8 / 19 | 0 | 15 |
| fl | 3 | 138 / 168 | 3570 | 27.8 / 44.3 / 59.3 | 0 | 8.9 / 13.5 | - / - | - / - | - / - | - / 7 | 12.9 | 0.0 | 9 / 20 | 0 | 15 |
| off | 1 | - | - | - | - | - | 60 / 101 | 55 / 59 | 1032 / 1664 | 3 / 6 | 11.3 | 0.0 | 11 / 18 | 20 | 1459 |
| off | 2 | - | - | - | - | - | 62 / 94 | 56 / 60 | 1128 / 1662 | 4 / 6 | 11.7 | 0.0 | 13 / 21 | 20 | 1459 |
| off | 3 | - | - | - | - | - | 62 / 99 | 54 / 58 | 1046 / 1662 | 3 / 6 | 11.4 | 0.0 | 11 / 17 | 20 | 1459 |
| onopt | 1 | 136 / 167 | 3551 | 29.4 / 46.7 / 58.1 | 0 | 8.9 / 12.9 | 62 / 101 | 53 / 60 | 1104 / 1664 | 4 / 7 | 22.8 | 0.0 | 11 / 24 | 19 | 1459 |
| onopt | 2 | 135 / 163 | 3594 | 28.1 / 43.4 / 65.9 | 0 | 9.0 / 12.3 | 63 / 100 | 56 / 63 | 1109 / 1662 | 4 / 7 | 22.8 | 0.0 | 8 / 20 | 19 | 1459 |
| onopt | 3 | 142 / 175 | 3530 | 32.0 / 47.8 / 66.8 | 0 | 9.5 / 13.4 | 64 / 102 | 54 / 59 | 1111 / 1664 | 4 / 8 | 23.4 | 0.0 | 7 / 18 | 20 | 1459 |
| iso | 1 | - | - | - | - | - | 60 / 99 | 54 / 59 | 1064 / 1666 | 3 / 6 | 11.3 | 0.0 | 12 / 18 | 20 | 1459 |
| iso | 2 | - | - | - | - | - | 61 / 100 | 56 / 60 | 1163 / 1660 | 4 / 6 | 11.3 | 0.0 | 9 / 16 | 19 | 1459 |
| iso | 3 | - | - | - | - | - | 62 / 103 | 55 / 59 | 1104 / 1666 | 4 / 6 | 11.2 | 0.0 | 9 / 15 | 20 | 1459 |
| onimu | 1 | 136 / 169 | 3558 | 29.2 / 45.4 / 62.1 | 0 | 9.1 / 12.8 | 66 / 106 | 54 / 60 | 1213 / 1688 | 4 / 7 | 23.1 | 0.0 | 9 / 20 | 20 | 1459 |
| onimu | 2 | 139 / 169 | 3557 | 29.2 / 46.6 / 59.2 | 0 | 9.1 / 13.1 | 68 / 112 | 56 / 62 | 1096 / 1667 | 4 / 7 | 23.8 | 0.0 | 11 / 22 | 19 | 1459 |
| onimu | 3 | 149 / 183 (residual: sampler lost the pid) | - | 28.6 / 45.6 / 66.8 | 0 | 9.4 / 13.5 | 68 / 108 | 57 / 63 | 1136 / 1688 | 4 / 7 | 23.9 | 0.0 | (FL inside residual) | 20 | 1459 |
| onopthold | 1 | 140 / 170 | 3540 | 28.5 / 47.4 / 61.1 | 0 | 9.1 / 13.6 | 65 / 103 | 57 / 62 | 1102 / 1662 | 4 / 7 | 23.6 | 0.0 | 9 / 20 | 20 | 1459 |
| onopthold | 2 | 140 / 173 | 3575 | 30.0 / 47.1 / 65.4 | 0 | 9.2 / 14.0 | 64 / 104 | 55 / 62 | 1150 / 1675 | 4 / 8 | 23.3 | 0.0 | 8 / 17 | 20 | 1459 |
| onopthold | 3 | 142 / 173 | 3553 | 29.2 / 49.9 / 70.1 | 0 | 9.1 / 13.5 | 66 / 110 | 58 / 64 | 1093 / 1668 | 4 / 7 | 23.8 | 0.0 | 8 / 19 | 20 | 1459 |
| onopttx | 1 | 141 / 178 | 3719 | 29.2 / 47.8 / 63.3 | 0 | 9.2 / 13.7 | 66 / 103 | 58 / 67 | 1279 / 1662 | 5 / 8 | 24.0 | 0.0 | 10 / 19 | 20 | 1459 |
| onopttx | 2 | 141 / 169 | 3704 | 30.9 / 48.8 / 70.6 | 0 | 8.9 / 13.3 | 66 / 102 | 56 / 65 | 1264 / 1661 | 5 / 9 | 23.9 | 0.0 | 9 / 19 | 20 | 1459 |
| onopttx | 3 | 139 / 170 | 3691 | 28.6 / 47.6 / 65.6 | 0 | 9.2 / 13.3 | 67 / 105 | 58 / 64 | 1359 / 1660 | 4 / 8 | 23.9 | 0.0 | 10 / 21 | 20 | 1459 |

Node-internal stage times (ms, mean over the run; v0.5 rt B0 reference: PTv3 57.9, stage A 2.7, gate 3.0, de-skew 5.6, proj 4.6, map 23.1, pub 4.3, stage B 40.7, period 104.0):

| arm | rep | PTv3 worker mean / p95 | stage A | gate | de-skew | proj | map | pub | stage B mean / p95 | period mean / p95 / max | latency (submit -> done) p50 / p95 / max | snapshot ms mean / max (n) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| off | 1 | 57.9 / 60.4 | 2.62 | 2.67 | 5.15 | 4.04 | 21.70 | 4.80 | 38.5 / 46.0 | 103.9 / 117.8 / 158.8 | 96.2 / 105.6 / 154.1 | 272 / 635 (72) |
| off | 2 | 58.9 / 61.6 | 2.95 | 2.78 | 5.24 | 4.22 | 22.98 | 5.19 | 40.5 / 48.8 | 103.9 / 118.0 / 163.4 | 98.7 / 109.4 / 163.8 | 276 / 673 (72) |
| off | 3 | 57.4 / 60.4 | 2.90 | 2.82 | 5.30 | 4.28 | 23.31 | 5.29 | 41.1 / 51.3 | 103.9 / 118.2 / 166.3 | 98.0 / 110.8 / 165.4 | 277 / 664 (72) |
| onopt | 1 | 57.5 / 62.6 | 2.64 | 2.75 | 5.50 | 4.10 | 22.16 | 4.92 | 39.6 / 48.0 | 103.9 / 120.3 / 162.4 | 96.7 / 109.1 / 162.0 | 293 / 847 (69) |
| onopt | 2 | 59.9 / 65.2 | 2.93 | 2.85 | 5.75 | 4.29 | 23.09 | 5.17 | 41.3 / 50.2 | 103.9 / 120.4 / 169.5 | 100.3 / 113.4 / 160.7 | 286 / 698 (70) |
| onopt | 3 | 58.0 / 62.2 | 2.98 | 2.90 | 5.61 | 4.32 | 23.50 | 5.33 | 41.8 / 52.8 | 103.9 / 120.9 / 171.1 | 98.8 / 114.6 / 160.2 | 287 / 681 (70) |
| iso | 1 | 57.3 / 60.5 | 2.59 | 2.71 | 5.22 | 4.06 | 21.86 | 4.73 | 38.7 / 45.8 | 103.9 / 118.2 / 179.2 | 95.5 / 109.4 / 174.7 | 275 / 663 (72) |
| iso | 2 | 58.9 / 62.4 | 2.86 | 2.83 | 5.29 | 4.25 | 22.70 | 5.06 | 40.2 / 48.1 | 103.9 / 117.9 / 177.8 | 98.7 / 109.8 / 176.8 | 280 / 679 (72) |
| iso | 3 | 58.0 / 61.6 | 2.86 | 2.75 | 5.35 | 4.31 | 22.87 | 5.07 | 40.5 / 49.5 | 103.9 / 118.3 / 153.6 | 98.3 / 109.4 / 157.6 | 278 / 669 (72) |
| onimu | 1 | 57.9 / 62.4 | 2.82 | 3.03 | 5.56 | 4.41 | 23.84 | 5.24 | 42.2 / 52.7 | 103.9 / 120.2 / 183.0 | 98.9 / 116.4 / 182.3 | 305 / 743 (68) |
| onimu | 2 | 59.7 / 65.1 | 2.90 | 3.01 | 5.66 | 4.53 | 24.24 | 5.36 | 43.0 / 52.2 | 103.9 / 120.0 / 149.4 | 101.8 / 117.6 / 162.5 | 300 / 825 (68) |
| onimu | 3 | 60.5 / 66.2 | 2.84 | 2.90 | 5.68 | 4.52 | 24.33 | 5.41 | 43.0 / 51.2 | 103.9 / 118.9 / 170.5 | 103.0 / 117.3 / 162.9 | 304 / 707 (68) |
| onopthold | 1 | 60.1 / 64.9 | 2.99 | 2.90 | 5.39 | 4.40 | 23.41 | 5.37 | 41.6 / 51.6 | 103.9 / 119.7 / 197.0 | 100.5 / 115.3 / 203.2 | 283 / 695 (71) |
| onopthold | 2 | 59.2 / 64.2 | 2.97 | 2.90 | 5.32 | 4.28 | 23.81 | 5.39 | 41.8 / 52.6 | 103.9 / 120.6 / 168.7 | 99.0 / 116.9 / 167.6 | 300 / 871 (68) |
| onopthold | 3 | 60.7 / 66.9 | 2.90 | 2.91 | 5.40 | 4.34 | 23.79 | 5.30 | 41.9 / 53.2 | 103.9 / 120.3 / 164.6 | 101.4 / 118.7 / 157.5 | 288 / 755 (69) |
| onopttx | 1 | 61.1 / 72.4 | 3.37 | 2.84 | 5.78 | 4.35 | 23.59 | 6.52 | 43.2 / 53.7 | 104.0 / 115.7 / 204.1 | 103.5 / 122.2 / 176.0 | 292 / 728 (69) |
| onopttx | 2 | 59.7 / 70.2 | 3.09 | 2.85 | 5.54 | 4.40 | 23.61 | 6.59 | 43.1 / 54.4 | 104.0 / 115.0 / 207.8 | 100.3 / 117.9 / 174.3 | 299 / 800 (69) |
| onopttx | 3 | 61.0 / 67.2 | 3.37 | 2.94 | 5.75 | 4.40 | 23.66 | 6.77 | 43.7 / 55.6 | 104.0 / 114.2 / 205.3 | 104.2 / 119.7 / 175.8 | 289 / 706 (70) |

## M3 -- end-to-end latency and throughput

In-node latency = sweep arrival (callback) -> /semantic_scan published (stage B done), per processed sweep from the frame log; external latency = the same sweep's /semantic_scan message arriving at the probe process (only the delivered ones).  Throughput = processed sweeps against the 1101 the bag holds (114.8 s, 9.59 Hz); the bag never exceeds the pipeline, so every arm processes every sweep that had a pose.

| arm | rep | received | processed | bp-drop | no-pose | stale | in-node latency p50 / p95 / p99 / max | external latency p50 / p95 / p99 / max | sweep delivery jitter p95 / p99 / max | processed / 1101 |
|---|---|---|---|---|---|---|---|---|---|---|
| off | 1 | 1101 | 1096 | 0 | 5 | - | 99 / 110 / 130 / 159 | 101 / 111 / 130 / 153 | 10 / 13 / 28 | 99.5 % |
| off | 2 | 1101 | 1096 | 0 | 5 | - | 101 / 113 / 130 / 167 | 104 / 116 / 134 / 174 | 10 / 16 / 34 | 99.5 % |
| off | 3 | 1101 | 1096 | 0 | 5 | - | 101 / 114 / 129 / 170 | 103 / 116 / 128 / 164 | 13 / 20 / 34 | 99.5 % |
| onopt | 1 | 1101 | 1095 | 0 | 6 | 0 | 99 / 113 / 126 / 164 | 102 / 114 / 128 / 166 | 3 / 11 / 22 | 99.5 % |
| onopt | 2 | 1101 | 1095 | 0 | 6 | 0 | 103 / 117 / 139 / 163 | 105 / 118 / 140 / 185 | 10 / 19 / 21 | 99.5 % |
| onopt | 3 | 1101 | 1095 | 0 | 6 | 0 | 102 / 118 / 140 / 165 | 105 / 123 / 158 / 174 | 4 / 12 / 31 | 99.5 % |
| iso | 1 | 1101 | 1096 | 0 | 5 | - | 98 / 113 / 129 / 177 | 101 / 114 / 133 / 179 | 10 / 13 / 28 | 99.5 % |
| iso | 2 | 1101 | 1096 | 0 | 5 | - | 101 / 113 / 130 / 182 | 104 / 115 / 134 / 206 | 10 / 13 / 27 | 99.5 % |
| iso | 3 | 1101 | 1096 | 0 | 5 | - | 101 / 113 / 134 / 160 | 103 / 116 / 139 / 177 | 10 / 21 / 40 | 99.5 % |
| onimu | 1 | 1101 | 1095 | 0 | 6 | 0 | 102 / 119 / 137 / 188 | 105 / 123 / 146 / 189 | 3 / 11 / 53 | 99.5 % |
| onimu | 2 | 1101 | 1095 | 0 | 6 | 0 | 105 / 121 / 136 / 166 | 108 / 123 / 145 / 165 | 2 / 11 / 34 | 99.5 % |
| onimu | 3 | 1101 | 1095 | 0 | 6 | 0 | 106 / 120 / 141 / 166 | 109 / 123 / 143 / 192 | 11 / 20 / 46 | 99.5 % |
| onopthold | 1 | 1101 | 1095 | 0 | 6 | 0 | 103 / 120 / 143 / 206 | 106 / 123 / 150 / 223 | 9 / 17 / 38 | 99.5 % |
| onopthold | 2 | 1101 | 1095 | 0 | 6 | 0 | 102 / 121 / 138 / 170 | 105 / 125 / 144 / 173 | 8 / 11 / 30 | 99.5 % |
| onopthold | 3 | 1101 | 1095 | 0 | 6 | 0 | 104 / 122 / 136 / 160 | 107 / 125 / 139 / 173 | 11 / 19 / 23 | 99.5 % |
| onopttx | 1 | 1100 | 1094 | 0 | 6 | 0 | 107 / 125 / 150 / 180 | 111 / 128 / 153 / 187 | 4 / 6 / 23 | 99.4 % |
| onopttx | 2 | 1100 | 1094 | 0 | 6 | 0 | 104 / 121 / 147 / 182 | 107 / 126 / 151 / 192 | 4 / 7 / 32 | 99.4 % |
| onopttx | 3 | 1100 | 1094 | 0 | 6 | 0 | 108 / 124 / 139 / 178 | 111 / 127 / 143 / 182 | 3 / 5 / 46 | 99.4 % |

## M4 -- topic stability at a subscriber (/semantic_scan BEST_EFFORT/KEEP_LAST(1) publisher, ~3.2 MB per sweep; /semantic_map RELIABLE/TRANSIENT_LOCAL/KEEP_LAST(1), up to 15 MB)

Probe: opt/topic_probe_v06.py (BEST_EFFORT/KEEP_LAST(10) on the scan, RELIABLE/TRANSIENT_LOCAL/KEEP_LAST(5) on the map; records stamp + wall arrival, nothing else).  Intervals in ms between consecutive DELIVERED messages; delivery = received / published (scan: processed sweeps; map: `snapshot_ms.n`, the final snapshot excluded from the interval).

| arm | rep | transport | scan published | delivered | delivery % | scan interval p50 / p95 / p99 / p99.9 / max | map published | delivered | map interval p50 / p95 / max | node pub ms mean / max |
|---|---|---|---|---|---|---|---|---|---|---|
| off | 1 | LARGE_DATA | 1096 | 645 | 58.9 | 109 / 415 / 719 / 3532 / 3535 | 72 | 72 | 1206 / 3055 / 3127 | 4.8 / 40.0 |
| off | 2 | LARGE_DATA | 1096 | 890 | 81.2 | 105 / 221 / 417 / 612 / 615 | 72 | 72 | 1206 / 3189 / 3354 | 5.2 / 57.5 |
| off | 3 | LARGE_DATA | 1096 | 611 | 55.7 | 105 / 463 / 1433 / 3857 / 4668 | 72 | 72 | 1182 / 3095 / 3305 | 5.3 / 58.1 |
| onopt | 1 | LARGE_DATA | 1095 | 716 | 65.4 | 111 / 323 / 610 / 1137 / 1141 | 69 | 69 | 1179 / 3184 / 3441 | 4.9 / 27.1 |
| onopt | 2 | LARGE_DATA | 1095 | 863 | 78.8 | 106 / 235 / 363 / 606 / 1135 | 70 | 70 | 1186 / 3233 / 3464 | 5.2 / 25.5 |
| onopt | 3 | LARGE_DATA | 1095 | 700 | 63.9 | 106 / 416 / 942 / 2414 / 2949 | 70 | 70 | 1202 / 3289 / 3402 | 5.3 / 66.1 |
| iso | 1 | LARGE_DATA | 1096 | 707 | 64.5 | 107 / 415 / 531 / 832 / 833 | 72 | 72 | 1193 / 3027 / 3310 | 4.7 / 21.9 |
| iso | 2 | LARGE_DATA | 1096 | 779 | 71.1 | 106 / 311 / 515 / 765 / 921 | 72 | 72 | 1198 / 3160 / 3359 | 5.1 / 29.9 |
| iso | 3 | LARGE_DATA | 1096 | 795 | 72.5 | 106 / 315 / 509 / 740 / 837 | 72 | 72 | 1205 / 3088 / 3282 | 5.1 / 49.7 |
| onimu | 1 | LARGE_DATA | 1095 | 741 | 67.7 | 108 / 327 / 594 / 1480 / 3354 | 68 | 68 | 1209 / 3358 / 3655 | 5.2 / 39.8 |
| onimu | 2 | LARGE_DATA | 1095 | 759 | 69.3 | 107 / 322 / 695 / 1476 / 1874 | 68 | 68 | 1198 / 3510 / 3976 | 5.4 / 34.8 |
| onimu | 3 | LARGE_DATA | 1095 | 890 | 81.3 | 106 / 220 / 416 / 553 / 633 | 68 | 68 | 1208 / 3401 / 3526 | 5.4 / 49.3 |
| onopthold | 1 | LARGE_DATA | 1095 | 776 | 70.9 | 106 / 302 / 686 / 2618 / 3739 | 71 | 71 | 1203 / 3282 / 3404 | 5.4 / 13.9 |
| onopthold | 2 | LARGE_DATA | 1095 | 803 | 73.3 | 106 / 309 / 728 / 1285 / 1381 | 68 | 68 | 1196 / 3429 / 4320 | 5.4 / 20.7 |
| onopthold | 3 | LARGE_DATA | 1095 | 811 | 74.1 | 105 / 240 / 813 / 1546 / 3137 | 69 | 69 | 1183 / 3464 / 3702 | 5.3 / 16.5 |
| onopttx | 1 | LARGE_DATA?max_msg_size=5MB&amp;sockets_size=20MB&amp;non_blocking=false | 1094 | 1093 | 99.9 | 104 / 116 / 141 / 202 / 206 | 69 | 50 | 1133 / 3903 / 32301 | 6.5 / 18.8 |
| onopttx | 2 | LARGE_DATA?max_msg_size=5MB&amp;sockets_size=20MB&amp;non_blocking=false | 1094 | 1090 | 99.6 | 104 / 116 / 152 / 209 / 210 | 69 | 49 | 1150 / 3391 / 27350 | 6.6 / 14.3 |
| onopttx | 3 | LARGE_DATA?max_msg_size=5MB&amp;sockets_size=20MB&amp;non_blocking=false | 1094 | 1090 | 99.6 | 104 / 115 / 144 / 206 / 219 | 70 | 30 | 2433 / 11101 / 38467 | 6.8 / 57.3 |

Transport diagnosis (21 s smokes, `out/v06/runs/analysis_smoke_*.json`), delivered / published on /semantic_scan:

- BEST_EFFORT/KEEP_LAST(1), plain LARGE_DATA (ON-imu smoke, no `-d`): **102 / 176 (58 %)**, interval p50 / p95 / p99 / max 106 / 408 / 1248 / 1499 ms
- BEST_EFFORT/KEEP_LAST(1), plain LARGE_DATA (ON-opt smoke): **124 / 194 (64 %)**, interval p50 / p95 / p99 / max 106 / 423 / 786 / 1163 ms
- RELIABLE/KEEP_LAST(1): **69 / 195 (35 %)**, interval p50 / p95 / p99 / max 209 / 788 / 1281 / 1355 ms
- BEST_EFFORT/KEEP_LAST(10): **94 / 195 (48 %)**, interval p50 / p95 / p99 / max 106 / 656 / 1047 / 2187 ms
- RELIABLE/KEEP_LAST(10): **110 / 195 (56 %)**, interval p50 / p95 / p99 / max 109 / 443 / 719 / 1438 ms
- BEST_EFFORT/KEEP_LAST(1), no periodic /semantic_map publish (map-rate 0.05): **91 / 195 (47 %)**, interval p50 / p95 / p99 / max 111 / 688 / 901 / 1353 ms
- BEST_EFFORT/KEEP_LAST(1), `LARGE_DATA?max_msg_size=5MB&sockets_size=20MB&non_blocking=false`: **194 / 195 (99 %)**, interval p50 / p95 / p99 / max 104 / 114 / 133 / 205 ms

## M5 -- the three-way decomposition (map level, common-9 mIoU / point accuracy on SemanticKITTI seq07 GT, abstain-WRONG)

Scorer: opt/replay_v06.py.  **live**: every evaluated GT point with a valid pose is placed in the world with the poses the map was built with and looked up in the node-written map by 0.2 m voxel key (no voxel = wrong) -- v0.5's LIVE-map reading, with one addition: for the ON arms the placement uses the CAUSAL bin poses the node recorded (`--pose-used`), so a point is scored in the voxel the node actually put it in.  **replay**: the CPU replay of stage B from the cached per-point predictions (draw B0_r_i for repetition i), `map_all_lookup` reading, with the arm's trajectory -- v0.5's OFF number (77.49 +- 0.19 out-of-frustum) is exactly this reading with the offline TUM.  mean +- population std over the 3 repetitions.

### The decomposition (out-of-frustum mIoU-9 is the headline reading of v0.5; in-frustum and global beside it)

| arm | pose source | query | LIVE map mIoU-9 out / in / global (at the node's own placement) | LIVE out, read at the NON-causal placement of the same stream | LIVE acc out / in / global | replay mIoU-9 out / in / global (same trajectory, cached draws) | trajectory ATE m | live voxels | live / replay voxels | lookup hit |
|---|---|---|---|---|---|---|---|---|---|---|
| **off** | offline TUM (rate 0.5, alone) | non-causal | **78.10 +- 0.08** / 78.44 +- 0.13 / 78.24 +- 0.04 | (same) | 91.61 +- 0.01 / 91.98 +- 0.01 / 91.67 +- 0.01 | 77.49 +- 0.15 / 77.28 +- 0.33 / 77.49 +- 0.19 | 0.880 +- 0.000 | 2693577 +- 313 | 1.000 +- 0.000 | 0.9997 +- 0.0000 |
| **onopt** | live /aft_mapped_to_init | causal, cv | **77.96 +- 0.19** / 77.98 +- 0.22 / 78.02 +- 0.17 | 77.94 +- 0.18 | 91.56 +- 0.04 / 91.92 +- 0.08 / 91.62 +- 0.04 | 77.33 +- 0.11 / 77.21 +- 0.21 / 77.35 +- 0.12 | 1.018 +- 0.098 | 2792701 +- 11380 | 1.030 +- 0.003 | 0.9996 +- 0.0000 |
| **iso** | recorded ON-opt stream i | non-causal | **77.88 +- 0.12** / 78.16 +- 0.29 / 78.01 +- 0.12 | (same) | 91.58 +- 0.06 / 91.92 +- 0.07 / 91.63 +- 0.06 | 77.33 +- 0.11 / 77.21 +- 0.21 / 77.35 +- 0.12 | 1.018 +- 0.098 | 2711457 +- 18620 | 1.000 +- 0.000 | 0.9997 +- 0.0000 |
| **onimu** | live /LIVO2/imu_propagate | causal | **77.70 +- 0.14** / 78.35 +- 0.54 / 77.91 +- 0.16 | 77.70 +- 0.14 | 91.62 +- 0.06 / 91.89 +- 0.04 / 91.66 +- 0.06 | 77.17 +- 0.15 / 76.96 +- 0.61 / 77.18 +- 0.25 | 0.764 +- 0.097 | 2862435 +- 18435 | 1.000 +- 0.000 | 0.9996 +- 0.0000 |
| **onopthold** | live /aft_mapped_to_init | causal, hold | **77.87 +- 0.37** / 79.68 +- 0.58 / 78.28 +- 0.41 | 76.18 +- 0.38 | 91.62 +- 0.08 / 92.28 +- 0.10 / 91.72 +- 0.08 | 77.41 +- 0.25 / 77.21 +- 0.21 / 77.41 +- 0.24 | 0.878 +- 0.038 | 2831080 +- 56186 | 1.042 +- 0.016 | 0.9996 +- 0.0000 |
| **onopttx** | live /aft_mapped_to_init, tuned transport | causal, cv | **77.72 +- 0.11** / 78.20 +- 0.19 / 77.88 +- 0.12 | 77.72 +- 0.12 | 91.56 +- 0.04 / 91.94 +- 0.03 / 91.62 +- 0.04 | 77.27 +- 0.25 / 77.25 +- 0.49 / 77.30 +- 0.29 | 0.947 +- 0.059 | 2765550 +- 21890 | 1.020 +- 0.005 | 0.9996 +- 0.0000 |
| v0.5 OFF reference (out/v05) | offline TUM | non-causal | live, 1 draw: 77.82 / 77.69 / 77.85 | (same) | 91.54 / 91.94 / 91.60 | 77.49 +- 0.15 / 77.28 +- 0.33 / 77.49 +- 0.19 | 0.880 | 2677268 | - | 0.9994 |

Per repetition (out-of-frustum): LIVE mIoU-9 / acc, replay mIoU-9, the ON maps read with the NON-causal interpolation of their own stream, and the trajectory the map was built on:

| arm | rep | live mIoU-9 | live acc | replay mIoU-9 | live, non-causal placement | causal sweeps / fallback | no-pose frames | trajectory ATE m |
|---|---|---|---|---|---|---|---|---|
| off | 1 | 78.20 | 91.61 | 77.29 | (same) | 0 / 1096 | 5 | 0.880 |
| off | 2 | 78.00 | 91.60 | 77.52 | (same) | 0 / 1096 | 5 | 0.880 |
| off | 3 | 78.11 | 91.62 | 77.66 | (same) | 0 / 1096 | 5 | 0.880 |
| onopt | 1 | 77.78 | 91.62 | 77.20 | 77.78 | 1095 / 1 | 5 | 0.883 |
| onopt | 2 | 78.21 | 91.53 | 77.33 | 78.19 | 1095 / 1 | 5 | 1.056 |
| onopt | 3 | 77.88 | 91.55 | 77.47 | 77.86 | 1095 / 1 | 5 | 1.115 |
| iso | 1 | 77.95 | 91.66 | 77.20 | (same) | 0 / 1096 | 5 | 0.883 |
| iso | 2 | 77.99 | 91.55 | 77.33 | (same) | 0 / 1096 | 5 | 1.056 |
| iso | 3 | 77.71 | 91.53 | 77.47 | (same) | 0 / 1096 | 5 | 1.115 |
| onimu | 1 | 77.86 | 91.62 | 77.21 | 77.86 | 1095 / 0 | 6 | 0.751 |
| onimu | 2 | 77.72 | 91.54 | 76.97 | 77.72 | 1095 / 0 | 6 | 0.888 |
| onimu | 3 | 77.52 | 91.70 | 77.33 | 77.53 | 1095 / 0 | 6 | 0.652 |
| onopthold | 1 | 77.60 | 91.58 | 77.15 | 75.92 | 1095 / 1 | 5 | 0.826 |
| onopthold | 2 | 78.39 | 91.73 | 77.75 | 76.72 | 1095 / 1 | 5 | 0.916 |
| onopthold | 3 | 77.62 | 91.55 | 77.32 | 75.91 | 1095 / 1 | 5 | 0.891 |
| onopttx | 1 | 77.62 | 91.62 | 76.96 | 77.61 | 1094 / 2 | 5 | 0.931 |
| onopttx | 2 | 77.67 | 91.53 | 77.26 | 77.66 | 1094 / 2 | 5 | 0.883 |
| onopttx | 3 | 77.88 | 91.54 | 77.58 | 77.88 | 1094 / 2 | 5 | 1.026 |

**Decomposition (paired by repetition, so each difference is between maps built on the SAME trajectory draw where that applies; mean +- std of the 3 paired differences), mIoU-9 out / in / global:**

- ON-opt minus CAUSAL-ISO = the causal-query restriction alone (same trajectory, same run pairing): **+0.07 +- 0.18** / -0.18 +- 0.39 / +0.02 +- 0.20
- CAUSAL-ISO minus OFF = the online trajectory itself (live maps): **-0.22 +- 0.16** / -0.28 +- 0.40 / -0.23 +- 0.12; the same at replay level (identical cached draws, only the trajectory differs): -0.16 +- 0.04 / -0.06 +- 0.16 / -0.14 +- 0.07
- ON-opt minus OFF = everything: **-0.15 +- 0.27** / -0.46 +- 0.27 / -0.21 +- 0.22

The live reading at the node's own placement is a LABEL-CONSISTENCY reading: it asks whether the voxel a point was put in carries the right class, so a coherent geometric displacement of the whole sweep tail is invisible to it.  `hold` shows this: at its own placement it scores like `cv` (77.87 +- 0.37), read at the non-causal placement of its own stream it loses 76.18 +- 0.38; `cv` reads the same either way (77.96 +- 0.19 vs 77.94 +- 0.18).  The geometric footprint table below is the honest size of the causal error; the map-inflation ratio (live / replay voxels, same stream) is its map-level trace: 1.030 +- 0.003 (cv) and 1.000 +- 0.000 (imu stream) against 1.000 +- 0.000 (OFF).

- ON-imu minus ON-opt (the IMU-propagated stream against the optimised stream + extrapolation; different FAST-LIVO2 runs, so the trajectory draw differs too): -0.26 +- 0.25 / +0.37 +- 0.69 / -0.12 +- 0.30
- ON-opt-hold minus ON-opt (the extrapolation policy; different runs): -0.09 +- 0.19 / +1.70 +- 0.51 / +0.26 +- 0.23
- ON-opt-tx minus ON-opt (tuned transport; different runs): -0.23 +- 0.23 / +0.22 +- 0.40 / -0.15 +- 0.25

**Trajectory quality vs map quality across all 18 maps (each scored on the trajectory it was built with):** Pearson r = 0.29, fitted slope +0.6 mIoU-9 per metre of ATE RMSE over an ATE range of 0.652 .. 1.115 m -- no detectable relation: within this range the map metric does not see global trajectory error (it enters only as voxel mixing, and the maps are scored on their own trajectory).

### The geometric footprint of the causal query (per evaluated point: causal placement vs the non-causal interpolation of the SAME recorded stream at the same bin centres)

| arm | |dp| p50 / p95 / p99 / max (m) | points that change 0.2 m voxel | bin rotation p95 (mrad) | extrapolated span p50 / max (s) |
|---|---|---|---|---|
| onopt | 0.0004 +- 0.0001 / 0.0261 +- 0.0020 / 0.0689 +- 0.0081 / 0.975 +- 0.030 | 0.0417 +- 0.0021 | 5.04 +- 0.15 | 0.054 +- 0.000 / 0.158 +- 0.000 |
| onimu | 0.0000 +- 0.0000 / 0.0000 +- 0.0000 / 0.0000 +- 0.0000 / 0.097 +- 0.038 | 0.0001 +- 0.0001 | 0.00 +- 0.00 | 0.000 +- 0.000 / 0.033 +- 0.034 |
| onopthold | 0.0012 +- 0.0006 / 0.4099 +- 0.0145 / 0.8207 +- 0.2059 / 4.258 +- 0.124 | 0.3529 +- 0.0072 | 27.58 +- 0.48 | 0.054 +- 0.000 / 0.158 +- 0.000 |
| onopttx | 0.0002 +- 0.0001 / 0.0229 +- 0.0011 / 0.0531 +- 0.0093 / 0.586 +- 0.317 | 0.0373 +- 0.0019 | 4.55 +- 0.20 | 0.054 +- 0.000 / 0.158 +- 0.000 |

### Trajectories (ATE vs KITTI GT, `T_velo = Tr^-1 T_cam Tr`, single SE(3) Umeyama, GT interpolated to the estimate stamps -- src/eval_fastlivo2_ate.py; divergence = |dp| in the shared W frame, no alignment)

| trajectory | rep | poses | ATE RMSE m | mean | median | max | drift % | vs offline TUM: |dp| p50 / p95 / max m, within 0.2 m % | vs FL-alone (same rep): |dp| p50 / max m |
|---|---|---|---|---|---|---|---|---|---|
| offline TUM (rate 0.5, no other load, v0.5 baseline) | - | 1096 | 0.880 | 0.812 | 0.727 | 1.844 | 0.127 | 0 | - |
| FAST-LIVO2 alone, rate 1.0 (evo file) | 1 | 1096 | 0.907 | 0.822 | 0.712 | 2.234 | 0.131 | 0.832 / 1.325 / 1.439, 24 % | - |
| FAST-LIVO2 alone, rate 1.0 (evo file) | 2 | 1096 | 0.753 | 0.705 | 0.726 | 1.554 | 0.109 | 0.593 / 1.228 / 1.297, 17 % | - |
| FAST-LIVO2 alone, rate 1.0 (evo file) | 3 | 1096 | 0.835 | 0.754 | 0.692 | 2.037 | 0.121 | 0.188 / 1.044 / 1.070, 52 % | - |
| ON-opt run, live stream (= its evo file) | 1 | 1096 | 0.883 | 0.813 | 0.725 | 2.011 | 0.127 | 0.406 / 1.018 / 1.025, 37 % | 0.325 / 0.910 |
| ON-opt run, live stream (= its evo file) | 2 | 1096 | 1.056 | 0.933 | 0.832 | 2.651 | 0.153 | 0.730 / 1.428 / 1.471, 17 % | 1.235 / 2.170 |
| ON-opt run, live stream (= its evo file) | 3 | 1096 | 1.115 | 0.997 | 0.857 | 2.655 | 0.161 | 0.867 / 1.660 / 1.723, 21 % | 1.050 / 1.903 |
| ON-imu run, IMU-propagated stream | 1 | 8125 | 0.751 | 0.705 | 0.655 | 1.612 | 0.109 | 1.019 / 1.682 / 1.716, 16 % | - |
| ON-imu run, IMU-propagated stream | 2 | 8005 | 0.888 | 0.814 | 0.725 | 2.130 | 0.128 | 0.335 / 1.194 / 1.244, 25 % | - |
| ON-imu run, IMU-propagated stream | 3 | 7966 | 0.652 | 0.624 | 0.622 | 1.124 | 0.094 | 1.179 / 2.348 / 2.541, 33 % | - |
| ON-imu run, evo file (LIO updates) | 1 | 1096 | 0.751 | 0.703 | 0.649 | 1.597 | 0.108 | 1.019 / 1.682 / 1.716, 16 % | - |
| ON-imu run, evo file (LIO updates) | 2 | 1096 | 0.879 | 0.802 | 0.695 | 2.128 | 0.127 | 0.335 / 1.194 / 1.244, 25 % | - |
| ON-imu run, evo file (LIO updates) | 3 | 1096 | 0.660 | 0.632 | 0.629 | 1.111 | 0.095 | 1.179 / 2.348 / 2.541, 33 % | - |
| ON-opt-hold run, live stream | 1 | 1096 | 0.826 | 0.758 | 0.736 | 1.821 | 0.119 | 0.403 / 0.655 / 0.741, 21 % | - |
| ON-opt-hold run, live stream | 2 | 1096 | 0.916 | 0.802 | 0.707 | 2.619 | 0.132 | 0.929 / 1.865 / 1.947, 21 % | - |
| ON-opt-hold run, live stream | 3 | 1096 | 0.891 | 0.803 | 0.682 | 2.164 | 0.129 | 0.461 / 1.067 / 1.118, 32 % | - |
| ON-opt-tx run, live stream | 1 | 1095 | 0.931 | 0.869 | 0.762 | 1.907 | 0.134 | 0.944 / 1.578 / 1.637, 21 % | - |
| ON-opt-tx run, live stream | 2 | 1095 | 0.883 | 0.807 | 0.707 | 2.114 | 0.128 | 0.268 / 1.031 / 1.055, 36 % | - |
| ON-opt-tx run, live stream | 3 | 1095 | 1.026 | 0.934 | 0.876 | 2.418 | 0.148 | 0.648 / 1.353 / 1.608, 19 % | - |

Every rate-1.0 FAST-LIVO2 run's LIO trajectory (15 runs, alone and beside the node): ATE RMSE **0.888 +- 0.115 m** (min 0.660, max 1.115) against 0.880 offline at rate 0.5.  FAST-LIVO2 is not reproducible run to run at real-time rate: the same bag gives trajectories whose ATE differs by up to 0.45 m, and pairs of them diverge in the shared W frame by the amounts in the table.

## M6 -- RViz2 captures (DISPLAY=:0, latched /semantic_map re-published from the saved .npz after every timed run; class view = Intensity transformer on `class`, rainbow 0..15, legend out/v06/rviz_class_legend_rainbow0-15.png)

Views: top-down overview (focal 20,93,-3 / distance 360 / pitch 1.50) and the oblique street view (4.6,21.7,-0.5 / 55 / 0.42), each as class and RGB, for off_1 / iso_1 / onopt_1 / onimu_1 (montage_top_*/montage_obl_* = OFF | CAUSAL-ISO | ON-opt | ON-imu side by side); the 10 m zoom tile where CAUSAL-ISO and ON-opt disagree most, for off_1 / iso_1 / onopt_1 (montage_zoom_* = the causal-footprint triptych); and the 10 m tile where OFF and ON-opt disagree most (rviz_*_drift_class.png, montage_drift_class = the trajectory-divergence triptych).

Files (26): rviz_class_legend_rainbow0-15.png, rviz_iso_1_drift_class.png, rviz_iso_1_obl_class.png, rviz_iso_1_obl_rgb.png, rviz_iso_1_top_class.png, rviz_iso_1_top_rgb.png, rviz_iso_1_zoom_class.png, rviz_iso_1_zoom_rgb.png, rviz_off_1_drift_class.png, rviz_off_1_obl_class.png, rviz_off_1_obl_rgb.png, rviz_off_1_top_class.png, rviz_off_1_top_rgb.png, rviz_off_1_zoom_class.png, rviz_off_1_zoom_rgb.png, rviz_onimu_1_obl_class.png, rviz_onimu_1_obl_rgb.png, rviz_onimu_1_top_class.png, rviz_onimu_1_top_rgb.png, rviz_onopt_1_drift_class.png, rviz_onopt_1_obl_class.png, rviz_onopt_1_obl_rgb.png, rviz_onopt_1_top_class.png, rviz_onopt_1_top_rgb.png, rviz_onopt_1_zoom_class.png, rviz_onopt_1_zoom_rgb.png

**CAUSAL-ISO (A) vs ON-opt (B), same trajectory**: 2610458 voxels matched by key, 51316 differ (1.97 %) -- voxel keys live in the same frame, so every differing voxel is a semantic / coverage difference caused by the causal placement; the per-tile GT accuracy is valid for both.  Top 10 m tiles (centre; voxels; differing; accuracy of A / B on GT-labelled voxels; dominant changes):
- (14.9, -4.8, 0.4): 17886 voxels, 959 differ (5.4 %), acc A 89.3 / B 88.7 on 10347 GT voxels, vegetation->manmade x291, manmade->vegetation x159, large_vehicle->vegetation x49
- (24.3, -3.2, 0.3): 10065 voxels, 643 differ (6.4 %), acc A 87.1 / B 86.8 on 9742 GT voxels, car->manmade x170, manmade->car x131, vegetation->manmade x84
- (4.9, -24.2, 0.4): 8016 voxels, 491 differ (6.1 %), acc A 79.3 / B 79.1 on 7674 GT voxels, large_vehicle->car x132, car->large_vehicle x117, vegetation->manmade x47
- (13.4, -14.3, 1.1): 8059 voxels, 481 differ (6.0 %), acc A 91.0 / B 90.3 on 3929 GT voxels, vegetation->manmade x260, manmade->vegetation x138, sidewalk->road x12

**OFF (A) vs ON-opt (B), different trajectories**: 1463559 voxels matched by key, 149200 differ (10.19 %) -- matched by voxel key across two trajectories that drift apart -- only the match rate is meaningful; the 'differences' and B's per-tile accuracy (GT joined in A's frame) measure the trajectory divergence, not semantics.  This is what a VISIBLE online-vs-offline difference is made of (rviz_*_drift_class.png: the same tile in the OFF, CAUSAL-ISO and ON-opt maps).  Top 10 m tiles (centre; voxels; differing; accuracy of A / B on GT-labelled voxels; dominant changes):
- (-14.6, 5.1, -0.0): 10295 voxels, 2495 differ (24.2 %), acc A 97.1 / B 67.3 on 9950 GT voxels, car->road x1127, vegetation->manmade x220, manmade->vegetation x218
- (15.0, -5.1, 0.4): 14209 voxels, 2292 differ (16.1 %), acc A 90.4 / B 80.1 on 8334 GT voxels, manmade->vegetation x439, vegetation->manmade x390, manmade->sidewalk x364
- (-4.8, 4.8, -0.3): 9249 voxels, 2170 differ (23.5 %), acc A 96.8 / B 67.2 on 8925 GT voxels, car->road x996, sidewalk->terrain x263, manmade->terrain x228
- (-85.1, 155.1, -2.8): 4237 voxels, 2016 differ (47.6 %), acc A 99.9 / B 15.2 on 4190 GT voxels, road->car x1240, road->large_vehicle x449, car->large_vehicle x274

**ON-opt (A) vs ON-imu (B)**: 1320454 voxels matched by key, 185976 differ (14.08 %) -- two online runs, two pose sources, two trajectory draws.  Top 10 m tiles (centre; voxels; differing; accuracy of A / B on GT-labelled voxels; dominant changes):
- (14.9, -5.0, 0.4): 14759 voxels, 3106 differ (21.0 %), acc A 88.2 / B 77.5 on 8619 GT voxels, vegetation->manmade x598, manmade->vegetation x512, manmade->sidewalk x466
- (-85.7, 155.0, -3.7): 6664 voxels, 2138 differ (32.1 %), acc A 96.6 / B 45.8 on 6491 GT voxels, car->large_vehicle x721, road->car x627, road->large_vehicle x176
- (25.9, 174.8, -7.5): 5853 voxels, 2084 differ (35.6 %), acc A 88.9 / B 52.7 on 5532 GT voxels, manmade->vegetation x886, vegetation->manmade x287, terrain->manmade x275
- (-54.7, 4.5, 2.2): 6726 voxels, 2016 differ (30.0 %), acc A 96.4 / B 71.4 on 5783 GT voxels, road->sidewalk x449, vegetation->manmade x273, manmade->vegetation x226

## What the integration exposed (unknown before v0.6), in the order that matters

1. **FAST-LIVO2 is not reproducible at real-time rate.** 15 rate-1.0 runs of the same bag give ATE RMSE 0.888 +- 0.115 m (min 0.660, max 1.115; the offline rate-0.5 baseline, 0.880, is one draw from this spread), pairs of runs diverge by up to 1.9 m in the shared W frame, and the trajectory of the run beside the node is not systematically different from the run alone.  The map metric does NOT see it (r = 0.29 between a map's own-trajectory ATE and its mIoU-9 over 18 maps): the label-consistency reading is blind to global trajectory error in this range.  Consequence: 'the online trajectory's cost' is a draw, not a number, and a single online-vs-offline comparison is a lottery ticket; the 3-repetition paired design below is the minimum.
2. **The three-way decomposition (out-of-frustum mIoU-9, paired by repetition): causal-query restriction +0.07 +- 0.18 (ON-opt minus CAUSAL-ISO, same trajectory), online trajectory -0.22 +- 0.16 live / -0.16 +- 0.04 replay (CAUSAL-ISO minus OFF), total -0.15 +- 0.27 (ON-opt minus OFF).**  With constant-twist extrapolation the causal restriction is indistinguishable from zero at the label level (its geometric footprint: |dp| p95 2.6 cm, 4 % of points change voxel, +3 % voxels); the trajectory term is one to two run-to-run standard deviations and of the sign expected; the whole online cost is within the spread of a single arm.  Read against the hold ablation: HELD poses cost nothing at the node's own placement but 1.6-2.0 mIoU-9 when the map is read where the points should have been -- the metric must be read both ways or it lies.
3. **FAST-LIVO2's ROS 2 odometry carries no sensor time.** Every published pose / cloud is stamped `now()`; only the evo file uses `last_lio_update_time`.  An online consumer that de-skews cannot use the topic as shipped.  One ROS-glue line fixes it (LIVMapper.cpp:1417); the estimator is untouched and the streamed pose equals the evo file.
4. **The pose of a sweep exists only after the whole sweep plus ~30 ms of LIO, and it is stamped at the image instant, half-way through the sweep.** So at 10 Hz a causal consumer always has the first half of a sweep covered and never the second half: the in-sweep pose arrives 44 ms (p50) / 63 (p95) / 88 (max) after the sweep, stage B starts 64 ms (p50) after it, the newest usable sample sits at ~0.48 of the sweep, 100 % of sweeps extrapolate ~53 ms (max ~158 ms when stage B beats the pose, ~1 % of sweeps), and the sample that covers the sweep END is the NEXT sweep's update, 148 ms (p50) after arrival -- not affordable inside a 104 ms period.  `/LIVO2/imu_propagate` removes the extrapolation entirely (span 0, ~8 samples per sweep) at the price of being a forward propagation with correction jumps.
5. **/semantic_scan does not reach a subscriber under the transport v0.5 prescribed, and the fix breaks /semantic_map.** 3.2 MB samples at 10 Hz on plain `LARGE_DATA` deliver 65 % (OFF, mean of 3; per-run 59/81/56; 56-81 % over all 15 plain-transport runs) in bursts of losses up to 1-4.7 s, for BEST_EFFORT and RELIABLE alike, with writer depth 1 or 10, with or without the 15 MB map publishes (M4 smokes) -- so it is the transport's fragment/socket budget, not the QoS.  `LARGE_DATA?max_msg_size=5MB&sockets_size=20MB&non_blocking=false` delivers 99.7 % (ON-opt-tx, mean of 3; interval p99 ~150 ms) at +1.5 ms per publish -- but the RELIABLE 15 MB /semantic_map sample then blocks: only 50/49/30 of 69/69/70 published maps reach the subscriber and the delivered intervals stretch to 27-38 s (M4).  The two topics need different transports or the map needs to be sent in increments; v0.5 never saw any of this because nothing subscribed during its timed runs.
6. **The port's camera-parameter fetch is a 100 ms discovery race.** vikit's `getRemoteParam` gives the parameter-service client 100 ms; under `LARGE_DATA` fastlivo_mapping aborted 4/4 times at startup, 2/4 on the default transport.  The offline trajectory of v0.1-v0.5 came from a run that happened to win the race.  Fixed: 10 s.
7. **`/LIVO2/imu_propagate` is off by default** (`uav.imu_rate_odom: false`) and, when enabled, delivers ~71 Hz from a 4 ms wall timer (the 100 Hz IMU, ~29 % of its samples coalesced); it re-bases on every LIO update, and the map built on it is 6 % larger (2.86 M vs 2.71 M voxels for the same scene) and -0.26 +- 0.25 mIoU-9 below ON-opt -- the jumps cost more than the extrapolation they avoid.
8. **Start-up discovery loss.** `ros2 bag play` publishes immediately; the v0.5 runs lost their first ~10 sweeps to subscription matching (recv 1091/1101), and FAST-LIVO2 loses its first IMU second the same way (the no-`-d` smoke's first pose came 1.9 s after the offline one).  `-d 3` removes it; every v0.6 run received 1100-1101 of 1101 sweeps.
9. **`pkill -f` self-kill.** A shell whose own command line contains a pattern given to `pkill -f` kills itself; three ssh sessions died that way during setup.  The run scripts only ever pkill from a file whose cmdline does not contain the patterns.
10. **What did NOT happen: no resource competition on this 12-core box.** FAST-LIVO2 1.36 +- 0.02 cores alone vs 1.38 +- 0.03 beside the node (LIO p50 28.0 +- 0.5 vs 29.8 +- 1.6 ms, p95 42.5 +- 1.4 vs 46.0 +- 1.9, never above 104 ms); node 0.61 +- 0.01 -> 0.63 +- 0.01 cores, PTv3 58.1 +- 0.6 -> 58.5 +- 1.0 ms; system CPU 11 +- 0 % -> 23 +- 0 %; GPU util ~19 +- 0 %; zero back-pressure or staleness drops in any ON run; the node's period is 103.9 ms in every arm.  The online cost is in the pose chain, not in the machine.

## Files

- `out/v06/REPORT.md`, `out/v06/summary.json` (this)
- `out/v06/runs/`: per run `stats_*.json` (node), `flog_*.npz` (frame log), `stream_*.tum/npz` (received pose stream), `pused_*.npz` (causal bin poses), `map_*.npz` (node-written map), `probe_*.npz`, `res_*.csv`, `fl_evo_*.tum`, `analysis_*.json`, `replay_*.json`, `diag_*.npz` (causal geometry per sweep), `ate_*.json/png`, `trajdiff_*.json`
- `out/v06/zoom_*.json`, `out/v06/rviz_*.png`
- `logs/v06_*.log`, `logs/node_*.log`, `logs/fl_*.log`, `logs/replay_v06_*.log`, `logs/verify_v06.log`
