# Online geometric loop backend

The existing Scan Context-style retrieval, robust Open3D registration and Pose3
graph now have a streaming ROS 2 path. GTSAM iSAM2 adds odometry/loop factors as
keyframes arrive. The frontend stays in its continuous local frame; a separate
process publishes global correction, corrected poses and a rebuilt keyframe map.

This is an online **geometric pose graph driven by LIVO odometry**. It does not
add independent visual residuals, IMU preintegration, velocity/bias graph states,
or a joint fixed-lag LIVO smoother. The shared semantic-map design is retained in
the [design review](../results/semantic_frontend_20261025/enhanced/README.md),
and the failed semantic frontend remains disabled in the online launch profile.

## Data and frame contract

The optional [frontend patch](../frontend/patches/fastlivo2-online-backend.patch)
exports two topics with the **same sensor timestamp**, derived from the upstream
double-valued update time (not wall time):

| Topic | Type | Coordinates / content |
|---|---|---|
| `/backend/input/odometry` | `nav_msgs/Odometry` | `T_camera_init_backend_imu`, each completed update |
| `/backend/input/cloud` | `sensor_msgs/PointCloud2` | deskewed, downsampled XYZ in `backend_imu`, at most 1 Hz / 12,000 points by default |
| `/backend/pose` | `geometry_msgs/PoseStamped` | current global body pose, for each received ordered odometry message |
| `/backend/path` | `nav_msgs/Path` | optimized historical keyframe poses |
| `/backend/map` | `sensor_msgs/PointCloud2` | bounded geometric display cloud rebuilt from local keyframe clouds |
| `/backend/status` | `std_msgs/String` JSON | processing age, correction revision, queue drops, worker failures and capacity |

In LIVO mode the export runs **after VIO at the same update time**. LiDAR-only
mode exports after LIO. Existing legacy topics and their timestamp conventions
are unchanged. The input covariance is the frontend covariance, not a graph
posterior; the corrected output uses PoseStamped to avoid advertising a false
posterior covariance.

Clouds are paired by exact timestamp and checked frame names. No nearest-time
or arrival-order pairing is used. Point coordinates already include `T_I_L`, so
the graph and ICP measurements both use IMU body frames. Registration returns
`T_historical_current`, matching `Between(historical, current)`.

The live transform is:

```
T_map_odom = T_map_keyframe * inverse(T_odom_keyframe)
T_map_body(t) = T_map_odom * T_odom_body(t)
TF: map -> camera_init -> backend_imu
```

Only corrections already received for a keyframe at or before `t` are applied.
Raw frontend poses define all odometry factors; corrected poses only initialize
new graph variables. Nothing is injected into the IEKF as another pose sensor.
Before any loop, correction is exactly identity. Past live pose messages cannot
be rewritten; `/backend/path` and `/backend/map` publish the corrected history.

Old geometric clouds remain in their own keyframe frames. A loop therefore moves
each retained cloud with its optimized pose. Applying one TF to an already fused
dense semantic map would not fix its internal drift: `/semantic_map` is **not**
automatically corrected by this module. Semantic fusion/retraction remains a
separate integration task; `/backend/map` currently has XYZ only.

## Bounded work and failure behavior

The frontend reuses existing downsampled geometry, caps serialization, and uses
best-effort depth-two publishers. It does no retrieval or graph optimization.
The adapter keeps at most 32 unmatched poses and 32 unmatched clouds. Evictions
of ordinary high-rate poses without a keyframe cloud are expected, not a count
of missing keyframes.

The worker is a separate, lower-priority process with single-thread native BLAS
and OpenMP settings. Its pending queue holds two keyframes. Full queues reject
new keyframes without waiting; queued items older than three wall-clock seconds
are discarded. The next accepted node connects to the previous accepted node
using original odometry. Edge variance scales with the elapsed number of nominal
keyframe intervals, rather than assigning a long gap the confidence of one step.
This is an engineering noise model, not calibrated frontend cross-covariance.

Defaults in [online.json](../tools/loop_closure/online.json) cap a session at
1,000 keyframes, 12,000 points per keyframe, two ICP attempts per keyframe, and
200,000 display points. Map materialization caps points **before** concatenation.
At capacity, graph growth stops with explicit `capacity_reached` status; current
poses continue with the last valid correction. This is not indefinite mapping or
automatic marginalization. Start a fresh session after sensor clock/frame resets;
duplicate/backwards odometry is rejected rather than mixing coordinate systems.

A response timeout or worker failure is reported, the worker is terminated, and
the adapter continues current-pose publication using the last valid correction.
It does not silently restart a disconnected global graph. Shutdown only stops
the processes belonging to this launch. Isolation and bounds prevent waiting on
the worker; they do not prove zero CPU contention or sustained realtime behavior.

Retrieval and registration retain the prior offline geometric gates and odometry
candidate radius. There is no kidnapped-robot/global-relocalization guarantee.
The detector uses only already arrived keyframes, including historical submaps.
No GT, labels or evaluation scores are read by the online backend.

The patch also fixes a ROS image ownership bug uncovered by the integration
fixture: `toCvShare(...)->image` alone loses the owner of native `bgr8` pixels
when the callback's ROS message dies. Queued images now use `toCvCopy`, preserving
pixels for delayed visual processing. This optional workspace is separate from
the frozen semantic-front-end experiment; its old source hashes are unchanged.

## Build and run

Use the existing ROS 2 Jazzy build dependencies from the
[frontend guide](../frontend/README.md). The ROS adapter uses system Python,
NumPy, `transforms3d`, `sensor_msgs_py`, and `tf2_ros`; no GTSAM import occurs in
the ROS process. The worker uses Python 3.10 or 3.11 with the existing pinned
`tools/loop_closure/requirements.txt` (GTSAM 4.2, Open3D CPU 0.19, NumPy 1.26).
Do not use NumPy 2 with the pinned GTSAM wheel, and do not resolve a venv Python
symlink to its base executable.

```bash
python3 frontend/bootstrap.py --workspace "$HOME/semantic_livo_backend_ws" --online-backend
bash frontend/build.sh "$HOME/semantic_livo_backend_ws"

# In a separate Python 3.10/3.11 venv:
python -m pip install -r tools/loop_closure/requirements.txt

# From the repository; OUTPUT_DIR must be new.
bash tools/loop_closure/run_online.sh \
  "$HOME/semantic_livo_backend_ws" /absolute/path/to/worker-venv/bin/python \
  /absolute/path/to/new-run
```

The launch defaults to this repository's KITTI calibration and topics. Its
optional fourth and fifth arguments replace the frontend and camera YAMLs.
It starts the camera parameter service, the backend and FAST-LIVO2; provide the
LiDAR, IMU and image streams in the same ROS domain. The included backend profile
disables the unadmitted semantic frontend. No data downloads or model training
are triggered by launching it.

To run the adapter separately (for inspection or different topic/frame names):

```bash
source /opt/ros/jazzy/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
/usr/bin/python3 -m tools.loop_closure.online_node \
  --worker-python /absolute/path/to/worker-venv/bin/python \
  --log /absolute/path/to/new-backend.jsonl
```

Use `--config` for JSON overrides to online/backend parameters. Match frontend
export point/interval limits if changed. Frame overrides identify existing
frames; they are not implicit extrinsic transformations.

## Checks and evidence

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /path/to/worker-venv/bin/python \
  -m unittest tools.loop_closure.test_backend tools.loop_closure.test_online -v
/path/to/worker-venv/bin/python -m tools.loop_closure.validate_archived_graph \
  --out /tmp/archived-graph-check.json
source /opt/ros/jazzy/setup.bash
source "$HOME/semantic_livo_backend_ws/install/setup.bash"
ROS_DOMAIN_ID=83 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp OMP_NUM_THREADS=2 \
  /usr/bin/python3 -m tools.loop_closure.validate_online_ros \
  --workspace "$HOME/semantic_livo_backend_ws" \
  --worker-python /path/to/worker-venv/bin/python --out /tmp/new-ros-check
```

Tests cover incremental/no-loop and loop-direction math, actual descriptor/ICP
closure, historical map deformation, timestamp/frame contracts, bounded state,
gap uncertainty and worker lifecycle. The archived seq07 check replays previously
accepted edges at their original keyframes and compares incremental optimization
with batch optimization. It is not new online loop-detection or accuracy evidence.
The ROS fixture uses synthetic LiDAR/IMU/images through the actual frontend,
then pauses its own worker to check that pose publication continues under overload.
See [preserved results](../results/online_backend_20261025/README.md).

Algorithm/API references: [GTSAM iSAM2](https://borglab.github.io/gtsam/isam2/),
[Scan Context authors](https://github.com/gisbi-kim/scancontext),
[Open3D ICP](https://www.open3d.org/docs/release/tutorial/pipelines/icp_registration.html).
