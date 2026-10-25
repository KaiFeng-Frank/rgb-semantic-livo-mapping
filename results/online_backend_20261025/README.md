# Online backend engineering checks — 2026-10-25

The geometric loop backend now accepts sensor-time LIVO keyframes, incrementally
optimizes a Pose3 graph in a separate process, and publishes corrected current
poses, keyframe trajectory and geometric map. The shared semantic-map proposal
and original failed semantic frontend are preserved separately.

## Results

| Check | Result | What it establishes |
|---|---|---|
| C++/ROS build | 5 packages built | Optional export patch compiles with the pinned frontend |
| Numerical/lifecycle contracts | 16 tests passed | Frames, no-loop identity, actual descriptor/ICP closure, map movement, limits, gap uncertainty, worker shutdown/failure |
| Archived seq07 graph | 110 nodes, one previously accepted loop | Incremental solver agrees with the old batch graph |
| Actual frontend + ROS synthetic sensors | 65/65 poses, seven keyframe clouds | Sensor-time pairing and continuous corrected output |
| Worker deliberately paused | 12/12 poses still published, nine keyframes rejected at full queue | Pose callback does not wait for backend optimization |
| Combined launcher | Three owned child processes started and stopped | Launch and shutdown integration |

The archived graph's maximum position difference from a fresh batch solve is
**0.00005194 m**. Incremental and batch graph errors are 2.480523590 and
2.480523578. Its measured update median/max are 0.051/1.988 ms on this machine;
these exclude retrieval, ICP, point-cloud preparation, ROS and map publication.
This reuses the existing loop edge at its original keyframe. It does not retest
online loop detection or establish a new ATE improvement.

The final ROS fixture uses 70 synthetic scans at nominal 10 Hz with IMU and
`bgr8` images, through the actual compiled frontend, in isolated ROS domain 83.
After initialization, all 65 exported sensor-time poses reach `/backend/pose` in
order. No-loop position difference is zero; quaternion difference is below
4.5e-15 (allowing equivalent quaternion signs). All seven clouds have exact pose
timestamp matches and finite, capped body-frame XYZ. This synthetic scene is an
interface fixture, not a visual tracking or driving-accuracy evaluation.

The subsequent overload probe pauses only its own worker and publishes 12
synthetic keyframe/pose pairs faster than the normal source rate. All current
poses continue, while nine keyframes are explicitly dropped at the bounded
queue. Its artificially advancing sensor clock explains the reported correction
age; neither that age nor the pause duration is a field-latency measurement.
Adapter odometry callback p95 was 0.580 ms in this fixture. It is not a measure
of total frontend-to-display latency or added frontend CPU cost.

These checks pass for **engineering integration**, not sustained online driving
admission. There is no new dataset experiment, model training, GT tuning or raw
dataset download. A live LiDAR/visual/IMU joint global graph, dense semantic-map
correction and long-duration deployment validation remain outside this version.

## Development failures and fixes

The first integration attempt exposed a publisher initialization error: the
upstream entry point passes a null local node reference to publisher setup, so
the new publisher must use the mapper's owned `this->node`. This was fixed.

A subsequent native-`bgr8` fixture crashed in the visual stage. Code inspection
identified the upstream `getImageFromMsg` ownership defect: copying only the
`cv::Mat` from a temporary `toCvShare` wrapper loses the ROS message owner. The
buffer can be freed while the queued image is still in use. Debugger runs could
pass before that fix and are **not** treated as proof of correctness. The new
optional patch uses `toCvCopy`, so the queued matrix owns its pixels. The normal
non-debug fixture then passed and the final run also passed after integration
refinements. Prior logs are retained in `development-diagnostics.tar.gz`.

The patch was reapplied to an isolated copy of the preserved frontend and its
resulting source compared byte-for-byte with the built workspace. The original
`fastlivo2-semantic.patch` remains unchanged. The optional patch hash and all
online source hashes are recorded in `runtime.json`.

## Files and reproduction

- `tests.log`, `build.log`: final contract/build results.
- `archived_graph.json`, `keyframe_times_ns.json`: solver comparison and the
  preserved official image timestamp subset; existing odometry/loop inputs are
  linked by hashes, not copied as new experimental observations.
- `ros_assessment.json`, `ros-evidence.tar.gz`: final ROS assessment, all frontend
  and worker logs, configuration, and per-keyframe processing records.
- `development-diagnostics.tar.gz`: earlier failures/debugger runs.
- `launcher.json`: combined startup/shutdown check.
- `runtime.json`, `SHA256SUMS`: dependency, source and artifact identity.

Local build: `/home/frank/semantic_livo_backend_ws`. Worker Python:
`/home/frank/semantic_backend_runtime/venv/bin/python`. The worker venv supplies
NumPy 1.26.4 and reads existing GTSAM 4.2 / Open3D 0.19 / SciPy 1.15.3 libraries
from the existing base environment without modifying that environment. ROS uses
system Python 3.12. The standalone pinned installation recipe and exact run
commands are in the [online backend guide](../../docs/online_loop_backend.md).

No test processes remain running after these checks. No new raw data or weights
are included in these files.
