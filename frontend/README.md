# Semantic observations inside FAST-LIVO2

This implementation adds semantic observations to FAST-LIVO2's frontend error-state
filter. It does not estimate another ICP pose and feed it into a pose graph. The
original 19-state prior, cross covariance, and LiDAR/visual sequential update remain
the estimator. Dataset improvement and online throughput require separate evidence.

## Implementation

* A scan-preserving input bridge adds native nuScenes `semantic_label` (uint16) and
  `semantic_confidence` (float32). Coordinates, intensity, ring, sensor timestamps,
  and point order are preserved. A bounded queue/deadline emits a scan exactly once;
  late predictions are discarded. `--fixed-delay` in both `online` and `geometry`
  modes gives a common emission deadline for timing controls. Inference is on the raw body-frame scan and does not require an estimated
  map or trajectory.
* The pinned Velodyne non-feature preprocessing path carries label/confidence through
  time sorting, scan splitting, and deskew in otherwise unused PCL normal channels.
  The ABI stays unchanged. Discrete label evidence is reaggregated after the original
  PCL VoxelGrid; class IDs are never interpreted as averaged numerical measurements.
  Other sensor types and feature extraction are rejected when semantics is enabled.
* Historical surface class evidence conditions association likelihood and geometric
  measurement information. A confident conflict triggers a bounded neighbouring
  surface search. One update contributes at most one normalized semantic vote per
  octree node. Unknown semantics are neutral. Evidence is committed after estimation.
* Locally planar boundaries between native ground classes supply an additional
  **tangential metric residual**, using a line fitted to matching historical class-pair
  boundaries. The residual and right-SO(3) Jacobian enter `H^T R^-1 H` and `H^T R^-1 r`
  in `VoxelMapManager::StateEstimation`, before the original state/covariance update.
  The ground-normal component remains the original LiDAR plane observation.
* Boundary history is frozen throughout the current update, then committed once.
  At least two older updates must support a match. Confidence, planarity, adjacency,
  class pair, surface-normal agreement, line shape, distance and robust residual gates
  apply. Correlated boundary information is capped by class pair and frame. History
  and boundary count have hard memory bounds.

An unknown/uniform semantic field supplies no boundary information. A straight
boundary does not constrain translation along itself. All noise/gating constants
are initial engineering choices, not calibrated uncertainty or tuned benchmark
results. The current implementation is conditional filtering against a frozen map,
not a full joint covariance over robot pose and every semantic map variable.

## Build and contracts

Use Ubuntu 24.04 / ROS 2 Jazzy, Eigen, PCL, OpenCV, glog, fmt, and colcon. The source
and dependency revisions are locked in `upstream-files.json`, `vikit-files.json`,
and `dependencies.json`. Bootstrap retrieves only the source files needed to build,
with Git blob hash verification; it does not download upstream videos or datasets.

```bash
python3 frontend/bootstrap.py --workspace "$HOME/semantic_livo_ws_clean"
bash frontend/build.sh "$HOME/semantic_livo_ws_clean"
source /opt/ros/jazzy/setup.bash
source "$HOME/semantic_livo_ws_clean/install/setup.bash"
ctest --test-dir "$HOME/semantic_livo_ws_clean/build/fast_livo" --output-on-failure
python3 -m unittest frontend.tests.test_input_bridge -v
cmake -S frontend -B /tmp/semantic-contracts -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/semantic-contracts -j2
ctest --test-dir /tmp/semantic-contracts --output-on-failure
```

The x86 build uses portable SSE code generation and 16-byte Eigen alignment throughout
to match the system PCL library. Mixing native AVX allocation rules with the distro's prebuilt PCL can
otherwise crash on destruction after a successful filter. This build setting must
also be used for the geometry control.

## Input and launch

The live path uses the existing verified Pointcept v1.5.1 worker. Set
`PTV3_PYTHON`, `PTV3_POINTCEPT`, `PTV3_CONFIG`, and `PTV3_CHECKPOINT` to the verified
runtime and weights. Strict state-dict loading and native 16-class order remain
mandatory; a 19-class checkpoint is not interchangeable.

```bash
python3 -m frontend.input_bridge --mode online --wait-ms 80 \
  --input /velodyne_points --output /semantic_frontend/points \
  --log /tmp/semantic-input.jsonl
```

Run `python3 frontend/parameter_blackboard.py --ros-args --params-file
config/camera_kitti_cam2.yaml` and FAST-LIVO2 with the original KITTI
calibration, overriding `common.lid_topic:=/semantic_frontend/points` and
`semantic.enabled:=true`. `semantic.association` and `semantic.boundaries` provide
separate ablations; `semantic.enabled:=false` is the unchanged observation path.
Do not feed `/semantic_scan` from the existing mapping node: that topic has already
been transformed/deskewed and lacks the original scan timing contract.

`[ Semantic ]` logs report frame, extracted boundaries, accepted constraints and
history size. Bridge logs record sensor stamp, arrival, completion, emission and
timeout outcome. Cached-label replay is only a mechanism control; it cannot stand
in for measured inference latency or an online result.

## Remaining validation and scope

The protocol is frozen in `protocol.json`. No accuracy claim follows from the
synthetic fixture. It must be followed by a short paired real-data check, and a
measured online run before calling the pipeline validated in driving scenes.

Actual object-motion tracking, transient object maps, semantic masking of visual
patches, and rebuilding maps after non-rigid loop correction are not implemented
in this first frontend increment. Movable categories are not treated as permanent
semantic boundary landmarks. A rigid map-frame rebase is supported; a non-rigid
correction requires clearing/rebuilding the boundary history. None of those pending
capabilities is inferred from the presence of semantic labels or a changed ATE.

The FAST-LIVO2 patch remains subject to its upstream GPL-2.0 license and notices.
