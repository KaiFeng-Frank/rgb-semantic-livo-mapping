# Semantic observations inside FAST-LIVO2

This implementation adds semantic observations to FAST-LIVO2's frontend error-state
filter. It does not estimate another ICP pose and feed it into a pose graph. The
original 19-state prior, cross covariance, and LiDAR/visual sequential update remain
the estimator. Dataset improvement and online throughput require separate evidence.

The separate [global backend](../tools/loop_closure/backend.py) is an offline
LiDAR-geometric pose graph: Pose3 nodes, an initial prior, odometry edges and ICP
loop edges. Its odometry can inherit visual information from the LIVO frontend,
but the graph has no independent visual residuals, IMU preintegration, velocity
or bias states. Semantic pose refinement weights geometric registration before
calling this same graph. This is not a validated joint global LIVO backend.
The author's separate [Global-LVBA](https://github.com/xuankuzcr/Global-LVBA) offers
offline LiDAR/visual refinement; [OKVIS2-X](https://github.com/ethz-mrl/OKVIS2-X)
is a candidate for joint visual/inertial/LiDAR optimization. Neither has been
integrated or validated in this repository.

The enhanced profile is `frontend/enhanced.yaml`. It adds three independently
switchable paths: ego-compensated motion decisions in geometric residuals and map
admission, image-space motion masks in the actual photometric update, and a
pose-dependent categorical likelihood in the same 19-state estimator. The old
failed sequence09 result evaluates only the earlier association/boundary variant;
it does not evaluate these additions or establish that semantic coupling is
ineffective.

All arms also initialize/recompute the upstream `blind_sqr` from `blind`; an
uninitialized shared preprocessing threshold was found by the integration fixture.
This is a new common baseline. The old failed experiment is preserved unchanged.

The enhanced profile disables the old boundary residual to avoid counting the
same class transitions twice. Every new switch defaults to false when omitted,
preserving the earlier controls. New real-data admission is separately frozen in
`enhanced-diagnostic.json`: 96 development frames, five arms, one repeat, no GT
scoring, no training, no parameter search and no automatic large experiment.

The frozen replay passed on the temporary GitHub CPU runner: all five arms produce
the same 91 pose timestamps. The combined arm uses categorical observations on
89 updates, detects moving points on 81, invalidates four old map nodes, and
generates moving-region image masks on 72 visual updates. Boundary constraint
rows remain zero in every arm. These are mechanism activation checks, not
accuracy or online throughput results. [Complete logs, hashes, settings and
trajectories](../results/semantic_frontend_20261025/enhanced/README.md) are preserved.

Full live-inference 1x replay subsequently fails realtime admission on the local
5070 Ti laptop: 87–89% timely semantics and 3.28–3.36 s pose-delivery p95 over three
paired repetitions. Input coverage and 114-pose timestamps match in all arms, but
the semantic LIO update takes roughly 123 ms median and the queue grows. See
[all online runs and timing scope](../results/semantic_frontend_20261025/enhanced/online/README.md).

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
* Boundary extraction uses the deskewed cloud before geometry voxelization;
  geometric association retains its original downsampled input. Locally planar
  boundaries between native ground classes supply an additional
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

Use Ubuntu 24.04 / ROS 2 Jazzy, Eigen, PCL, OpenCV, glog, Ceres, fmt, and colcon.
The validated large-cloud transport profile also needs `ros-jazzy-rmw-cyclonedds-cpp`. The source
and dependency revisions are locked in `upstream-files.json`, `vikit-files.json`,
and `dependencies.json`. Bootstrap retrieves only the source files needed to build,
with Git blob hash verification; it does not download upstream videos or datasets.

```bash
python3 frontend/bootstrap.py --workspace "$HOME/semantic_livo_ws_clean"
bash frontend/build.sh "$HOME/semantic_livo_ws_clean"
source /opt/ros/jazzy/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
source "$HOME/semantic_livo_ws_clean/install/setup.bash"
ctest --test-dir "$HOME/semantic_livo_ws_clean/build/fast_livo" --output-on-failure
python3 -m unittest frontend.tests.test_input_bridge -v
python3 -m unittest frontend.tests.test_ros_bridge -v
python3 -m unittest frontend.tests.test_replay_identity -v
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
  --input-reliability reliable \
  --input /velodyne_points --output /semantic_frontend/points \
  --log /tmp/semantic-input.jsonl
```

This complete-replay profile requires a reliable input publisher, such as the
tested rosbag replay. For best-effort-only sensors, select
`--input-reliability best_effort` (the compatibility default). Actual indexed
scans still lost messages with best-effort input on this setup even under
Cyclone DDS; reliable input passed three paired repetitions. See
[the complete online transport evidence](../results/semantic_frontend_20261025/enhanced/online/README.md).

Run `python3 frontend/parameter_blackboard.py --ros-args --params-file
config/camera_kitti_cam2.yaml` and FAST-LIVO2 with the original KITTI
calibration, overriding `common.lid_topic:=/semantic_frontend/points` and
`semantic.enabled:=true`. `semantic.association` and `semantic.boundaries` provide
separate ablations; `semantic.enabled:=false` is the unchanged observation path. For the enhanced profile,
append `--params-file frontend/enhanced.yaml` after the KITTI parameter file.
Do not feed `/semantic_scan` from the existing mapping node: that topic has already
been transformed/deskewed and lacks the original scan timing contract.

`[ Semantic ]` logs report frame, extracted boundaries, accepted constraints and
history size. Bridge logs record sensor stamp, arrival, completion, emission and
timeout outcome. Cached-label replay is only a mechanism control; it cannot stand
in for measured inference latency or an online result.

The exactly-once rule applies to scans received by the bridge. Its input uses ROS
sensor-data best-effort QoS; network loss before receipt is outside that buffer.
Allow DDS discovery to settle before replay (the validation uses `ros2 bag play
--delay 3`) and compare received timestamps, rather than assuming delivery from
the publisher's count.

## Remaining validation and scope

The protocol is frozen in `protocol.json`. No accuracy claim follows from the
synthetic fixture. The first short real-data replay completed, but accepted zero
boundary rows: see [the failed admission](../results/semantic_frontend_20261025/README.md).
The revised pre-voxel source is checked with `--diagnostic`: geometry, full B0 and
full ZS once each, with no GT fetch or scoring. The old window is now development
data. A new frozen frontend holdout and measured online run are required before
claiming driving accuracy or online performance.

`holdout09.json` freezes the first 240 frames of sequence 09, five ablations and
three repetitions, with algorithm/configuration hashes checked before recovery.
This sequence was not used to develop this frontend, but earlier segmentation
evaluations did use it. All 15 trajectories must be frozen before reading pose GT.
The admission rule requires paired ATE improvements beyond the repeat ranges,
non-regressing mean 1-second translation/rotation RPE and active boundary support.
The `holdout09` workflow input runs this bounded validation without training.

For local GPU validation, `check_gpu_runtime.py` compares CUDA reduction, sparse
convolution and variable-length attention against separate reference operations,
then strictly loads the archived B0 checkpoint. `online_admission.py` measures the
existing worker and the actual ROS input bridge with a fixed 80 ms deadline. Its
six warm-up and twenty measurement scans are read from the official KITTI archive
into RAM; only hashes, timings, class histograms and transport logs are saved.
Set the four `PTV3_*` paths above, then run:

```bash
"$PTV3_PYTHON" -m frontend.check_gpu_runtime --out /tmp/semantic-runtime-ready.json
source /opt/ros/jazzy/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
/usr/bin/python3 -m frontend.online_admission --out /tmp/semantic-online-admission
```

The local bridge gate requires at least 95% of measured scans to carry semantic
predictions completed within the deadline, complete ordered timestamp coverage,
and unchanged original point bytes. It measures input admission rather than the
combined live LIO pipeline; passing it alone does not establish online localization.

Cached replay carries `raw_point_index` from the original KITTI bag conversion in
every arm, including geometry. Annotation verifies a complete index permutation
and exact xyz/intensity equality before applying archived labels. Existing sensor
fields and timestamps stay byte-identical; the converter's default output is
unchanged unless `--point-index-field` is requested. Recomputing a permutation is
kept only as a diagnostic, never as the point identity used to attach semantics.

The enhanced implementation distinguishes class candidates from measured motion.
Class-conditioned clusters are compared in the world frame, with one-to-one
association, shape/ambiguity gates, pose uncertainty, repeated movement evidence
and a stationary confirmation period. Unknown candidates retain 0.25 geometric
weight and are withheld from map insertion; confirmed motion has zero weight;
confirmed stationary candidates retain full weight. Old associated map surfaces
and visual landmarks are invalidated when an object starts moving. These are
bounded centroid tracks, not a validated general multi-object tracker. Confidence
thresholds and motion uncertainty floors are engineering choices, not calibrated
probabilities.

Image masks are projections of the current 3-D motion regions, with a freshness
limit and nearer observed static-depth protection. They are **not dense 2-D
semantic segmentation** and may miss unobserved regions. Full patch footprints,
including pyramid/gradient support, gate both photometric update implementations,
new visual landmarks and reference-patch updates. Unknown regions are downweighted;
confirmed moving regions contribute no photometric rows. An all-masked update
preserves the entry state and does not shrink covariance. Expired masks are cleared.

The categorical field uses smoothed historical class-vote frequencies, not
reconstructed full network posteriors. Its negative-log class likelihood changes
with pose, so its spatial derivative contributes directly to the IEKF Jacobian.
A uniform class field contributes no pose information. History remains frozen
throughout an update, requires two older observations, and has point/history and
information caps. Tests cover finite-difference Jacobians, input-order invariance,
and actual state/covariance/velocity response with boundary constraints disabled.
The optional `adaptive_boundary_noise` switch adds measured line scatter and a
confidence penalty to the old boundary noise floor; it is not used in the enhanced
profile and is not claimed as calibrated uncertainty.

Non-rigid loop corrections still require clearing/rebuilding all geometry,
semantic histories and motion tracks. The older boundary map's rigid rebase helper
alone does not rebase the new motion/likelihood state; live loop integration is not
enabled by this profile. Passing synthetic integration tests is not evidence of
real-data accuracy benefit or online throughput.

The FAST-LIVO2 patch remains subject to its upstream GPL-2.0 license and notices.
