# Offline loop closure for the semantic map

This experiment adds a geometric backend to the frozen FAST-LIVO2 trajectory,
then replays exactly the archived v0.6 B0 seed-1 pass-1 predictions into a fresh
semantic voxel map. It does not retrain the segmentation model. The initial
scope is one paired KITTI sequence 07 feasibility run.

## Components and interfaces

- A small NumPy adapter implements the ring/sector height descriptor, ring-key
  retrieval and yaw-invariant cosine distance from [Scan Context](https://github.com/gisbi-kim/scancontext).
  This adapter is an independent implementation of the published representation,
  not the authors' C++ module. Candidate search is also limited to 20 m by the
  existing odometry, so this is not a kidnapped-robot/global-localization test.
- [Open3D](https://www.open3d.org/docs/release/tutorial/pipelines/icp_registration.html)
  supplies robust point-to-plane ICP against a historical submap. Descriptor,
  overlap, residual and correction-size checks must pass before adding a loop.
- [GTSAM](https://github.com/borglab/gtsam) supplies Pose3 odometry/loop factors,
  a first-pose anchor, Huber loop noise and Levenberg–Marquardt optimization.
- `src/sem_core.py` supplies the unchanged IMU–LiDAR extrinsic, trajectory
  interpolation and confidence-weighted semantic fusion. A streaming second
  pass scores the finished map without retaining 125 million point records.

The graph operates on LiDAR poses. Its world-frame corrections are interpolated
onto the original IMU TUM samples and applied on the **left**. The output keeps
the input timestamps; `T_W_L = T_W_I @ T_I_L` remains valid. Maps are rebuilt from
scratch because changing the published poses cannot move already fused voxels.
The graph adapter projects the rounded calibration rotation onto SO(3), which
GTSAM requires. The frozen decimal extrinsic stays unchanged in semantic replay.

## Frozen protocol and checks

All parameters are in `tools/loop_closure/protocol.json`, committed before the
first execution. The backend reads only raw scans, timestamps and odometry.
Official KITTI poses and SemanticKITTI labels are confined to the evaluators.
No threshold or loop is selected using ground-truth scores.

1. Synthetic tests check a nonzero extrinsic under a rotated world correction,
   ICP measurement direction, yaw retrieval, an odometry-only graph, and the
   real calibration's rounded rotation on a long synthetic trajectory.
2. A second odometry-only graph on the actual sequence must return the input
   poses within 1e-6 matrix-entry error.
3. Baseline replay must reproduce the archived B0 run's voxel count,
   out-of-frustum point count and scan/map mIoU (within 1e-8 percentage point).
4. Scan-level confusion matrices must remain identical in both arms.

ATE uses the **official KITTI odometry ground truth**, interpolated to the TUM
timestamps, with one global rigid alignment and no scale. The historical
`results/seq07_ate.json` instead uses SemanticKITTI reference poses; those two
numbers must not be compared as if they used the same reference.
The odometry `times.txt` differs from the raw image clock by up to 4.788 ms on
seq07. As in the original evaluator, association uses the raw image clock.
The auxiliary time-table check records that difference and rejects a deviation
of half a frame; a regression test covers both sub-frame jitter and a frame slip.

Held-out frames ending in 5 supply early/late static building/fence/pole/sign
surfaces for a revisit diagnostic. None enters the backend's keyframes, which
are multiples of 10. Symmetric nearest-neighbour overlap is measured in a fixed
25 m world ROI without fitting another transform. This is evidence within one
sequence, not an independent test sequence or a guarantee of better absolute
map accuracy.

Semantic map scores use the existing common-9 taxonomy and P3 raw-camera-frustum
definition, 0.20 m voxels, confidence gate 0.5 and 128-bin deskew. The historical
metric omits points with invalid timestamps inside partially supported sweeps;
`map_all_lookup` reproduces that denominator for parity, and `map_all_complete`
also reports those omitted points as abstentions. Both abstain-excluded and
abstain-as-wrong mIoU are saved. These scores measure semantic fusion under each
arm's own geometry; they are not an independent measure of global drift.

## Execution and preservation

The GitHub workflow `seq07-loop-closure.yml` restores inputs on an ephemeral CPU
runner, executes the tests and paired experiment, and saves checksums, loop
decisions, trajectories, complete semantic maps, figures and evaluation JSON
to a draft release. Failure diagnostics are preserved too: a draft release's
existence alone does not mean that all checks passed. `execution.json` links
the exact commit and workflow run. Raw KITTI data remains on the runner.

For an existing remote CPU environment:

```sh
python -m pip install -r tools/loop_closure/requirements.txt
python -m unittest tools.loop_closure.test_backend -v
python -m tools.loop_closure.restore --root /tmp/loop-inputs --out out/loop
python -m tools.loop_closure.backend --root /tmp/loop-inputs --out out/loop
python -m tools.loop_closure.replay --root /tmp/loop-inputs --out out/loop --arm baseline
python -m tools.loop_closure.replay --root /tmp/loop-inputs --out out/loop --arm corrected
python -m tools.loop_closure.evaluate --root /tmp/loop-inputs --out out/loop
```

Results are pending the first completed run. Online ROS integration and broader
sequence validation are outside this first feasibility experiment.
