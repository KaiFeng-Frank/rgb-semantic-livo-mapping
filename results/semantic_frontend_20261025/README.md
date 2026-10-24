# Semantic frontend verification

These are local **synthetic integration and startup** results for the first
semantic frontend increment. They are not driving accuracy or online throughput
results. The implementation commit and source hashes are in
[`local-verification.json`](local-verification.json).

The fixture calls the actual pinned FAST-LIVO2 `StateEstimation` method. Its flat
ground cannot geometrically observe lateral translation. Adding historical
road/sidewalk boundary observations to the same filter gives:

| Quantity | Geometry | Semantic boundary |
|---|---:|---:|
| Lateral position error, m | 0.200000 | 0.057159 |
| Lateral position variance, m² | 0.100000 | 0.028580 |
| Accepted boundary observations | 0 | 25 |

The semantic innovation also updates lateral velocity to -0.028568 m/s through
the prior cross covariance. This confirms state coupling; it is not a measured
vehicle velocity improvement. With no admitted semantic observations, the pose
and covariance match the geometry path within 1e-12 in this fixture.

Other checks cover actual alternative surface association, raw point identity,
PCL voxel tag aggregation without coordinate changes, the right-SO(3) Jacobian,
historical-only evidence, information caps, bounded history, and six input-field
and timing tests. Geometry and semantic ROS startup both load the expected KITTI
camera calibration successfully.

The subsequent [isolated build check](clean-build-verification.json) disables the
CMake package registry and forces all three packages to use the pinned Sophus.
It passes the actual estimator test and both ROS startup modes. A live ROS bridge
test also verifies five received scans, preserved point bytes and timestamps, and
one geometry fallback emission per scan after the configured deadline.

The remote build, actual estimator test, input tests and both startup modes passed
in [run 40124920386](https://github.com/KaiFeng-Frank/rgb-semantic-livo-mapping/actions/runs/40124920386).
Its first data preparation was stopped by a point-order mismatch before any
trajectory replay or scoring. The complete small evidence archive and
[`ci-attempts.json`](ci-attempts.json) preserve that result.

Cached replay now carries original point IDs from bag conversion and checks
exact xyz/intensity correspondence before attaching predictions. Three added
tests cover MCAP transport, unchanged sensor bytes and invalid/duplicate IDs.
The frozen [120-frame protocol](../../frontend/protocol.json) completed in
[run 40125959560](https://github.com/KaiFeng-Frank/rgb-semantic-livo-mapping/actions/runs/40125959560).
Raw points and images stayed on the ephemeral runner. All 15 trajectories have the
same 115 timestamps and were hash-frozen before reading pose GT. All source point
IDs passed exact xyz/intensity checks; the diagnostic recomputed orders also agreed
in this run. The prior preparation failure's exact cause remains unestablished.

## Real-data admission: failed boundary activation

**Every run accepted zero historical semantic boundary constraints.** The build and
execution succeeded, but the mechanism gate failed. Changes in error therefore
cannot be attributed to the boundary residual. No longer experiment is admitted.

| Arm | Mean ATE, m (min–max over 3 runs) | Mean 1 s translation RPE, m | Boundary constraints |
|---|---:|---:|---:|
| geometry | 0.074106 (0.073545–0.074490) | 0.094358 | 0 |
| association_B0 | 0.073180 (0.072615–0.073656) | 0.093129 | 0 |
| boundary_B0 | 0.074496 (0.073440–0.076388) | 0.094350 | 0 |
| full_B0 | 0.074326 (0.073403–0.076050) | 0.092839 | 0 |
| full_ZS | 0.071848 (0.071550–0.072177) | 0.093204 | 0 |

Boundary-only's ATE range is 2.947 mm even though its boundary rows were all zero.
Full ZS has 2.258 mm lower mean ATE than geometry on this window, but this cannot
support a boundary-tight-coupling claim. See every repetition in [scores.json](scores.json)
and the explicit decision in [admission.json](admission.json). The complete compressed
evidence is `remote-smoke-40125959560.tar.gz` (815 kB, SHA-256 in `admission.json`).
It contains trajectories, configurations, logs and input hashes, not raw scans,
images or model archives.

The first implementation extracted semantic edges after the geometry's 0.5 m voxel
filter. Only 0–8 candidates survived per frame. The next diagnostic will inspect
boundary extraction from the deskewed cloud before that filter, preserving the
geometry input and existing acceptance gates. This window is now development data;
any revised accuracy claim needs a separately frozen, unseen window. Cached labels
do not establish live inference latency.
