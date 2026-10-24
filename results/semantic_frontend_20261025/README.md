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
The frozen [120-frame protocol](../../frontend/protocol.json) is being rerun in
[run 40125959560](https://github.com/KaiFeng-Frank/rgb-semantic-livo-mapping/actions/runs/40125959560).
Raw points and images stay on the ephemeral runner. All trajectories must be
frozen before reading pose GT; cached predictions cannot establish live inference latency.
