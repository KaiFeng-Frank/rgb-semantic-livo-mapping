# Semantic frontend verification

The frontend passes actual-estimator and **boundary activation** checks, but the
subsequent frozen sequence 09 accuracy admission **failed**. Across three repeats,
full B0 mean ATE is 0.463356 m versus geometry's 0.429595 m (7.86% higher), and mean
rotation RPE also regresses. All 15 runs preserve the same 235 timestamps;
boundary rows are active on 233/235 B0 updates. See the
[complete five-arm holdout results and admission checks](holdout09/README.md).
No localization benefit or larger experiment is admitted by these results.

The initial synthetic evidence and source hashes are in
[`local-verification.json`](local-verification.json); revised runtime evidence is
in [pre-voxel-diagnostic.json](pre-voxel-diagnostic.json).

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
filter. Only 0–8 candidates survived per frame. The follow-up uses
boundary extraction from the deskewed cloud before that filter, preserving the
geometry input and existing acceptance gates. This window is now development data;
any revised accuracy claim needs a separately frozen, unseen window. Cached labels
do not establish live inference latency.

## Revised source: boundary activation confirmed

[Run 40127490465](https://github.com/KaiFeng-Frank/rgb-semantic-livo-mapping/actions/runs/40127490465)
uses commit `8ffa14f62d9258ceb21a341404709557f6c7e150`. All build, integration,
ROS transport and startup tests pass, including a new fixture retaining dense
semantic input alongside 0.5 m geometric voxelization. The three development
replays each retain the same 115 timestamps. Pose GT was not fetched or scored.

| Arm | Updates accepting boundary rows | Rows summed over updates | Maximum history points | Median LIO time, ms | LIO p95, ms |
|---|---:|---:|---:|---:|---:|
| geometry | 0/115 | 0 | 0 | 21.71 | 29.87 |
| full_B0 | 112/115 | 2554 | 568 | 50.07 | 67.24 |
| full_ZS | 106/115 | 1611 | 393 | 50.54 | 70.97 |

B0 accepts boundary rows on 97.4% of updates and ZS on 92.2%. Those rows enter the
same state/covariance solve; the synthetic integration test separately verifies
pose and cross-correlated velocity response. Row counts refer to the final IEKF
iteration and are not counts of statistically independent measurements.

Boundary extraction alone has median cost 19.44 ms (B0) and 19.98 ms (ZS). These
are instrumented CPU-stage times on the GitHub runner, with cached predictions,
not sensor-to-output latency or an online model benchmark. LIO time excludes
segmentation inference and does not establish realtime operation.

The full 249 kB evidence archive is `remote-diagnostic-40127490465.tar.gz`; its
SHA-256 is in [pre-voxel-diagnostic.json](pre-voxel-diagnostic.json). The executed
[diagnostic protocol](executed-diagnostic-protocol.json), all three trajectories,
input identity audits, configurations and logs are preserved. Raw scans, images
and model archives remain outside Git source.

This completes the first frontend implementation and activation check. An unseen,
separately frozen paired evaluation is still needed to measure accuracy benefit;
the live GPU runtime also needs latency/admission validation. Dynamic-object
tracking, visual semantic masking and non-rigid loop-map rebuilding remain outside
this increment. No larger experiment is launched by this diagnostic.
