# Live semantic input validation

The first actual-scan Cyclone DDS replay still failed with best-effort input:
geometry received 16/26 scans and online received 22/26. The received online
measurement scans all completed inference before the 80 ms deadline, but missing
scans make that run inadmissible. Its complete evidence is retained in
[best-effort-failed](best-effort-failed/result.json). An earlier launcher with an
incorrect config path was stopped before inference and is also retained.

An explicit reliable-input option was then added, without changing the estimator,
model or inference deadline. The publisher must offer reliable QoS as well.
Default best-effort sensor compatibility is still available; it does not guarantee
complete large-cloud replay on this setup.

The [three repeated reliable checks](reliable-three-repeats/result.json) pass:

| Repetition | Geometry scans | Online scans | Timely semantic measurement scans | Online emission median / p95, ms |
|---|---:|---:|---:|---:|
| 1 | 26/26 | 26/26 | 20/20 | 86.18 / 86.92 |
| 2 | 26/26 | 26/26 | 20/20 | 85.31 / 86.50 |
| 3 | 26/26 | 26/26 | 20/20 | 85.37 / 86.72 |

Every received scan retains its original point bytes, timestamp and order. The
80 ms deadline applies to inference completion after bridge receipt. Emission
includes field assembly and ROS serialization, so it occurs later than 80 ms;
these figures are not sensor-to-pose or driving-display latency. Six warmup scans
and twenty measured scans are used per run, with arm order reversed in repetition
two. Original scans remain in RAM only.

The full LiDAR/IMU/image plus live-inference replay is separately frozen in
`frontend/online_lio_validation.py` and is pending. It uses a task-owned tmpfs
directory, 120 development frames at 1x, three paired repetitions and the actual
FAST-LIVO2 executable. Bridge-only success does not establish end-to-end success.

[Assessment](assessment.json), [model/runtime checks](runtime-ready.json), and
[full-size indexed-cloud contracts](reliable-contracts.log) accompany all raw logs.
