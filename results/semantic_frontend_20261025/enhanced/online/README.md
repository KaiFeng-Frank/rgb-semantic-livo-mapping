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

The full LiDAR/IMU/image plus live-inference replay **failed online admission**.
It uses 120 development frames at 1x, three paired repetitions and the actual
FAST-LIVO2 executable. All six runs receive 120/120 scans, preserve original point
bytes, and produce the same 114 finite pose timestamps. The raw tmpfs directory
was removed after completion. One duplicated/nonmonotonic raw IMU sample was
removed by the existing converter, leaving 1,286 IMU samples at 99.92 Hz.

| Repetition | Geometry pose latency median / p95, ms | Online pose latency median / p95, ms | Timely online semantic scans | Online latency growth, ms |
|---|---:|---:|---:|---:|
| 1 | 109.96 / 121.41 | 1458.00 / 3318.90 | 89/100 | 3070.77 |
| 2 | 110.07 / 116.50 | 1486.06 / 3282.11 | 87/100 | 3013.14 |
| 3 | 109.62 / 114.99 | 1503.32 / 3360.06 | 88/100 | 3079.52 |

These timings start when the complete scan needed by the update reaches the
observer and end at corresponding Odometry receipt. Upstream Odometry uses wall
timestamps, so correspondence to sensor-time TUM rows is accepted only after
counts and published positions match. Both controls have a fixed 80 ms input
wait; geometry here is not an undelayed deployment baseline. The first twenty
sensor scans are excluded from steady-state timing, after a separately declared
six-scan inference pre-roll before estimator startup. Observer overhead is included.

The [stage timings](full-lio/stage-profiles.json) show roughly 123 ms median in
the online LIO update versus 13 ms in geometry; VIO remains about 3–4 ms. This
supports an estimator throughput bottleneck. Boundary extraction still consumes
about 15 ms despite its observation switch being off. Exact attribution among
the new semantic components needs the separate six-arm timing evidence; the
categorical field's repeated neighbourhood queries are a code-level candidate,
not an independently measured share of the cost.

[Full assessment](full-lio/assessment.json), [interpretation](full-lio/interpretation.json),
[frozen protocol](full-lio/protocol.json), and the
[complete compressed logs/configs/trajectories](full-lio/online-lio-evidence.tar.gz)
preserve every run. This result rejects current online deployment readiness;
it does not establish that semantic tight coupling is ineffective.

[Assessment](assessment.json), [model/runtime checks](runtime-ready.json), and
[full-size indexed-cloud contracts](reliable-contracts.log) accompany all raw logs.
