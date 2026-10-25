# Visual instance association: implementation and validation

The optional online path now produces session-scoped object IDs on current
semantic scans. PTv3 is retained. There is no new training, distillation,
instance-conditioned semantic fusion, or change to persistent map evidence.
This is an association baseline with measured identity failures, not a semantic
accuracy improvement or a leaderboard result.

[Interface and deployment](../../tools/visual_instances/README.md)
· [implementation PR](https://github.com/KaiFeng-Frank/rgb-semantic-livo-mapping/pull/12)

## Real observations with people and vehicles

The frozen YOLO11n-seg / BoT-SORT configuration runs on six previously inspected
nuScenes development scenes, 242 keyframes at 2 Hz. No ground-truth labels or
boxes enter detection or association; supplied ego poses align camera and LiDAR.
Object motion between the two acquisition times is not compensated.

| Diagnostic | Result | Interpretation |
|---|---:|---|
| Points assigned a visual instance | 14,746 / 8,402,272 | Front camera only; this is not full-view coverage |
| Supported thing-class points associated in camera frustum | 12,610 / 21,811 = 57.81% | Denominator includes occluded points; no claim of visible-surface recall |
| Visual class agrees with point GT among assigned, labeled points | 11,423 / 14,726 = 77.57% | Visual hypothesis precision, not PTv3 mIoU |
| Wrong identity in checkable adjacent predicted-track pairs | **22 / 180** | Majority box-supported identity changes despite the same predicted ID |
| ID switches in checkable adjacent GT-object pairs | **33 / 191** | Same GT object acquires another predicted identity |
| Predicted instance observations with fewer than 3 GT-box-supported points | 42 | Excluded from identity checks; not assumed correct |
| Detector + tracking + mask construction latency | median 24.55 / P95 34.38 / max 39.75 ms | Warm model, standalone diagnostic |
| Point association latency | median 2.13 / P95 2.36 / max 3.79 ms | Separate from inference and ROS transport |

Box + lidarseg identity is an evaluation proxy, not official panoptic/LSTQ ground
truth. Within-mask box purity is 97.54%, but omits unsupported/background points
and does not establish temporal identity correctness. The failures above are why
instance history does not overwrite current semantic predictions.

Exact counts, per-scene/per-class diagnostics and frame records are in
[`nuscenes/`](nuscenes/). The input index is the existing
[`index.json.gz`](../semantic_consistency_20261026/index.json.gz); its decompressed
SHA-256 matches `nuscenes/protocol.json`. Raw imagery, scans and weights are not
duplicated here.

## Actual ROS delivery on a dynamic scene

`real_ros.py` sends scene-0519's original camera and LiDAR through a separate
visual GPU worker, live PTv3, the semantic node and actual ROS subscribers.
The graph receives supplied ego poses; this is not a live-SLAM accuracy test.

- All **41/41** current-scan instance outputs arrived; **40** contain associated
  points, **4,826** points in total. The first frame lacks a causal pose bracket
  for the camera timestamp and abstains.
- Subscriber-observed semantic labels exactly match the corresponding raw scan.
- LiDAR publication to instance-cloud reception: median **79.15 ms**, P95
  **112.16 ms**, max **127.01 ms** at the original **2 Hz** cadence.
- Camera/scan offset bound is explicitly 60 ms for this asynchronous dataset.
  Ego motion uses only received pose brackets. Object motion is not compensated.

See [`real_ros/report.json`](real_ros/report.json) and frame-level records.
The report's periodic status snapshot is one frame behind; the shutdown
`vision/final.json` and `semantics/final.json` record all 41 processed frames.
The recorded subscriber counts are the delivery authority.

## Integrated runtime, including adverse runs

All HKU runs replay the same 127.696 s sensor payload at original speed:
1,277 camera frames, 1,277 LiDAR messages and 25,871 IMU messages. Semantic
keyframes remain approximately **1 Hz**, not 10 Hz. Each run delivers 122 semantic
outputs. The machine is the local RTX 5070 Ti Laptop GPU, with 32 logical CPUs.

CPU isolation gives vision physical cores 14/15 and SMT siblings 30/31; other
pipeline processes inherit CPUs 0–13,16–29. Visual ROS runs at nice 10 and its
GPU worker at nice 15, with OpenCV/torch worker threads limited to one. These CPU
indices are specific to this host. Both isolated control and active runs use the
same restricted non-vision CPU set.

| Run | Vision | CPU isolation | Corrected pose median / P95 / P99 / max (ms) | Missing initialized poses |
|---|---|---|---|---:|
| `control` | off | no | 37.07 / 50.30 / 57.13 / 71.59 | 0 |
| `active` | on | no | 52.25 / 85.54 / 109.67 / 165.66 | 0 |
| `active-isolated` | on | yes | 37.86 / 51.64 / 60.50 / 93.77 | 0 |
| `control-final` | off | yes | 37.50 / 51.57 / 109.06 / **879.22** | **2** |
| `active-final` | on | yes | 37.86 / 51.64 / 57.60 / 70.35 | 0 |
| `control-repeat` | off | yes | 37.90 / 51.91 / 62.72 / 78.98 | 0 |

The unisolated active run materially worsens pose latency. Both isolated active
runs process **1,277/1,277 images**, deliver **1,274/1,274 initialized poses** and
publish **122/122 instance clouds**, with no visual superseded/stale/invalid
frames. This supports the tested deployment configuration, not a hard deadline
guarantee or a claim that asynchronous GPU work is free.

The isolated control stalls around 56 s. Two image stamps at 55.799 and 55.900 s
have no pose; the frontend reports 762 ms camera delivery and subsequent
IMU/LiDAR synchronization warnings. The source replay's maximum scheduling delay
is 64 ms, which alone does not explain the stall. Root cause is unresolved. The
[numbered frontend excerpt](hku/control-final/frontend-stall-excerpt.txt), failed
run and all timestamp pairs are retained. The single additional same-configuration
control (`control-repeat`) delivers all initialized poses, with P95 51.91 ms and
max 78.98 ms. This does not explain or erase the earlier stall. There are too few
repeats to infer that the active path is faster from its slightly lower numbers.

HKU contains only one image with a supported detection, outside the selected
semantic keyframes. **Zero points receive an instance ID in HKU.** It validates
runtime and empty-detection handling, not moving-object association quality. The
real nuScenes ROS replay supplies the nonzero delivery check.

[`runtime-summary.json`](runtime-summary.json) is recomputed from compressed
subscriber logs by `python3 results/visual_instances_20261026/summarize.py`.
All runs have the same payload hash. `hku/*/assessment.json` retains broader
runtime counters; model metadata records actual worker affinity and weight hash.

## Checks, provenance and reproduction

The semantic-memory/association suite passed **209 tests + 128 subtests**, with
one ROS-only skip. The geometry suite passed **79 tests**, with two ROS-only
skips. A separate cancellation test verifies that a real owned subprocess is
reaped during startup cancellation. Actual ROS fixture checks cover ID delivery,
measured depth occlusion, future-frame rejection, session reset, unchanged raw
semantic labels and exact map rebuild parity.

The final detector, geometry and evaluation hashes are recorded in
`nuscenes/protocol.json` and match commit `2aa2c25`. Subsequent commits
`5ef4f7d` and `52412cb` add worker cancellation cleanup and its regression test;
they do not change association or detector output. Pre-freeze development runs
are retained for resource-contention evidence; exact per-run source hashes were
not recorded for those early runs. `implementation-sha256.json` records the
final reviewed runtime sources. Configuration/weight metadata is archived per
run. The official visual weight SHA-256 is
`55ed65c56c91713d23e8402371c6c49a6fd84f257f7dce452e8d70e41dcbe152`.

Reproduction commands, using the documented optional environment:

```bash
python -m unittest tools.visual_instances.test_association tools.visual_instances.test_lifecycle -v
python -m tools.visual_instances.evaluate --index /path/to/index.json \
  --weights /path/to/yolo11n-seg.pt --out /path/to/new-diagnostic
# Source ROS and provide its NumPy/OpenCV/SciPy dependencies first:
python3 -m tools.visual_instances.validate_ros --out /path/to/new-ros-fixture
python3 -m tools.visual_instances.real_ros --index /path/to/index.json \
  --runtime /path/to/semantic-runtime --weights /path/to/yolo11n-seg.pt \
  --out /path/to/new-real-ros
```

Dataset path resolution follows `tools/semantic_memory/nuscenes_pilot_data.py`;
restore the existing nuScenes manifest/index and data at their configured roots.
For HKU, use `tools.loop_closure.realtime_replay` with `--backend --semantics`,
the existing semantic runtime, and optional `--visual-instances --visual-weights`.
The exact local launch recipes are preserved alongside these results. GPU checks
were local; no released remote instance was used.
