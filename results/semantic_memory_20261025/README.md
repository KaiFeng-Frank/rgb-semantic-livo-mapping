# Causal semantic memory: engineering checks, 2026-10-25

The first implementation and opt-in shadow path pass the checks below. **There is
no real-data fusion training result, segmentation gain, event-recovery result or
sustained driving realtime admission.** Existing B0 weights are unchanged.

## Implementation and numerical contracts

The [implementation guide](../../tools/semantic_memory/README.md) describes the
bounded raw-observation ledger, causal local attention, loss-improvement gate,
hard single-frame fallback at inference, visibility-based withdrawal and training
split rules. [Source hashes](source-manifest.json) identify the final code and
frozen backbone; the pre-existing base commit is recorded separately.

[Unit-test output](unit-tests.log): 155 passed, one ROS-specific test skipped in
the PTv3 interpreter, plus 128 parameter subtests. The separate system-Python ROS
geometry command passed both checks (padded/endian point layout and quaternion
transform). The skipped test was therefore exercised in the proper interpreter.

Checks cover raw-only writes despite changed fused predictions, strict causal
ordering, epoch reset, capacity/age bounds, occlusion versus missing returns versus
free-space contradiction, no-history identity, exact inference rejection fallback,
gate gradients based on loss improvement, and frozen-feature training. Additional
guards cover train/validation/test separation, ontology and provenance mismatches,
mutated caches, synthetic deployment refusal, worker deadlines, mixed synchronous
and asynchronous requests, and cleanup with outstanding shared-memory views.

Two initially failing numerical regressions exposed finite features overflowing
float16 and finite float64 logits overflowing float32. Inputs are now checked in
their actual storage/computation dtypes before history is changed. Review also
identified synthetic-flag propagation, execution-configuration compatibility,
unbounded reads and stale publication; these were fixed before the final run.

## Fixed real scan on the local GPU

[Full result and per-call measurements](runtime/result.json),
[baseline log](runtime/baseline-worker.log),
[shadow log](runtime/shadow-worker.log).

Hardware: NVIDIA GeForce RTX 5070 Ti Laptop GPU; Torch
`2.11.0.dev20260201+cu128`. The frozen strict B0 model has 488 state tensors.
One existing nuScenes scan (34,720 points, 21,920 voxels) was repeated with identity
transforms and synthetic 0.1 s timestamps. Its content hash is in the JSON.
Intensities were divided by 255 and the worker scale was 1.0. No scan or model was
copied into this repository.

| Measured path | Median | p95 | Maximum |
|---|---:|---:|---:|
| Memory module alone, retained raw features | 6.09 ms | 6.17 ms | 6.20 ms |
| Baseline client round trip | 40.45 ms | 42.80 ms | 43.67 ms |
| Shadow client round trip | 47.00 ms | 56.28 ms | 58.17 ms |

Each row uses 20 measurements after five warm-up calls. Extra peak allocated
Torch GPU memory during the memory-only path was 122,955,264 bytes (117.3 MiB).
This includes temporary allocations; it is not the permanent ledger size or
total device usage. History was eligible for 21,882 of 21,920 query voxels.

The captured features reproduced **their own forward's** head logits exactly.
Shadow output also equalled that same forward's baseline exactly. Separate
backbone calls are not asserted bitwise identical. Baseline and shadow timings
were measured sequentially, not randomized/interleaved; laptop power/thermal
variation and this single scan limit comparison. A 5 ms memory-overhead budget
would **not pass** this run. These numbers do not establish driving throughput.

The [initial runtime check](initial-runtime/result.json) exposed a shared-memory
`BufferError` at worker cleanup, preserved in its
[log](initial-runtime/baseline-worker.log). Exported buffer references are now
released before closing. The final baseline and shadow logs contain no such
cleanup error or leaked-shared-memory warning. Initial logs predate the final
source manifest and are retained as diagnostics.

## Actual ROS transport

[ROS result](ros/result.json), [node log](ros/node.log),
[adapter records](ros/node/adapter.jsonl), [worker log](ros/node/worker.log).

One isolated Jazzy/Cyclone DDS run on domain 65 passed:

- Four accepted 34,720-point messages retained exact source nanosecond headers,
  frame IDs and XYZI point ordering. Labels were uint16 native16 IDs; confidence
  was finite float32 in [0,1]. All four outputs arrived before the adapter's 300 ms
  receipt-to-publication limit.
- A cloud/odometry frame mismatch and duplicate sensor timestamp were rejected
  without publication. A subsequent valid scan still succeeded.
- History retrieval activated, the ledger stayed within its bound, and the node
  exited cleanly without forced termination.

The fixture sends one pair at a time; it does **not** stress overload, prove
sustained 10 Hz delivery or evaluate moving scenes. Its identity odometry and
timestamps are synthetic, and passing the deskew declaration does not validate
deskewing. It leaves raw source intensity unchanged and uses the adapter's 0.2
scale, so it is a wire/geometry test rather than a semantic comparison to the
separate normalized-intensity timing fixture. Original command paths in the JSON
identify the run before its small log directory was relocated for archiving.

## Synthetic training only

[Training provenance and epoch metrics](synthetic-training/metrics.json) and
[held-out generated-fixture evaluation](synthetic-training/test-evaluation.json).

The CLI generated four tiny frames each for train/validation/test, trained only
the fusion network for two epochs (four optimizer steps), wrote a checkpoint and
loaded it for evaluation. `scope=synthetic`, `trained=False`, `deployable=False`;
the checkpoint cannot activate the worker's fused output. No real dataset label
was used. This short fixture has zero repaired and zero introduced errors after
the inference gate; it verifies the training/serialization/evaluation chain and
is not evidence of semantic improvement.

Run commands, using the configured PTv3 interpreter:

```bash
$PTV3_PYTHON -m pytest -q tools/semantic_memory/test_memory.py \
  tools/semantic_memory/test_training.py tools/semantic_memory/test_interfaces.py
source /opt/ros/jazzy/setup.bash
/usr/bin/python3 -m unittest tools.semantic_memory.test_interfaces.GeometryTests -v
$PTV3_PYTHON -m tools.semantic_memory.train smoke --out /path/to/new-smoke
$PTV3_PYTHON -m tools.semantic_memory.check_runtime \
  --scan /path/to/existing-nuscenes-scan.pcd.bin --out /path/to/new-check --repetitions 20
/usr/bin/python3 -m tools.semantic_memory.check_ros --out /path/to/new-ros-check
```

Next scientific admission needs suitable chronological training and held-out data,
point-level and event-level comparison, and a sustained load test. SemanticKITTI
training annotations were not found locally. Existing nuScenes annotations need
a separate scene-split/ontology/geometry adapter; they were not substituted for
KITTI training data in this implementation.
