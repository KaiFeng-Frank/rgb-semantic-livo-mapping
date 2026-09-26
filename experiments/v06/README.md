# v0.6 experiment source snapshot

This directory archives the v0.6 code, captured on 2026-09-26: online integration, the
evaluation-unit analyses, and the training queue with the seq-09 hold-out. It accompanies
[online integration](../../docs/v06_online_integration.md),
[the evaluation unit](../../docs/v06_evaluation_unit.md) and
[the training protocol](../../docs/v06_training_protocol.md). Files under `source/` keep
their original relative paths; each one's original path and SHA-256, and the published
SHA-256 where the two differ, are recorded in
[`results/v06/snapshot_manifest.json`](../../results/v06/snapshot_manifest.json).

Edits against the originals: the host-specific data root is rewritten to `/data/`, and
two scripts carry a marked redaction of a reference to an unrelated job that shared the
host (`opt/prep_seq09.sh`, `opt/train_queue_v06_resume.sh`). Nothing else is changed.

## Online integration

| source path | role |
|---|---|
| `src/semantic_map_node.py` | the v0.6 node: `--pose-topic` live pose buffer, causal query with `cv`/`hold` extrapolation, staleness gate, optional wait; frame, stream and used-pose records |
| `src/_pre_v06_backup/semantic_map_node.py` | the v0.5 node, loaded beside the v0.6 node by the verifier |
| [`patches/fast_livo2_lio_sensor_stamp.patch`](patches/fast_livo2_lio_sensor_stamp.patch) | FAST-LIVO2 ROS 2 port: `/aft_mapped_to_init` stamped with the LIO update's sensor time instead of `now()`; `patch -p1` in the FAST-LIVO2 checkout |
| [`patches/vikit_remote_param_wait.patch`](patches/vikit_remote_param_wait.patch) | rpg_vikit: remote-parameter wait 100 ms → 10 s; `patch -p1` in the rpg_vikit checkout |
| `opt/run_v06.sh` | one run of one arm (`fl`, `off`, `onopt`, `onimu`, `iso`; `EXTRAP=hold` and the transport string select the two ablations) |
| `opt/v06_all.sh` | the run matrix: three interleaved repetitions of all seven arms, one run at a time |
| `opt/v06_post.sh` | replays, trajectory divergence, zoom tiles and captures, after every timed run |
| `opt/res_sampler_v06.py`, `opt/topic_probe_v06.py` | per-process CPU/RSS and GPU sampler; subscriber-side delivery probe |
| `opt/replay_v06.py` | the v0.5 map scorer plus causal-placement scoring and the geometric footprint |
| `opt/verify_v06.py` | default-path identity: v0.5 and v0.6 node in one process on identical cached predictions |
| `opt/fl_start_test.sh` | the startup-race measurement (4/4 aborts under `LARGE_DATA`, 2/4 on the default transport) |
| `opt/capture_v06.sh` | RViz2 captures from the latched map |
| `tools/v06_analyze.py`, `tools/v06_traj_diff.py`, `tools/v06_online_report.py` | per-run measurement extraction, trajectory divergence, and the report generator |

Modules v0.6 did not touch (`sem_core.py`, `ptv3_client.py`, `ptv3_loader_verified.py`,
`ptv3_worker.py`) are listed with their hashes under `unchanged_by_v06` in the manifest;
the v0.5 snapshot and the repository's `src/` hold them.

## Evaluation unit

| source path | role |
|---|---|
| `tools/map_eval.py` | the v0.5 map scorer generalised to any sequence, with the point- and cell-weighted anatomy readouts |
| `tools/residual_anatomy.py`, `tools/anatomy_report.py`, `opt/run_anatomy.sh` | point-weighted residual anatomy (preregistered) |
| `tools/residual_anatomy_vox.py`, `tools/anatomy_voxel_report.py`, `opt/run_anatomy_vox.sh` | per-cell anatomy (post-hoc, preregistered before its numbers) |
| `tools/sparse_diag.py`, `tools/sparse_diag_report.py`, `opt/run_sparse_diag.sh` | sparse-cell diagnosis on B0 and zero-shot |
| `tools/sparse_diag_confirm.py`, `opt/run_sparse_confirm.sh` | amendment A2's confirmation on five unseen arms |
| `tools/propagate_check.py`, `opt/run_propagate.sh` | map-propagated pseudo-label premise check, with its control run and source guard |

The two anatomy tools are `opt/replay_v05.py` verbatim plus appended readout blocks;
`tools/sparse_diag.py` is `tools/map_eval.py` plus insertions only; `tools/propagate_check.py`
is new, built on `map_eval`'s trajectory, de-skew and voxel code. Each run script logs its
preregistration's SHA-256 before the first replay. The `map_eval` control and its checks
are in [`results/v06/mapeval/`](../../results/v06/mapeval/).

## Training with seq 09 held out

| source path | role |
|---|---|
| `src/split_v2.py` | the v0.6 split as subclasses of the v0.4 datasets, with the static and path-level guards |
| `src/Pointcept_v151/configs/semantic_kitti/arm_*_v2_*.py` | seven configurations: B0 ×3 seeds, B0 without KL ×3 seeds, R′ without KL ×1 |
| `tools/train_distil_v2.py` | the single-GPU training entry point with the v0.6 split registered |
| `opt/prep_seq09.sh` | the seq-09 rig: 100 Hz IMU stream, bags, FAST-LIVO2 trajectory, then hand-off to the queue |
| `opt/train_queue_v06.sh` | the serial queue with its split self-check |
| `opt/train_queue_v06_resume.sh` | the resumable restart after the host out-of-memory event |
| `opt/score_v06.sh`, `tools/v06_report.py` | extraction, 32 prediction caches, map-level replay, and the four-question report |

The v2 configurations inherit the v0.4 arm configurations archived in
[`experiments/v04/source/`](../v04/source/src/Pointcept_v151/configs/semantic_kitti/).

Results: [v0.6 results](../../docs/v06_results.md). Two notes on running these files:

- The scoring ran `tools/map_eval.py` and `opt/score_v06.sh` with two edits made after this
  snapshot: the voxel-map capacity constants doubled, and replay parallelism went from 4 to
  2. The as-run hashes are in the manifest's `updates` record.
- `opt/train_queue_v06_resume.sh` resumes through Pointcept v1.5.1's `CheckpointLoader`. On
  one GPU that loader loaded the frozen anchor into the student instead of restoring it
  ([deviation 1](../../docs/v06_results.md#protocol-deviations-and-disclosures)).

## Environment and replay

A research source snapshot, not an installer. The experiment root was `/data/livo_sem`
(after sanitising): Pointcept v1.5.1 under `src/Pointcept_v151`, the ROS 2 workspace with
the FAST-LIVO2 port under `ros2_ws/`, bags under `bags/`, prediction caches under `out/`.
GPU jobs ran in the `ptv3` conda environment (Python 3.10, PyTorch 2.5.1+cu124, NumPy
2.2.6, spconv-cu124 2.3.8); CPU replays and reports in a separate environment; ROS nodes
under the system Python of ROS 2 Jazzy. Checkpoints, caches, bags, per-run records, maps
and logs are not bundled. Read [`CRITICAL_CONSTRAINTS.md`](../../CRITICAL_CONSTRAINTS.md)
O1–O10 before running the online path.

The original entry points, once the layout exists:

```bash
bash opt/v06_all.sh           # the 21 timed runs, one at a time
bash opt/v06_post.sh          # replays, divergence, zoom tiles, captures
python3 tools/v06_online_report.py
source /opt/ros/jazzy/setup.bash && python3 opt/verify_v06.py --frames 40
```

Verify the published files against the manifest:

```bash
python3 - <<'PY'
import hashlib, json, pathlib
for e in json.load(open("results/v06/snapshot_manifest.json"))["files"]:
    d = pathlib.Path(e["published"]).read_bytes()
    assert len(d) == e["bytes"] and hashlib.sha256(d).hexdigest() == e["sha256"], e["published"]
print("ok")
PY
```
