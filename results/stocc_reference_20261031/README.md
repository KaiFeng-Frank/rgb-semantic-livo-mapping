# ST-Occ official pretrained validation — completed

**The released ST-Occ R50 model has completed all 150 validation scenes / 6,019
frames. Accuracy and temporal variability closely reproduce the published
reference: 42.1361% mIoU and 8.6770% mSTCV.** The independent CPU audit verifies
all saved prediction hashes, complete scene coverage and the official mIoU.
No training, architecture adaptation or validation-driven parameter tuning was
performed. This is a validated official-system reference for the next research
step; it does not establish a memory gain in our PTv3 pipeline.

Completed: **2026-10-31 06:35:18 +08:00**; independent audit finished at 06:36:09.

| Metric | Authors' published R50 | This run | Interpretation |
|---|---:|---:|---|
| Occupancy mIoU, %, higher is better | 42.13 | 42.1361 | Numerically close; official evaluator rounds to 42.14 |
| mSTCV, %, lower is better | 8.68 | 8.6770 | Same value at the authors' two-decimal reporting precision |
| Validation coverage | Full validation | 150 scenes / 6,019 frames | Every scene is complete and ordered |

Reference: [official model zoo at the pinned source revision](https://github.com/matthew-leng/ST-Occ/tree/1633f62e2e6677a5fa474905977acfeca4e7819e).
The small numerical differences are not claimed as improvements over the
released model. The source is pinned at
`1633f62e2e6677a5fa474905977acfeca4e7819e`; the official checkpoint SHA-256 is
`68491f55a2208fad884e3b664d0e34b8311688376380f7722e30cac6bde5ffa0`.
Strict loading matched all keys for **68,794,628 parameters**.

## Accuracy, temporal metric and independent verification

The mIoU is computed from one global camera-masked 18×18 confusion matrix,
averaging the 17 non-free classes. It is not the mean of per-scene mIoUs.
The [independent audit summary](full-validation-workers8/audit-summary.json)
recomputes it from exported uint8 voxel predictions without calling the official
evaluator, and agrees with every rounded official class score. It also checks
all 6,019 prediction digests, token/index mappings and scene boundaries.

mSTCV is the **official implementation's** result and was not independently
recomputed. It measures aligned-voxel classification variability with GT-change
correction and the official GT-speed exclusion. It is not instance identity,
re-ID, dynamic-object recovery or a point-semantic segmentation score.

| Class | IoU, % |
|---|---:|
| others | 14.3627 |
| barrier | 49.6063 |
| bicycle | 27.7692 |
| bus | 46.2767 |
| car | 52.5747 |
| construction_vehicle | 26.8749 |
| motorcycle | 29.7933 |
| pedestrian | 29.8064 |
| traffic_cone | 31.3531 |
| trailer | 35.4113 |
| truck | 39.0305 |
| driveable_surface | 84.2687 |
| other_flat | 47.8113 |
| sidewalk | 56.1131 |
| terrain | 59.8510 |
| manmade | 45.2841 |
| vegetation | 40.1271 |

The free-space IoU is 90.6114%, reported separately and excluded from mIoU.

## Measured runtime

Single remote RTX 4090, batch size one, official FP16 configuration, eight CPU
loader workers and a 75% CUDA allocator cap. No other training or GPU process
was active at launch; the GPU is idle after completion. The local GPU was not
used. TASeg remains stopped following the user's earlier shutdown; its epoch-33
checkpoint is preserved and was not automatically resumed.

| Measurement | Result |
|---|---:|
| CUDA-synchronized model call, p50 | 147.06 ms |
| CUDA-synchronized model call, p95 | 162.82 ms |
| CUDA-synchronized model call, mean | 149.73 ms |
| First cold model call / maximum | 3213.66 ms |
| Data-loader wait, p50 / p95 | 13.72 / 33.87 ms |
| Inference, prediction export and final official evaluation | 1187.71 s |
| Throughput over that entire measured phase | 5.068 frames/s |
| PyTorch peak allocated / reserved GPU memory | 2.493 / 4.510 GiB |

Model-call timings include official temporal-metric bookkeeping and exclude
loading. Whole-phase throughput includes compressed output writing and the final
CPU metric pass; it excludes environment/model construction and the subsequent
50-second independent audit. Allocated/reserved values are PyTorch statistics,
not total driver-reported device usage. These measurements do not establish a
5 ms memory plugin, a 10 Hz pipeline or an online SLAM deployment.

## Correctness controls and preserved partial runs

- All **36,114 camera images** and **6,019 occupancy labels** passed saved hash,
  token, shape and file-integrity checks. Metadata contains 150 complete scenes
  with contiguous indices and unique tokens.
- A complete 40-frame scene-0003 smoke run passed before full validation. Its
  29.0665% mIoU / 6.6510% mSTCV are subset diagnostics, not full-set scores.
- On the fixed five-frame smoke prefix, replacing evaluation occupancy by a
  constant and GT velocity by zero leaves every prediction hash unchanged
  ([record](evidence/label-and-speed-independence.json)). This tests those input
  boundaries on that prefix; it is not a universal leakage guarantee.
- The initial full run with two CPU workers was superseded after 564 frames to
  remove data-loading stalls on the now idle machine. All 564 overlapping
  predictions match the eight-worker run exactly
  ([record](evidence/loader-parity.json)); the initial outputs remain in
  [full-validation/](full-validation/). No model or metric setting changed.

## Runtime and data qualifications

Python 3.8.20, torch 1.12.1+cu113 and the released weights run on the RTX 4090.
MMCV 1.6.0 replaces the documented 1.5.2; the input wrapper preserves the CPU
metadata container expected by the original code. One exact integer-shape
operation is patched from `prod` to height × width to avoid the old NVRTC/Ada
failure. CPU/CUDA equality passed on 10,003 integer pairs. The complete
[patch](evidence/compatibility.patch), [digests](evidence/compatibility.json)
and [environment](evidence/runtime-versions.txt) are saved. Learned arithmetic,
weights, geometry, memory and class mapping are unchanged.

Original nuScenes camera archives were authenticated against current official
S3 ETags. Occ3D labels came from a pinned public transfer mirror, with archive
SHA-256 and complete local audits; they were **not separately matched against the
authors' inaccessible Google Drive archive**. The close metric reproduction is
additional empirical evidence, not independent archive authentication. Source
URLs, archive and per-file manifests are retained in [evidence/](evidence/).

The official supplied ego poses and full-scene memory bounds are retained.
The released model uses six cameras; the shared test pipeline's LiDAR point
field is not consumed by a LiDAR encoder in this configuration. This is not a
reproduction with estimated SLAM poses or unknown future scene extent.

## What is complete and what this does not test

The first step is complete: an intact published temporal occupancy system now
runs and reproduces its reported metrics in our environment. We have a credible
reference before changing the current/history representation or unfreezing PTv3.
This run alone does not isolate memory's causal contribution. Same-weight
history removal would be a sensitivity check; a matched independently trained
single-frame baseline is a different experiment. No PTv3 migration or new
training was started as part of this step.

## Saved artifacts

- [Official full-run summary](full-validation-workers8/summary.json),
  [independent summary](full-validation-workers8/audit-summary.json),
  [raw independent audit, compressed](full-validation-workers8/independent-audit.json.gz).
- [Per-frame records](full-validation-workers8/frames.jsonl),
  [effective configuration](full-validation-workers8/effective-config.py),
  [evaluator as run](full-validation-workers8/evaluate-as-run.py),
  [official results CSV](full-validation-workers8/official-metrics/results.csv),
  [run log, compressed](full-validation-workers8/run.log.gz),
  [artifact digests](full-validation-workers8/artifact-manifest.json).
- [Protocol and executable tools](../../tools/stocc_reference/README.md).
- [Data audit](evidence/data-audit.json), [metadata audit](evidence/metadata-audit.json),
  [source provenance](evidence/source.json), [weight provenance](evidence/model-download.json).
- [Smoke evidence](smoke-0003/), [label control](label-control-0003/),
  [label-and-speed control](label-and-speed-control-0003/).
- [Pre-shutdown observations](observed_preflight.json) are historical and are
  superseded by these completed post-restart results.

Remote run: `/home/wuyou/stocc_reference/runs/full-validation-workers8`.
Images, labels, weights and full voxel prediction arrays stay on the remote
disk. Git contains scripts, configurations, digests, controls and aggregate
results. The final exported artifact hashes were also verified after copying
these small evidence files to the local repository.
