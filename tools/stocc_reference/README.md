# Official ST-Occ pretrained validation

This is an isolated reproduction of the official six-camera occupancy model.
It does not train a model, replace PTv3, or establish a benefit for our existing
point-semantic/instance-memory pipeline.

## Frozen references

- [Official repository](https://github.com/matthew-leng/ST-Occ), commit
  `1633f62e2e6677a5fa474905977acfeca4e7819e`.
- [Official R50 checkpoint](https://github.com/matthew-leng/ST-Occ/releases/download/v1.0/stocc-r50-256x704.pth),
  SHA-256 `68491f55a2208fad884e3b664d0e34b8311688376380f7722e30cac6bde5ffa0`.
- Base config `occupancy_configs/st_occ/stocc-r50-256x704-36e.py`; use the
  official `_flicker_eval.py` extension for mSTCV, including its precise global
  coordinates and GT-speed exclusion of moving voxels in metric bookkeeping.
- Reference model-zoo numbers: 42.13% mIoU and 8.68% mSTCV. These are author
  results, not our measurements. Evaluate all 150 official validation scenes
  (6,019 frames) before comparing full-set scores.
- Retain six cameras, 256×704 network input, official coordinate transforms,
  voxel resolution, predicted motion, uncertainty, memory and class mapping.

## Data provenance and resource policy

All bulk downloads and inference run directly on the remote host at
`/home/wuyou/stocc_reference`. The local GPU remains unused. Existing TASeg
training is not stopped or modified. The inference CUDA allocator is initially
capped at 40% of GPU memory when sharing with TASeg. After the user shut down and
restarted the host, TASeg was no longer running (checkpoint epoch 33 survives).
ST-Occ's resumed validation uses a 75% allocator cap on the idle GPU. The final
full run uses eight CPU loader workers (the initial shared-resource setting of
two workers was CPU-bound); overlapping predictions are checked for equality.
No training
job is automatically restarted as part of this inference task.

`acquire_cameras.py` obtains the original nuScenes keyframe archives from
Motional S3/CloudFront, verifies the S3 multipart ETag and records SHA-256,
retains only the 36,114 validation-camera JPEGs and deletes each verified
temporary archive after extraction. Metadata and LiDAR keyframes are read-only
symlinks to the existing nuScenes installation. `official_s3.py` preserves the
verified acquisition functions from the TASeg tooling; its digest is recorded
in `camera-plan.json`. The first remote download used that original helper path.

The authors' Occ3D Google Drive is unreachable from this remote network.
`acquire_occ3d.py` therefore uses a pinned public transfer mirror,
`RoyYao233/occ3d-nuscenes-transfer` revision
`b679e0ace1dc03d0a86737cb73fa790df03a0ad5`, through hf-mirror.com. Both archive
and annotation bytes are checked against that revision's LFS SHA-256. This
verifies mirror integrity, **not independent identity with the author archive**.
The retained label manifest must cover all 6,019 validation frames and use the
official `(200,200,16)` grid, 18 labels and camera/LiDAR masks.

## Runtime

Separate Python 3.8.20 environment, PyTorch 1.12.1+cu113, torchvision
0.13.1+cu113, CUDA 11.3 compiler, GCC 9, mmdet 2.24.0 and mmsegmentation
0.24.0. Runtime compatibility exception: mmcv-full **1.6.0**, within this
repository's accepted range, uses the official CUDA 11.3/PyTorch 1.12 wheel;
the installation instructions request 1.5.2 but publish no matching wheel at
that index. This is recorded explicitly, not counted as a method change.
The evaluation wrapper preserves the original CPU metadata `DataContainer`
expected by ST-Occ's `forward_test`; MMCV 1.6 would otherwise unwrap it during
single-GPU scatter. Measurement tensors still use ordinary MMCV CUDA transfer.
This compatibility adapter passed the real 40-frame inference smoke test.

One source-level runtime compatibility patch is needed on the 4090:
`spatial_shapes.prod(1)` triggers the known torch 1.12/cu113 NVRTC architecture
failure ([PyTorch issue 87595](https://github.com/pytorch/pytorch/issues/87595)).
`apply_compat.py` replaces this two-integer product by explicit height × width,
checks exact CPU/CUDA equality on 10,003 pairs, and saves the source diff and
before/after digests. No learned arithmetic or weights are changed.

`env.sh` selects only the dedicated environment and matching NVIDIA driver
library. After reboot, the host's kernel and system libraries both report
595.99.02; the old 595.84 user-space workaround is no longer used. No system
driver, local VPN, or running TASeg environment is changed. The official BEV
pooling extension is compiled for sm_86, supported by
the 4090, because CUDA 11.3 predates sm_89.

## Execution and acceptance

1. Run `prepare_validation.py`: it calls the original converter,
   `add_ann_adj_info` and `add_global_info`, with `max_sweeps=0`. It selects only
   the 6,019 validation frames and reuses one full SDK instance; transforms,
   annotations and scene bounds use the original source functions. A proxy
   limits iteration without changing the SDK's full token lookup tables.
   It emits an early complete-scene smoke subset, then full validation metadata.
2. Strictly load every checkpoint key; no missing/unexpected keys are accepted.
3. Use a complete available scene for the smoke test. Check six-camera input,
   increasing frame indices, memory population, predicted-flow production,
   finite class IDs, resource use and official metrics.
4. On a short fixed sequence, repeat with evaluation labels replaced by a
   constant and GT speed replaced by zero. Prediction hashes should remain unchanged. This diagnoses the
   boundary between ground-truth metric bookkeeping and inference, not accuracy.
5. Run the full 150-scene validation with the official sampler, pipeline, model
   and metrics. Require exactly 6,019 unique frames in scene order. Report mIoU,
   per-class IoU, mSTCV, timing quantiles and peak memory. Investigate a mismatch
   with published scores rather than tuning on this validation set.

Example, after sourcing `env.sh`:

```bash
cd /home/wuyou/stocc_reference/upstream
python /home/wuyou/stocc_reference/apply_compat.py
python /home/wuyou/stocc_reference/prepare_validation.py
python /home/wuyou/stocc_reference/verify_data.py
python /home/wuyou/stocc_reference/evaluate.py \
  --ann-file data/nuscenes/stocc-smoke-nuscenes_infos_global_val.pkl \
  --memory-fraction .75 --out /home/wuyou/stocc_reference/runs/smoke
python /home/wuyou/stocc_reference/evaluate.py \
  --memory-fraction .75 --workers 8 \
  --out /home/wuyou/stocc_reference/runs/full-validation-workers8
python /home/wuyou/stocc_reference/audit_results.py \
  --run /home/wuyou/stocc_reference/runs/full-validation-workers8 \
  --data-root /home/wuyou/stocc_reference/data/nuscenes \
  --metadata /home/wuyou/stocc_reference/data/nuscenes/bevdetv4-nuscenes_infos_global_val.pkl
```

The wrapper uses official flicker bookkeeping, the source model plus the exact
integer-shape compatibility patch, and saves each uint8 prediction separately. It records
CUDA-synchronized model-call time and data-loading time separately. No training,
architecture adaptation, additional initialization weights or GT flow inputs
are used. Enabling metric bookkeeping is included in the measured call cost.
The CPU audit reloads exported predictions, verifies every prediction digest and
scene/index boundary, and recomputes the camera-masked confusion matrix without
calling the official evaluator. Its class IoUs and 17-class mean must match the
rounded official results. It does not independently recompute mSTCV.

## Interpretation boundaries

- Official metadata uses supplied ego poses and the full scene trajectory to
  preallocate global memory bounds. Reproducing this protocol is not a claim of
  deployment with estimated SLAM poses or unknown future scene extent.
- The official inference function receives GT occupancy for auxiliary metric
  bookkeeping. The source's `self.gt_flow` name is misleading: inference
  overwrites it with the model's **predicted** flow. Verify this boundary in the
  running label-control check; never feed training GT flow at test time.
- The shared test pipeline also loads LiDAR points. The released configuration
  has no `pts_voxel_encoder`, and `extract_feat` only uses points when that
  component exists. Image depth is predicted by `depth_net`; test inputs contain
  no GT depth or GT flow. These are source/config checks, not a claim that all
  possible leakage has been ruled out by the five-frame intervention.
- mSTCV subtracts GT class-change counts from predicted class-change counts at
  aligned world voxels, excludes moving voxels using the official GT-speed
  mask, and normalizes by non-free voxels. It is not an instance
  identity/re-ID metric, nor an object-recovery metric.
- Occupancy mIoU cannot be compared directly with our PTv3 point segmentation
  mIoU. A pretrained ST-Occ run alone does not isolate memory's causal benefit.
- Same-weight history removal, if later run, is only a sensitivity diagnostic;
  it does not substitute for an independently trained single-frame baseline.

The first deliverable is an evidence-backed official-system validation, with
any incomplete coverage or numerical discrepancies stated explicitly. Data and
large model files stay outside Git; scripts, provenance, logs and aggregate
results belong in the existing semantic research branch.
