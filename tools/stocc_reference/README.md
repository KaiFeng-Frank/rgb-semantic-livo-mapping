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
capped at 40% of GPU memory; timing while sharing the GPU is not isolated latency.

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
This compatibility adapter remains subject to the real inference smoke test.

`env.sh` selects only the dedicated environment and matching user-space NVIDIA
driver library. No system driver, local VPN, or running TASeg environment is
changed. The official BEV pooling extension is compiled for sm_86, supported by
the 4090, because CUDA 11.3 predates sm_89.

## Execution and acceptance

1. Run the original `tools/create_data_bevdet.py`, with `max_sweeps=0` as in its
   entry point, to produce the official global validation metadata.
2. Strictly load every checkpoint key; no missing/unexpected keys are accepted.
3. Use a complete available scene for the smoke test. Check six-camera input,
   increasing frame indices, memory population, predicted-flow production,
   finite class IDs, resource use and official metrics.
4. On a short fixed sequence, repeat with evaluation labels replaced by a
   constant. Prediction hashes should remain unchanged. This diagnoses the
   boundary between ground-truth metric bookkeeping and inference, not accuracy.
5. Run the full 150-scene validation with the official sampler, pipeline, model
   and metrics. Require exactly 6,019 unique frames in scene order. Report mIoU,
   per-class IoU, mSTCV, timing quantiles and peak memory. Investigate a mismatch
   with published scores rather than tuning on this validation set.

Example, after sourcing `env.sh`:

```bash
cd /home/wuyou/stocc_reference/upstream
python /home/wuyou/stocc_reference/evaluate.py \
  --scenes scene-0003 --out /home/wuyou/stocc_reference/runs/smoke
python /home/wuyou/stocc_reference/evaluate.py \
  --out /home/wuyou/stocc_reference/runs/full-validation
```

The wrapper enables the official `flickering_cnt` metric bookkeeping, preserves
the source model, and saves each uint8 prediction separately. It records
CUDA-synchronized model-call time and data-loading time separately. No training,
architecture adaptation, additional initialization weights or GT flow inputs
are used. Enabling metric bookkeeping is included in the measured call cost.

## Interpretation boundaries

- Official metadata uses supplied ego poses and the full scene trajectory to
  preallocate global memory bounds. Reproducing this protocol is not a claim of
  deployment with estimated SLAM poses or unknown future scene extent.
- The official inference function receives GT occupancy for auxiliary metric
  bookkeeping. The source's `self.gt_flow` name is misleading: inference
  overwrites it with the model's **predicted** flow. Verify this boundary in the
  running label-control check; never feed training GT flow at test time.
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
