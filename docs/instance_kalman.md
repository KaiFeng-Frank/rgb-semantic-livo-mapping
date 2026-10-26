# Feature-conditioned instance Kalman pilot

Status: implementation and CPU contracts checked; GPU extraction, training,
accuracy evaluation and real ROS replay have **not run**. The user requested use
of a new remote machine; SSH has not completed its banner exchange. No model
accuracy or runtime improvement is claimed.

## Model and reuse boundary

Frozen PTv3 decoder features are pooled over the existing geometry-validated
instance components. The observation network receives 64 feature means, 64
log-standard-deviations, the native16 raw posterior mean, COCO80 independent
sigmoid class scores, and eight observation-quality statistics. A 32-unit hidden
layer predicts a bounded residual in centered class-evidence space and positive
diagonal observation variances. The state is continuous evidence, not an integer
class ID. The final classification remains categorical.

The design uses the learned-observation/uncertainty idea in
[RKN](https://proceedings.mlr.press/v97/becker19a.html) and elementary analytic
Kalman updates. It is an adaptation, not an implementation or reproduction of
the full RKN architecture. No upstream neural-filter weights are imported.
[KalmanNet](https://arxiv.org/abs/2107.10043) and
[GLFS](https://arxiv.org/abs/2311.10018) are related baselines/design references;
the inverse-variance control here is not a reproduction of GLFS.

Identity remains `(visual session, predicted track ID, geometric epoch)`.
Appearance/geometry breaks start new generations. Repeated similar observations
have inflated measurement variance; prior covariance has a floor and forgets
with elapsed time. Large disagreement alone is not an outlier test. Strong
current competing object classes veto history-dependent overrides; background
predictions may be corrected when the remaining acceptance conditions pass.
These are approximate robustness policies, not a proof that systematic or
correlated errors are calibrated away.

Only raw observations enter the retained evidence window. Recursive filter
posteriors are state estimates, never reinserted as independent observations.
Track count is capped at 128, evidence windows at four observations, TTL at two
seconds. Persistent map fusion continues to use raw independent frame evidence.

## Online transport

`PTv3Client(instance_features=True, probability_classes=16)` enables the extra
transport explicitly. Ordinary inference retains one decoder feature tensor and
its inverse point-to-voxel map on GPU. Once the asynchronous visual result is
available, `pool_last_instances(groups, count)` computes only requested component
means/stds. The worker checks the exact completed sequence before pooling.
No second PTv3 forward and no all-point feature copy to the ROS process occurs.
Legacy label/confidence and posterior interfaces keep their existing contracts.

The visual predictor retains COCO80 scores by NMS indices; the same rows survive
empty-mask removal, tracker reordering, mask filtering, and packet encoding.
Scores are not treated as a calibrated categorical distribution. Unsupported
native classes are retained through the raw 3D branch.

The optional ROS node flags are:

```sh
--visual-instances --instance-semantics shadow \
--instance-kalman-checkpoint /path/to/learned.json
```

`active` publishes the enhanced observed scan/instances; `shadow` leaves scan
labels unchanged. This pilot is native16 and requires real full-score visual
evidence. Prediction fixtures cannot fabricate the missing feature evidence.

## Frozen experiment protocol

See [protocol](../results/instance_kalman_20261027/protocol.json). Eight training,
two development, and four test scenes are selected by collection-log hashes,
excluding logs of all previously inspected pilot/diagnostic scenes. The previous
six instance scenes are diagnostic regressions only. Test files are not opened
by the trainer until both heads finish selection on development scenes.

Both heads use **official 3D training labels**. This is supervised head training,
not a no-3D-label result or a new PTv3 backbone training run. Feature normalization
uses training observations only. The single-frame control has the same network
initialization, data, capacity, epoch count and optimizer as the temporal arm.
Same-head current, inverse-variance weighting, fixed-variance KF and learned KF
controls isolate the fusion decision from new input evidence.

Per-scene caches contain compact instance statistics and selected-point raw
posteriors/labels; raw scans and images are never copied. Cache manifests bind
inputs, backbone checkpoint, effective model settings and source versions.

```sh
bash tools/visual_instances/run_kalman_local.sh -m tools.visual_instances.kalman_cache \
  prepare --out /path/to/index
bash tools/visual_instances/run_kalman_local.sh -m tools.visual_instances.kalman_cache \
  extract --index /path/to/index/index.json --out /path/to/cache \
  --weights /path/to/yolo11n-seg.pt
CUDA_VISIBLE_DEVICES='' bash tools/visual_instances/run_kalman_local.sh \
  -m tools.visual_instances.train_kalman --cache /path/to/cache --out /path/to/run
```

The local convenience script names this machine's verified runtime. Remote
deployment must first resolve its own runtime and dataset paths. GPU admission
checks refuse to compete with an existing foreign CUDA process. Do not launch a
large run until real feature transport and one scene extraction pass.

## Validation completed before remote deployment

- 232 tests passed, one skipped, plus 128 subtests across instance/memory tools.
- 52 ROS semantic fusion, output, delivery and lifecycle tests passed under the
  system Python with ROS Jazzy. An earlier invocation under non-ROS Python failed
  two message-schema imports; rerunning under the correct interpreter passed.
- Neural export agrees between PyTorch and NumPy; gradients reach the observation
  and variance heads. Tests cover stale/retired identities, covariance recovery,
  repeated observations, feature indexing and shared-memory slot ownership.
- GPU transport, semantic gain, calibrated uncertainty, and real driving latency
  remain unverified. Defaults are not promoted based on unit tests.
