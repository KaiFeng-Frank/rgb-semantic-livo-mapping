# Bounded instance feature memory, 2026-10-29

This experimental branch retains the frozen B0 PTv3 and YOLO11n-seg frontends.
It reads same-forward PTv3 decoder summaries and YOLO P3 appearance features,
associates objects in world coordinates, and uses a small query reader to update
current point semantics from raw recent/past object observations. Static semantic
mapping and pose estimation are unchanged. No accuracy or real-time improvement
is established by implementation or unit tests. The first trained experiment is
now [complete and failed its advancement gates](../results/instance_memory_20261029/README.md).
Actual populated ROS delivery passed at 2 Hz, but held-out semantic/identity gains
did not. The optional module is not promoted to the default pipeline.

The branch starts from `ac80da1`, retaining the unused opt-in Kalman prototype's
feature transport. The present method does not use its Kalman semantic filter.

## Fixed first experiment

The [protocol](../config/instance_memory_20261029.json) fixes the first candidate
and its controls before training. Selection uses metadata, not predictions. All
18 official validation collection logs were already represented in earlier
experiments. The former Kalman prototype mistakenly labelled four previously
inspected confirmation scenes as new holdout; this experiment does not reuse that
split. Instead, 24 uninspected official-training collection logs provide 16
training scenes (642 frames), four development scenes (161), and four project
holdout scenes (161). The **head-level holdout is not a whole-system blind test**:
the released PTv3 backbone was pretrained on the official nuScenes training pool.

Both trained semantic heads receive identical observations, supervision,
initialization, epochs, optimizer and capacity. A descriptor is first trained on
training Panoptic nuScenes instance labels, then frozen and shared across all
arms. Semantic training visits its own predicted association states; GT does not
force track matches. One head reads historical raw features; the other is given
only its current token. Selection is minimum development point CE, with epoch
zero eligible. Three seeds are run. Holdout extraction/scoring follows frozen
selection; it is not used to redesign, tune thresholds or pick epochs.

The first export has 474 training and 90 development object observations. Training
contains 89 strongly supported instances, including ten pedestrians with 32
observations, but only one bus and one motorcycle. These coverage limits are
reported instead of treating the total point count as independent examples.

## Identity and memory

Visual track IDs are small matching hints, not permanent identities. A learned
descriptor and coarse motion prediction drive one-to-one matching without a
class-equality gate. Ambiguous matches start a new identity. Up to 128 active
objects retain four recent observations and two older representative views, with
eight-second expiry. Only raw encoder evidence is stored; enhanced predictions
are never reintroduced as fresh independent observations. This limits recursive
feedback but does not make correlated observations independent or correct.

Current points query the matched object's history before the current observation
is written. The current token is always available. The decoder supplies a class
logit correction to the raw point prediction, permitting within-object point
differences when projection contains mixed surfaces. Empty/missing observations
do not create detections. No unobserved instance is painted into the current scan.
Retained sensor anchors can be revised when causal pose-graph updates arrive.

The frozen detector feature extraction hook operates on the existing P3 map.
Masks follow the same square letterbox geometry. There is no second image or
PTv3 forward. Very small masks without a P3 cell carry an explicit missing-feature
flag. A pre-existing empty-NMS float-index error was fixed and applied to every
control; it is an interface repair, not a learned-memory gain.

## Ground truth and evidence boundaries

The official Panoptic nuScenes archive was downloaded remotely through a mirror
and checked against its published content-addressed SHA-256:
`dab40d850a0c35024f2422a8d2a9ae508544bfefc4b2cbc50a84813dcd07844d`.
Only 964 selected annotation files (4.74 MB compressed) were recovered locally.
Existing images and scans stay at their original dataset location. No raw data
are copied into Git. Every selected annotation is individually hashed.

Panoptic category IDs use the original lidarseg ontology. Some lidarseg-labelled
points are ignored by the panoptic annotation; these remain in semantic scoring
and are excluded from identity scoring. Nonignored labels must align exactly.
Predictions and targets occupy separate cache files. GT instance labels enter
only training losses or post-prediction metrics, never matching inputs.

Report native16 full-scan mIoU, observed-point accuracy, corrections/new errors,
point identity F1, association quality, coverage and per-log results. These
explicit point identity diagnostics are **not official nuScenes PAT or a
leaderboard submission**. Missing instance coverage remains in denominators;
same-class switches count as identity failures. A 2 Hz annotation gap is not
proof of physical occlusion. Cached module timings are not end-to-end latency.

## Reproduction

```bash
bash tools/visual_instances/run_kalman_local.sh -m tools.instance_memory.data prepare --out INDEX
bash tools/visual_instances/run_kalman_local.sh -m tools.instance_memory.data extract \
  --index INDEX/index.json --annotations ANNOTATIONS --out CACHE --parts train dev
CUDA_VISIBLE_DEVICES='' python -m tools.instance_memory.train --cache CACHE --out RUN --seed 0
# After every seed completes frozen selection:
python -m tools.instance_memory.evaluate --cache CACHE --run RUN --out RESULT.json --split test
```

Training requires PyTorch, NumPy and SciPy; feature extraction uses the existing
verified PTv3/Ultralytics runtime. Memory and cache are bounded. Source snapshots,
failed first extractions, train/dev coverage and all selected checkpoints are
retained. Actual populated-instance ROS delivery and end-to-end timing have been
checked; active deployment remains experimental because the accuracy gates failed.

## Related implementation boundaries

The instance-query/long-short memory design is motivated by
[AutoSeg3D](https://github.com/AutoLab-SAI-SJTU/AutoSeg3D), whose released system is
for indoor scene segmentation. No claim of reproducing its architecture or
benchmark scores is made. This code uses a minimal adaptation suited to the
existing frozen PTv3 interfaces. [4DSegStreamer](https://llada60.github.io/4DSegStreamer/)
motivates explicit motion alignment and latency-aware streaming; its official
repository still has no implementation at this audit. Neither adding attention
nor introducing a memory bank alone is claimed as a new contribution.
