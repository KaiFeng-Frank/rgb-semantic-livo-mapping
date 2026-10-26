# Online visual instance association

This optional path preserves PTv3 and adds visual **instance association** to
current semantic scans. It does not modify class predictions, train a new head,
distill a model, or feed instance hypotheses into the persistent map.

The implementation uses the official COCO-pretrained **YOLO11n-seg** model with
Ultralytics **BoT-SORT**, sparse optical-flow camera-motion compensation, and
appearance ReID disabled. The implementation and weights are external optional
dependencies; they are not vendored into this repository.

Sources: [YOLO11](https://docs.ultralytics.com/models/yolo11/),
[tracking API](https://docs.ultralytics.com/modes/track/),
[official weight](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-seg.pt).

## What runs online

1. A separate ROS process receives the original camera topic with a one-frame
   pending queue. A separate GPU subprocess runs segmentation and tracking.
   Neither process calls the LIVO frontend or its pose update.
2. Observed instance-mask interiors are published, with overlapping masks and a
   two-pixel boundary removed. Unobserved tracker predictions never produce a mask.
3. The semantic process uses only **already received** camera results at or before
   the scan timestamp, within 20 ms by default. It never waits for vision. It
   retains at most 16 camera results. Missing, future, expired, wrong-size or
   invalid packets produce no instance association.
4. Sensor XYZ is projected into the original, full-resolution distorted image
   using `Rcl/Pcl`, raw camera intrinsics, and distortion. All points enter a
   three-pixel depth buffer before semantic filtering. A point more than 0.4 m
   behind the closest measured point in its cell is rejected. An instance needs
   at least three surviving points.

Supported visual classes map to native16: person→6, bicycle→1, car→3,
motorcycle→5, bus→2, truck→9. The visual class is a **separate hypothesis**, not a
replacement for the PTv3 class. The detector uses a fixed 640×640 letterboxed
input; no ground-truth data enter inference or association.

Identity is the pair **(session UUID, instance_id)**. A worker restart, new
sequence, or image gap over 750 ms starts a new session. This prevents an ID
counter reset from joining unrelated histories. IDs can still switch *within*
a session: BoT-SORT is not a physical identity guarantee. No semantic history
accumulation is enabled in this implementation.

## Outputs

- `/backend/visual_instances`: bounded compressed uint16 mask-index image and
  per-instance metadata; camera acquisition timestamp.
- `/backend/visual_status`: queue/drop/failure/readiness counters.
- `/backend/semantic_instances`: current scan in `backend_imu`, preserving XYZ,
  `label`, and `confidence`; adds `instance_id` (uint32), `visual_label` (uint16),
  and `instance_score` (float32). ID 0 means unassociated; visual label 65535 means
  unknown. The score is the detector score, **not** calibrated association confidence.
- `/backend/semantic_instance_info`: pair by exact cloud timestamp to obtain the
  session UUID and association diagnostics. Do not use bare IDs across sessions
  or consume a cloud with missing identity metadata as a temporal observation.

Existing `/backend/semantic_scan` and map interfaces retain their original schema.
Association and the old point-level temporal fusion cannot be enabled together.

## Run with the existing launcher

Use an existing GPU Python environment compatible with the machine's CUDA/PyTorch.
The validated environment adds `ultralytics==8.3.228`, `lap==0.5.12`, and
`ultralytics-thop==2.0.18` in a separate dependency directory. It reuses the existing
PyTorch, torchvision, NumPy, OpenCV, SciPy and YAML installations. Do not replace
the PTv3 environment's CUDA/PyTorch stack just to install the optional detector.
The ROS interpreter also needs OpenCV, NumPy, SciPy and YAML, but does not import
torch. SciPy supplies pose interpolation for nonzero camera/scan timestamp gaps.

Set the existing `SEMANTIC_RUNTIME` and, when needed, `SEMANTIC_ROS_PYTHONPATH`.
Add these environment settings before `tools/loop_closure/run_online.sh`:

```bash
export VISUAL_INSTANCE_WEIGHTS=/path/to/yolo11n-seg.pt
export VISUAL_PYTHONPATH=/path/to/isolated/visual-deps
export SEMANTIC_CONSISTENCY=off
```

Keep perception CPU work away from the pose cores on a resource-constrained
machine. `VISUAL_CPU_AFFINITY=14,15,30,31` is the tested local assignment (physical
cores 14/15 and their SMT siblings); the rest of the pipeline runs on
`0-13,16-29`. Choose the assignment from the host's actual CPU topology, not by
copying these numbers blindly. The visual ROS process runs at nice 10 and its
GPU worker at nice 15. Affinity is opt-in and inherited only by owned processes.
Without isolation, the measured pose latency was materially higher.

The launcher reads the image topic/transport and full-resolution calibration from
the supplied frontend/camera YAML files. Wait for both semantic and visual status
to report `ready=true` before publishing sensors. Logs include the exact weight
SHA-256, library version, tracker configuration and camera calibration.

For the existing paced real-sensor replay, add:

```bash
--visual-instances --visual-weights /path/to/yolo11n-seg.pt
```

## Validation and limits

[Recorded results, including failed associations and runtime outliers](../../results/visual_instances_20261026/README.md).

`test_association.py` checks nearby people, measured foreground occlusion, mask
overlap, rejected foreground, boundary rejection, camera axes, malformed transport,
causality, expiry and identity reset. `validate_ros.py` checks actual ROS delivery,
including a conflicting visual class that must not change semantic predictions,
future-camera rejection, restart identity and exact raw-map rebuild parity.

`real_ros.py` feeds real nuScenes images and raw LiDAR at their original 2 Hz
timestamps through the separate visual worker, live PTv3 and the same semantic
ROS node. It checks nonzero instance delivery and exact preservation of semantic
labels at the actual subscribers. Its graph poses are supplied nuScenes ego
poses, not a live SLAM result; GT semantic labels and boxes are not sent to nodes.

`evaluate.py` runs real images and LiDAR with labels/boxes used **only after
inference** for diagnostics. The six existing nuScenes scenes are development
data, not a fresh held-out evaluation. Box-supported identity checks are a proxy,
not official panoptic/LSTQ scores. The 2 Hz keyframe cadence is stated explicitly.

The HKU replay evaluates integrated runtime at 10 Hz camera / 1 Hz semantic
keyframes. It contains almost no detected traffic participants and cannot validate
dynamic-object identity accuracy. There is no HKU semantic ground truth.

`evaluate.py --audit-instances` additionally records each observed ID's current
visual class, GT-box support, ambiguous votes and adjacent identity/label changes.
These are different events: two different cars can both be classified correctly
while a track ID moves between them. Conversely, the same physical object and ID
can receive a different current-frame class, because no track-level semantic
state is accumulated. [The reproduced 242-frame audit](../../results/visual_instance_audit_20261026/README.md)
separates these cases and qualifies sparse/tied GT-box-majority evidence.

Only camera-visible surfaces are associated. Sparse depth checks cannot detect
every occlusion; image masks can include background, and object motion or ego motion
between timestamps can break projection. The exact-synchronized HKU input avoids
an inter-sensor timestamp gap; a nonzero tolerated gap is an approximation and does
not perform object-motion compensation. For a nonzero gap, the ROS node compensates
ego motion by interpolating only the already received graph poses bracketing the
camera time; it abstains if no bracket exists or its interval exceeds 1.5 seconds.
This is an approximation to within-scan motion, not deskewing raw nuScenes scans.
The offline nuScenes diagnostic uses supplied ego poses at both acquisition times.

Do not infer semantic improvement from having instance IDs. The next semantic
fusion experiment must separately measure corrected points, newly wrong points,
identity failures, coverage and latency, with a raw single-frame control.
