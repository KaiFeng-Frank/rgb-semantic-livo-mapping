# FAST-LIVO2 trajectory metrics — independent reproduction

Reproduced 2026-09-25 on a fresh host, starting from the public KITTI archives and the
source commits recorded in [`HANDOFF_fastlivo2_trajectory.md`](../HANDOFF_fastlivo2_trajectory.md).
The claim under test is README criterion 1: **seq 07 ATE 0.880 m, drift 0.127 % over 693 m**
(and the seq 04 smoke figure of 2.556 m).

**Short answer.** seq 07 reproduces. Rebuilt and re-run from raw data, FAST-LIVO2 scores
**ATE RMSE 0.858 ± 0.030 m over three runs (drift 0.121–0.129 %)** against KITTI odometry GT; the
committed trajectory scores **0.837 m** against the same GT, inside that spread. The published
0.880 m itself cannot be re-derived on this host: it was scored against SemanticKITTI's
`poses.txt`, and `semantic-kitti.org` is unreachable from here. seq 04 does **not** reproduce: all
three re-runs diverge in the cold-start transient (ATE 35–41 m).

Two harness defects were found on the way and are fixed in
[`run_fastlivo2_kitti.sh`](../run_fastlivo2_kitti.sh). Neither raised an error.

---

## 1. What the published number was scored against

`src/eval_fastlivo2_ate.py` reads its ground truth from `sequences/<seq>/poses.txt`.
[`fetch_kitti.sh`](../fetch_kitti.sh) never downloads KITTI's own `data_odometry_poses.zip`. The
only archive it unpacks into `sequences/` that contains a `poses.txt` is SemanticKITTI's
`data_odometry_labels.zip`. The committed numbers were therefore computed against **SemanticKITTI's
poses**, not KITTI's odometry GT. Scored both ways, the committed trajectory gives:

| seq 07, committed trajectory | published (`results/seq07_ate.json`) | re-scored here |
|---|---:|---:|
| GT source | SemanticKITTI `poses.txt` | KITTI `data_odometry_poses.zip` |
| matched pairs | 1096 | 1096 |
| EST path length | 694.35 m | **694.35 m** |
| GT path length | 692.68 m | 693.89 m |
| ATE RMSE | **0.880 m** | **0.837 m** |
| ATE mean / median / max | 0.812 / 0.727 / 1.844 m | 0.752 / 0.690 / 1.704 m |
| ATE min | 0.259 m | 0.017 m |
| drift (RMSE / GT length) | 0.127 % | 0.121 % |
| *seq 04 ATE RMSE* | *2.556 m* | *2.545 m* |

The estimate side matches exactly, so the committed trajectory and the evaluator are what the
handoff describes. The GT is the only difference: KITTI's poses give a GT path 1.21 m longer and a
different residual. The SemanticKITTI column could not be re-derived here because the host's network
policy denies `semantic-kitti.org`.

*Note for the other documents:* `src/score_2d_vs_3d.py` describes the same file as "KITTI odometry GT
(sequences/07/poses.txt)". On a host provisioned by `fetch_kitti.sh`, that file is SemanticKITTI's.

## 2. Full re-run from raw data

Everything was rebuilt rather than copied. The inputs were the KITTI raw drives 0027 and 0016
(`_sync` clouds and images, `_extract` 100 Hz OXTS). Bags were generated with the committed
`src/kitti_to_ros2bag.py --time-unit us`. FAST-LIVO2 was the ROS 2 port at `837b7bb`, run with the
unchanged `config/kitti_velodyne64.yaml`. `src/fastlivo2_extrinsics.py` re-derived the extrinsics
from KITTI's calibration, and they match the yaml digit for digit. The bags match the handoff's
counts. seq 07 has 1101 clouds, 1101 images and 11 482 IMU samples, after the one documented
duplicate is dropped. seq 04 has 271, 271 and 2 861.

This host cannot reach `packages.ros.org`, so ROS 2 Jazzy came from RoboStack/conda-forge.
[`tools/setup_fastlivo2_conda.sh`](../tools/setup_fastlivo2_conda.sh) recreates the workspace with
the handoff's three build-system-only patches. It pins the libraries that plausibly matter
numerically to Ubuntu 24.04's versions: Eigen 3.4.0, gcc 13, CMake 3.28 and Sophus from the
`ros-jazzy-sophus` 1.22.9102 release. No estimator source file was modified. Versions are listed in
[`results/fastlivo2_repro_20260925/environment.txt`](../results/fastlivo2_repro_20260925/environment.txt).

| seq 07, KITTI odometry GT | ATE RMSE | mean | median | max | drift | EST length | max ‖p − p_committed‖ |
|---|---:|---:|---:|---:|---:|---:|---:|
| committed trajectory | 0.837 m | 0.752 | 0.690 | 1.704 | 0.121 % | 694.35 m | — |
| re-run 1 | 0.838 m | 0.758 | 0.722 | 1.726 | 0.121 % | 693.97 m | 0.51 m |
| re-run 2 | 0.843 m | 0.763 | 0.695 | 1.966 | 0.122 % | 694.02 m | 0.99 m |
| re-run 3 | 0.893 m | 0.766 | 0.641 | 2.581 | 0.129 % | 694.07 m | 2.35 m |
| **re-runs, mean ± SD** | **0.858 ± 0.030 m** | | | | | | |

* Every re-run writes 1096 poses. Their timestamps are identical to the committed file's, starting at
  `1317386426.075841`.
* The three re-runs are identical to each other for the first poses. From pose 1 onward they all
  differ from the committed trajectory by the same amount (1.7 mm at pose 1, 14 mm at pose 6). That
  is a deterministic platform difference, not noise.
* **FAST-LIVO2 on this harness is not run-to-run deterministic.** With the same binary and the same
  bag, runs 2 and 3 first differ by more than 1 mm at pose 71 and end 1.7 m apart. A single run is
  therefore not a measurement. The committed file is one such draw, and it lies within the spread.
* An earlier run, made before the intrinsics check of §4 existed, scored 0.785 m. It is excluded
  because its node log was overwritten and its camera parameters cannot be confirmed.

Scores and per-run comparisons are in
[`results/fastlivo2_repro_20260925/summary.json`](../results/fastlivo2_repro_20260925/summary.json).
[`tools/score_fastlivo2_repro.py`](../tools/score_fastlivo2_repro.py) regenerates the file with the
unchanged evaluator.

## 3. seq 04 does not reproduce, and why

| seq 04, KITTI odometry GT | ATE RMSE | EST length (GT 387.06 m) |
|---|---:|---:|
| committed trajectory | 2.545 m | 371.52 m |
| re-runs 1 / 2 / 3 | 35.38 / 41.12 / 35.68 m | 403.04 / 441.60 / 394.27 m |

The inputs are not the problem. Poses 0–3 agree with the committed trajectory to within 0.01 mm
(TUM files carry 1 µm), and the vehicle is already moving at 12.7 m/s. A scan-deskew or timing
difference would show up as centimetres to decimetres on the very first update. The difference
first appears at pose 4, at 0.55 mm, and grows about 7× per sweep: 11 mm at pose 5, 84 mm at pose 6,
210 mm at pose 7. This is the cold-start transient of handoff §5.1: `vel_end = 0` while the car is
doing 12.7 m/s. On this host the filter never recovers. The estimate stalls and then runs backwards.
The committed recovery is one realisation of an unstable transient, and it does not carry across
builds.

The cause is not AVX-512 code generation. An AVX2-only build (`-march=x86-64-v3`) departs even
earlier, by 151 mm at pose 3. The remaining platform differences are library versions and the
compiler patch level. This host used PCL 1.15.1 and OpenCV 4.12 from conda-forge; Ubuntu 24.04's apt
packages are PCL 1.14 and OpenCV 4.6. The handoff already advises discarding seq 04's first 3 s.
This result goes further: seq 04 started at frame 0 is not a usable benchmark for this
configuration.

## 4. Two silent harness defects, fixed

**Bag playback started before the subscriber was matched.** `ros2 bag play` publishes as soon as
its publishers exist. On this host, DDS discovery took longer than that. The first run lost the
first 0.52 s of the bag (5 sweeps, 5 images, 52 IMU samples) without any error. The node then
initialised on later data and produced 261 poses for seq 04 instead of 266, on a different
trajectory. The fix is `--delay ${PLAY_DELAY:-3}`. With it, every run produced the committed pose
count and identical timestamps.

**Camera intrinsics could silently become zero.** The port's `vk::getRemoteParam` creates a fresh
client for each camera parameter, waits once for 100 ms for `parameter_blackboard` and otherwise
returns the default. If `cam_model` is missed, the node aborts. That happened in 7 of 18 node starts
here. If `cam_fx`, `cam_cx` or `cam_width` were missed, the value would silently become `0` and the
VIO would run on a broken camera. This was not observed here, but nothing in the port prevents it.
The runner now requires the node's printed `intrinsic:` line to equal the yaml and restarts the node
otherwise, up to five attempts. It also deletes the previous `out/<seq>_fastlivo2_tum.txt` first.
Before this change, a failed run left the previous run's file in place, and a wrapper that copied
`out/` produced three "results" that were byte-identical to the preceding run before the failure was
noticed.

## 5. Reproduce

```bash
tools/setup_fastlivo2_conda.sh                 # ROS 2 Jazzy via RoboStack + workspace build
./fetch_kitti.sh && ./fetch_oxts100.sh         # plus data_odometry_poses.zip for KITTI GT
./build_bags.sh                                # the _us bags are the ones used
./run_fastlivo2_kitti.sh kitti_seq07 /data/livo_sem/bags/kitti_seq07_us 0.5
python3 tools/score_fastlivo2_repro.py <dir with kitti_seq0?_run*.txt> summary.json
```

## 6. Still open

* Score the re-runs against the SemanticKITTI `poses.txt` behind the published 0.880 m. That needs
  `semantic-kitti.org` to be reachable, or a copy of `sequences/07/poses.txt` from the original host.
* Locate FAST-LIVO2's run-to-run nondeterminism. Two candidates remain: thread scheduling (the
  OpenMP reduction in `vio.cpp`) and message-arrival timing in `sync_packages`. Comparing a
  single-threaded build (`MP_PROC_NUM=1`) with a rate-independent replay would separate them.
