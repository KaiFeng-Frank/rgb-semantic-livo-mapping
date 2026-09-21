# rgb-semantic-livo-mapping

**A reproducible baseline that turns an offline rosbag into a world-frame point cloud where every
point carries `(x, y, z, R, G, B, class, confidence)`.**

FAST-LIVO2 supplies the pose. PTv3 supplies the semantics. The camera supplies the real colour.
ROS 2 Jazzy, KITTI + SemanticKITTI, every number checked against ground truth.

![semantic map](docs/img/rviz_class_final.png)

*The full 693 m loop of KITTI seq 07, coloured by predicted class. Cyan is `driveable_surface`,
yellow is `car`, magenta is `vegetation`, purple is `manmade`. The continuous yellow ribbon down the
middle of the road is not an error — it is one moving vehicle's accumulated trail, and the repo
measures exactly that (see criterion 5).*

---

## Why this exists

Semantic mapping stacks are easy to assemble and hard to trust. Every component "runs". The map
looks plausible. Nothing throws. And yet the result can be quietly, structurally wrong.

This repo is the opposite bet: **a baseline where every claim is nailed to ground truth**, so that
the interesting problems can be discovered instead of assumed. That choice paid off immediately —
two of the three failures documented below produce *no error message whatsoever*, and would have
survived any amount of looking at RViz.

---

## What it does today

| # | Acceptance criterion | Result | How it was proven |
|---|---|---|---|
| 1 | LIVO runs stably, trajectory matches upstream | **PASS** | ATE 0.880 m, drift **0.127 %** over 693 m; estimator core differs from upstream by 3.7 %, all mechanical ROS 1→2 changes |
| 2 | Per-point semantics on the same LiDAR frame, near real time | **PASS (6.9 Hz, not 10)** | PTv3 87.7 ms; full frame 144.4 ms mean / 166.5 ms p95 |
| 3 | Timestamps correspond, no frame mismatch | **PASS** | 1096 trajectory/image stamp pairs, **max &#124;Δ&#124; = 477 ns** |
| 4 | RViz shows a continuously accumulating world-frame cloud | **PASS** | 2.70 M voxels @ 0.20 m published natively, real RViz2 captures |
| 5 | Dynamic objects do not smear from sync error | **PASS** | controlled experiment, see below |
| 6 | Full bag, no crash / VRAM growth / sustained drops | **PASS** | 1096/1099 scans, VRAM flat at 3798 MiB from warm-up to frame 1101 |

Semantic quality against SemanticKITTI ground truth, 20 frames of seq 07, zero-shot
(nuScenes-trained weights, never fine-tuned on KITTI): **point accuracy 86.3 %, coarse mIoU 61.3 %**.

| class | IoU | class | IoU |
|---|---|---|---|
| car | 91.2 | road | 84.1 |
| manmade | 79.7 | terrain | 72.0 |
| sidewalk | 69.1 | vegetation | 64.7 |
| truck | 40.2 | person | 37.8 |

---

## The three traps this repo documents

Each one was found by measurement, each one is silent, and each one is written down so nobody pays
for it twice. Full detail in [`CRITICAL_CONSTRAINTS.md`](CRITICAL_CONSTRAINTS.md).

### 1. A constant that deletes the road

PTv3's nuScenes weights consume `strength` (LiDAR intensity) with **no normalisation anywhere in the
test pipeline**. Pointcept's nuScenes loader does `strength = intensity / 255`; KITTI intensity is
*already* in `[0, 1]`. Both are "legal" — but the distributions differ by an order of magnitude.

Feeding KITTI intensity unscaled classifies **the entire road surface as `terrain`**. No error, no
warning, mean confidence 0.93.

| strength scale | point acc | road IoU | vegetation IoU |
|---|---|---|---|
| 1.0 (raw KITTI) | 57.5 % | 7.8 % | 65.9 % |
| 0.0 (zeroed) | — | 98.6 % recall | **17.1 %** |
| **0.2** | **86.3 %** | **84.1 %** | **64.7 %** |

Zeroing the channel is *not* the fix: it rescues the road and destroys vegetation, which proves the
channel carries real signal and only its scale was wrong. Ruled out by ablation: ground-plane height
(z offsets −0.70 … +0.21 m change nothing), point density (coarser grids are worse), range clipping.

![intensity ablation](docs/img/semantic_intensity_ablation.png)

### 2. "strict load succeeded" is not "the model matches"

The released PTv3 checkpoint targets **Pointcept v1.5.1**. On HEAD, `PointTransformerV3.__init__`
no longer accepts `cls_mode`. Rename the config key and `load_state_dict(..., strict=True)` reports
**0 missing / 0 unexpected** — and the model is broken:

| load path | point acc | coarse mIoU | car IoU | points predicted `car` |
|---|---|---|---|---|
| HEAD + key rename | 25.7 % | 7.9 % | **0.00 %** | **5** of 2.4 M |
| **v1.5.1** | **86.3 %** | **61.3 %** | **91.2 %** | 227 920 |

A model that finds 5 car points where ground truth has 6.5 % car is not suffering a domain gap; it
is a different model wearing the same weights. Upstream says so itself: *"Released model weights are
temporarily invalid as the model structure of PTv3 is adjusted."*

**Rule: if you had to rename, remap or drop any key to make a checkpoint load, the load is not
evidence. Verify behaviour on data with ground truth.**

### 3. A right-multiplication that only shows up in corners

FAST-LIVO2's state is the **IMU** pose, so its trajectory is `T_{W←IMU}`, not `T_{W←LiDAR}`.
The fusion must right-multiply by the extrinsic:

```
T_W_L = T_W_I @ T_I_L
```

Omitting it is invisible in ATE — a left-multiplied global alignment absorbs a *left* constant, but
this one is on the right. It is also invisible while driving straight, where it is a pure global
translation. It only blurs the map when `R_W_I(t)` changes. Measured over a 113.3° turn:

| metric | with `T_I_L` | without | ratio |
|---|---|---|---|
| local surface thickness, median | **0.0803 m** | 0.1080 m | 1.35× |
| ground thickness, median | **0.1056 m** | 0.1659 m | **1.57×** |
| occupied 10 cm voxels | **1.086 M** | 1.420 M | 1.31× |

---

## Criterion 5, done as a controlled experiment

SemanticKITTI labels *stationary* cars `10` and *moving* cars `252`, so smear does not have to be
judged by eye. Both groups go through the identical transform chain; we measure the **world-frame
footprint length of a single sweep**:

| | clusters | footprint length median / mean / p90 |
|---|---|---|
| moving cars (252) | 51 | **3.31 / 4.24 / 9.50 m** |
| stationary cars (10) | 383 | **4.07 / 4.66 / 8.12 m** |

Moving cars are **not** longer than stationary ones. That is the right test: a stationary car keeps
its own length no matter how wrong the pose is, while a 14 m/s car placed at the wrong instant
stretches along its direction of travel. No stretching ⇒ no sync-induced smear.

Accumulation trails are a different thing and are physically inevitable without dynamic-object
removal: one tracked vehicle travels 23.0 m in 3.32 s while each sweep's footprint stays 8.39 m,
so the map keeps a ~31 m ribbon. Visible in `docs/img/rviz_dynamic_class_zoom.png`.

---

## Architecture

```
rosbag (KITTI -> ROS 2)          FAST-LIVO2 (ROS 2 port)
  /velodyne_points  10 Hz  ───────►  TUM trajectory  T_{W<-IMU}
  /camera/image_raw 10 Hz            keyed on TRUE sensor time
  /camera/camera_info                (evo/pose_output_en, zero code change)
  /imu             100 Hz
        │                                     │
        │                                     ▼
        │                        interpolate (slerp + linear)
        │                        T_W_L = T_W_I @ T_I_L
        ▼                                     │
   PTv3 (nuScenes, v1.5.1)                    │
   intensity × 0.2                            │
   per-point class + confidence               │
        │                                     │
        └──────────────┬──────────────────────┘
                       ▼
            project into rectified cam2 -> RGB
            transform to world -> voxel hash
                       ▼
        /semantic_map   PointCloud2, point_step 28
        x f4 | y f4 | z f4 | rgb f4 | class u16 | confidence f4 | has_rgb u8
```

FAST-LIVO2 is a **pose source only**. Its `/cloud_registered` is deliberately not consumed: in LIVO
mode ~52 % of those messages are empty and the rest carry only the camera-frustum subset of the scan.

**Fusion rule.** Class: confidence-weighted vote, `score[voxel][c] += softmax_max_prob`, argmax wins.
Confidence: winner's share of total vote weight. RGB: mean over observations that actually had camera
coverage; voxels with none get `has_rgb = 0` and a sentinel grey rather than a misleading black.
37.7 % of map voxels end up with RGB (a single sweep only sees 16.2 %).

---

## Reproduce

Requires ROS 2 Jazzy, an NVIDIA GPU, and ~50 GB for KITTI + bags.

```bash
./setup_ros2.sh          # ROS 2 Jazzy desktop
./mkenv.sh               # conda env: torch + spconv + torch_scatter + flash-attn
./fetch_kitti.sh         # KITTI raw drives 0027 / 0016 + SemanticKITTI labels
./fetch_ptv3.sh          # Pointcept v1.5.1 + nuScenes PTv3 checkpoint
./build_bags.sh          # KITTI -> ROS 2 mcap  (NOTE: use the _us bags, see below)

./run_fastlivo2_kitti.sh seq07 bags/kitti_seq07_us 0.5     # -> TUM trajectory
./run_pipeline.sh seq07 bags/kitti_seq07_us \
    results/kitti_seq07_fastlivo2_tum.txt 0.3 \
    --map-voxel 0.20 --map-rate 0.5 --conf-gate 0.5 --reliable

rviz2 -d rviz/semantic_map.rviz      # toggle "Map Class" to switch views
```

Two things that will bite you if skipped:

* **Use the `_us` bags.** FAST-LIVO2's `preprocess.cpp` reads `curvature = time / 1000 // ms`, i.e.
  it wants **microseconds**. A control run with seconds gives ATE **50.6 m** and a 25 %-short
  trajectory; microseconds gives 2.56 m.
* **Export `FASTDDS_BUILTIN_TRANSPORTS=LARGE_DATA`.** With default transports a 2.7 MB
  `BEST_EFFORT` sample loses 63 % of scans even when the consumer is completely idle and the CPU is
  99 % free. `run_pipeline.sh` does this for you.

---

## Known limits, stated plainly

* **6.9 Hz, not 10 Hz.** PTv3's 87.7 ms dominates the 144.4 ms frame.
* **Per-point deskew is currently neutral** on this data (within ±5 % on every sharpness metric).
  It is kept because camera projection needs per-point world coordinates, not because it buys accuracy.
* **No dynamic-object removal**, so moving vehicles leave trails. By design for this milestone.
* **RGB covers 37.7 % of map voxels.** The camera is a narrow forward frustum; the LiDAR is 360°.
* **PTv3 on spconv is not bit-deterministic** (atomic accumulation). At `confidence ≥ 0.5`,
  run-to-run repeatability is 99.2 % — which is why the confidence field exists and is gated on.

---

## Roadmap

**v0.1 — this release.** The honest baseline: four components, six criteria, every number against
ground truth, three silent failure modes documented.

**v0.2 — close the real-time gap.** Reach a sustained 10 Hz. The budget is known and the bottleneck
is not in doubt; TensorRT export, fp16, sparse-conv backends and frame-skipping with pose-keyed
catch-up are all on the table.

**v0.3 — make the map honest about time.** Dynamic-object handling, so a moving car is a moving car
and not a 31 m ribbon. The ground truth for this already exists in SemanticKITTI's moving-class
labels, so it can be scored, not eyeballed.

**v0.4 — cross-sensor semantics that survive the transfer.** Trap #1 is a symptom of something
larger: a segmentation network trained on one LiDAR consumes raw geometry and raw intensity from
another. This repo can measure that transfer precisely, on any sequence with per-point ground truth.

**v0.5 — beyond the rotating scanner.** Non-repetitive solid-state patterns (Livox) break the
implicit assumptions of every model trained on spinning LiDAR. The measurement harness here is the
prerequisite for saying anything defensible about it.

---

## Credits

Built on work by others, none of it claimed here:

* [FAST-LIVO2](https://github.com/hku-mars/FAST-LIVO2) — Chunran Zheng et al., HKU MARS Lab (GPLv2)
* ROS 2 port: [Robotic-Developer-Road/FAST-LIVO2](https://github.com/Robotic-Developer-Road/FAST-LIVO2), branch `humble`
* [Point Transformer V3](https://github.com/Pointcept/PointTransformerV3) / [Pointcept](https://github.com/Pointcept/Pointcept) — Xiaoyang Wu et al. (**pin v1.5.1**)
* [KITTI](https://www.cvlibs.net/datasets/kitti/) and [SemanticKITTI](http://www.semantic-kitti.org/)

This repository contains the integration, the measurement harness, and the failure analysis.
