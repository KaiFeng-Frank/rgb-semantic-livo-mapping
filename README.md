# rgb-semantic-livo-mapping

**A reproducible RGB-semantic mapper that sustains 10 Hz rosbag replay using a precomputed
FAST-LIVO2 trajectory, producing a world-frame point cloud with class, confidence and RGB coverage.**

FAST-LIVO2 supplies the pose. PTv3 supplies the semantics. The camera supplies the real colour.
ROS 2 Jazzy, KITTI + SemanticKITTI, RTX 4090.

The measured workflow has two passes: first run FAST-LIVO2 and save its TUM trajectory,
then replay the bag through the semantic/RGB mapper with that trajectory loaded at startup.
v0.6 also runs both concurrently on bag replay, the mapper fed by live FAST-LIVO2 poses
([below](#v06--online-integration-live-fast-livo2-poses)); that online mode is archived in
`experiments/v06/`, and `src/` still carries the two-pass node.

![semantic map](docs/img/rviz_class_final.png)

*The full 693 m loop of KITTI seq 07, coloured by predicted class. Cyan is `driveable_surface`,
yellow is `car`, magenta is `vegetation`, purple is `manmade`. The continuous yellow ribbon down the
middle of the road is not an error — it is one moving vehicle's accumulated trail, and the repo
measures exactly that (see criterion 5).*

---

## Why this exists

Semantic mapping stacks are easy to assemble and hard to trust. Every component "runs". The map looks
plausible. Nothing throws. And yet the result can be quietly, structurally wrong.

**Six of the failures documented here produce no error message at all.** One of them — a single
constant — deletes the road from the map while the model reports 0.93 mean confidence. Another makes
`load_state_dict(..., strict=True)` report `0 missing / 0 unexpected` on a model that has been
silently replaced. A third is a source-code *comment* that was wrong, and cost 15 ms per frame for as
long as anyone had been reading it.

This repo is the opposite bet: **a baseline where every number is checked against ground truth**, so
the interesting problems can be discovered instead of assumed.

---

## What it does today

| # | Acceptance criterion | Result |
|---|---|---|
| 1 | LIVO runs stably, trajectory matches upstream | **PASS** — ATE 0.880 m, drift **0.127 %** over 693 m; estimator core differs from upstream by 3.7 %, all mechanical ROS 1→2 changes |
| 2 | Mapper throughput with precomputed poses | **PASS** — saturated frame period **60.98 ms mean / 65.99 ms p95 ⇒ 16.4 Hz ceiling** against 10 Hz replay |
| 3 | Timestamps correspond, no frame mismatch | **PASS** — 1096 trajectory/image stamp pairs, **max &#124;Δ&#124; = 477 ns** |
| 4 | RViz shows a continuously accumulating world-frame cloud | **PASS** — 2.70 M voxels @ 0.20 m, real RViz2 captures |
| 5 | Dynamic objects do not smear from sync error | **PASS** — controlled experiment, see below |
| 6 | Full mapping replay, no crash / VRAM growth / sustained drops | **PASS** — **1092 / 1092 scans, 0 dropped**, VRAM flat |

**Semantic quality**, all 1101 frames of seq 07 against SemanticKITTI ground truth, zero-shot
(nuScenes-trained weights, never fine-tuned on KITTI):

| metric | value |
|---|---|
| point accuracy | **87.3 %** |
| coarse mIoU, 11 GT-present classes | **53.2 %** |
| coarse mIoU, 9 common classes | **61.0 %** |

Per-class IoU: car 92.9 · road 83.5 · manmade 81.9 · terrain 74.6 · sidewalk 68.7 · vegetation 68.3 ·
truck 39.6 · person 38.0 · motorcycle 31.5 · bicycle 4.5 · other_vehicle 1.9

*Both mIoU columns are given on purpose. The 9-class figure is the one usually quoted for a
nuScenes→KITTI transfer; the 11-class figure is what the ground truth on these frames actually
contains. Reporting only the first is how a transfer number gets quietly inflated.*

---

## Mapper replay throughput: 152.8 ms → 61.0 ms

| stage | before | after |
|---|---|---|
| PTv3 worker (incl. its numpy prep) | 90.57 ms | **56.43** |
| IPC | 5.73 ms (1.96 MB over stdio) | 10-byte control msg (`/dev/shm` ring) |
| camera projection | 10.20 ms | **4.14** |
| voxel-hash map insert | 26.52 ms | **22.19** |
| **frame period, saturated** | **152.83 / 178.64 p95** | **60.98 / 65.99 p95** |
| scans processed @ rate 1.0 | 738 / 1096 (67.3 %) | **1092 / 1092 (0 dropped)** |

These measurements cover the mapper with recorded poses already available. The saturated
frame period measures throughput; in-node latency, pose arrival and resource contention with
FAST-LIVO2 running beside the mapper are measured in [v0.6](#v06--online-integration-live-fast-livo2-poses).
`/semantic_scan` publishes each processed sweep unless disabled. `/semantic_map` uses
`--map-rate 1.0` by default, with adaptive throttling for expensive snapshots; its actual
publication rate can be lower. Scan processing and full-map publication have separate rates.

Accuracy across that change: point acc 87.319 → 87.311 %, mIoU(11) 53.065 → 53.180 %.
**All three deltas are inside the arms' own repeat spread.** The speedup is free.

Where it came from — note that the two biggest wins were *not* algorithmic:

* **Two-stage pipeline** — PTv3 on the GPU for scan *k* overlapping projection + map insert for scan
  *k−1*. Turns `sum(stages)` into `max(stages)`. Zero accuracy cost by construction: pose lookup is
  keyed on each scan's own stamp, so late processing is still correct.
* **Voxel prep moved to the GPU** — −15 ms. See trap #4; this one was blocked by a wrong comment.
* **fp16 weights + fp16 feats** — −12 ms. Not `torch.autocast`, which spconv rejects. The checkpoint
  was *trained* under fp16 autocast (`enable_amp = True`), so this is in-distribution, not a gamble.
* **Hilbert serialisation rewritten** — −7 ms. v1.5.1 ships a pure-Python bit loop; replaced with
  Skilling's transform on packed int64 lanes. Bit-identical over 2.2 M codes.
* **Exact CPU rewrites** — projection −6.1 ms, map insert −4.3 ms. Bit-identical on real frames.

---

## The six silent failures this repo documents

Full detail with commands in [`CRITICAL_CONSTRAINTS.md`](CRITICAL_CONSTRAINTS.md).

### 1. A constant that deletes the road

PTv3's nuScenes weights consume `strength` (LiDAR intensity) with **no normalisation anywhere in the
test pipeline**. Pointcept's nuScenes loader does `strength = intensity / 255`; KITTI intensity is
*already* in `[0, 1]`. Both are "legal" — the distributions differ by an order of magnitude.

Feeding KITTI intensity unscaled classifies **the entire road surface as `terrain`**, with mean
confidence 0.93 and no warning.

| strength scale | point acc | road IoU | vegetation IoU |
|---|---|---|---|
| 1.0 (raw KITTI) | 57.5 % | 7.8 % | 65.9 % |
| 0.0 (zeroed) | — | 98.6 % recall | **17.1 %** |
| **0.2** | **86.3 %** | **84.1 %** | **64.7 %** |

Zeroing is *not* the fix: it rescues the road and destroys vegetation, proving the channel carries
real signal and only its scale was wrong.

![intensity ablation](docs/img/semantic_intensity_ablation.png)

### 2. "strict load succeeded" is not "the model matches"

The released PTv3 checkpoint targets **Pointcept v1.5.1**. On HEAD, `PointTransformerV3.__init__` no
longer accepts `cls_mode`. Rename the config key and `load_state_dict(..., strict=True)` reports
**0 missing / 0 unexpected** — on a broken model:

| load path | point acc | coarse mIoU | car IoU | points predicted `car` |
|---|---|---|---|---|
| HEAD + key rename | 25.7 % | 7.9 % | **0.00 %** | **5** of 2.4 M |
| **v1.5.1** | **87.3 %** | **61.0 %** | **92.9 %** | 227 920 |

A model that finds 5 car points where ground truth has 6.5 % car is not suffering a domain gap; it is
a different model wearing the same weights.

**Rule: if you had to rename, remap or drop any key to make a checkpoint load, the load is not
evidence. Verify behaviour on data with ground truth.**

### 3. A right-multiplication that only shows up in corners

FAST-LIVO2's state is the **IMU** pose, so its trajectory is `T_{W←IMU}`, not `T_{W←LiDAR}`:

```
T_W_L = T_W_I @ T_I_L
```

Omitting it is invisible in ATE — a left-multiplied global alignment absorbs a *left* constant, not a
right one. It is also invisible while driving straight, where it is a pure global translation. It
only blurs the map when `R_W_I(t)` changes. Measured over a 113.3° turn:

| metric | with `T_I_L` | without | ratio |
|---|---|---|---|
| local surface thickness, median | **0.0803 m** | 0.1080 m | 1.35× |
| ground thickness, median | **0.1056 m** | 0.1659 m | **1.57×** |
| occupied 10 cm voxels | **1.086 M** | 1.420 M | 1.31× |

### 4. A comment that was wrong, and cost 15 ms a frame

Voxel preparation stayed in single-threaded numpy because of a source comment: CUDA's float64 divide
supposedly disagrees with numpy's. Measured over 22 real scans and **8 034 645 cell indices: zero
differing entries.** The comment is true of the *float32* form — which this code never used.

Moving the whole prep to the device: **−15 ms**, bit-identical on `coord`, `strength`, `grid_coord`
and `inverse`. An unverified comment had been load-bearing.

### 5. Launch-bound, not compute-bound — which kills the obvious lever

`torch.profiler` on the forward: **40.4 ms of device time inside a 56 ms forward, 1924 kernel
launches, 13.98 ms of CPU sitting in `cudaLaunchKernel`.**

Consequence: coarsening `grid_size` — the most obvious way to make a point-cloud network faster —
**buys almost nothing here**, because voxel count is not what the clock is spent on. This was
measured as a proper sweep (`opt/sweep_grid.sh`, 200 frames, on the fixed harness) and rejected on
the numbers, after an earlier sweep run under the broken intensity scale had already been voided.

Corollary, found in the same profile: **12.44 ms of "GPU inference" was two pure-Python bit loops**
computing Hilbert codes.

### 6. The measuring instrument was measuring the wrong model

`eval_report.py` imported `PTv3Segmenter` from `ptv3_infer.py`, whose `POINTCEPT_ROOT` defaults to
Pointcept **HEAD** at intensity **1.0** — exactly the pairing trap #1 and trap #2 identify as broken.
Run as shipped, it reports ~25.7 % and car IoU 0 **regardless of what is being tested**.

Any optimisation campaign run against that instrument would have "discovered" that every change was a
regression, and reverted the good ones. The instrument is now the deployed class itself
(`ptv3_worker.Segmenter`), not a parallel implementation of it.

### Bonus: know your noise floor before you trust a delta

The model is **non-deterministic run to run**. With `shuffle_orders` pinned off, two forward passes
over the same scan in the same process still disagree on ~4–5 % of points. Bisected with forward
hooks to `SerializedPooling`'s unstable `torch.sort(cluster)`; forcing `stable=True` did not remove
it, so at least one more source remains unidentified.

**On 20 frames the instrument's own spread is ~1 point of coarse mIoU — the same size as the
acceptance gate.** Every verdict in this repo is therefore taken over all 1101 frames with repeats,
and the spread is reported next to every delta.

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
removal: one tracked vehicle travels 23.0 m in 3.32 s while each sweep's footprint stays 8.39 m, so
the map keeps a ~31 m ribbon. Visible in `docs/img/rviz_dynamic_class_zoom.png`.

---

## Dynamic objects (v0.3): semantics is not a policy

A moving vehicle used to leave a ~31 m ribbon in the map. v0.3 removes it with **semantic candidate
gating + range-image free-space evidence + publish-time withholding** (no deletion, no ray casting).

The control arm is the point. "Just delete everything classified as a vehicle" is the obvious policy,
and it fails — *not* because the classifier is imperfect, but because the policy is wrong:

| arm | dynamic recall | **static false-kill** | damage ratio |
|---|---|---|---|
| no-op (v0.2) | 0 % | 0 % | — |
| naive class-delete, real PTv3 output | 97.2 % | **9.83 %** | 12.18 |
| **naive class-delete, PERFECT classifier** (analytic bound) | 100 % | **9.81 %** | 11.80 |
| **v0.3** | 60.1 % | **0.161 %** | **0.33** |

Even a zero-error classifier destroys 9.81 % of static points, because a parked car and a moving car
are the same class. seq 07 has 383 stationary car clusters against 51 moving ones. Ribbon: 2228 → 199
voxels (−91.1 %). Real-time gate met with 28 ms to spare.

**The precise negative — the false-kill floor is the trajectory, not the mechanism.** Swapping only
the pose source, identical config and frames:

| pose source | static false-kill | vehicle-parked FK | recall |
|---|---|---|---|
| FAST-LIVO2 (ATE 0.879 m) | 0.260 % | 2.454 % | 63.49 % |
| SemanticKITTI GT poses | **0.052 %** | **0.468 %** | 63.99 % |

**5.0×**, at unchanged recall. ATE 0.879 m is 4–9 voxels at 0.2 m — a map-vs-sweep visibility test
cannot beat the relative pose error between the sweep that wrote a voxel and the sweep that tests it.
That factor is bought in the pose layer, not in the carver.

Honest caveats: pooled recall is 60.1 %, not 95 % (truck 91.3 / bicyclist 67.7 / car 56.8 /
**person 17.1** — a pedestrian displaces 0.04–0.17 m per sweep, less than one 0.2 m voxel, so a
visibility test has no separation to work with). All three pre-registered gates were *just* missed
(0.161 vs ≤0.1 %, 1.504 vs ≤1 %, 91.1 vs ≥95 %) and were not relaxed afterwards.

One more instrument trap: conventional per-point mIoU **goes down** after dynamic removal
(59.42 → 58.20, 11-class), because a coarse label space scores a correctly-classified ghost as a true
positive. SemanticKITTI's moving-class ids are the only honest instrument here.

---

## "Why not project 2D segmentation onto the points?"

The obvious challenge, answered with a boundary instead of a defence.
**Full experiment, including the 14 concessions made to the 2D arm: [`docs/2d_vs_3d.md`](docs/2d_vs_3d.md).**

All 1101 frames of seq 07, common-9 label space, one mask built once and passed to both arms:

| arm | coverage | in-frustum mIoU | global mIoU (unlabelled = wrong) |
|---|---|---|---|
| **3D** — PTv3 | 100 % | 65.43 ±0.22 | **65.00 ±0.09** |
| **2D** — EoMT-L (Cityscapes 84.2) projected | **15.24 %** | **77.59** | 14.24 |
| **Hybrid** — 2D in frustum, 3D outside | 100 % | 75.89 | **66.60 ±0.04** |

**2D wins inside the frustum by 12.2 mIoU — 55× the 3D arm's own run-to-run noise.** It wins 8 of 9
classes (`person` 75.4 vs 39.8, `sidewalk` 87.9 vs 66.5). That is not a result to hide.

**3D wins the map, entirely on coverage, and the crossover is arithmetic:** at 93.74 % accuracy where
it speaks, the 2D arm would need to label 93.6 % of all points to match 3D globally. It labels 15.2 %.
The "but the camera sweeps as you drive" rebuttal gets a number too — over the whole trajectory only
**35.77 %** of occupied map voxels are *ever* seen by the camera.

**The structural weakness is geometric, and only there.** At depth discontinuities 2D drops 11.60
accuracy points against 3D's 2.73 (**4.2×**) — the only in-frustum subset 3D wins. At *semantic*
boundaries 2D still wins, so it is the projection, not the recognition.

The honest conclusion is **both**: the hybrid beats 3D alone on every global metric (66.60 vs 65.00
mIoU, 18× noise) at full coverage — for 3× the latency (335 ms serial / 253 ms parallel vs 83 ms) and
a second network. For a global map at 10 Hz, 3D is the backbone; the camera's job here is colour.

*Self-refuting bonus:* the preparation reasoned that KITTI must be upscaled 3.2× to focal-match
Cityscapes. Measured on a held-out sequence, 3.20× is **worse than native**; the optimum is 1.30×.
Following the reasoned default would have handicapped the 2D arm by ~2.3 mIoU.


## Architecture

```
Pass 1: rosbag -> FAST-LIVO2 (ROS 2 port) -> saved TUM trajectory T_{W<-IMU}
                                          keyed on true sensor time

Pass 2: rosbag replay                     saved TUM loaded at node startup
  /velodyne_points  10 Hz                      │
  /camera/image_raw 10 Hz                      │
  /camera/camera_info                          │
        │                                     │
   ┌────┴─────────────── stage A (executor thread) ──────────────┐
   │  parse, per-point time, pose gate, write /dev/shm slot      │
   │  submit to PTv3 coprocess (non-blocking)                    │
   └────┬────────────────────────────────────────────────────────┘
        │                          ▼  GPU, scan k
        │              PTv3 (nuScenes, Pointcept v1.5.1, fp16)
        │              intensity x 0.2, GPU voxel prep
        │              per-point class + confidence
        │                          │
   ┌────┴─────────── stage B (worker thread, scan k-1) ──────────┐
   │  confidence gate -> de-skew (128 bins) -> project to cam2   │
   │  -> T_W_L = T_W_I @ T_I_L -> voxel hash insert -> publish   │
   └─────────────────────────────────────────────────────────────┘
                       ▼
        /semantic_scan  every processed sweep
        /semantic_map   default 1 Hz setting, adaptively throttled
        PointCloud2, point_step 28
        x f4 | y f4 | z f4 | rgb f4 | class u16 | confidence f4 | has_rgb u8
```

FAST-LIVO2 is a **pose source only**, via the saved TUM file in this implementation.
`SemanticMapNode` constructs `TrajInterp(args.traj)` at startup; the pose lookup interpolates
that complete file. Its `/cloud_registered` is deliberately not consumed: in LIVO mode ~52 %
of those messages are empty and the rest carry only the camera-frustum subset of the scan.
v0.6 adds a live-pose mode (`--pose-topic`, causal query over the poses that have arrived),
archived in [`experiments/v06/`](experiments/v06/README.md).

**Fusion rule.** Class: confidence-weighted vote, `score[voxel][c] += softmax_max_prob`, argmax wins.
Confidence: winner's share of total vote weight. RGB: mean over observations that actually had camera
coverage; voxels with none get `has_rgb = 0` and a sentinel grey rather than a misleading black.
37.7 % of map voxels end up with RGB (a single sweep only sees 16.2 %).

---

## Reproduce

Requires ROS 2 Jazzy, an NVIDIA GPU, and ~50 GB for KITTI + bags.
The commands below run trajectory generation and semantic mapping in separate passes.

```bash
./setup_ros2.sh          # ROS 2 Jazzy desktop
./mkenv.sh               # conda env: torch + spconv + torch_scatter + flash-attn
./fetch_kitti.sh         # KITTI raw drives 0027 / 0016 + SemanticKITTI labels
./fetch_ptv3.sh          # Pointcept v1.5.1 + nuScenes PTv3 checkpoint
./build_bags.sh          # KITTI -> ROS 2 mcap   (use the _us bags, see below)

# Pass 1: generate and save the trajectory.
./run_fastlivo2_kitti.sh seq07 bags/kitti_seq07_us 0.5
# Pass 2: replay at sensor rate using the saved trajectory.
./opt/run.sh after3 src bags/kitti_seq07_us \
    out/kitti_seq07_fastlivo2_tum.txt 1.0 \
    --reliable --conf-gate 0.5 --expect-voxels 4000000
python3 opt/summ.py opt/out/stats_after3.json

rviz2 -d rviz/semantic_map.rviz      # toggle "Map Class" to switch views
```

Accuracy, on the shipped path, all 1101 frames:

```bash
python src/eval_report.py --nframes 1101 --stride 1 --tag AFTER \
    --shuffle 0 --half 1 --fast-voxel 1 --gpu-voxel 1 --fast-hilbert 1
```

Every numeric optimisation ships with a verifier rather than an argument — `opt/verify_fast.py`,
`opt/verify_voxel_fast.py`, `opt/verify_cpu_exact.py`, `opt/bisect_nondet.py`.

Two things that will bite you if skipped:

* **Use the `_us` bags.** FAST-LIVO2's `preprocess.cpp` reads `curvature = time / 1000 // ms`, i.e. it
  wants **microseconds**. A control run with seconds gives ATE **50.6 m** and a 25 %-short trajectory.
* **Export `FASTDDS_BUILTIN_TRANSPORTS=LARGE_DATA`.** With default transports a 2.7 MB `BEST_EFFORT`
  sample loses 63 % of scans even when the consumer is completely idle and the CPU is 99 % free.
  "Frames not received" is not the same as "the node cannot keep up".

---

## Known limits, stated plainly

* **Dynamic removal recalls 60.1 % of moving points**, and only 17.1 % for pedestrians — they move
  less than one voxel per sweep. Its false-kill floor is set by trajectory error, not by the mechanism.
* **RGB covers 37.7 % of map voxels.** The camera is a narrow forward frustum; the LiDAR is 360°.
* **The model is not bit-deterministic** and one source of it remains unidentified (see above).
  Confidence is meaningful, so the map is gated at `conf ≥ 0.5`.
* **The 10 Hz acceptance uses a precomputed trajectory.** v0.6 runs the online path on bag
  replay (live poses, FAST-LIVO2 alongside, in-node latency p50 98–108 ms) from its snapshot;
  live-sensor operation is untested, and FAST-LIVO2 itself varies run to run at real-time
  rate (ATE 0.888 ± 0.115 m over 15 runs).
* **Map metrics before the evaluation-unit analysis are per point.** Per map cell the same maps
  read 6–8 points lower; see [the evaluation unit](#the-evaluation-unit-points-are-not-where-the-map-is).
* **Per-point deskew is currently accuracy-neutral** on this data (within ±5 % on every sharpness
  metric). It is kept because camera projection needs per-point world coordinates.
* **Live `/semantic_map` is capped at 1 M published points** for the visualiser; the saved `.npz`
  keeps full resolution.

---

## v0.4 — camera-supervised adaptation, first results

**Camera pseudo-label fine-tuning improves outside-frustum mIoU-9 from 65.01 to
72.98 (+7.96 points)** on all 1,101 frames of SemanticKITTI seq 07. The adapted
student uses point clouds alone for semantic inference.

| Arm | Training supervision | Outside mIoU-9 ± SD | Δ vs zero-shot |
|---|---|---:|---:|
| A | Frozen nuScenes model | 65.01 ± 0.25 | — |
| B1 | Camera pseudo-labels, head only | 65.90 ± 0.22 | +0.88 |
| **B0** | **Camera pseudo-labels, full fine-tuning** | **72.98 ± 0.17** | **+7.96** |
| D | GT inside the camera frustum | 74.68 ± 0.15 | +9.67 |
| R | GT on random raw points | 82.75 ± 0.14 | +17.74 |
| R′ | GT on random surviving voxels | 83.36 ± 0.11 | +18.34 |
| C | GT everywhere, with KL | 85.94 ± 0.06 | +20.93 |

Each adapted arm is **one training run with three inference draws**; A has six
inference draws. ± is inference SD, not variation across training seeds. D/R/R′/C
are target-3D-GT diagnostic references. B0 is the camera-supervised method; its
outside-frustum mIoU-8 delta is +8.11 when `two_wheeler` is removed for decomposition.

The [original preregistration](docs/v04_preregistration.md) remains unchanged.
B0 exceeds the aggregate score threshold, but its predicted class-sign pattern
matches only **5/8** cells, so that mechanism explanation did not pass. R′ matches
D's total supervision after voxelisation (ratio **0.999920**) and scores 8.68 points
higher, with class composition and KL-region geometry still differing.

**Status, 2026-09-24:** the paired no-KL diagnostic is complete. D_noKL reaches
81.44 ± 0.55 outside-frustum mIoU-9 and Rprime_noKL reaches 84.90 ± 0.12; the
supervision-count gate passes at ratio 1.000000, and the gap falls from 8.68 to
3.46 points. The result remains exploratory because each arm has one training run
and the historical comparison uses different seeds. The existing 10 Hz mapping
result is from v0.2/v0.3; deployment of the fine-tuned student is a later gate.

See [full results, evaluation conditions and next steps](docs/v04_results.md),
[all metrics and per-class results](results/v04/completed_summary.md), and the
[experiment source snapshot](experiments/v04/README.md). Checkpoint selection uses
seq 08 GT; independent training repeats and evaluation on data unseen during
method development remain outstanding.

Verify the archived numbers without a GPU or dataset:

```bash
python3 tools/summarize_v04.py --check
```


## v0.5 — trained checkpoints in the mapper: the map beats the scan

**The fused map scores higher than the per-scan predictions it is built from, for every
checkpoint. The margin collapses as the classifier gets stronger, and the strongest
checkpoint loses road and sidewalk at map level.**

Three checkpoints go through the same mapper with one change, a `--ptv3-ckpt` selector;
without the flag the node loads the released weights exactly as before. Each trained
student is proven to be the same model before it is deployed: strict 488/488 load, every
tensor equal, argmax agreement inside the same-model repeat band, a different student as
the negative control.

| checkpoint | per-scan prediction | map, all evaluated points | live ROS map (one run) |
|---|---:|---:|---:|
| zero-shot (released nuScenes) | 65.01 ± 0.25 | 70.37 ± 0.37 | 72.00 |
| **B0** — camera pseudo-labels | 72.98 ± 0.17 | **77.49 ± 0.19** | 77.82 |
| R′ without KL — random target GT, reference | 84.89 ± 0.24 | 85.25 ± 0.14 | 85.54 |

*Out-of-frustum mIoU-9, seq 07, per point. A point that lands in no voxel counts as wrong.
Per map cell the order holds on the one draw measured so far: B0's map 70.97, per scan 68.74.*

The map wins by confidence-weighted voting over many views of each voxel, and voting can
only average away noise that exists. R′'s map scores below its own per-scan predictions on
road (96.9 → 95.3), sidewalk (93.5 → 90.5), car, terrain and manmade: the pipeline's
geometric cost — 0.20 m voxels on a 0.88 m-ATE trajectory — surfaces once per-scan noise is
gone. The camera-frustum split dissolves at map level: B0's in-minus-out accuracy gap goes
from −2.65 per scan to +0.44 in the map.

Latency follows the architecture, not the weights: saturated frame period 57.46 / 57.95 /
61.34 ms (zero-shot / B0 / R′, mean of three; one R′ run stepped mid-run and did not
reproduce), peak VRAM 1459 MiB for all three.

[v0.5 results](docs/v05_results.md) · [report](results/v05/REPORT.md) ·
[source snapshot](experiments/v05/README.md)

---

## v0.6 — online integration: live FAST-LIVO2 poses

**FAST-LIVO2 does not reproduce itself at real-time rate, and the map metric cannot see
it.** Fifteen rate-1.0 runs of the same bag give ATE RMSE 0.888 ± 0.115 m (min 0.660, max
1.115); v0.1's 0.880 m is one draw from that spread. Across 18 maps, a map's
own-trajectory ATE and its mIoU-9 correlate at r = 0.29.

Four arms separate the costs. B0, three paired repetitions, live map, out-of-frustum
mIoU-9 per point:

| arm | pose source | query | mIoU-9 |
|---|---|---|---:|
| OFF | offline trajectory, the v0.5 input | non-causal | 78.10 ± 0.08 |
| CAUSAL-ISO | the stream an online run received, replayed | non-causal | 77.88 ± 0.12 |
| **ON-opt** | live `/aft_mapped_to_init` | causal, constant-twist extrapolation | **77.96 ± 0.19** |
| ON-imu | live `/LIVO2/imu_propagate` | causal | 77.70 ± 0.14 |

* **The causal query costs nothing measurable.** ON-opt − CAUSAL-ISO on the same
  trajectory: +0.07 ± 0.18. Footprint: |dp| p95 2.6 cm, 4 % of points change voxel.
* **The online cost is a trajectory draw.** CAUSAL-ISO − OFF: −0.22 ± 0.16, one to two
  run-to-run standard deviations; the total, ON-opt − OFF: −0.15 ± 0.27, sits inside the
  spread of a single arm.
* **The IMU-propagated pose is worse.** ~71 Hz, not 100; its correction jumps grow the map
  by 6 % (2.86 M vs 2.71 M voxels) and cost 0.26 ± 0.25.
* **The FAST-LIVO2 ROS 2 port stamps every pose `now()`.** One line of glue gives the
  odometry its sensor time; the estimator is untouched.
* **A sweep's pose arrives ~44 ms after the sweep (p50), stamped mid-sweep.** Covering the
  sweep end would mean waiting 148 ms, longer than the 104 ms period.
* **`/semantic_scan` reaches a subscriber 56–81 % of the time under plain `LARGE_DATA`.**
  The tuned transport delivers 99.6–99.9 %, and then the 15 MB RELIABLE `/semantic_map`
  blocks: 30–50 of ~70 maps arrive, at intervals stretching to 27–38 s. The two topics need
  different transports, or the map needs increments. v0.5 never saw this: nothing
  subscribed during its timed runs.
* **The port's camera-parameter fetch is a 100 ms race.** `fastlivo_mapping` aborted at
  startup 4/4 under `LARGE_DATA` and 2/4 on the default transport; the v0.1–v0.5 trajectory
  came from a run that won it. Now 10 s.
* **No resource competition.** FAST-LIVO2 1.36 → 1.38 cores beside the node, PTv3 58.1 →
  58.5 ms, frame period 103.9 ms in every arm, in-node latency p50 98–108 ms. The online
  cost is in the pose chain and the transport, not the machine.
* **Holding the last pose is invisible where the node put the points** (77.87) and costs
  1.6–2.0 where they belong (76.18). A causal-pose change is read at both placements.

[Online integration](docs/v06_online_integration.md) · [report](results/v06/online/REPORT.md) ·
[source snapshot](experiments/v06/README.md)

---

## The evaluation unit: points are not where the map is

**The headline map metric weights every scored point equally, and 90.7 % of B0's scored
points sit in voxels within 10 m of the trajectory. Voxels observed ≤ 3 times are 42 % of
the map and hold 1.1 % of its points.** Scored per map cell, the same maps read 6–8 points
lower:

| map (out-of-frustum, one draw) | mIoU-9 per point | mIoU-9 per cell |
|---|---:|---:|
| zero-shot | 70.91 | 64.54 |
| B0 | 77.83 | 71.85 |
| R′ without KL | 85.88 | 78.17 |

*Both columns are given on purpose. Per point is the LiDAR-segmentation convention and the
unit of every earlier map number in this README; per cell is what a consumer of the map
gets, and how semantic scene completion benchmarks score. Map results carry both from here
on.*

**The residual is classification, not discretisation.** Preregistered anatomy of B0's map
errors, per point: 80.3 % DENSE — wrong labels on well-observed geometry — 17.3 % MIX,
voxel-boundary mixing that a perfect per-voxel classifier would also get wrong, and
2.5 % SPARSE. The voxel-majority ceiling is 94.82 for every model; B0 sits at 77.83.

**Whether a completion head has a target depends on the unit.** Labelling every sparse
voxel perfectly moves B0 from 77.83 to 78.40 per point, under the 0.60 decision band, and
from 71.85 to 84.12 per cell.

**Sparse cells fail at roughly the network's ordinary per-scan error rate.** The
preregistered test (two amendments) returned NEITHER on B0, and its out-of-sample CONTEXT
test passed on one of five arms; what follows is the reading of the numbers. Fusion is not
the cause:
9.2 % of B0's wrong sparse cells have split votes. Camera reach is not: 21 % were ever
camera-visible. Isolation is a tail effect: per-scan accuracy is 3–6 points lower where a
point's 8th neighbour is ≥ 0.5 m away, the same sign on all five unseen arms, past the
5-point bar on one. The lever is per-scan accuracy, which is supervision.

[The evaluation unit](docs/v06_evaluation_unit.md)

---

## Map-propagated pseudo-labels: the premise holds

Camera pseudo-labels cover a scan's frustum; the accumulated map remembers them. Voting the
filter-E labels through the map, with each point's own sweep left out, raises supervised
coverage of seq 07's evaluated points from **14 % to 68 %**, at **93.4 %** precision on the
propagated out-of-frustum points against **95.9 %** for per-scan labels inside the frustum.
The preregistered bar was twice the coverage at precision within 5 points.

| propagated label, out-of-frustum | precision |
|---|---:|
| road / sidewalk / terrain / manmade | 96.72 / 94.96 / 97.18 / 94.66 |
| vegetation | 87.08 |
| car / person | 85.41 / 90.11 |
| large_vehicle / two_wheeler | 64.90 / 51.48 |

Stuff propagates precisely; things do not. Measured on seq 07; a seq-09 replication is
running. It reads seq-09 ground truth to measure label precision only: the operating point
(k ≥ 2, majority ≥ 2/3) and the stuff-only policy were fixed on seq 07 before it ran, and no
design choice waits on it. The v0.6 protocol's stronger statement, that no design decision
has looked at seq 09, therefore no longer holds for the propagation premise; it is recorded
here rather than quietly kept. [Details](docs/v06_evaluation_unit.md#4-map-propagated-camera-pseudo-labels--premise-check-preregistered)

---

## Roadmap

**v0.1 — the honest baseline.** Four components, six criteria, every number against ground truth.

**v0.2 — mapper replay throughput. ✅ done.** 152.8 → 61.0 ms, a 16.4 Hz ceiling against 10 Hz replay, accuracy
unchanged. Along the way: the obvious lever (`grid_size`) was measured and rejected, and two of the
three biggest wins turned out to be a wrong comment and a pure-Python loop.

**v0.3 — make the map honest about time. ✅ done.** Dynamic-object handling scored against
SemanticKITTI's moving-class ids. Ribbon −91.1 %, static false-kill 0.161 % against the naive
control arm's 9.83 % — 61× — and 62× against that arm's *perfect-classifier* bound. The finding that
matters is the precise negative: the false-kill floor is the trajectory (5.0× better on GT poses),
not the removal mechanism.

**v0.4 — camera-supervised transfer. ✅ diagnostic stage complete.** Camera pseudo-labels
lift outside-frustum mIoU-9 from 65.01 to 72.98 with zero target 3D labels; the paired
no-KL diagnostic is complete. The [engineering gates](docs/v04_results.md#next-milestones)
it defined became v0.5 and v0.6.

**v0.5 — trained checkpoints in the mapper. ✅ done.** The fixed-trajectory gate. The map
beats the per-scan prediction for every checkpoint, the margin collapses as the classifier
strengthens, and the strongest checkpoint loses road and sidewalk at map level. Latency
unchanged.

**v0.6 — online integration, the evaluation unit, a held-out sequence. 🔄 training running.**
Online integration ✅ on bag replay: live FAST-LIVO2 poses, a causal query at no measurable
cost, no resource competition, timestamp and startup patches to the FAST-LIVO2 port.
Evaluation unit ✅: map results per cell beside per point; the residual is classification;
sparse cells fail at roughly the ordinary per-scan error rate, so the lever is supervision.
Training 🔄: seq 09 held out, B0 with and without the KL anchor on three seeds each, R′ as
the reference, scored for decision contamination, the KL anchor's cost, seed-versus-pass
spread, and whether v0.5's map findings survive on seq 09. Results pending;
[protocol](docs/v06_training_protocol.md).

**v0.7 — map-propagated stuff-label distillation. Design in progress.** Carry the camera
pseudo-labels through the accumulated map to points the camera never labels in their own
scan, for the stuff classes where propagation is precise. The premise holds on seq 07; the
seq-09 replication is running and decides nothing about the design.

**v0.8 — beyond the rotating scanner.** Non-repetitive solid-state patterns (Livox) break the
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
