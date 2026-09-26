# CRITICAL CONSTRAINTS — read before touching the PTv3 track

Established by independent adversarial verification. Violating these produces a pipeline that
runs and looks plausible but is silently wrong.

## C1. PIN POINTCEPT TO v1.5.1 (commit 72a7993)
Upstream Pointcept states: "Released model weights are temporarily invalid as the model structure
of PTv3 is adjusted." The published nuScenes checkpoint was trained against Pointcept v1.5; HEAD is
v1.7.0 and the PTv3 module structure changed. Loading v1.5 weights into a v1.7 model either throws
on state_dict mismatch or — worse — loads with unexpected/missing keys silently.
ACTION: `git checkout v1.5.1` (or 72a7993) after cloning. Treat the config.py shipped ALONGSIDE the
checkpoint on HuggingFace (7497 bytes) as authoritative over anything in the repo's configs/.
ASSERT: load_state_dict(..., strict=True) must succeed with ZERO missing and ZERO unexpected keys.
If it does not, STOP and report — do not paper over it with strict=False.

## C2. THE DOMAIN SHIFT IS ON 3 OF 4 INPUT CHANNELS, NOT 1
The nuScenes config uses feat_keys=('coord','strength') with NO CenterShift, NO Normalize, NO
PointClip in the test pipeline. The network therefore consumes RAW SENSOR-FRAME XYZ. The checkpoint
learned nuScenes' sensor geometry: LiDAR mounted ~1.84 m above a flat ground plane, z up, gravity
aligned. Feed it geometry from a different mount and the ground plane lands at the wrong z and the
whole scene prior shifts.
  - KITTI HDL-64E sits at ~1.73 m. Close to nuScenes but NOT identical — quantify the ground-plane
    z in a real KITTI scan (fit a plane to the road points) and state it. If it is off by more than
    ~0.1 m from what nuScenes implies, apply a fixed z translation and SAY SO.
  - KITTI velodyne frame is x-forward, y-left, z-up. Confirm nuScenes' LIDAR_TOP convention matches
    before assuming no rotation is needed. State what you confirmed and how.

## C3. INTENSITY SCALE — THE 255x TRAP
KITTI .bin intensity is a FLOAT in [0, 1]. nuScenes 'intensity' is conventionally [0, 255].
If the checkpoint was trained on 0-255 and you feed 0-1, the strength channel is effectively dead
and you lose a quarter of the input.
ACTION: (a) print the actual min/max/mean/percentiles of intensity in a real KITTI .bin;
(b) find how nuScenes' loader in Pointcept v1.5.1 populates 'strength' (read
pointcept/datasets/nuscenes.py and any preprocessing script) and what range it produces;
(c) match them explicitly. Report both ranges and the scaling you applied.
DO NOT GUESS. This is a measurable fact in both codebases.

## C4. PROVE IT WITH THE CLASS HISTOGRAM, NOT WITH "IT RAN"
A KITTI urban scan MUST come out dominated by driveable_surface / manmade / vegetation / terrain /
sidewalk, with a modest car count. If the histogram is near-uniform, or collapses onto one or two
classes, the preprocessing is wrong. Ablate C2 and C3 (with/without the z shift, with/without the
intensity scaling) and report the histogram for each — that ablation is the deliverable, not an extra.

## C5. FRAME CORRESPONDENCE FOR THE GT CHECK
SemanticKITTI seq 07 == raw drive_0027 frames 000000..001100 (1101 labels, but the raw _sync dir
holds 1106 velodyne frames). seq 04 == raw drive_0016 frames 000000..000270 (271 labels vs 279 raw).
Align from index 0; the extra raw frames are at the TAIL and have no labels. Do not zip blindly.

## C6. NEVER UPLOAD FROM THE LOCAL MACHINE
All downloads originate here on the remote. Local bandwidth costs the user money; remote is free.
GitHub: prefix https://gh-proxy.com/ . HuggingFace: export HF_ENDPOINT=https://hf-mirror.com .

---
## MEASURED RESULT (2026-09-21) — C1 is CONFIRMED, not theoretical

Tested directly on this machine:

| Pointcept | result with nuscenes-semseg-pt-v3m1-0-base |
|---|---|
| HEAD 1342eda | `TypeError: PointTransformerV3.__init__() got an unexpected keyword argument 'cls_mode'` — cannot even construct. Also logs `[PointROPE] CUDA implementation unavailable`, i.e. the architecture changed. |
| **v1.5.1 (72a7993)** | **488/488 tensors, 0 missing, 0 unexpected, 0 shape mismatch, `strict=True` OK, 46.2M params** |

v1.5.1 is already cloned at /data/livo_sem/src/Pointcept_v151 .
A verified loader is at /data/livo_sem/src/ptv3_loader_verified.py — import `build_ptv3` from it.

Additional measured fact: `pointcept.models.__init__` eagerly imports every model and therefore drags in
CUDA extensions PTv3 never uses. Only TWO need stubbing: `pointops`, `pointgroup_ops` (plus pip install ocnn,
already done). Do NOT try to compile pointops/pointgroup_ops. The stub's `__getattr__` MUST raise
AttributeError for dunder names or `inspect` breaks with "type object '_Any' has no attribute 'endswith'".

---
## C3 RESOLVED — MEASURED (2026-09-21). The intensity channel must be scaled by 0.2

The "255x trap" was real but NOT in the direction first assumed. Pointcept's nuScenes loader already
normalises: `nuscenes.py:80  strength = points[:,3]/255  # scale strength to [0,1]`. KITTI intensity is
ALSO natively [0,1]. So both are "legal" [0,1] — but the DISTRIBUTIONS differ by an order of magnitude,
because real LiDAR returns concentrate at low raw values. KITTI: mean 0.293, p50 0.31, p90 0.45.
nuScenes after /255: concentrated near ~0.06. Feeding KITTI intensity unscaled poisons the prediction
silently — no error, no warning, just a wrong map.

Measured on SemanticKITTI seq07 GT, 5 frames (0/200/500/800/1100), per-point agreement over the
classes with an unambiguous nuScenes<->SemanticKITTI correspondence:

| strength scale | overall | road | sidewalk | building | car | vegetation |
|---|---|---|---|---|---|---|
| 1.0 (raw KITTI) | 56.5% | 5.1%  | 4.7%  | 93.5% | 93.8% | 77.4% |
| 0.0 (zeroed)    | 70.6% | 98.6% | 54.2% | 88.5% | 94.1% | 17.1% |
| 0.1             | 80.3% | 98.3% | 63.1% | 89.3% | 95.5% | 52.6% |
| **0.2**         | **87.6%** | **98.3%** | **63.3%** | **94.6%** | **95.6%** | **86.6%** |
| 0.3             | 86.2% | 94.9% | 57.3% | 93.6% | 94.4% | 84.7% |
| 0.5             | 76.3% | 69.6% | 27.7% | 93.5% | 94.6% | 86.0% |

USE strength_scale = 0.2 FOR KITTI. Do NOT zero the channel — zeroing rescues road but destroys
vegetation (86.6% -> 17.1%), proving the channel carries real information and only its scale was wrong.

RULED OUT by ablation (each tested, each made no difference or made it worse):
  - ground-plane height: z offsets from -0.70 to +0.21 m left driveable_surface pinned at 0.8-2.0%
  - point density: grid_size 0.10 / 0.15 / 0.20 all made road recall WORSE (0.8-1.6%)
  - range clipping to nuScenes-like +-50 m / +-40 m: no effect (3.7% -> 4.5%)

Other measured facts from the same run:
  - inference 70.4 ms mean / 71.3 ms p95 for a 122626-point KITTI scan (82655 voxels at grid 0.05)
    on the 4090, peak VRAM 0.56 GB -> 10 Hz is feasible.
  - The config's test pipeline uses a 10-way multi-scale+flip TTA. DISABLE it for real-time; the
    numbers above are single forward pass, no TTA.
  - Classes that transfer well even unscaled: building->manmade 97%, car->car 94%, vegetation->vegetation 87%.
  - GT fence is genuinely ambiguous: 55% vegetation / 26% barrier. Expected, not a bug.
  - raw drive_0027 frame N .bin and SemanticKITTI seq07 frame N .label have IDENTICAL point counts
    (122626 for frame 0) -> index correspondence confirmed, offset 0.

---
## C1 ESCALATED (2026-09-21) — "state_dict keys match" does NOT mean "the model matches"

A sibling track loaded the SAME checkpoint on Pointcept **HEAD** by renaming the config key
`cls_mode` -> `enc_mode` (HEAD renamed it). That made load_state_dict report strict OK,
0 missing / 0 unexpected — and the resulting model is BROKEN. HEAD's PTv3 is a different
architecture (PointROPE among other changes); upstream says so itself: "Released model weights are
temporarily invalid as the model structure of PTv3 is adjusted."

Measured side by side, identical 20 frames of seq07 (every 55th), identical coarse-class mapping:

| load path | point acc | coarse mIoU | car IoU | points predicted `car` |
|---|---|---|---|---|
| HEAD + cls_mode->enc_mode rename, x1.0 | 25.66% | 7.86% | **0.00%** | **5** |
| **v1.5.1**, x1.0 | 57.48% | 32.14% | 86.18% | 218400 |
| **v1.5.1, x0.2** | **86.34%** | **61.26%** | **91.22%** | 227920 |

Per-class IoU, v1.5.1 @ x0.2: car 91.22 / truck 40.17 / other_vehicle 12.58 / person 37.79 /
road 84.10 / sidewalk 69.14 / terrain 71.97 / vegetation 64.66 / manmade 79.68.

That track concluded from its numbers that this is a "nuScenes->KITTI sensor domain gap".
IT IS NOT. A model that predicts `car` on 5 points out of 2.4M, when GT is ~6.5% car, is a broken
model, not a domain gap. The domain gap is real but much smaller: coarse mIoU 61.26% zero-shot.

COROLLARY: that track also calibrated the intensity scale on the broken model and concluded 1.0.
A hyperparameter optimum found on a broken model carries no information. On the correct v1.5.1
model, 0.2 wins decisively over the same 20 frames (road IoU 7.84 -> 84.10, point acc 57.48 -> 86.34).

RULE: never accept "strict load succeeded" as proof of a correct pairing when you had to rename,
remap, or drop ANY key to get there. Verify behaviour on data with ground truth.

DO NOT USE /data/livo_sem/src/ptv3_infer.py (it targets HEAD).
USE /data/livo_sem/src/ptv3_loader_verified.py (it targets Pointcept_v151).

---
## REAL-TIME WORK (2026-09-21) — measured facts that change earlier conclusions

### R1. The old accuracy harness scored a DIFFERENT MODEL from the one deployed
`src/eval_report.py` used to `from ptv3_infer import PTv3Segmenter` and instantiate it
with defaults, i.e. Pointcept **HEAD** at intensity **1.0** — the exact pairing C1 marks
as broken. Run as-is it reports ~25 % point accuracy and car IoU 0 no matter what is
being tested. It now imports `ptv3_worker.Segmenter`, literally the class the node's
co-process runs. Anything scored before this fix against "86.3 % / 61.3 %" was scored
with an instrument pointed at the wrong model.

### R2. The pipeline output is NON-DETERMINISTIC run to run, and shuffle_orders is only
### half the reason
`weights/.../config.py:52` ships `shuffle_orders=True`; `build_ptv3` never overrode it,
and v1.5.1 applies it inside `forward()` ungated by `self.training`, so `.eval()` does
not disable it. `build_ptv3` now takes `shuffle_orders=` and the worker pins it False.
That is NOT sufficient: with shuffle pinned off, two forwards of the SAME scan in the
SAME process still disagree on ~4-5 % of points. Bisected with forward hooks to
`SerializedPooling` — `torch.sort(cluster)` (unstable on CUDA) picks a different
representative per pooled cell, so `serialized_code`/`serialized_order` differ from run
to run. Forcing `stable=True` there did NOT remove it, so there is at least one more
source; not chased further. CONSEQUENCE FOR MEASUREMENT: on 20 frames the instrument's
own run-to-run spread is ~1 point of coarse mIoU — the same size as the acceptance gate.
Use >= 100 frames for exploration and the full 1101 for a verdict, and always report the
repeat spread next to the delta.

### R3. The quoted "61.3 % coarse mIoU" is a 9-class mean, not an 11-class one
GT over the 20 canonical frames contains 11 coarse classes, not 9: bicycle (1579 pts)
and motorcycle (8077 pts) are present. The 9 IoUs listed under "C1 ESCALATED" average to
exactly 61.26. eval_report now prints both `coarse_miou` (all GT-present classes) and
`coarse_miou9` (that subset) so the comparison is like for like.

### R4. "CUDA float64 divide disagrees with numpy" is FALSE for this operation
The worker kept the voxel divide+floor in numpy on those grounds. Measured over 22 real
seq07 scans, 8 034 645 cell indices: `torch.floor(coord.double()/grid)` on cuda and
`np.floor(coord.astype(f8)/grid)` differ in ZERO entries. Both are IEEE-754
correctly-rounded doubles. The hazard is real for the FLOAT32 form (2904 of 122626 points
change cell) — which this code never used. The whole voxel prep is now on the device
(`src/ptv3_fast.py:voxelize_gpu`), verified bit-identical on coord/strength/grid_coord/
inverse over 22 scans: 10.0 -> 1.0 ms.

### R5. The forward is LAUNCH-bound, and 12.4 ms of it was a pure-python bit loop
torch.profiler, one warm frame, fp16: 40.4 ms of device time inside a 56 ms forward,
1924 kernel launches, 13.98 ms of CPU in cudaLaunchKernel. `Point.serialization` costs
13.76 ms, of which hilbert + hilbert-trans are 12.44 (z / z-trans are LUT-based, 0.43
each). v1.5.1's `serialization/hilbert.py:encode` runs `for bit: for dim:` with ~8
elementwise kernels per iteration over an (N,3,depth) byte tensor. `src/ptv3_fast.py:
hilbert_encode_fast` does the same transform on three packed int64 lanes with a Morton
spread for the interleave: bit-identical over 2.2 M codes (random coords, depths 10-13,
both axis orders, plus real scans), 7.6 -> 3.7 ms.

### R6. fp16 weights are accuracy-neutral and worth ~12 ms
The checkpoint was trained under fp16 autocast (`config.py: enable_amp = True`) and
flash-attention already casts qkv to fp16 inside every one of the 22 blocks, so fp32 mode
paid a round trip for nothing. `build_ptv3(half=True)`. Do NOT use torch.autocast
(spconv implicit_gemm rejects fp16 activations with fp32 weights).

### R7. Presize the map. The 415.7 ms max frame was ONE reallocation
`cap0 = 1<<21` rows and `VoxelHash(cap = 1<<22)` both grow at ~2.07 M voxels, and seq07
ends at 2.70 M — so both fire in the SAME frame, copying ~490 MB while rehashing 2.1 M
keys. `--expect-voxels` presizes both.

### R8. Over stdio, submit/collect CANNOT be split
A POSIX pipe holds 64 KiB; the request is 1.96 MB, so the write blocks until the worker
drains it and the worker only drains after finishing the previous frame. The payload now
rides a /dev/shm slot ring and the pipe carries 10-byte requests / 22-byte responses.

---
## v0.3 DYNAMIC OBJECTS (2026-09-21) — what is now fixed, and what is pose-limited

### D1. The deployed mechanism
Semantic candidate gate -> range-image free-space EVIDENCE -> publish-time WITHHOLDING.
`sem_core.FreeSpaceCarver` + `SemanticVoxelMap(dyn=True)` + `snapshot(k_free=...)`.
No voxel is ever deleted and `VoxelHash` is untouched; a withheld voxel keeps its score,
rgb and centroid and reappears the moment the evidence is reset.  `--dyn` off allocates
nothing and is BIT-IDENTICAL to v0.2 (opt/verify_v03.py V1).
FROZEN operating point: 64 x 150 image, margin 0.6 m, r in [3, 25] m, dilation (1,1),
reset_on_seen 0, k_free 24.  Every value sits on a measured curve in opt/out/sweep/.

### D2. SEMANTICS SELECT, GEOMETRY DECIDES — measured, not asserted
The naive "delete every point predicted potentially-movable" arm, run with the node's OWN
PTv3 argmax over 1086 frames: dynamic recall 97.25 %, static false-kill **9.83 %**,
parked-vehicle/person false-kill **93.75 %**, 12.18 static points destroyed per moving
point removed.  The analytic bound with a PERFECT classifier is no better: 100 % / 9.81 %
/ 96.59 %.  A class-only policy deletes 211 whole parked objects.  Never ship one.

### D3. THE FALSE-KILL FLOOR IS THE TRAJECTORY, NOT THE MECHANISM
out/seq07_ate.json: the deployed FAST-LIVO2 trajectory has **ATE RMSE 0.879 m** (max 1.84 m)
— four to nine 0.2 m voxels.  A map-vs-sweep visibility test cannot be more accurate than
the relative pose error between the sweep that WROTE a voxel and the sweep that TESTS it.
Measured directly, identical config and frames, only the pose source swapped:
    FAST-LIVO2 poses   static false-kill 0.260 %   vp 2.454 %   recall 63.49 %
    SemanticKITTI GT   static false-kill 0.052 %   vp 0.468 %   recall 63.99 %
i.e. 5.0x / 5.2x lower false-kill at unchanged recall.  Do NOT try to buy that factor back
by tuning the carver; it is bought in the pose layer.

### D4. reset_on_seen MUST BE OFF ON THIS PIPELINE
Dynablox's "ever-free = k CONSECUTIVE free observations" rule collapses here: with a 0.88 m
ATE the "still occupied" observation is itself unreliable, so one spurious re-occupation
wipes real evidence.  MEASURED at matched false-kill, reset_on_seen 1 leaves 1217 of 2228
ribbon voxels against 96 for reset_on_seen 0.  Cumulative evidence, not consecutive.

### D5. COARSER AZIMUTH IS SAFER, AND src/eval_report.py CANNOT SEE ANY OF THIS
Azimuth resolution at k_free 12, dil (1,1) — static false-kill vs ribbon-7 voxel reduction:
56 col 0.167 %/88.6 %, 75 col 0.189 %/92.5 %, 112 col 0.260 %/93.8 %, 150 col 0.327 %/94.3 %,
225 col 0.430 %/94.4 %, 450 col 0.646 %/94.6 %, 900 col 0.932 %/94.6 %.  A wider bin keeps a
nearer return and therefore cannot carve; fine azimuth is the dangerous direction.
src/eval_report.py scores per-point INFERENCE and never touches the map, and SK_TO_COARSE
maps 252 and 10 both to `car`, so a map-level change registers as EXACTLY ZERO there.  Its
three v0.3 repeats (87.247 / 87.280 / 87.334 % point accuracy) sit inside their own K4 spread
and prove only that the inference path is untouched.  Map-level accuracy is in
opt/replay_v03.py (`map_accuracy`): withholding moves map point accuracy 91.038 -> 90.999 %
and coarse mIoU 59.42 -> 58.20 (11 cls) / 68.18 -> 66.91 (9 cls) — it goes DOWN, because the
coarse label space scores a correctly-classified ghost as a true positive.  That is a property
of the metric, not a regression.

### D6. PEDESTRIANS ARE OUT OF SCOPE AND SAID SO IN ADVANCE
Per-moving-class recall at the frozen point: moving-truck 91.25 %, moving-bicyclist 67.66 %,
moving-car 56.78 %, **moving-person 17.12 %**.  A walking person displaces 0.04-0.17 m per
sweep, under one 0.2 m voxel, so a free-space test has no separation to work with.  Always
report recall PER MOVING CLASS; a pooled number hides this.

---
## v0.5 TRAINED MODELS IN THE LIVE PIPELINE (2026-09-24) -- measured facts, out/v05/REPORT.md

### V1. A DistilSegmentorMiB checkpoint is the released model structure plus a frozen anchor
exp/sk/arm*/model/model_best.pth holds 976 tensors (no `module.` prefix): backbone.* (486) + seg_head.* (2) =
the 488-tensor student, IDENTICAL key set / shapes / dtypes to the released checkpoint; frozen_backbone.* +
frozen_head.* = the fp16 anti-forgetting anchor (never deploy it); plus optimizer/scheduler/scaler state.
tools/extract_student.py keeps backbone.*+seg_head.* only -> weights/v05/<tag>_student.pth; build_ptv3(ckpt_path=)
loads it strict=True 488/488.  VERIFY BEHAVIOUR, NOT JUST KEYS (the C1 lesson): every tensor torch.equal to the
DistilSegmentorMiB student (fp32 and fp16), identical module tree, and on 101 seq07 frames the argmax agreement
extracted-vs-original equals the SAME-MODEL repeat agreement at every logit-margin stratum (a different student
drops to 0.90 / 0.92-0.96 at margin>2).  The forward is non-deterministic (R2): never demand bitwise equality.

### V2. Latency is architecture-bound, not weight-bound -- but one run stepped
Saturated (rate 2.0) frame period, 3 reps each: ZS 57.8/56.5/58.1 ms, B0 57.7/58.6/57.6, Rprime_noKL 65.5/58.1/60.4.
The 65.5 is a mid-run STEP in the worker stage (58.3 ms for the first 15 s, 68.2 ms after, peak VRAM 1413 instead of
1459 MiB) that the same checkpoint did not reproduce; charge it to the run, not the model.  Rprime always ran third
in each interleaved triplet (warmest GPU) -- rotate the order next time.  Rate 1.0: 1075-1086 processed, 6-7
backpressure drops ALL inside the first ~1.5 s (RELIABLE/KEEP_LAST(10) start-up burst), 0 afterwards.

### V3. The map is BETTER than the per-scan prediction it is built from (offline -> map, out-of-frustum mIoU-9)
zero-shot 65.01 -> 70.37 (+5.4), B0 72.98 -> 77.49 (+4.5), Rprime_noKL 84.89 -> 85.25 (+0.35); decomposition =
confidence-gate selection (+2.3 / +2.6 / +0.35) + confidence-weighted multi-view voting on the inserted points
(+4.4 / +3.5 / +0.7) - coverage of gated-out and no-pose points (-1.3 / -1.6 / -0.7).  The gain shrinks as the
classifier improves, and for Rprime the map LOSES on the flat/boundary classes (road 96.9->95.3, sidewalk 93.5->90.5,
manmade 93.2->92.4, car 98.2->97.3): that is the geometric cost of the pipeline (0.2 m voxels + 0.88 m ATE) and it is
only visible once per-scan noise is gone.  The in-frustum/out-of-frustum split dissolves at map level (B0 in-out
point-acc gap -2.65 offline -> +0.44 map).  Scorer: opt/replay_v05.py (extends replay_v03.map_accuracy; common-9;
its offline_all reading reproduces the frozen harness to 3 decimals per draw).  Live maps agree with the replays
except the two_wheeler abstain cell.

---
## v0.6 ONLINE INTEGRATION (2026-09-24) -- measured facts, out/v06/REPORT.md

### O1. FAST-LIVO2's ROS 2 topics carry NO sensor time.  Patched: one ROS-glue line
`/aft_mapped_to_init`, `/cloud_registered`, `/path`, `/mavros/vision_pose/pose`, `/rgb_img` are all stamped
`now()` (LIVMapper.cpp 1202/1264/1379/1397/1417/1436/1445); only the evo FILE uses `last_lio_update_time`.
An online consumer that de-skews cannot associate a `now()`-stamped pose with a sweep.  LIVMapper.cpp:1417
now stamps the odometry with `sec2Stamp(LidarMeasures.last_lio_update_time)`; the pose is the same post-LIO
state the evo file writes (the recorded stream's ATE equals the run's evo-file ATE).  Estimator untouched.
Backup: src/_pre_v06_backup/LIVMapper.cpp.

### O2. The port's camera-parameter fetch is a 100 ms discovery race.  Patched: 10 s
vikit `getRemoteParam` (rpg_vikit/vikit_ros/include/vikit/params_helper.h:117) waits 100 ms for the
parameter service of `parameter_blackboard`.  MEASURED (opt/fl_start_test.sh): fastlivo_mapping aborts at
startup with "Camera model not correctly specified" 4/4 under FASTDDS_BUILTIN_TRANSPORTS=LARGE_DATA and
2/4 on the default transport.  The offline trajectory of v0.1-v0.5 came from a run that won the race.
Rebuilt with `colcon build --packages-select vikit_ros fast_livo`.

### O3. The pose of a sweep exists only after the whole sweep + ~30 ms of LIO, stamped at the IMAGE instant
FAST-LIVO2's LIVO scheduler cuts the LiDAR stream at each image (mid-sweep, 0.48 of the way through); the
LIO update for sweep k needs the whole sweep (delivered at its end) and lands 44 ms (p50) / 63 (p95) / 96
(max) after the sweep arrives; stage B starts 64 ms (p50) after arrival.  So a causal consumer at 10 Hz
always has the FIRST half of a sweep covered and NEVER the second half: 100 % of sweeps extrapolate ~53 ms
(158 ms in the ~1 % where stage B beats the pose).  The sample covering the sweep END is the NEXT sweep's
update, 148 ms (p50) after arrival -- a wait-for-coverage policy is not affordable inside a 104 ms period.
Constant-twist extrapolation from the two newest samples costs |dp| p50 0.4 mm / p95 2.6 cm / p99 6.9 cm
(max ~1 m on the 158 ms cases), 4 % of points change 0.2 m voxel, +3 % voxels in the map, and NOTHING
measurable at the label level (ON-opt minus CAUSAL-ISO +0.07 +- 0.18 mIoU-9 out-of-frustum, paired).
HOLDING the newest pose instead costs |dp| p95 41 cm, 35 % of points change voxel, and 1.6-2.0 mIoU-9
when the map is read where the points should have been -- while scoring the SAME as cv at the node's own
placement.  RULE: the live label-consistency reading is blind to a coherent displacement of the sweep
tail; any causal-pose change must be read both at the node's placement and at the corrected placement
(opt/replay_v06.py `live_lookup_causal` vs `live_lookup_interp`) and with the geometric footprint.

### O4. `/LIVO2/imu_propagate` is off by default and delivers ~71 Hz, not 100
`uav.imu_rate_odom: false` -> `imu_prop_enable` false -> the topic never publishes.  With
`-p uav.imu_rate_odom:=true` it publishes from a 4 ms wall timer whenever a new IMU sample arrived (stamp
spacing p50 10 ms but ~8000 samples in 114 s: ~29 % of the 100 Hz IMU is coalesced), stamped with the IMU
time, re-based on every LIO update.  It covers the
sweep end at delivery (nothing extrapolated, ~7.4 samples per sweep) but its correction jumps inflate the map
by 6 % (2.86 M vs 2.71 M voxels for the same scene) and the map scores 0.26 +- 0.25 mIoU-9 below ON-opt.
Use the optimised stream + constant-twist extrapolation, not the IMU stream.

### O5. FAST-LIVO2 is NOT reproducible at real-time rate
15 rate-1.0 runs of the same bag (alone and beside the node): ATE RMSE 0.888 +- 0.115 m, min 0.660, max
1.115; pairs of runs diverge by up to 1.9 m in the shared W frame.  The offline 0.880 (rate 0.5) is ONE
draw from this spread; running beside the node does not shift it.  The map metric does not see it
(r = 0.29 between a map's own-trajectory ATE and its mIoU-9 over 18 maps) -- the label-consistency reading
is blind to global trajectory error in this range.  CONSEQUENCE: any online-vs-offline map comparison needs
>= 3 paired repetitions; "the online trajectory's cost" (CAUSAL-ISO minus OFF: -0.22 +- 0.16 live,
-0.16 +- 0.04 replay, out-of-frustum) is a draw, and a single-run comparison is a lottery ticket.

### O6. /semantic_scan does not reach a subscriber under plain LARGE_DATA, and the fix breaks /semantic_map
3.2 MB samples at 10 Hz on `FASTDDS_BUILTIN_TRANSPORTS=LARGE_DATA` deliver 56-81 % (15 runs) in bursts of
losses up to 1-4.7 s, for BEST_EFFORT and RELIABLE alike, writer depth 1 or 10, with or without the 15 MB
map publishes (out/v06/runs/analysis_smoke_*.json) -- the transport's fragment/socket budget, not the QoS.
`LARGE_DATA?max_msg_size=5MB&sockets_size=20MB&non_blocking=false` (every process) delivers 99.6-99.9 %
with interval p99 ~150 ms at +1.5 ms per publish -- but then the RELIABLE/TRANSIENT_LOCAL 15 MB
/semantic_map sample blocks: 30-50 of ~70 maps reach the subscriber and delivered intervals stretch to
27-38 s.  The two topics need different transports, or the map must be sent in increments.  v0.5 never saw
any of this because nothing subscribed during its timed runs (and its ~4.3 ms publish cost was the cost of
serialising into an unmatched writer).

### O7. Start-up: `ros2 bag play` publishes before discovery.  Use `-d 3`
The v0.5 runs lost their first ~10 sweeps to subscription matching (recv 1091/1101), and FAST-LIVO2 loses
its first IMU second the same way (the no-`-d` smoke's first pose came 1.9 s after the offline one).  With
`-d 3` every v0.6 run received 1100-1101 of 1101 sweeps and FAST-LIVO2 initialised at the same bag time as
the offline run (first pose 1317386426.0758).

### O8. No resource competition on this 12-core box
FAST-LIVO2 1.36 cores alone vs 1.38 beside the node (LIO p50 28.0 vs 29.8 ms, p95 42.5 vs 46.0, never above
104 ms); node 0.61 -> 0.63 cores, PTv3 worker 0.55 cores, PTv3 time 58.1 -> 58.5 ms; system CPU 11 % ->
23 %; GPU util ~19 %, memory exclusively the worker's 1459 MiB; zero back-pressure or staleness drops in any
ON run; period 103.9 ms in every arm; in-node latency (sweep arrival -> /semantic_scan published)
p50 99-108 / p95 110-125 / p99 126-150 / max 159-206 ms in every arm.  The online cost is in the pose
chain and the transport, not in the machine.

### O9. `pkill -f` self-kill
A shell whose own command line contains a pattern handed to `pkill -f` kills itself -- the `[p]attern`
trick only protects the pkill command, not the enclosing `bash -c "..."` or ssh command line.  Three ssh
sessions died that way.  Only pkill from a script FILE whose cmdline does not contain the patterns.

### O10. Default path unchanged, verified bit for bit
opt/verify_v06.py: the v0.6 node without `--pose-topic` and the v0.5 node (src/_pre_v06_backup) in one
process, same cached predictions (PTv3 stub), 40 real sweeps + images: every map array, the hash table,
the written .npz and the non-timing stats fields identical (19/19 PASS, logs/verify_v06.log).

## H1 (2026-09-25) bags_ros1/ deleted to make room for the v0.6 queue
The four ROS1 .bag files (seq04, seq04_us, seq07, seq07_us, 11 GB) were removed while the
v0.6 training queue was running: /data had fallen to 24 GB and the queue still needed
~17.5 GB (4 training jobs + 32 prediction caches).  They are reproducible from
data/raw/2011_09_30/ at any time; the ROS2 bags in bags/ are built independently from the
same raw drives by src/kitti_to_ros2bag.py, not converted from these, so nothing in the
v0.1-v0.6 chain reads them.  Only src/gen_handoff.py mentions the path, for documentation.
NOT touched: out/v04/ (holds the B0_r1..r3 prediction caches that every v0.5/v0.6 replay
reads).

## H2 (2026-09-26) KITTI raw drive archives deleted
The ten *_sync.zip archives (~92 GiB) were removed after verifying each was already
extracted under data/raw/<day>/<drive>_sync/ (frame counts checked) and that
data/pointcept_sk symlinks into data/raw, not into the archives.  Re-downloadable from the
KITTI S3 bucket.  Kept: the calib zips, data_odometry_labels.zip, data/raw itself.

## T1 (2026-09-26) NEVER resume a DistilSegmentorMiB run through Pointcept's CheckpointLoader
MECHANISM.  Pointcept v1.5.1 CheckpointLoader.before_train (src/Pointcept_v151/pointcept/engines/hooks/misc.py
lines 227-236) adds "module." only when world_size > 1, but whenever world_size == 1 it does `key = key[7:]` on
EVERY key, prefixed or not, and loads with strict=False.  Our checkpoints have no "module." prefix (V1), so on
this one-GPU box: frozen_backbone.X -> backbone.X (the fp16 anchor = the RELEASED weights overwrite the student
trunk), backbone.X -> e.X, seg_head.X -> d.X, frozen_head.X -> head.X (490 keys dropped as unexpected), and the
student head keeps the released head the model constructor loaded.  477 keys are reported missing = 473
frozen_backbone + 2 seg_head + 2 frozen_head (the other 13 frozen_backbone num_batches_tracked are filled silently
by BatchNorm's version-compat path -- hence 477, not 490).  Epoch, best_metric_value, optimizer, scheduler and
scaler are then restored correctly, so the run LOOKS resumed.  `weight=<ckpt>` without resume=True does the same
load.
WHICH RUN.  Job 5 of the v0.6 queue, armB0_noKL_s2, resumed 2026-09-25 23:35 from its epoch-9 model_last.pth by
opt/train_queue_v06_resume.sh (`--options resume=True weight=...`).  Its epoch 10 trained the released model.  At
00:15:04 CheckpointSaver overwrote model_last.pth AND model_best.pth with that state, so the epoch-9 checkpoint and
the seed's own best checkpoint no longer exist; both files now hold epoch 10, best 0.6179504518350944 -- the
released model plus one epoch, not seed 2 of B0_noKL, never to be scored as such.  queue.log's "job 5
armB0_noKL_s2  OK   40 min  Best mIoU: 0.6180" is this run.  Clean from-scratch rerun: opt/rerun_B0_noKL_s2.sh ->
exp/sk2/armB0_noKL_s2_clean.
HOW TO TELL FROM A LOG.  (1) After `Loading weight at:` a line `misc.py line 240 ... Missing keys: [...]` with a
non-empty list (477 entries; out/v06_train/armB0_noKL_s2.resume_ep9.log line 220).  A correct run logs `No weight
found at: None` there -- every other v0.6 training log does.  (2) The first step of the resumed epoch sits at the
EPOCH-1 level, not where the previous epoch ended: `Train: [10/10][1/7818] loss: 0.8021 ce: 0.5367 lovasz: 0.2654`
vs epoch 1 step 1 `0.8018 / 0.5366 / 0.2653` and ~0.07 at the end of epoch 9; epoch train mean 0.1010 (ep 9) ->
0.3868 (ep 10); val mIoU 0.5090 (ep 9) -> 0.6180 (ep 10).
ALSO MEASURED -- inherent to ANY resume, not to the defect: default_setup re-seeds at start, so the first resumed
epoch replays EPOCH 1's sample order and augmentations exactly (job 5's epoch-10 rare_class_audit counts equal
epoch 1's -- 160763879 supervised points, every class -- and no other epoch's).
FIX.  tools/train_distil_v2_resume.py --resume-from <model_last.pth>; tools/train_distil_v2.py is unchanged for
fresh runs (.pre_resume = its byte copy).  opt/train_queue_v06_resume.sh resumes only through it (.pre_fix = the
old script) and fails any job whose log contains "Missing keys".  The tool refuses --options resume= / weight=,
forces cfg.resume=False, cfg.weight=None (the hook loads nothing), builds the trainer with TRAINERS.build and
restores BEFORE .train(), i.e. before every before_train hook: state_dict strict=True ("module." removed only if
EVERY key carries it; a mix is refused), then every tensor torch.equal to the same-name checkpoint tensor;
optimizer / scheduler / scaler after group-size, moment-shape and total_steps checks; start_epoch and
best_metric_value from the checkpoint; a KL arm's calibrated kl_lambda from <run>/rare_class_audit.jsonl (it is
NOT in the checkpoint -- left alone, the first resumed step re-calibrates it on the trained student); train.log
appended, not truncated; config.py not re-dumped, and the current config must equal it.
VERIFIED ON CPU (tools/verify_resume_restore.py: the tool's own build_resumed_trainer, then the hooks'
before_train, then an independent load of the checkpoint).  resume_restore_verify.txt (this file): 0 missing /
0 unexpected, 976/976 tensors equal by name, trunk = checkpoint backbone.* 486/486 (backbone.* differs from
frozen_backbone.* in 485/486), optimizer 449 entries at step 78148, scheduler and scaler equal, start_epoch 10,
best 0.6179504518350944.  resume_restore_negctl.txt (Pointcept's path, same file): 477 missing / 490 unexpected,
trunk = frozen anchor 486/486, head = released head, while optimizer / scheduler / epoch / best all "restore".
resume_restore_guards.txt: 18/18 refusals.  resume_restore_verify_armB0_noKL_s3.txt and _armB0_s1.txt: clean
10-epoch checkpoints (trunk vs anchor differ 486/486), incl. the KL arm's kl_lambda 0.6693611546824091.
Never pass resume= / weight= to a DistilSegmentorMiB run.
IN THIS REPOSITORY.  experiments/v06/source/opt/train_queue_v06_resume.sh is the OLD script, as it ran on
2026-09-25 (the host's .pre_fix); do not resume with it.  The resume tool and its verifier are in
experiments/v06/source/tools/, the five resume_restore_*.txt outputs in results/v06/training/.

## P1 (2026-09-26) sequences/<seq>/poses.txt is SemanticKITTI's, not KITTI's odometry ground truth
fetch_kitti.sh never fetched data_odometry_poses.zip; the poses.txt under data/odometry/dataset/
sequences/ comes from SemanticKITTI's labels archive. Against KITTI's own poses (now in
data/odometry_gt/dataset/poses/) the per-frame position differs by 0.48 m mean on seq 07, 0.68 m on
seq 04, 2.08 m on seq 09 and 13.8 m (max 17.5 m) on seq 08. Trajectory metrics scored against
poses.txt (v0.1 ATE 0.880 m, the v0.3 pose-source ablation, the 2D-vs-3D coverage bound) are
therefore against a SLAM-quality reference; the same seq-07 trajectory scores 0.837 m against
KITTI's ground truth (docs/fastlivo2_reproduction.md).
