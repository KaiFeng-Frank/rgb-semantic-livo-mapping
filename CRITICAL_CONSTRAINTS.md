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
