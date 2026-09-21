# 2D projection vs 3D point-cloud semantics — the measured boundary

> **The challenge.** "You have a calibrated RGB camera. Why not run 2D semantic segmentation on the
> image and project the labels onto the points — the mature and usually more accurate approach —
> instead of segmenting the point cloud directly?"

It is a fair challenge and it deserves numbers, not an argument. This document is the experiment.

**Short answer: 2D wins inside the camera frustum, by a lot. 3D wins the map, entirely on coverage.
The honest conclusion is "both", and the cost of "both" is 3x the latency.**

---

## How the deck was stacked — against us, on purpose

A comparison like this is worthless if the losing arm was handicapped. Every contestable choice was
therefore made in the 2D arm's favour, and they are all listed here so a reader can audit it.

1. **The strongest practical model.** Two Cityscapes checkpoints were run: **EoMT-L** (Cityscapes val
   84.2 mIoU single-scale, CVPR 2025) and **Mask2Former Swin-L** (83.3). The stronger is quoted per
   metric. The best *published* Cityscapes number (~84.8 single-scale) needs mmcv-1.x custom CUDA ops
   that cannot coexist with this stack — so the 2D arm is ~0.6 mIoU below the best runnable model,
   and that is disclosed rather than hidden.
2. **A deliberately asymmetric domain gap.** Cityscapes → KITTI (German urban street, forward vehicle
   camera, daytime) is a far smaller shift than nuScenes → KITTI. The 2D arm gets the easier transfer.
3. **A taxonomy that fits the 2D arm.** SemanticKITTI's 19 classes were defined *after* Cityscapes'
   19, so the 2D vocabulary maps almost 1:1 onto the ground truth while nuScenes-16 does not.
4. **Input scale and TTA chosen by measurement, not by reasoning.** 38 configurations (2 checkpoints
   × 8 scales × {none, hflip}, plus multi-scale+flip at the best 3 scales of each) were swept on
   **seq 04** — same rig, same calibration day, a disjoint drive — then frozen and never revisited
   against seq 07. Tuning on the test sequence would have invalidated everything.
5. **Sub-pixel sampling at the network's own resolution**, not rounded onto the native pixel grid.
6. **Occlusion counts as abstention, not error.** A point the camera cannot see (z-buffer, 5×5 window,
   tolerance max(0.5 m, 2 %)) is the 2D arm declining to answer, not answering wrongly.
7. **The occlusion window is the physical value, not the flattering one.** 2 px is the HDL-64E's
   vertical sampling pitch in camera pixels. At 3 px the 2D arm would score *better* (94.26 / 78.74).
8. **`manmade` merges six Cityscapes classes** (building, wall, fence, pole, traffic light, traffic
   sign) because nuScenes-lidarseg has a single `static.manmade`. This **understates the 2D arm**:
   separating poles and signs is something it can genuinely do and the 3D arm structurally cannot.
9. **Ground-truth exclusions all remove points the 2D arm would have been charged for** (other-vehicle,
   parking, other-ground, other-structure, other-object — 5.76 % of all points).
10. **Both abstention conventions are reported, never one.**

The one thing *not* conceded: Cityscapes `sky` is counted **wrong**, not abstained. No LiDAR return is
sky, so abstaining there would let the 2D arm delete its own errors — the abstention leak that the
EXCLUDED/UNMAPPED split exists to close. It is worth 0.01 mIoU here, so it decides nothing.

**Instrument fairness was verified, not asserted**: a 49-check self-test on synthetic arms (oracle 2D,
oracle 3D, constant arm, a deliberate leakage probe) plus a post-hoc audit confirming every arm's
`n_eval` is bit-identical across all 17 subsets.

---

## Setup

SemanticKITTI seq 07 (KITTI raw `2011_09_30_drive_0027_sync`), **all 1101 frames**, identical frames
for every arm. Common-9 label space — car, large_vehicle, two_wheeler, person, road, sidewalk,
terrain, vegetation, manmade — every class natively expressible in SemanticKITTI, nuScenes-16 **and**
Cityscapes-19. Scored point set = (GT class not EXCLUDED); in-frustum columns add (in frustum).
**One mask, built once, passed to both arms.** `mean ± half-range` over 3 draws of the
non-deterministic 3D arm; the 2D arm is bitwise deterministic (verified over 40 frames).

---

## Table 1 — the result

| arm | coverage % | in-frustum acc % | in-frustum mIoU % | global acc % (ii) | global acc % (i) | global mIoU % (i) |
|---|---|---|---|---|---|---|
| **A. 3D** — PTv3 | 100.00 | 89.39 ±0.00 | 65.43 ±0.22 | 87.79 ±0.05 | **87.79 ±0.05** | **65.00 ±0.09** |
| **B. 2D** — EoMT-L projected | **15.24** | **93.74** | **77.59** | 93.74 | 14.29 | 14.24 |
| B2. 2D — Mask2Former-L projected | 15.24 | 92.08 | 76.09 | 92.08 | 14.04 | 14.42 |
| B′. 2D — EoMT-L, no occlusion test | 16.06 | 91.92 | 74.34 | 91.92 | 14.76 | 14.51 |
| **C. Hybrid** — 2D in frustum, 3D outside | 100.00 | 93.29 ±0.00 | 75.89 ±0.02 | **88.41 ±0.05** | **88.41 ±0.05** | **66.60 ±0.04** |

**(i)** unlabelled counted WRONG — *a map needs a label everywhere*.
**(ii)** unlabelled ABSTAINED, removed from the denominator — *judge it where it speaks*, the
convention that flatters the 2D arm. The 3D and hybrid arms never abstain, so (i) = (ii) for them.

### Per-class IoU, in frustum

| arm | car | large_veh | two_wheeler | person | road | sidewalk | terrain | vegetation | manmade |
|---|---|---|---|---|---|---|---|---|---|
| A. 3D PTv3 | 89.28 | 49.52 | 18.99 | 39.84 | 89.30 | 66.50 | 77.91 | 74.93 | 80.81 |
| B. 2D EoMT-L | 88.75 | **75.57** | 25.38 | **75.44** | **96.86** | **87.85** | **81.34** | **81.74** | **85.41** |
| B2. 2D Mask2Former-L | 87.99 | 68.31 | **59.53** | 64.89 | 95.59 | 77.44 | 65.14 | 81.67 | 84.20 |
| *GT support* | *13.4 %* | *1.1 %* | *0.26 %* | *0.17 %* | *28.6 %* | *10.4 %* | *6.3 %* | *17.3 %* | *22.6 %* |

The 12.2-point in-frustum mIoU gap is **55× the 3D arm's own run-to-run half-range**. It is real.
2D wins 8 of 9 classes; `car` is a tie inside the 3D arm's noise.

---

## Three findings

### 1. The coverage crossover is arithmetic, not rhetorical

The 2D arm answers for 15.24 % of scored points (16.06 % in frustum, minus 5.07 % lost to occlusion).
At 93.74 % accuracy where it speaks, **it would have to label 93.6 % of all points to match the 3D
arm globally.** It labels 15.2 %. The 3D arm answers for **6.56×** as many points.

The natural rebuttal — *but the camera sweeps the scene as the vehicle drives* — is answered with a
measurement rather than an opinion: over the whole 1101-frame trajectory, **only 35.77 % of the
2,584,582 occupied 0.2 m map voxels are ever seen by the camera at all**, and a voxel is in frustum
on just 18.49 % of the sweeps that observe it. (Computed with KITTI GT poses, so it is an upper
bound; a drifting trajectory can only make it worse.)

### 2. The structural weakness is geometric, and sits exactly where predicted

At LiDAR **depth discontinuities** — where a foreground pixel's label can slide onto a background
point — the 2D arm drops **11.60 accuracy points** (94.76 → 83.16) against the 3D arm's 2.73
(89.71 → 86.98). **4.2× worse, and the only in-frustum subset where 3D beats 2D.** Without the
z-buffer test the 2D drop is 21.70 points.

At **GT semantic boundaries** the 2D arm still wins (68.01 vs 58.58). So the failure is not about
recognising things near class borders — it is the projection itself. Both boundary sets are defined
without reference to any arm's predictions, so neither can be gamed.

### 3. Range: the camera does not get wronger, it goes quieter

The expected "a camera loses angular resolution with range" does **not** appear as an accuracy
collapse. Under (ii) the 2D arm degrades gently (95.61 → 90.69 from 0–10 m to 30–50 m) and **beats
the 3D arm in every range bin**. What collapses is *coverage*, because occlusion grows with range:

| 2D coverage within the frustum | 0–10 m | 10–20 m | 20–30 m | 30–50 m |
|---|---|---|---|---|
| | 99.18 % | 95.29 % | 88.93 % | 81.93 % |

So the (i)−(ii) gap inside the frustum widens from 0.8 points at 0–10 m to 16.4 points at 30–50 m.

---

## Latency (RTX 4090, exclusive, batch 1, fp16, 100 frames)

| arm | mean ms | p95 ms | Hz at mean |
|---|---|---|---|
| **A. 3D** PTv3 end to end | **82.7** | 86.7 | 12.1 |
| **B. 2D** EoMT-L @1.30 + hflip + project + z-buffer + sample | 252.6 | 272.6 | 4.0 |
| B. 2D EoMT-L @1.30, no TTA | 158.4 | 172.7 | 6.3 |
| **C. Hybrid, serial** (one GPU) | 334.6 | 355.9 | 3.0 |
| **C. Hybrid, parallel** (two streams) | 252.6 | 272.6 | 4.0 |

3D breakdown: read 0.7 · voxelize 19.2 · **forward 61.5** · devoxelize 1.3.
2D breakdown: read 11.5 · **forward 217.3** · project+z-buffer+sample 23.9 (timed separately so a
reader who disagrees can subtract it).

A second GPU does **not** rescue the hybrid: the 2D forward dominates the 3D forward on *every one*
of the 100 frames, so the parallel bracket equals the 2D arm's own cost.

---

## The sweep that refuted its own premise

The preparation reasoned, correctly-sounding, that since Cityscapes has fx ≈ 2262 px and KITTI's
rectified `image_02` has fx = 707.09, the same object subtends 3.2× fewer pixels than in training, so
the image must be upscaled ~3.2× to restore the trained regime.

**Measured on seq 04, this is wrong.** The focal-matched 3.20× is *worse than native* for both
checkpoints (EoMT 67.50 vs 68.93 mIoU). The optimum is a mild **1.30×**. Multi-scale TTA does not
beat single-scale hflip for either model.

Had the 2D arm been run at the reasoned default it would have been handicapped by ~2.3 mIoU — and
the headline in-frustum gap would have been quietly inflated in our favour.

---

## Threats to validity

- The input scale was frozen on **6 of 9 classes**: seq 04 GT contains no `person`, `two_wheeler` or
  `large_vehicle` points, so the 38-configuration argmax saw only car / road / sidewalk / terrain /
  vegetation / manmade.
- 9-class mIoU gives `two_wheeler` (0.26 % support) and `person` (0.17 %) the same weight as `road`
  (28.6 %). The two 2D checkpoints disagree wildly on `two_wheeler` (25.38 vs 59.53 IoU), which alone
  moves 9-class mIoU by 3.8 points.
- The `manmade` merge **understates the 2D arm**, as noted above.
- One sequence, one rig, daytime, dry, no night. The *coverage* results are geometric and transfer to
  any forward-camera + 360° LiDAR rig; the *accuracy* results are one drive.
- Both networks are zero-shot. This is the deployment question, not the ceiling question.
- Map coverage (35.77 %) uses GT poses and is therefore an upper bound.

---

## What this means for the system

Not "2D is wrong". The measured position is:

> Inside the frustum, 2D is 12.2 mIoU stronger. But it covers 15.2 % of points and 35.8 % of map
> voxels, and it degrades 4.2× at depth discontinuities. A hybrid takes both, at 3× the latency and
> a second network.

For a system whose output is a **global** semantic map at **10 Hz**, the 3D arm is the one that can
be the backbone, and the camera's real job is what it is already doing here: supplying per-point
colour. If the deployment target were "label what the camera sees, fast enough", the answer would
flip — and that is the point of measuring instead of asserting.

There is also a forward-looking reason this repo segments the point cloud: the questions it exists to
ask next — **how LiDAR semantics transfer across sensors, and how they degrade under non-repetitive
solid-state scan patterns** — do not exist at all if the semantics come from a camera.
