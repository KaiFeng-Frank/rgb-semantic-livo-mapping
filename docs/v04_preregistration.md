# v0.4 DECISION RULE — pre-registered on the instrument, before any arm is trained

**Written 2026-09-22, after measuring arm A only. No arm has been trained. No training data was
touched. seq 07 was read, never written, never selected on.**

Instrument: PTv3 (Pointcept v1.5.1, nuScenes PTv3-m1 weights, strict 488/488, fp16,
intensity x0.2, `shuffle_orders=False`), zero-shot, SemanticKITTI seq 07, all 1101 frames,
common-9 label space, GT-EXCLUDED points dropped for every arm (P2). Arm A never abstains, so
abstain-excluded and abstain-wrong coincide and there is one accuracy number.

Spread is measured over **6 independent forward passes** of the identical frozen configuration on
the identical 1101 frames: `out/vs2d/ptv3_r{1,2,3}` (published v0.3) and `out/v04/ptv3_r{4,5,6}`
(fresh, this session). The evaluated-point set is bit-identical across all six (`n_eval` guard).
Between two draws 2.5–3.3 % of point labels change, the known unstable `torch.sort` in
`SerializedPooling`.

## Reproduction check (step 2 gate) — PASSED

| in-frustum | published v0.3 | this session (3 fresh draws) | diff of means |
|---|---|---|---|
| point acc % | 89.3910 ± 0.0025 | 89.3732 ± 0.0273 | −0.018 |
| mIoU-9 %    | 65.4324 ± 0.2200 | 65.2615 ± 0.1223 | −0.171 |

Both differences sit inside the published half-range. The instrument is the one that produced
89.39 / 65.43 ± 0.22. Proceeding.

## Arm A, 6-draw pooled

| subset | n_eval | point acc % | mIoU-9 % | mIoU-freq7 % |
|---|---|---|---|---|
| **out-of-frustum (PRIMARY)** | 105 671 053 | 87.4541 ± 0.0713 | **65.0125 ± 0.3198** (sd 0.2472) | 72.5820 |
| in-frustum | 20 215 061 | 89.3821 ± 0.0273 | 65.3469 ± 0.2641 (sd 0.1857) | 75.3260 |
| global | 125 886 114 | 87.7637 ± 0.0633 | 65.0648 ± 0.2725 (sd 0.2083) | 73.1010 |

Coverage 16.06 % in-frustum / 83.94 % out-of-frustum of evaluated points.

## ABSTAIN CELLS — declared in advance, uninterpretable for v0.4

| subset | class | IoU | half-range | sd | observed range |
|---|---|---|---|---|---|
| out-of-frustum | two_wheeler | 38.644 | **3.396** | 2.430 | 36.193 – 42.984 |
| in-frustum | two_wheeler | 21.076 | **2.636** | 1.867 | 18.991 – 24.263 |
| global | two_wheeler | 35.173 | **2.710** | 2.069 | 33.310 – 38.730 |

`two_wheeler` is the only class over the ~1-point line, in every subset. It alone contributes
**52.5 %** of the primary metric's standard deviation: drop it and the same metric is 2.11x
quieter (sd 0.247 → 0.117). The single highest arm-A draw on the primary metric (r4, 65.510) is
a two_wheeler outlier (42.98 against a 36.2–39.4 body), not a real shift.

**Consequence for P3, recorded before any number is seen: the pre-registered sign prediction for
two_wheeler is NOT TESTABLE in v0.4.** P3 is therefore an 8-cell test, not 9. `person` (0.945) and
`large_vehicle` (1.078) survive but are expensive cells — their predicted UP must clear ~1 IoU
point to count at all. Nothing else in P3 is at risk from noise.

## Per-class detection thresholds

Each arm is scored as the mean of its own independent draws. Arm B will get 3; arm A has 6. So
`se(Δ) = sd · sqrt(1/3 + 1/6) = 0.7071 · sd`, and a change counts at `3·se(Δ)`:

| class | sd | 3σ threshold | | class | sd | 3σ threshold |
|---|---|---|---|---|---|---|
| road | 0.070 | **0.15** | | terrain | 0.208 | **0.44** |
| car | 0.121 | **0.26** | | person | 0.446 | **0.95** |
| manmade | 0.122 | **0.26** | | large_vehicle | 0.508 | **1.08** |
| sidewalk | 0.138 | **0.29** | | two_wheeler | 2.430 | 5.16 → ABSTAIN |
| vegetation | 0.171 | **0.36** | | | | |

| aggregate | sd | 3σ threshold |
|---|---|---|
| **out-of-frustum mIoU-9 (PRIMARY)** | 0.2472 | **0.524** |
| out-of-frustum mIoU-8, two_wheeler removed (decomposition only) | 0.1174 | 0.249 |
| out-of-frustum point accuracy | 0.0509 | 0.108 |

## THE DECISION RULE

On the primary metric — out-of-frustum-only mIoU over the common-9 space, arm B minus arm A, each
the mean of at least three independent forward passes of its own — the run-to-run standard
deviation of the instrument is 0.247 mIoU points, giving a difference-of-means standard error of
0.175, so the noise band is **±0.52 mIoU points and I round it out to ±0.6 to cover the fact that
arm B's own draw-to-draw spread has not yet been measured and may exceed arm A's**: a change of
**+0.6 or more counts as PROPAGATION** (the 15 % of supervised points made the other 85 % better),
a change of **−0.6 or more counts as FORGETTING** (the distillation damaged what the frozen model
already knew out of frustum), and anything strictly inside **±0.6 counts as NEITHER** — which under
P4 is read as "a teacher signal of this magnitude is not enough", never as "distillation does not
work", and only arm D can tell those two apart. A verdict of PROPAGATION additionally requires
that the P3 sign pattern hold on the eight testable classes, each judged against its own per-class
3σ threshold above and not against zero; if the primary metric clears +0.6 while the sign pattern
disagrees, the pre-registered mechanism story is wrong and that disagreement is itself the
finding, to be reported and not retrofitted. `two_wheeler` in every subset is an abstain cell and
is quoted but never used to support or refute anything, and because it carries half the primary
metric's variance, every primary-metric verdict must be published next to the same delta computed
without it, as decomposition and never as the verdict. In-frustum and global deltas are reported
in the same table as decomposition only: at 15.24/84.76 coverage the teacher's +12.2 in-frustum
mIoU mechanically buys ~+1.8 global with zero propagation, so no global number may be quoted as
evidence for or against propagation.

## Out-of-frustum GT class distribution — where the primary metric is computed

| class | out-of-frustum pts | out % | in-frustum % | global % |
|---|---|---|---|---|
| manmade | 36 177 805 | 34.24 | 22.59 | 32.37 |
| road | 19 423 836 | 18.38 | 28.57 | 20.02 |
| vegetation | 17 652 670 | 16.71 | 17.28 | 16.80 |
| sidewalk | 16 145 411 | 15.28 | 10.40 | 14.50 |
| car | 9 436 225 | 8.93 | 13.35 | 9.64 |
| terrain | 5 614 732 | 5.31 | 6.25 | 5.46 |
| large_vehicle | 874 156 | 0.83 | 1.12 | 0.88 |
| **two_wheeler** | 214 722 | **0.20** | 0.26 | 0.21 |
| **person** | 131 496 | **0.12** | 0.17 | 0.13 |
| TOTAL | 105 671 053 | 100.00 | 100.00 | 100.00 |

No class is empty out of frustum, so mIoU-9 is well defined there and the primary metric never
silently becomes an 8-class mean. Two things to state now rather than after the fact. First, the
three classes the rare-class sampling is built for are exactly the three thinnest cells out of
frustum: person 0.12 %, two_wheeler 0.20 %, large_vehicle 0.83 % — together 1.16 % of the points
the verdict is computed on, yet one third of the mIoU-9 average. Second, the mix rotates across
the frustum boundary in a way that matters: out of frustum manmade grows 22.6 → 34.2 % and
sidewalk 10.4 → 15.3 %, while road shrinks 28.6 → 18.4 % and car 13.4 → 8.9 %. The primary metric
is therefore weighted towards precisely the two classes the recon measured as net-negative teacher
signal (terrain and manmade, the Cityscapes/SemanticKITTI taxonomy boundary), which is the
strongest available argument for their exclusion from the distillation loss in arm B — and it also
means that if the exclusion fails to hold them flat, the primary metric will show it loudly.

## Files

- `out/v04/armA_spread.json` — 3 fresh draws, full protocol + per-draw metrics (the scorer's own output)
- `out/v04/armA_spread_report.txt` / `.json` — 3-draw report as specified in the task
- `out/v04/armA_spread_pooled.txt` / `.json` — 6-draw pooled spread
- `out/v04/armA_noise.txt`, `out/v04/armA_thresholds.json` — noise decomposition and per-class thresholds
- `out/v04/ptv3_r{4,5,6}/` — the three fresh per-point caches
