# Map-propagated camera pseudo-labels: premise check -- RESULT (seq 07)

PREREG: `/data/livo_sem/out/v06_propagate/PREREG.md` (sha256 fbbcd3cfed7cb29c..., written 2026-09-26 04:19:02). Tool: `tools/propagate_check.py`; run: `opt/run_propagate.sh`; numbers: `out/v06_propagate/result.json`.

Instrument: seq 07, FAST-LIVO2 trajectory `/data/livo_sem/out/kitti_seq07_fastlivo2_tum.txt`, 0.20 m cells, 128-bin deskew, no confidence gate; cells = keys of evaluated points (GT not EXCLUDED) with a valid pose. Filter E = `{"require_visible": true, "range_lt50": false, "teachers_agree": true, "conf_min": 0.9, "drop_depth_edge": false}` (from the B0 config, equal to `filterE_08.json` and to the PREREG). Majority rule: n_votes >= k AND 3*top >= 2*n_votes, after removing the point's own frame's votes (leave-own-frame-out).

## GATE 1 -- the per-scan reading is right: PASS

What was reproduced and why: filter E is NOT the 2D arm's operating point: the 2D arm answers every z-buffer-visible in-frustum point with the EoMT label and counts sky as an answer; filter E additionally requires EoMT == Mask2Former and EoMT p1 >= 0.90 and never keeps sky. PREREG: 'otherwise reproduce out/pseudo's own audit numbers'. The only out/pseudo audit numbers defined on seq 07 are the '07 (published)' row of out/pseudo/INVENTORY_NOTES.md (candidate % of all points, two-teacher agreement % of candidates).

| number | target | reproduced | abs diff | within 0.1 |
|---|---|---|---|---|
| candidate % of all points (out/pseudo seq-07 row) | 15.14 | 15.1432 (20229087 / 133585213) | 0.0032 | True |
| two-teacher agreement % of candidates (out/pseudo seq-07 row) | 94.0 | 94.0330 (19022018 / 20229087) | 0.0330 | True |

Also reproduced (not the gate; they check the same code path end to end):

| check | published | reproduced | result |
|---|---|---|---|
| 2D arm coverage % of evaluated points | 15.2442 (19190299 / 125886114) | 15.2442 (19190299 / 125886114) | exact |
| 2D arm in-frustum mIoU-9 (abstain excluded) | 77.5934 | 77.5934 | within 0.1 |
| recon joint table (9x10x2x3x20x5 counts, out/distil/signal_07.npz) | 20229087 candidates | sum |diff| = 0 | bit-identical |
| filter E on seq 07 from that table: kept / evaluated kept / correct | 18292379 / 17756157 / 17032490 | 18292379 / 17756157 / (see below) | exact |
| cell keys vs `map_eval --conf-gate 0 --save-map` | 2737213 voxels | 2737213 voxels; key set equal True, points/key equal True, evaluated/key True, in-frustum/key True | identical |

Label-source sensitivity (PREREG fixes the frozen label caches; the confidence caches carry their own argmax from a second GPU pass): argmax differs at 2114 / 21389101 sampled EoMT points and 11953 / 21389101 M2F points; filter E with the conf-cache argmax instead changes the kept set by 0 (EoMT), 2805 (M2F), 2805 (both) points.

Vote arithmetic: brute-force recount of 4000 random non-labelable points (all three vote readings, k = 1, 2, 3) against the vectorised pass: mismatches {"lofo": 0, "incl": 0, "r2": 0} -> OK.

## GATE 2 -- accounting: PASS

Every evaluated point is exactly one of {labelable in own frame, propagated-only, unlabelled}. Evaluated total 125886114 (published n_eval 125886114, equal: True).

| k | labelable own | propagated-only | unlabelled | sum | evaluated | closes |
|---|---|---|---|---|---|---|
| >=1 | 17756157 | 71088071 | 37041886 | 125886114 | 125886114 | True |
| >=2 | 17756157 | 68124873 | 40005084 | 125886114 | 125886114 | True |
| >=3 | 17756157 | 66202110 | 41927847 | 125886114 | 125886114 | True |

## Coverage / precision (%, all evaluated points of seq 07, GT = SemanticKITTI common-9)

| supervision | all: coverage / precision | in-frustum | out-of-frustum | supervised points |
|---|---|---|---|---|
| per-scan filter E | 14.10 / 95.92 | 87.84 / 95.92 | 0.00 / n/a | 17756157 |
| propagation k>=1 (own label, else LOFO cell majority) | 70.58 / 92.86 | 97.03 / 94.30 | 65.51 / 92.45 | 88844228 |
| propagation k>=2 (own label, else LOFO cell majority) | 68.22 / 93.65 | 96.37 / 94.50 | 62.84 / 93.40 | 85881030 |
| propagation k>=3 (own label, else LOFO cell majority) | 66.69 / 94.09 | 95.88 / 94.63 | 61.11 / 93.93 | 83958267 |

Propagated-only points (not labelable in own frame), leave-own-frame-out:

| k | in-frustum: count / precision | out-of-frustum: count / precision | all: count / precision |
|---|---|---|---|
| >=1 | 1859489 / 78.76 | 69228582 / 92.45 | 71088071 / 92.09 |
| >=2 | 1724236 / 79.79 | 66400637 / 93.40 | 68124873 / 93.06 |
| >=3 | 1627031 / 80.47 | 64575079 / 93.93 | 66202110 / 93.60 |

## Decision (PREREG, applied literally; primary operating point k >= 2, share >= 2/3)

- (a) propagated coverage of all evaluated points 68.221 % vs 2 x per-scan 14.105 % = 28.210 % -> **holds** (ratio x4.837)
- (b) propagated-only out-of-frustum precision 93.405 % (n = 66400637) vs per-scan filter-E in-frustum precision 95.924 % - 5 = 90.924 % -> **holds**

**DECISION: PREMISE HOLDS**

Smallest k at which (b) holds: 1.

| k | (a) prop coverage | x per-scan | (a) | (b) PO-out precision | threshold | (b) |
|---|---|---|---|---|---|---|
| >=1 | 70.575 | 5.004 | True | 92.450 | 90.924 | True |
| >=2 | 68.221 | 4.837 | True | 93.405 | 90.924 | True |
| >=3 | 66.694 | 4.728 | True | 93.928 | 90.924 | True |

Reading of (a): propagated coverage = (labelable-in-own-frame + propagated-only) / evaluated, i.e. the coverage of the propagation scheme whose accounting is GATE 2; the propagated-only share alone is reported next to it. Propagated-only share alone: k>=1 56.470 %, k>=2 54.116 %, k>=3 52.589 %.

## Per class

Precision by label class (%; n = points labelled as the class) and coverage by GT class.

| class | per-scan E in: prec (n) | per-scan E: GT coverage all | prop k>=2 all: prec (n) | prop k>=2: GT coverage all | PO k>=2 out: prec (n) | PO k>=2 out: GT coverage of non-labelable |
|---|---|---|---|---|---|---|
| car | 92.87 (2649017) | 20.53 | 87.40 (8696451) | 64.61 | 85.41 (5851078) | 55.01 |
| large_vehicle | 90.87 (188721) | 17.34 | 77.44 (414142) | 37.85 | 64.90 (211481) | 23.40 |
| two_wheeler | 68.75 (15164) | 5.53 | 55.46 (74358) | 45.62 | 51.48 (56403) | 38.55 |
| person | 95.85 (22747) | 15.32 | 92.06 (69160) | 46.26 | 90.11 (42257) | 35.04 |
| road | 99.20 (5625052) | 22.34 | 97.20 (23114441) | 90.57 | 96.72 (17357055) | 87.91 |
| sidewalk | 97.68 (1787548) | 9.96 | 95.08 (12627739) | 70.04 | 94.96 (10636040) | 66.61 |
| terrain | 98.55 (795964) | 13.26 | 97.19 (4119986) | 67.38 | 97.18 (3187605) | 63.07 |
| vegetation | 91.87 (2979240) | 13.72 | 87.73 (12338294) | 56.89 | 87.08 (8882083) | 49.36 |
| manmade | 95.35 (3692704) | 9.26 | 94.57 (24426459) | 61.74 | 94.66 (20176635) | 57.47 |

## Diagnostics (not part of the decision)

Propagated-only out-of-frustum precision by LiDAR range (k>=2):

| range m | non-labelable evaluated | propagated | coverage | precision |
|---|---|---|---|---|
| 0-10 | 74821168 | 52898537 | 70.70 | 93.66 |
| 10-20 | 22906177 | 11071929 | 48.34 | 92.35 |
| 20-30 | 5190453 | 1828122 | 35.22 | 93.18 |
| 30-50 | 2747952 | 601242 | 21.88 | 91.09 |
| 50+ | 5303 | 807 | 15.22 | 92.57 |

Per-scan filter-E in-frustum precision by range:

| range m | evaluated in-frustum | supervised | coverage | precision |
|---|---|---|---|---|
| 0-10 | 6926919 | 6542368 | 94.45 | 97.18 |
| 10-20 | 9259986 | 8144111 | 87.95 | 95.44 |
| 20-30 | 2803542 | 2222336 | 79.27 | 94.54 |
| 30-50 | 1224448 | 847234 | 69.19 | 94.55 |
| 50+ | 166 | 108 | 65.06 | 93.52 |

Propagated-only precision under two other vote readings (count / precision):

| k | LOFO (the instrument) out | incl. own-frame votes out | LOFO + GT-excluded voters out | LOFO + GT-excluded voters: prop coverage all |
|---|---|---|---|---|
| >=1 | 69228582 / 92.45 | 69234095 / 92.45 | 69429773 / 92.39 | 70.740 |
| >=2 | 66400637 / 93.40 | 66405719 / 93.41 | 66547088 / 93.36 | 68.343 |
| >=3 | 64575079 / 93.93 | 64580165 / 93.93 | 64707206 / 93.89 | 66.804 |

LOFO vote reach of non-labelable points (cells with >= 1 / 2 / 3 votes from other frames, before the 2/3 share test): in-frustum 83.58 / 78.06 / 73.60 %, out-of-frustum 68.03 / 65.34 / 63.46 %.

Points: {"evaluated": 125886114, "no_pose_frames_eval": 578892, "pose_invalid_in_posed_frames_eval": 111718, "gt_excluded_pose_valid": 7654284}. Cells: {"n_cells": 2319864, "evaluated_with_cell": 125195504, "evaluated_without_cell": 690610, "gt_excluded_pose_valid_in_evaluated_cell": 3046571}. Votes: {"votes_evaluated": 17670168, "votes_gt_excluded": 228946, "labelable_without_cell": 85989, "cells_with_votes": 779296}.

## Caveats (computed)

- filter-E labels on GT-EXCLUDED points: 536222 of 18292379 (2.93 %); never scored, and they vote only in the 'LOFO + GT-excluded voters' diagnostic, whose decision would read: PREMISE HOLDS
- evaluated points without a cell (sweep without pose, or point outside the trajectory span): 690610 (0.549 % of evaluated); they can only be labelled in their own frame
- stricter reading of (a) -- propagated-only share alone >= 2x per-scan: True -> decision would read: PREMISE HOLDS
- B0's loss additionally drops terrain and manmade (arm_B0.py exclude_classes); filter E itself does not, and the PREREG measures all nine classes

## Post-hoc views (appended after the decision was read; NOT part of the decision)

Computed from the per-class counts in result.json. The PREREG precision is point-weighted over all nine classes and that reading alone decides. These rows show which classes carry it; the same (b)-style test (propagated-only out-of-frustum >= per-scan in-frustum - 5) is applied to each row for information only.

| view | per-scan filter E in-frustum | PO out k>=1 | PO out k>=2 | PO out k>=3 | (b)-style test at k>=2 |
|---|---|---|---|---|---|
| point-weighted, 9 classes (the PREREG reading; decides) | 95.92 | 92.45 | 93.40 | 93.93 | holds |
| point-weighted, the 7 classes in B0 loss (terrain, manmade excluded) | 95.93 | 91.40 | 92.54 | 93.15 | holds |
| class-macro mean of per-class precision, 9 classes | 92.33 | 83.38 | 84.72 | 85.34 | fails |
| point-weighted, things only (car, large_vehicle, two_wheeler, person) | 92.64 | 81.27 | 84.43 | 86.17 | fails |

Per-class precision by label, per-scan filter E in-frustum vs propagated-only out-of-frustum at k>=2 (the full table is under Per class above):

| class | per-scan E in | PO out k>=2 | PO out k>=2 points |
|---|---|---|---|
| car | 92.87 | 85.41 | 5851078 |
| large_vehicle | 90.87 | 64.90 | 211481 |
| two_wheeler | 68.75 | 51.48 | 56403 |
| person | 95.85 | 90.11 | 42257 |
| road | 99.20 | 96.72 | 17357055 |
| sidewalk | 97.68 | 94.96 | 10636040 |
| terrain | 98.55 | 97.18 | 3187605 |
| vegetation | 91.87 | 87.08 | 8882083 |
| manmade | 95.35 | 94.66 | 20176635 |

Other caveats (not computed):

- Votes come from ALL other frames, past and future (offline label generation, as for training). A causal online variant would see only past votes and reach fewer points.
- Leave-own-frame-out removes only the query point's own sweep; adjacent sweeps vote. For out-of-frustum points the votes necessarily come from sweeps in which the cell was inside the camera frustum.
- Point-weighted numbers are dominated by near range: at k>=2, 52898537 of the 66400637 propagated-only out-of-frustum points lie within 10 m.
- One sequence, and it is the TEST sequence: this check read seq-07 GT. A replication on a train sequence that has a FAST-LIVO2 trajectory (seq 09: out/kitti_seq09_fastlivo2_tum.txt, per-point teacher signal in out/pseudo/09, raw order) would test transfer without touching the test set.
- Peak RSS of the run 8.0 GB (limit about 10 GB); wall 306 s after a 2 min 15 s map_eval control.
