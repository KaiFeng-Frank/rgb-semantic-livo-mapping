# The evaluation unit: what the map metric measures, and where the residual lives

Updated 2026-09-26. Four analyses of the v0.5 seq-07 maps (FAST-LIVO2 trajectory,
confidence gate 0.5, 0.20 m voxels), in the order they were run. Each one was
preregistered before its numbers existed; the per-cell check was preregistered after the
point-weighted verdict had been read, and is labelled post-hoc. The preregistrations,
amendments and reports are published unchanged apart from path sanitising and one marked
redaction ([manifest](../results/v06/snapshot_manifest.json)).

| # | question | files |
|---|---|---|
| 1 | Would a point-cloud completion head have a target in this map? | [`results/v06/anatomy/`](../results/v06/anatomy/) `PREREG.md`, `REPORT.txt` |
| 2 | Does the answer survive when every map cell counts once? | [`results/v06/anatomy/`](../results/v06/anatomy/) `PREREG_voxel.md`, `REPORT_voxel.txt` |
| 3 | Why are sparse cells wrong? | [`results/v06/sparsediag/`](../results/v06/sparsediag/) `PREREG.md` (A1, A2), `REPORT.txt`, `REPORT_confirm.txt` |
| 4 | Does propagating camera pseudo-labels through the map give more supervision at comparable precision? | [`results/v06/propagate/`](../results/v06/propagate/) `PREREG.md`, `REPORT.md`, `result.json` |

Arms, all on seq 07: ZS = zero-shot, B0 = camera pseudo-labels (deployable), RP = R′
without KL (random target GT, reference); B1, D, C, R as in [v0.4](v04_results.md). `_r1`,
`_r2` are independent inference draws of the same checkpoint.

## 1. Residual anatomy — point-weighted, preregistered

Recommendation under test: "do not add a completion head", because the remaining
structural bottleneck is 0.20 m voxel mixing at class boundaries, which completion does
not touch. The analysis was built to be able to refute that.

**Instrument.** `tools/residual_anatomy.py` is `opt/replay_v05.py` verbatim plus one
readout block. Negative control: its `map_all_lookup` must equal the v0.5 json bit for
bit. Accounting: every error must be classified, and the headline confusion matrix must be
rebuilt exactly from the anatomy's histograms. Both hold for all four arms.

**Definitions** (per voxel r; m_r = GT majority of all points scored in r):

- **MIX** — model error on a point whose GT ≠ m_r. A perfect per-voxel classifier gets it
  wrong too: irreducible at 0.20 m.
- **CLS** — model error on a point whose GT = m_r; reducible by a better classifier. Split
  into **SPARSE** (n_obs ≤ 4 confident observations over the whole drive; sensitivity 2
  and 8) and **DENSE** (n_obs > 4: well-observed geometry, wrong label).
- **ceiling** — mIoU when every voxel is labelled m_r.
- **sparse-oracle** — mIoU when only the sparse voxels are relabelled m_r: the most a
  completion head could buy at map level.

Out-of-frustum, the headline subset, sparse := n_obs ≤ 4:

| arm | map mIoU-9 | ceiling | sparse-oracle | errors | MIX | SPARSE | DENSE | floor |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ZS_r1 | 70.91 | 94.82 | 71.54 | 10566028 | 14.6 % | 2.2 % | 83.2 % | 2167321 |
| B0_r1 | 77.83 | 94.82 | 78.40 | 8778422 | 17.3 % | 2.5 % | 80.3 % | 2166978 |
| B0_r2 | 78.05 | 94.82 | 78.63 | 8881173 | 17.1 % | 2.4 % | 80.5 % | 2167072 |
| RP_r1 | 85.88 | 94.82 | 86.33 | 4581071 | 35.5 % | 2.9 % | 61.5 % | 2167468 |

**Verdict, as preregistered (B0).** Rule 1: perfect classification of every sparse voxel
moves B0 from 77.83 to 78.40 (r2: 78.05 to 78.63), below the 0.60 decision band — no
target at the point-weighted headline. Rule 2: the largest residual category is DENSE at
T = 2, 4 and 8 — outcome R3. Completion has no target, but the stated reason was worded
wrongly for B0: its residual is systematic classification on well-observed geometry,
which neither completion nor finer voxels address. Mixing outweighs classification error
only in car and road for B0, and in car, road and sidewalk for RP (per-class tables).

Consistency predictions, all PASS: P1 the floor is model-independent (max deviation 0.0 %);
P2 the MIX share rises ZS → B0 → RP (14.6 → 17.3 → 35.5 %); P3 RP's road and sidewalk map
errors are majority MIX (road MIX 301282 vs CLS 118952; sidewalk MIX 417698 vs CLS 277483).

Where B0_r1's out-of-frustum errors sit:

| n_obs | voxels | scored points | CLS % | MIX % |
|---|---:|---:|---:|---:|
| 1 | 627803 | 364165 | 16.20 | 0.17 |
| 2–3 | 501662 | 777805 | 14.00 | 0.59 |
| 4–7 | 382756 | 1404800 | 12.60 | 0.86 |
| 8–15 | 291448 | 2345530 | 11.38 | 1.11 |
| 16–31 | 228352 | 3887885 | 10.20 | 1.19 |
| 32–63 | 187906 | 6609431 | 9.27 | 1.31 |
| 64–127 | 181892 | 12968112 | 7.20 | 1.26 |
| 128–255 | 173235 | 24165124 | 5.26 | 1.35 |
| 256+ | 118354 | 52515627 | 6.54 | 1.62 |

| closest approach of the trajectory | voxels | scored points | CLS % | MIX % |
|---|---:|---:|---:|---:|
| 0–10 m | 1383907 | 95302810 | 6.73 | 1.51 |
| 10–20 m | 620121 | 7565986 | 7.08 | 0.82 |
| 20–30 m | 330753 | 1559159 | 12.30 | 0.60 |
| 30–50 m | 282005 | 610504 | 19.05 | 0.29 |
| 50 m+ | 76622 | 20 | 40.00 | 0.00 |

Per class, B0_r1 out-of-frustum IoU against the ceiling, and where the errors are:

| class | model | ceiling | MIX points | CLS points |
|---|---:|---:|---:|---:|
| car | 96.99 | 98.19 | 95625 | 81672 |
| large_vehicle | 70.88 | 89.23 | 14201 | 150913 |
| two_wheeler | 51.72 | 93.84 | 8789 | 87447 |
| person | 67.53 | 94.84 | 4358 | 27326 |
| road | 87.21 | 96.62 | 220693 | 88982 |
| sidewalk | 79.27 | 93.62 | 401584 | 2318204 |
| terrain | 83.38 | 95.02 | 145934 | 469475 |
| vegetation | 76.06 | 95.30 | 263463 | 2533330 |
| manmade | 87.39 | 96.74 | 361242 | 1505184 |

The same table for RP_r1 is in [`REPORT.txt`](../results/v06/anatomy/REPORT.txt).

## 2. Per-cell weighting — post-hoc, preregistered before its numbers

**Why.** The headline map metric weights every scored point equally. In the B0 map,
voxels within 10 m of the trajectory hold **90.7 %** of the scored points; voxels observed
≤ 3 times are **42 %** of the map's voxels but **1.1 %** of its scored points. The metric
mostly examines the scene within 10 m of the vehicle; completion acts exactly where it
barely looks. SemanticKITTI's semantic scene completion benchmark scores each voxel once.

**Instrument.** `tools/residual_anatomy_vox.py` is `tools/residual_anatomy.py` verbatim
plus one block. Gates: `map_all_lookup` equals the v0.5 json; the point-weighted anatomy
equals the preregistered run bit for bit; the voxel-weighted ceiling is 100 for every class.
All pass. Voxel GT = m_r; voxel prediction = the fused class; outside = voxels whose scored
points are majority out-of-frustum. MIX cannot exist at voxel level.

| arm | outside voxels | wrong | SPARSE / DENSE share | **per cell** mIoU-9 | per-cell sparse-oracle | **per point** mIoU-9 | per-point sparse-oracle |
|---|---:|---:|---|---:|---:|---:|---:|
| ZS_r1 | 1910107 | 12.70 % | 47.5 / 52.5 % | **64.54** | 79.09 | **70.91** | 71.54 |
| B0_r1 | 1904525 | 10.98 % | 49.1 / 50.9 % | **71.85** | 84.12 | **77.83** | 78.40 |
| B0_r2 | 1904446 | 11.01 % | 49.0 / 51.0 % | **71.92** | 84.23 | **78.05** | 78.63 |
| RP_r1 | 1918988 | 7.10 % | 54.0 / 46.0 % | **78.17** | 88.97 | **85.88** | 86.33 |

Sparse voxels (n_obs ≤ 4) are 39.7 % of B0's outside voxels.

**Verdict, post-hoc rules.** Pass-to-pass |B0_r1 − B0_r2| = 0.073, so the band stays at
0.60. Rule 1: perfect sparse cells would lift B0 from 71.85 to 84.12 — a target, 12.27
against a band of 0.60. Rule 2: the largest category is DENSE at T = 2 and 4 and SPARSE at
T = 8 — threshold-sensitive. **The completion question depends on the evaluation unit:**
no target at the point-weighted headline, a real one per map cell. The expectation written
into the preregistration was a flip; it was met.

B0_r1, wrong-voxel rate by observation count and by range, outside:

| n_obs | voxels | wrong % | | closest approach | voxels | wrong % |
|---|---:|---:|---|---|---:|---:|
| 1 | 345115 | 14.40 | | 0–10 m | 1036602 | 9.70 |
| 2–3 | 317962 | 13.12 | | 10–20 m | 461101 | 12.28 |
| 4–7 | 270520 | 12.09 | | 20–30 m | 246790 | 12.75 |
| 8–15 | 218094 | 11.16 | | 30–50 m | 160016 | 12.83 |
| 16–31 | 177402 | 10.24 | | 50 m+ | 16 | 37.50 |
| 32–63 | 151994 | 9.56 | | | | |
| 64–127 | 156485 | 7.57 | | | | |
| 128–255 | 156274 | 5.83 | | | | |
| 256+ | 110679 | 6.42 | | | | |

Per class, outside, voxel IoU (B0_r1 / RP_r1): car 91.29 / 94.38, large_vehicle
67.97 / 61.59, two_wheeler 53.01 / 69.07, person 57.89 / 56.02, road 81.56 / 95.55,
sidewalk 63.08 / 84.06, terrain 65.68 / 67.26, vegetation 83.39 / 86.85, manmade
82.80 / 88.73. Sparse-wrong and dense-wrong voxel counts per class are in
[`REPORT_voxel.txt`](../results/v06/anatomy/REPORT_voxel.txt).

**The map still beats the scan per cell.** The `map_eval` control run (B0_r1, one draw)
reports the per-cell form of v0.5's comparison: every voxel carries weight 1, spread evenly
over its scored points, and each point is scored against its own label, so only the
prediction differs between the two sides. Out-of-frustum: map 70.97, per scan 68.74;
global: 71.14 against 69.19 ([`ctl_ast.txt`](../results/v06/mapeval/ctl_ast.txt)).

**Consequence.** Every map-level number in this repository before this analysis, v0.5 and
the v0.6 online decomposition included, is per point. Map results are reported per cell
from here on, with the per-point reading beside it; per-point segmentation metrics stay
per point.

## 3. Sparse-cell failure diagnosis — preregistered, amendments A1 and A2

Per cell, B0's sparse cells (n_obs ≤ 4, about 40 % of cells) are wrong about 14 % of the
time. Three causes, three mechanisms:

- **CONTEXT** — the per-scan network sees the cell's points with too few neighbours →
  completion or shape-prior head, or multi-scan context at the input.
- **SUPERVISION** — the region was never reachable by camera pseudo-labels → propagate
  pseudo-labels through the map over time.
- **FUSION** — the cell's observations disagree and the vote lands wrong → fusion or
  calibration; the network is not the lever.

**Instrument.** `tools/sparse_diag.py` is `tools/map_eval.py` plus insertions only. Gate:
every section shared with `map_eval` equals the `map_eval` control run on B0_r1 bit for
bit ([`results/v06/mapeval/`](../results/v06/mapeval/)); the isolation histogram
reproduces `offline_all`'s point and correct counts exactly. Both hold. Amendment A1,
before any run: isolation is measured as d8, the distance to the 8th-nearest same-scan
neighbour in the sensor frame (a neighbour count within 0.5 m would have cost ~1 h per arm).

Sparse wrong cells (n_obs ≤ 4, fused class ≠ GT majority), all cells:

| arm | cells | SINGLE | CONSISTENT | SPLIT | camera-reachable | non-split camera-reachable | median d8_min wrong / correct |
|---|---:|---:|---:|---:|---:|---:|---|
| B0_r1 | 143957 | 46.5 % | 44.3 % | **9.2 %** | **21.0 %** | 20.6 % | 0.500 / 0.444 m |
| B0_r2 | 144232 | 46.5 % | 44.2 % | 9.2 % | 21.2 % | 20.8 % | 0.500 / 0.444 m |
| ZS_r1 | 141677 | 47.4 % | 42.8 % | 9.8 % | 14.3 % | 13.6 % | 0.474 / 0.448 m |

Dense wrong cells, for contrast: B0_r1 136365 cells, 1.5 % SINGLE / 68.4 % CONSISTENT /
30.1 % SPLIT, 42.4 % camera-reachable.

B0_r1 per-scan accuracy against isolation, out-of-frustum points:

| d8 | points | accuracy |
|---|---:|---:|
| 0.00–0.02 m | 6677 | 78.06 % |
| 0.02–0.05 m | 9803906 | 90.04 % |
| 0.05–0.10 m | 56204556 | 90.03 % |
| 0.10–0.20 m | 26234619 | 90.80 % |
| 0.20–0.50 m | 11372149 | 90.20 % |
| 0.50–1.00 m | 1716965 | 86.58 % |
| 1.00–2.00 m | 280982 | 80.08 % |
| 2.00–5.00 m | 46811 | 73.86 % |
| 5.00 m+ | 4388 | 65.91 % |

Supervision reach: 15.3 % of scored points are camera-labelable in their own scan; via the
map, 69.6 % of out-of-frustum points sit in a cell the camera labelled at some time;
36.1 % of labelled cells are camera-labelable at some time.

**Preregistered decision on B0: NEITHER** — FUSION is not primary (SPLIT 9.2 %),
SUPERVISION is not supported (21.0 % camera-reachable), CONTEXT fails criterion (i)
(least-isolated quartile 89.30 % against most-isolated 90.23 %) while (ii) holds. B0_r1
and B0_r2 agree. The verdict stands as the result of that test.

**Amendment A2, after the B0 result and before any other arm ran.** Criterion (i) used
"the most isolated quartile of points", and that quartile is dominated by dense near-range
points: its boundary is d8 = 0.131 m, where accuracy is still flat. Accuracy falls only
beyond d8 = 0.5 m, and B0's failing sparse cells sit at median d8_min = 0.50 m. Because the
0.5 m threshold was read off B0, it is tested on five arms whose diagnosis nobody had
seen; CONTEXT is confirmed if at least 4 of 5 pass both (i′) accuracy at d8 ≥ 0.5 m at
least 5 points below d8 < 0.5 m and (ii) wrong sparse cells more isolated than correct ones.

| arm | accuracy d8 < 0.5 m | accuracy d8 ≥ 0.5 m | (i′) | d8_min wrong / correct | (ii) | SPLIT | non-split camera-reachable |
|---|---:|---:|---|---|---|---:|---:|
| B1_r1 | 87.93 % | 83.55 % | fail | 0.479 / 0.446 m | pass | 10.0 % | 13.3 % |
| D_r1 | 90.95 % | 88.23 % | fail | 0.458 / 0.451 m | pass | 8.3 % | 9.8 % |
| C_r1 | 95.74 % | 90.88 % | fail | 0.492 / 0.449 m | pass | 6.9 % | 11.1 % |
| R_r1 | 94.50 % | 89.74 % | fail | 0.491 / 0.448 m | pass | 7.8 % | 11.7 % |
| RP_r1 | 96.06 % | 90.36 % | **pass** | 0.489 / 0.450 m | pass | 6.4 % | 11.2 % |

**CONTEXT not confirmed: 1 of 5 arms pass, 4 required.**

**Reading.** Sparse cells fail at roughly the network's ordinary per-scan error rate,
without the averaging that fusion gives well-observed cells. Fusion is not the cause
(SPLIT 6–10 %), camera reach is not the cause (10–13 % of failing sparse cells on the five
confirmation arms were ever camera-visible, 21 % on B0), and isolation is a tail effect:
per-scan accuracy on the d8 ≥ 0.5 m tail is 3–6 points lower, in the same direction on all
five arms, below the 5-point bar on four of them. Completion or a shape prior addresses
that tail only. The lever that moves sparse and dense cells alike is per-scan accuracy
itself, which is supervision.

## 4. Map-propagated camera pseudo-labels — premise check, preregistered

Camera pseudo-labels supervise about 15 % of a scan's points, yet via the map 69.6 % of
out-of-frustum points sit in a cell the camera labelled at some time. Premise: propagating
the pseudo-labels through the accumulated map yields more supervision at comparable
precision. A method is designed on it only if it holds.

**Instrument.** `tools/propagate_check.py`, built on `tools/map_eval.py`'s trajectory,
de-skew and voxel code: seq 07, FAST-LIVO2 trajectory, 0.20 m cells from all evaluated
points with a valid pose (no confidence gate, model-independent). Camera labels = the two
2D teachers' projected labels (EoMT-L, Mask2Former-L) through the same filter E as the B0
pseudo-labels (visible, teachers agree, confidence ≥ 0.9), common-9. A cell's label is the
majority of the filter-E labels of all its points from all frames, with at least k votes
and a majority share ≥ 2/3; for a point not labelable in its own frame, its own frame's
votes are removed first (leave-own-frame-out), so the number measures propagation from
other frames.

**Gates, both PASS.** Gate 1 reproduces the pseudo-label audit's seq-07 row: candidates
15.1432 % of all points (target 15.14) and two-teacher agreement 94.0330 % (target 94.0);
the 2D arm of [2d_vs_3d](2d_vs_3d.md) is reproduced exactly on the way (coverage 15.2442 %,
in-frustum mIoU-9 77.5934), and the cell keys equal `map_eval --conf-gate 0`'s (2737213
voxels). Gate 2: every one of the 125886114 evaluated points is exactly one of labelable
in its own frame, propagated-only, or unlabelled.

| supervision | coverage, all points | precision, all | in-frustum coverage / precision | out-of-frustum coverage / precision |
|---|---:|---:|---|---|
| per-scan filter E | **14.10 %** | 95.92 % | 87.84 / **95.92 %** | 0.00 % / — |
| propagation k ≥ 1 | 70.58 % | 92.86 % | 97.03 / 94.30 % | 65.51 / 92.45 % |
| **propagation k ≥ 2** (primary) | **68.22 %** | 93.65 % | 96.37 / 94.50 % | 62.84 / 93.40 % |
| propagation k ≥ 3 | 66.69 % | 94.09 % | 95.88 / 94.63 % | 61.11 / 93.93 % |

Propagated-only points, leave-own-frame-out:

| k | in-frustum count / precision | out-of-frustum count / precision |
|---|---|---|
| ≥ 1 | 1859489 / 78.76 % | 69228582 / 92.45 % |
| ≥ 2 | 1724236 / 79.79 % | 66400637 / **93.40 %** |
| ≥ 3 | 1627031 / 80.47 % | 64575079 / 93.93 % |

**Decision, applied literally: PREMISE HOLDS.** (a) Propagated coverage 68.221 % against
twice the per-scan 14.105 % = 28.210 % (×4.837). (b) Propagated-only out-of-frustum
precision 93.405 % against per-scan in-frustum precision 95.924 % − 5 = 90.924 %.
(b) already holds at k ≥ 1.

**Stuff propagates precisely; things do not.** Precision of propagated-only
out-of-frustum labels at k ≥ 2, by label class, against per-scan filter E in-frustum:

| class | per-scan filter E, in-frustum | propagated-only, out-of-frustum | propagated points |
|---|---:|---:|---:|
| road | 99.20 | 96.72 | 17357055 |
| sidewalk | 97.68 | 94.96 | 10636040 |
| terrain | 98.55 | 97.18 | 3187605 |
| manmade | 95.35 | 94.66 | 20176635 |
| vegetation | 91.87 | 87.08 | 8882083 |
| car | 92.87 | 85.41 | 5851078 |
| person | 95.85 | 90.11 | 42257 |
| large_vehicle | 90.87 | 64.90 | 211481 |
| two_wheeler | 68.75 | 51.48 | 56403 |

Post-hoc views, not part of the decision: point-weighted over things only (car,
large_vehicle, two_wheeler, person), 84.43 % at k ≥ 2 against 92.64 % per scan — the
(b)-style test fails; the class-macro mean, 84.72 % against 92.33 %, fails too.

Diagnostics: precision holds with range while coverage falls (propagated-only
out-of-frustum at k ≥ 2: 0–10 m 70.70 % coverage / 93.66 % precision; 10–20 m
48.34 / 92.35; 20–30 m 35.22 / 93.18; 30–50 m 21.88 / 91.09). In-frustum points that
filter E did not label receive propagated labels of only 79.79 % precision (k ≥ 2);
propagation does not replace the in-frustum filter.

Caveats, as the report states them: votes come from all other frames, past and future
(offline label generation, as for training — a causal variant would reach fewer points);
leave-own-frame-out removes only the query point's own sweep; point-weighted numbers are
dominated by near range (52898537 of the 66400637 propagated-only out-of-frustum points lie
within 10 m); one sequence, and it is the test sequence. **A seq-09 replication is pending.**

## Files

| path | content |
|---|---|
| `results/v06/anatomy/` | `PREREG.md` (one clause redacted, marked), `PREREG_voxel.md`, `REPORT.txt`, `REPORT_voxel.txt` |
| `results/v06/sparsediag/` | `PREREG.md` with amendments A1 and A2, `REPORT.txt`, `REPORT_confirm.txt` |
| `results/v06/propagate/` | `PREREG.md`, `REPORT.md`, `result.json` |
| `results/v06/mapeval/` | the `map_eval` control: it reproduces v0.5's `map_all_lookup` and both anatomy runs bit for bit, and gives the per-cell map-versus-scan reading |
| [`experiments/v06/`](../experiments/v06/README.md) | `tools/map_eval.py`, the anatomy, sparse-diagnosis and propagation tools and their run scripts |
