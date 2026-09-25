# Sparse-cell failure diagnosis -- preregistered before any number

## Why
The per-cell (voxel-weighted) reading found a large objective target: B0's sparse map
cells (n_obs <= 4, ~40% of cells) are wrong ~14% of the time, and labelling them perfectly
would lift B0's per-cell map mIoU from 71.85 to 84.12.  Whether a completion head is the
right mechanism depends on WHY those cells are wrong.  Three causes, three mechanisms:
- CONTEXT     the per-scan network sees the cell's points with too few neighbours
              -> completion / shape-prior head, or multi-scan context at the input
- SUPERVISION the region was never reachable by camera pseudo-labels
              -> propagate pseudo-labels through the map over time
- FUSION      the cell's observations disagree and the vote lands wrong
              -> fusion / calibration; changing the network is not the lever
The method is designed around what this finds.  The completion head enters the method
only if CONTEXT is supported.

## Instrument
tools/sparse_diag.py = tools/map_eval.py + insertions only (one module-level import alias,
loop accumulators, one end block).  Gate: every section shared with map_eval
(map_all_lookup, anatomy, anatomy_voxel, reweighted_map_vs_scan) must equal
out/v06_mapeval/ctl_B0_r1.json bit for bit on B0_r1.  Accounting: the isolation histogram
must reproduce offline_all's point counts and correct counts exactly.
Arms: B0_r1, B0_r2 (pass stability), ZS_r1 (contrast).  seq 07, FAST-LIVO2 trajectory,
conf gate 0.5, voxel 0.20 m.

## Measurements (per map cell; wrong = fused class != GT majority of the cell)
- vote pattern of a cell, from the per-scan predictions of all its scored points:
  SINGLE (one point) / SPLIT (>= 2 points, top prediction < 2/3) / CONSISTENT (>= 2/3)
- camera-labelable: >= 1 of the cell's points was in the camera frustum AND passed the same
  z-buffer test the 2D arm used (score_2d_vs_3d.visible_mask, np.rint pixels, OCC_WIN) in
  some frame -- a camera pseudo-label could have reached it at some time.
- isolation: nb = same-scan neighbours within 0.5 m in the sensor frame (self included);
  per cell nb_max = the best context any of its observations had.
- per-scan accuracy vs nb over all evaluated points, split in / out of the frustum.

## Decision (B0_r1; B0_r2 must agree; population = sparse wrong cells, all cells)
1. FUSION is the primary lever if SPLIT >= 50% of sparse wrong cells.
2. Otherwise, over the non-SPLIT sparse wrong cells:
   - SUPERVISION supported if >= 50% of them are camera-labelable.
   - CONTEXT supported if BOTH (i) per-scan accuracy of the most-isolated quartile of
     evaluated out-of-frustum points (by nb) is >= 5 points below the least-isolated
     quartile's, AND (ii) their median nb_max is below the median nb_max of sparse CORRECT
     cells (wrong cells are more isolated than correct cells of the same sparsity class).
   - Both supported -> the method combines both.  Neither -> these three causes do not
     explain the sparse errors; report that, and the design goes back to analysis.
written 2026-09-26 03:22:16

## Amendment A1 -- before any run, before any number
Counting same-scan neighbours within 0.5 m is O(10^3) per near-range point (a surface at
5 m holds ~2000 points in that radius), i.e. ~1 h per arm.  Replaced, with the same intent
(local sampling density the per-scan network sees), by:
  d8 = distance to the 8th nearest same-scan neighbour (sensor frame, self excluded).
Larger d8 = more isolated.  Per cell: d8_min = the best context any observation had.
Criteria restated in d8 (nothing else changes):
- CONTEXT (i): per-scan accuracy of the most-isolated quartile of evaluated out-of-frustum
  points (largest d8) is >= 5 points below the least-isolated quartile's (smallest d8).
- CONTEXT (ii): median d8_min of the non-SPLIT sparse wrong cells is LARGER than that of
  sparse CORRECT cells (wrong cells are more isolated than correct cells of equal sparsity).
amended 2026-09-26 03:24:27

## Amendment A2 -- after the B0 results, before any other arm was run
The preregistered verdict on B0 is NEITHER, and it stands as the result of that test.
It came from a flaw in CONTEXT (i): "the most isolated quartile of POINTS" is dominated by
dense near-range points -- the same point-weighting bias this work set out to remove.  On
B0 the quartile boundary is d8 = 0.131 m, where per-scan accuracy is still flat (~90%);
accuracy falls only beyond d8 = 0.5 m (86.6 / 80.1 / 73.9 / 65.9% in the 0.5-1 / 1-2 / 2-5 /
>5 m bins), and B0's failing sparse cells sit at median d8_min = 0.50 m.  The 0.5 m threshold
below is chosen FROM B0's data, so B0 cannot test it.  It is tested on arms whose
sparse_diag output nobody has seen: B1_r1, D_r1, C_r1, R_r1, RP_r1 (seq 07), and later the
v0.6 seq 09 runs.
CONTEXT confirmed if, on >= 4 of these 5 arms, BOTH hold:
  (i')  per-scan accuracy of out-of-frustum points with d8 >= 0.5 m is >= 5 points below that
        of out-of-frustum points with d8 < 0.5 m, AND
  (ii)  (unchanged) median d8_min of non-SPLIT sparse wrong cells > that of sparse correct cells.
FUSION and SUPERVISION are reported for these arms but decide nothing new.
amended 2026-09-26 03:47:35
