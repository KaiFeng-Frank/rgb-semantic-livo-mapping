# Voxel-weighted residual anatomy -- POST-HOC check, preregistered before its numbers

STATUS: written AFTER the point-weighted verdict (REPORT.txt under PREREG.md) was read,
and BEFORE any voxel-weighted number existed.  It is a sensitivity analysis prompted by
something seen in that result, and must be reported as such -- not folded into the
original preregistered test.

Why: the headline metric is point-weighted.  In the B0 map, voxels within 10 m of the
trajectory hold 90.7% of the scored points; voxels observed <= 3 times are 42% of the map's
voxels but 1.1% of its scored points.  Completion acts exactly where the metric barely
looks.  SemanticKITTI's SSC benchmark scores each voxel once.  Question: does "no
completion target" survive when every map voxel counts equally?

## Instrument
tools/residual_anatomy_vox.py = tools/residual_anatomy.py verbatim + one appended block.
Gates: (a) map_all_lookup == v0.5 json bit for bit; (b) the point-weighted `anatomy`
section == the preregistered run's anat_<tag>.json bit for bit; (c) voxel-weighted
ceiling == 100 for every present class (a voxel scored against its own majority).

## Definitions
Voxel GT = m_r (majority of all scored points in the voxel); voxel prediction = fused
class; each voxel with >= 1 scored point counts once.  Subsets: outside = voxels whose
scored points are majority out-of-frustum (headline analogue); global = all voxels.
MIX cannot exist at voxel level by construction; every wrong voxel is CLS, split SPARSE
(n_obs <= 4; sensitivity 2, 8) / DENSE.

## Verdict (B0, outside; r2 for stability)
band_vox = max(0.60, 2.1 x |B0_r1 - B0_r2| voxel-weighted mIoU).  2.1|d| ~ 3 sd for two
passes.  The 0.60 band came from point-weighted noise; sparse voxels hang on one or two
votes, so voxel-weighted noise may be larger and the band must be allowed to widen.
1. NO TARGET if sparse-oracle(T=4) - model < band_vox.
2. Largest category among SPARSE / DENSE, required to agree at T = 2, 4, 8.

## What each outcome means for the recommendation
- Both rules say no target -> "do not add completion" is robust to the evaluation unit;
  the paper may state it without qualification.
- Either rule says target -> the recommendation depends on the evaluation unit: none
  under the point-weighted headline, a real one per map cell.  Reason 2 ("the accumulated
  map already is the completion") does NOT cover these voxels -- they are still sparse
  after the whole drive -- so only reason 3 (novelty taken: JS3C-Net) would remain, and
  the decision becomes which unit the paper's claim lives in.

## Expectation, stated so it can be wrong
Per-point CLS error is 16.2% in n_obs = 1 voxels vs 6.5% at n_obs >= 256, and sparse voxels
are ~42% of the map, so I expect SPARSE to become the largest category and the
sparse-oracle gain to exceed the band -- i.e. the verdict to flip.
written 2026-09-26 01:50:33
