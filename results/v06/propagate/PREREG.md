# Map-propagated camera pseudo-labels: premise check -- preregistered before any number

## Why
The sparse-cell diagnosis (out/v06_sparsediag) found that sparse cells fail at roughly the
network's ordinary per-scan error rate: not fusion (SPLIT 6-10%), not camera reach (10-13%
of failing sparse cells were ever camera-visible), isolation only mildly (per-scan accuracy
3-6 points lower on the d8 >= 0.5 m tail, same direction on all 5 arms, below the 5-point
bar on 4 of 5).  The lever that moves sparse and dense cells alike is per-scan accuracy
itself, i.e. supervision.  Camera pseudo-labels supervise 15.3% of a scan's points; via the
map, 69.6% of out-of-frustum points sit in a cell the camera labelled at some time.
Premise to test: propagating camera pseudo-labels through the accumulated map yields MORE
supervision at COMPARABLE precision.  Only if it holds does a method get designed on it.

## Instrument
New tool tools/propagate_check.py, built on tools/map_eval.py's trajectory / deskew / voxel
code (copy what is needed; do not modify map_eval.py or any existing tool).  seq 07,
FAST-LIVO2 trajectory out/kitti_seq07_fastlivo2_tum.txt, 0.20 m cells, cells built from ALL
evaluated points with a valid pose (no confidence gate; model-independent).
Camera labels: the 2D teacher caches out/vs2d/seg2d_eomt and out/vs2d/seg2d_m2f (EoMT-L and
Mask2Former-L projected per point), passed through the SAME filter E the B0 pseudo-labels
used (require_visible, teachers_agree, conf_min 0.9; range_lt50 False, drop_depth_edge
False -- the generator lives under tools/ and src/distil_ext.py; read it, do not reinvent),
in common-9 via src/label_spaces.py.
Propagation: a cell's camera label = majority over the filter-E labels of all its points
from all frames; operating points k >= 1, 2, 3 votes with majority share >= 2/3.
For a point NOT labelable in its own frame, the propagated label must be computed with the
point's own frame's votes REMOVED from the cell (leave-own-frame-out), so the number
measures propagation from OTHER frames.
GATE 1 (the per-scan reading is right): the per-scan filter-E labels on seq 07 must
reproduce a published number within 0.1 -- the 2D arm's coverage 15.24% of evaluated points
and in-frustum mIoU-9 77.59 (docs/2d_vs_3d.md, out/vs2d/results_eomt.json) if filter E
equals that arm's operating point; otherwise reproduce out/pseudo's own audit numbers.  State
which was reproduced and why.
GATE 2 (accounting): every evaluated point is exactly one of {labelable in own frame,
propagated-only, unlabelled}; the three counts must add up to the evaluated total.

## Measurements (all evaluated points of seq 07; GT = SemanticKITTI in common-9)
For per-scan filter E, and for propagation at k >= 1, 2, 3:
  coverage  = supervised points / evaluated points   (in-frustum, out-of-frustum, all)
  precision = supervised points with label == GT / supervised points  (same subsets, per class)
  and, separately, precision and count of PROPAGATED-ONLY points (not labelable in own frame),
  leave-own-frame-out, in-frustum / out-of-frustum.

## Decision (primary operating point k >= 2, majority >= 2/3)
PREMISE HOLDS if BOTH:
  (a) propagated coverage of all evaluated points >= 2x the per-scan filter-E coverage, AND
  (b) precision of propagated-only out-of-frustum labels >= per-scan filter-E in-frustum
      precision minus 5 points.
PREMISE FAILS otherwise.  Also report the smallest k at which (b) holds, if any.
Expectation, stated so it can be wrong: (a) holds (69.6% cell reach); (b) is uncertain --
projection through occlusion and pose error at range may pull it down.
written 2026-09-26 04:19:02
