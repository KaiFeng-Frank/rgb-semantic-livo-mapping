# Residual anatomy -- preregistered before any number was looked at

Question: would a point-cloud completion head have a target in this map?
Recommendation under test: "do not add it".  Stated reason 1: the remaining structural
bottleneck is 0.20 m voxel mixing at class boundaries, which completion does not touch.
This analysis is built to be able to refute that.

## Instrument
tools/residual_anatomy.py = opt/replay_v05.py verbatim, plus ONE readout block appended
after the final map readings.  Two checks gate every number:
- NEGATIVE CONTROL: map_all_lookup must equal out/v05/map_<tag>.json bit for bit (same
  cache, same trajectory, same code path).  Otherwise the anatomy is of a different map.
- ACCOUNTING: the anatomy's answered / error / no-voxel counts must equal the headline
  confusion matrix exactly, and rebuilding the headline from the anatomy's histograms must
  reproduce its confusion matrix exactly.  Otherwise some errors are unaccounted for.

## Definitions (per voxel r; m_r = GT majority of ALL points scored in r, both subsets)
- MIX     model error on a point whose GT != m_r.  A perfect per-voxel classifier (label r
          with m_r) also gets it wrong: irreducible at 0.20 m.
- CLS     model error on a point whose GT == m_r.  Reducible by a better classifier.
- SPARSE  CLS in voxels with n_obs <= 4 (n_obs = confident inserted points over the whole
          drive).  Sensitivity: <= 2 and <= 8.  This is completion's target.
- DENSE   CLS in voxels with n_obs > 4.  Well-observed geometry, wrong label.
- CEILING mIoU when every voxel is labelled m_r (voxel-majority oracle).
- SPARSE-ORACLE  mIoU when ONLY the sparse voxels are relabelled m_r.  Its difference from
          the model's mIoU is the most a completion head could possibly buy at map level:
          a completion head that classified every sparse voxel perfectly.
- range   closest approach of the sensor trajectory to the voxel centroid (a lower bound
          on its observation range).

## Verdict (B0, out-of-frustum = the headline subset; B0_r2 shown for pass stability)
1. NO TARGET if SPARSE-ORACLE(T=4) exceeds the model by less than 0.60 mIoU (the project's
   decision band).  Completion cannot move the headline measurably even if it were
   perfect on every sparse voxel.  This decides the question, whatever the shares say.
2. What the residual IS, by the largest category, required to agree at T = 2, 4, 8, else
   reported as THRESHOLD-SENSITIVE:
   - R1 SPARSE largest -> reason 1 is refuted; completion has a target.
   - R2 MIX largest    -> reason 1 stands as stated.
   - R3 DENSE largest  -> completion has no target, BUT reason 1 is worded wrongly for B0:
                          its residual is systematic classification on well-observed
                          geometry, which neither completion nor finer voxels address.
                          The recommendation would stand on a different reason.

## Consistency predictions (the anatomy must satisfy these to be trusted)
- P1 the floor (oracle error count) is roughly model-independent: ZS / B0 / RP within 10%
     of their mean.  It depends on geometry and GT, not on the classifier -- only on which
     voxels exist, which the confidence gate changes slightly.
- P2 MIX share of the residual rises ZS -> B0 -> RP (classifier errors shrink, MIX does not).
- P3 Rprime's map-level errors on road and on sidewalk are majority MIX.  This is the v0.5
     cell "Rprime loses on planar classes at map level" [one clause referring to work
     outside this repository removed from the public copy].  An anatomy that does not
     reproduce it is measuring something else.

## Arms
ZS_r1 (out/vs2d/ptv3_r1), B0_r1, B0_r2 (out/v04/arms/), RP_r1 (out/v05/RP_r1).
seq 07, FAST-LIVO2 trajectory, conf gate 0.5, voxel 0.20 m -- the v0.5 operating point.
written 2026-09-26 00:48:57
