# Revisit semantic refinement at fixed geometry

Frozen B0 seed-1 predictions and the previously verified corrected seq07 trajectory. GT is read only after all semantic maps are built. Offline future-aware common-9 mapping; not a single-scan or 19-class leaderboard result.

| Primary: held-out revisit ROI | mIoU9, abstentions wrong |
|---|---:|
| ordinary | 78.387156% |
| group_balanced | 78.743365% |
| revisit_balanced | 78.731261% |

Revisit minus ordinary: +0.344104 percentage points. Revisit minus group-balanced: -0.012104 percentage points.

This is a one-sequence feasibility result. Keep all arms, harmful changes and unchanged predictions in the comparison. Do not interpret confidence-vote shares as calibrated probabilities or temporal groups as statistically independent.
