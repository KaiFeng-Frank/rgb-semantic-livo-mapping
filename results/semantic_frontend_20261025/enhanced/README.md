# Enhanced semantic frontend verification

The earlier association/boundary-only variant failed its own accuracy admission.
It did not implement dynamic point/map admission or visual semantic masking and
cannot establish that a more complete semantic coupling design is ineffective.
The previous results and failed attempts remain unchanged.

Commit `0d279c7` adds causal motion decisions, projected motion-region image masks,
and a pose-dependent categorical likelihood in the actual 19-state estimator.
The enhanced profile disables the old boundary residual. The same commit fixes
the upstream uninitialized `blind_sqr` threshold in the shared preprocessing path.
The old accuracy variation has not been causally attributed to that bug.

[Local evidence](local/assessment.json) includes actual LiDAR/VIO integration,
finite-difference and temporal contracts, poisoned-threshold preprocessing checks,
three startup modes, and small/120000-point ROS transport. The categorical-only
flat-plane fixture changes lateral error from 0.20 m to 0.0483 m with zero boundary
rows; this is synthetic mechanism evidence, not measured driving accuracy.

The frozen development window is sequence07 frames 730–825: geometry, motion-only,
visual-mask-only, categorical-only and combined, one repetition each. Selection
uses a previously documented moving-car segment. No GT scoring or parameter
search is allowed. [GitHub CPU run 40162901816](https://github.com/KaiFeng-Frank/rgb-semantic-livo-mapping/actions/runs/40162901816)
completed successfully on commit `0d279c7dd338a079e5a0887327f3684331c50f72`.
All five arms have 91 finite poses with identical timestamps. All 13 frozen source
hashes and all five frozen trajectory hashes were independently verified. Original
point IDs passed exact xyz/intensity correspondence checks for all 96 input scans.
Recomputing the sort order disagreed at 34 B0 and 121 ZS indices across the window;
that diagnostic order is not used for annotation. The carried original point IDs,
verified against the actual message coordinates/intensity, determine label lookup.

| Arm | Updates with categorical rows | Updates detecting moving points | Moving-point observations | Invalidated old map nodes | Visual updates with moving masks |
|---|---:|---:|---:|---:|---:|
| Geometry | 0 | 0 | 0 | 0 | 0 |
| Motion only | 0 | 78 | 2,076 | 5 | 0 |
| Visual mask only | 0 | 79 | 2,279 | 0 | 70 |
| Categorical only | 89 | 0 | 0 | 0 | 0 |
| Combined | 89 | 81 | 2,424 | 4 | 72 |

Every arm has zero accepted boundary constraints. Visual-only still tracks motion
to construct masks, but its geometric motion switch is off. Moving-point counts
sum frame-wise observations, not unique objects; masks indicate projected region
activation, not dense 2-D segmentation accuracy or the number of removed visual
residuals. Accepted photometric rows include estimator iterations. Actual-estimator
fixtures separately check that these switches affect geometric/photometric rows,
map admission, and the state/covariance solve.

This passes the bounded mechanism check. It does not measure an accuracy gain,
motion-classification quality, or online throughput: labels are cached, playback
is 0.125x, and there is one development repetition. The combined arm also enables
class-conditioned association, so its difference from an individual arm cannot be
attributed to a single new component. A separate frozen, repeated paired evaluation
is still needed before claiming a localization benefit.

[Assessment](replay/assessment.json), [all arm counters](replay/runs.json),
[exact protocol](replay/protocol.json), [workflow metadata](replay/workflow.json),
and [trajectory hashes](replay/frozen.json) are retained. The
[295,540-byte evidence archive](replay/frontend-evidence.tar.gz) contains logs,
configurations, trajectories and input hashes; raw points/images stayed on the
ephemeral runner. Its [SHA-256](replay/frontend-evidence.sha256) is
`0f1fc28d8a03b72f51aba971969902e88d85d7f1363d268b8704716db2a3d5ff`.

The earlier local GPU attempt was stopped before inference to preserve an
independent mapping experiment. Once the GPU was free, reliable input passed
three repeated bridge checks, but [full 1x online LIVO admission failed](online/README.md):
timely semantics cover 87–89% of measured scans and pose-delivery p95 reaches
3.28–3.36 s. The LIO estimation stage is too slow on this workload; all six runs
still preserve input coverage and identical 114-pose timestamps. All failures
and stage timing logs are retained.

[A separate frozen six-arm seq09 accuracy evaluation](accuracy09/README.md) is
pending. Projected 3-D masks are not dense 2-D segmentation; historical class votes
are not full model posteriors. No larger experiment is admitted by these checks.
