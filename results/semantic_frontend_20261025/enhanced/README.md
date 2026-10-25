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
is pending; raw points/images stay on its ephemeral runner.

Local GPU admission was stopped before inference to preserve an independent
mapping experiment already using 11366 MiB. Full online LIO and real-data accuracy
benefit remain unvalidated. Projected 3-D masks are not dense 2-D segmentation;
historical class votes are not full model posteriors. No larger experiment is
admitted by the local checks.
