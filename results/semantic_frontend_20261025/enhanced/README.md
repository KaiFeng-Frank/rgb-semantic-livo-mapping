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
attributed to a single new component. The separate frozen repeated accuracy check
below subsequently failed; mechanism activation does not establish a benefit.

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

[The separate frozen six-arm seq09 accuracy evaluation](accuracy09/README.md)
also failed: all eighteen runs preserve identical 235-pose coverage, but combined
mean ATE is 20.894547 m versus 16.204397 m for geometry, with worse translation
and rotation RPE. Categorical-only reduces ATE/translation RPE but worsens rotation
RPE; no semantic arm passes every declared comparison rule. The geometry baseline
itself has large errors after a cold start at frame 240, so this window is not a
normally initialized full-sequence benchmark. Its complete failure, all repetitions
and component timings are preserved without retuning or selecting a replacement.
Projected 3-D masks are not dense 2-D segmentation; historical class votes are not
full model posteriors. No larger experiment is admitted by these checks.

## Feasibility decision after the design review, 2026-10-25

This is an analysis of the current implementation, existing evidence and primary
literature. No new training, replay, parameter sweep or estimator change was made
for this review. The project objective is useful improvement in localization and
semantic mapping while retaining the direct LIVO pipeline's realtime behavior.
Adding more named components or independent residual blocks is not itself a goal.

**Decision:** reject further development of the current boundary/history-field
frontend structure. Prefer a bounded design that uses semantic and temporal
evidence to maintain reliability in the existing geometric/visual map and
measurement model. This is technically feasible as an engineering direction;
neither a new independent semantic pose observation nor a research contribution
has yet been established. Do not start another implementation merely by replacing
the expensive field with a faster one.

### Information available and what it can justify

The current [worker](../../../src/ptv3_worker.py) consumes XYZ and intensity, computes
16-class scores, and exports only the winning class and its score. It does not
consume RGB. [The class field](../../../frontend/include/semantic_likelihood.hpp)
smooths historical top-1 votes; it does not recover the missing class distribution
or a calibrated probability of a pose measurement being correct. The static field
uses only classes 10–15: driveable surface, other flat, sidewalk, terrain, manmade
and vegetation. There is no separate lane-marking, pole identity or persistent
object-instance measurement available to this frontend. Projected 3-D motion
regions are not an independent image-segmentation observation.

The worker already computes all class scores before taking the maximum. A future
interface could export scores for retained voxels without retraining the model,
but score calibration and efficient aggregation would still be required. Old
top-1 archives cannot reconstruct those scores. Exporting every raw point's full
distribution is not assumed free and is not part of the current interface.

For fixed model parameters, the labels are derived from the same scan. They can
provide learned scene priors and information discarded by a simplified geometric
residual, but cannot simply be counted as an independently noisy second sensor.
An added class factor needs a joint/conditional error model and correlation
analysis; a syntactically correct Jacobian does not supply either. In a locally
constant class region the class-field gradient is zero, so its pose Jacobian is
zero. Smoothing labels can create a gradient, but cannot establish that this
gradient corresponds to an accurate, independent metric constraint. Where a class
transition coincides with an already used geometric edge, independence also needs
justification. These limitations are visible before running another experiment.

For fixed retained geometric Jacobians, reweighting cannot constrain a direction
that is in the null space of every row. Changing correspondences or recovering
previously rejected static structure can improve the usable information, but
does not justify a blanket claim of resolving geometric degeneracy with semantics.

### Timing and estimator capabilities

Existing local measurements put direct inference medians around 63–70 ms. These
are observed medians, not worst-case execution guarantees. A current scan's final
semantic prediction cannot affect an already published pose for that scan without
either waiting for the result or revisiting the historical state. Asynchronous
execution alone does not remove this causal dependency or CPU contention.
Stable 10 Hz throughput and low output latency are different requirements: a
bounded wait can be compatible with throughput if the rest fits, but it changes
latency and must be stated. The preferred direction here preserves the original
geometry output path and avoids a fixed semantic wait.

Under that choice, the fast update uses only semantic context already available
at its timestamp. Late labels may update the original scan's surviving map
entries, through point identity, node generation/version and the historical pose,
for future updates. They must not be relabelled as a current observation or used
to reinject a completed geometric measurement. No claim of immediate semantic
rejection of a newly seen moving object follows from this design. Such objects
initially rely on the existing geometric/photometric robustness mechanisms.

The current [offline backend](../../../tools/loop_closure/backend.py) contains
Pose3 odometry/loop edges. It has no fixed-lag IMU/visual state history for delayed
measurement updates. Adding that capability would be a substantial estimator
project, not a small use of the existing backend. Loop corrections must also
change or invalidate the coordinate/version context of pending semantic updates;
the current online frontend has no validated implementation of this feedback.

### Route selection

| Route | Decision | Reason |
|---|---|---|
| Separate boundary extraction, matching and history in the high-rate frontend | Reject for this project | Adds a second feature pipeline without established incremental information; current cost and accuracy both fail. |
| Current smoothed top-1 class field as an extra high-rate pose factor | Reject as the next implementation | Coarse evidence, uncalibrated measurement meaning, correlation and repeated searches; caching alone addresses only the searches. |
| Semantic reliability in the shared direct LIVO map/measurements | Preferred design candidate | Reuses existing points, planes and visual patches; has a concrete route to reducing bad correspondence and map contamination without a second registration system. |
| Independent 2-D semantic factors plus delayed-state joint optimization | Defer | Current independent 2-D predictions and joint fixed-lag backend are absent; this would expand both perception and estimator scope. |
| Semantic support for verified loop associations and map consistency | Secondary direction | Can help long-term consistency, but does not replace a working realtime frontend or provide same-frame semantic fusion. |

The preferred candidate should represent whether a map surface is currently a
reliable static reference, not equate an object class with motion. Current motion
code [withholds unconfirmed candidates and assigns weight 0.25](../../../frontend/include/semantic_motion.hpp);
its centroid tests have shape/uncertainty safeguards but are not validated motion
probabilities. View changes, occlusion and pose error remain confounders. Likewise,
projecting an object's 3-D hull does not establish that every enclosed image patch
belongs to the object. These heuristics should not become unquestioned priors in
the replacement.

A conceptual observation model introduces a shared latent validity state `a_v`
for a map element and uses semantic/temporal context available before the update:

`p(z_v | x, h_<t) = sum_a p(a_v=a | h_<t) p(z_v | x, a_v=a, h_<t)`.

Here `z_v` groups the relevant geometric/photometric observations, and `h_<t`
includes timestamped semantic evidence and its reliability. This is a candidate
conditional robust model, not a calibrated model implemented in the repository.
It changes which existing measurements are trusted rather than inventing another
metric distance from a hard class label. A geometry-only/null hypothesis must
remain available for unknown or conflicting semantics. Applying the same semantic
prior independently to every point and again to the visual update would duplicate
evidence; a sequential implementation must carry the latent belief consistently
and account for the approximations in its covariance. Increasing the model's
complexity without solving this issue would repeat the current mistake.

The useful two-way mechanism is explicit: semantic/temporal reliability conditions
the direct pose update; pose uncertainty limits how strongly observations change
semantic class/motion/map-validity beliefs. During poor pose observability, map
evidence should become less decisive rather than falsely confirming motion or
class conflict. Changed-object/changed-voxel work replaces full-map invalidation;
existing visible map points/patches replace full-image masks per object wherever
their support suffices. Fresh semantics cannot be assumed available at zero cost.

The computational design must reuse existing geometric candidate queries, keep a
bounded active semantic cache, freeze consistent map snapshots during a solve,
and cap incremental maintenance. Semantic priors are computed per map element
when its context changes, not recomputed over all classes for every point in every
iteration. With `N` retained measurements, fixed class count `C=16`, existing
iteration count `K`, and bounded changed entries `B`, the target extra arithmetic
is `O(K*N + B*C)` with no additional full-map traversal or history-map
neighbourhood search. This is an operation-count target, not a runtime
prediction. A proposed local critical-path budget is at most 3 ms of added CPU work
per update, without reducing the original sensor rate or geometric point budget;
this is a design acceptance target, not a measured result. Background CPU/GPU,
serialization, memory and semantic evidence age remain part of eventual full
system accounting. No experiment is launched to test that target in this review.

### Research value and stopping conditions

[SuMa++](https://arxiv.org/abs/2105.11320) already uses semantics in scan matching and
dynamic filtering. [Semantic-weighted ICP](https://arxiv.org/abs/2606.03905) directly
studies class-dependent residual weighting and reports environment dependence.
[Probabilistic Data Association via Mixture Models](https://arxiv.org/abs/1909.11213)
already handles uncertain semantic association and a null hypothesis. These are
precedents, not evidence of novelty for the candidate above. Reimplementing
weighting, filtering or a mixture under a LIVO name is insufficient as the central
research contribution. This focused review is also not an exhaustive novelty audit.

A defensible research question is how to use delayed, fallible semantics without
letting semantic errors, pose errors and map updates reinforce each other in a
direct LIVO system. Our archived [pose-swap mapping evidence](../../../CRITICAL_CONSTRAINTS.md)
already reports static false-removal changing from 0.260% with estimated poses to
0.052% with GT poses at the same configuration. That supports investigating the
pose-to-semantic direction; it does not establish the reverse localization gain
or promise that the new candidate closes the gap.

The expected opportunity is reduced failure from dynamic contamination and
ambiguous association, plus more reliable semantic maps. There is no theoretical
promise of lower ATE in every clean static scene, new observability on a uniformly
labelled plane, or improved segmentation class discrimination from map consensus.
Temporal agreement can reinforce a wrong label and must not be equated with
accuracy. Before implementation, the candidate needs a concrete treatment of
prior calibration, shared-evidence correlation, delayed-map attachment, uncertainty
feedback and a contribution beyond the cited approaches. If it reduces to fixed
class weights and heuristics, classify it as an engineering baseline and stop
presenting it as a new semantic tight-coupling method. The unhealthy cold-start
geometry baseline also needs resolution before any later accuracy claim; the
current failed scores remain unchanged and are not used to choose a new window.

The resulting investment decision is deliberately narrower than "semantic LIVO is
possible": the shared-map reliability route is a plausible engineering component,
but implementing it now as the main research contribution is not justified by
this review. The next research investment should settle the joint uncertainty/
causality model and its distinction from prior work on paper. If those do not
yield a substantive mechanism beyond semantic weighting and map filtering, keep
the component as a baseline and prioritize the project's demonstrated semantic
mapping strengths instead of expanding the frontend study.
