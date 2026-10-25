# Frozen enhanced accuracy validation

Sequence09 frames 240–479 are the first contiguous window after the earlier
frontend validation window. This window was selected before enhanced scoring;
the dataset itself has appeared in historical segmentation/offline mapping work.
The estimator is unchanged from the successful mechanism replay.

Six arms (geometry, association, motion, visual masks, categorical likelihood,
combined) run three times at 0.25x with rotated arm order. All inputs and pose
timestamps must match. Only after all eighteen trajectories are hash-frozen may
the scorer read official pose GT. The combined-versus-geometry screen requires
three lower paired ATE values, nonoverlapping ATE ranges, and no worse mean 1 s
translation or rotation RPE. The increment over association is reported separately.

[Attempt 40164553378](attempt-40164553378/assessment.json), commit `b5b95db`,
stopped at its large-cloud best-effort bridge test (3/5 received), before data
recovery, trajectory replay or GT scoring. The complete artifact and test logs
remain preserved. Its protocol is not silently overwritten.

The [explicit reliable-transport amendment](../../../../frontend/enhanced-validation09-reliable.json)
adds optional reliable bridge input and tests large clouds with reliable input.
The estimator, selected window, arm settings, repetitions and scoring/decision
code remain unchanged. Local contract checks pass. The new frozen commit is
`fe7389e3c5a8671f7e876aeb3b97d148eb9f7a76`.

[GitHub CPU run 40165404469](https://github.com/KaiFeng-Frank/rgb-semantic-livo-mapping/actions/runs/40165404469)
completed, but the **accuracy admission did not pass**. Workflow execution success
means the experiment and assessment finished, not that the method passed.
All eighteen trajectories contain 235 finite poses with identical timestamps.
The seventeen frozen source hashes, eighteen trajectory hashes, archive checksum,
and recomputed assessment match. All boundary rows are zero; categorical rows are
active on 231–232 of 235 updates in the likelihood arm and 232/235 in the combined
arm. Original point identity is checked in all 240 input scans. The diagnostic
recomputed ordering disagrees at 86 B0 and 12 ZS indices; it is not used for labels.

| Arm | Mean ATE, m | Mean 1 s translation RPE, m | Mean 1 s rotation RPE, deg | LIO median range across repetitions, ms |
|---|---:|---:|---:|---:|
| Geometry | 16.204397 | 5.327161 | 1.304492 | 27.68–28.19 |
| Association | 33.514217 | 7.089274 | 1.440283 | 66.95–69.22 |
| Motion | 23.628483 | 5.946884 | 1.574615 | 128.14–136.36 |
| Visual masks | 20.081579 | 5.738287 | 1.495822 | 52.55–55.88 |
| Categorical likelihood | 15.609650 | 5.027462 | 1.414759 | 187.22–204.09 |
| Combined | 20.894547 | 5.921384 | 2.014813 | 267.49–287.25 |

The combined arm regresses against geometry in all three metrics and all three
repetitions. Its comparison to association alone also fails the rotation-RPE
rule. Categorical-only improves ATE in all three pairs with nonoverlapping ranges,
and lowers mean translation RPE, but worsens rotation RPE. No semantic arm meets
every declared comparison rule; none is selected as a successful replacement.

**The geometric baseline itself is unhealthy in this window.** Every arm starts
a fresh estimator at frame 240 without an earlier initialization replay. The
geometry log initializes gravity to `(-1.1721, 2.1191, -9.5064)` with initial mean
acceleration norm `10.3437`. Initialization is a hypothesis requiring a separate
check, not an established cause of these errors. This cold-start protocol cannot
stand in for a normally initialized full-sequence benchmark. Its failure and all
scores remain preserved; no window, settings or admission thresholds were changed
after seeing GT, and no replacement experiment was launched.

The stage timings are from the same temporary CPU runner at **0.25x with cached
labels**. They establish substantial overhead in both categorical-only and
motion-only paths but are not isolated function profiles, additive component
costs, or measurements of local GPU inference. VIO medians remain about 5–6 ms.
All semantic arms also waste about 18 ms extracting disabled boundary observations.
Repeated field searches and unconditional map traversal during dynamic invalidation
are code-level candidates for redesign. Independent [local 1x online validation](../online/README.md)
already failed with 3.28–3.36 s pose-delivery p95 and accumulating lag.

[Scores for every run](scores.json), [frozen-rule assessment](assessment.json),
[stage timings](stage-profiles.json), [interpretation](interpretation.json),
[independent verification](independent-verification.json), [protocol](protocol.json),
[trajectory hashes](frozen.json), and [workflow metadata](workflow.json) are retained.
The [2,109,015-byte evidence archive](frontend-evidence.tar.gz) includes all logs,
configs, trajectories, contracts and input hashes, without raw scans, images or
models. Its [SHA-256](frontend-evidence.sha256) is
`1d4ed421e6effbf530a556f6d5ad218c6a63f460031d13d8925143b4d0103536`.
[The verification helper](verify_evidence.py) accepts a downloaded artifact directory
and reproduces the checksum, source/trajectory checks, assessment and stage profiles.
No localization benefit, online readiness or larger experiment is admitted.
