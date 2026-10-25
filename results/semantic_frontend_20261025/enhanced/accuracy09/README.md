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
is pending. It is a temporary validation runner, not a replacement GPU server.
No accuracy claim or large experiment is admitted while results are pending.
