# v0.4 results and next steps

> **Follow-up, 2026-09-26.** Engineering gate 1 below ran as [v0.5](v05_results.md) and
> gate 2 as [v0.6 online integration](v06_online_integration.md). Independent training seeds
> and a held-out sequence are the [v0.6 training protocol](v06_training_protocol.md). The
> Livox phase named in milestone 5 is now v0.8 in the README roadmap.

Updated 2026-09-24. Six adapted arms and the paired no-KL diagnostic have completed
ten-epoch training and three full-sequence inference draws each.

Camera pseudo-label fine-tuning (B0) raises outside-frustum mIoU-9 from **65.01 to
72.98 (+7.96 points)** on SemanticKITTI sequence 07. The corresponding mIoU-8 delta,
with `two_wheeler` removed, is **+8.11**. These are results from one training run per
arm. The original class-sign mechanism prediction is not supported: B0 matches
only **5 of 8** tested cells.

## Evaluation contract

- Training sequences: **00, 01, 02, 04, 05, 06, 09, 10**. Sequence 03 is absent
  from every arm. Each epoch contains 17,228 frame draws with the same fixed,
  length-preserving rare-class resampling rule.
- Filter boundaries and best-checkpoint selection use **sequence 08 GT**. Sequence
  07 is excluded from training and checkpoint selection for these runs.
- Final scoring covers **all 1,101 frames of sequence 07**. Outside-frustum is the
  primary subset: 105,671,053 evaluated points, against 20,215,061 inside.
- The common-9 mapping drops GT-excluded classes for every arm. Predictions in
  unmapped classes remain errors. The student keeps the original 16-way nuScenes
  head; supervised probabilities are marginalised without renormalisation.
- Inference uses Pointcept **v1.5.1**, intensity ×0.2, 0.05 m voxels, fp16,
  `shuffle_orders=False`, and no TTA. Each trained checkpoint has three inference
  draws; the frozen A reference has six.
- **± below denotes sample SD across inference draws**, not uncertainty across
  independently trained models. The original ±0.60 decision band measures the
  inference instrument's variation, not training-seed variation.

The camera's 15.24% usable projected-label coverage and the geometric frustum's
16.06% share of evaluated points have different denominators and masks. Neither
should be used to linearly interpolate global mIoU: global IoUs come from pooled
class counts, not a coverage-weighted mean of subset mIoUs.

Sequence 07 was used in earlier pipeline development, including intensity and
filter inspection. It is therefore not an untouched final benchmark. B0 avoids
target 3D GT in its training loss, but the present development and selection
procedure is not entirely target-label-free. Independent evaluation and a
label-free selection protocol are separate future experiments.

## Completed arms

| Arm | Training supervision | Outside mIoU-9 ± SD | Δ vs A | Outside mIoU-8 | In-frustum mIoU-9 | Global mIoU-9 |
|---|---|---:|---:|---:|---:|---:|
| A | Frozen nuScenes model | 65.01 ± 0.25 | — | 68.31 | 65.35 | 65.06 |
| B1 | Camera pseudo-labels, head only | 65.90 ± 0.22 | +0.88 | 69.52 | 67.34 | 66.14 |
| **B0** | **Camera pseudo-labels, full fine-tuning** | **72.98 ± 0.17** | **+7.96** | **76.42** | **70.61** | **72.61** |
| D | GT inside the camera frustum | 74.68 ± 0.15 | +9.67 | 74.56 | 82.94 | 76.11 |
| R | GT on random raw points | 82.75 ± 0.14 | +17.74 | 82.78 | 84.38 | 83.07 |
| R′ | GT on random surviving voxels | 83.36 ± 0.11 | +18.34 | 83.38 | 85.05 | 83.68 |
| C | GT everywhere, with the existing KL anchor | 85.94 ± 0.06 | +20.93 | 86.33 | 87.81 | 86.34 |

The [generated results table](../results/v04/completed_summary.md) also gives
mIoU-8 deltas, all outside-frustum class IoUs, training seeds and realised KL
coefficients. The [machine-readable summary](../results/v04/completed_summary.json)
retains every inference draw.

B0 is the main camera-supervised method. B1 is its frozen-backbone control. D, R,
R′ and C consume target-domain 3D GT and are diagnostic references.

## What the comparisons establish

**B0 improves outside the camera frustum in this run.** Its gain is much larger
than the observed inference spread. Head-only adaptation reaches 65.90, compared
with 72.98 for full fine-tuning; independent training repeats remain necessary.
These arms also realised different automatically calibrated KL coefficients, so
their difference is not an isolated backbone-freezing effect.

**The preregistered mechanism story did not pass.** B0 improves road, terrain and
manmade, contrary to the recorded flat/down expectations. All eight testable
classes improve, but only five match the original predicted directions.
`two_wheeler` remains an abstain cell for the per-class sign interpretation;
mIoU-8 is a decomposition, never a replacement primary metric.

The archived `verdict_*.json` files preserve the historical helper output,
including its `PROPAGATION` string. That helper assigns the string from the
aggregate threshold and records `p3` separately. The
[original decision rule](v04_preregistration.md) requires both conditions, so the
string alone is not the full preregistered verdict. The aggregate score gain is
supported; the proposed class-sign explanation is not.

**R′ corrects R's supervision-count mismatch after voxelisation.** R matched raw
point counts but its sampled audit predicted only 83.66% of D's surviving
supervision. R′ selects label-valid voxels after `GridSample`. Across the full ten
epochs its mean supervised count is **0.999920 × D's**; all ten totals pass the
predeclared D mean ±5 SD gate. Its outside score exceeds D by **8.68 points**.

That difference concerns supervision distribution, including class composition
and the KL anchor's spatial support. It does not isolate geometry alone: D
anchors the camera-exterior region, whereas R′ anchors the random unsupervised
complement. The class mix is deliberately not matched, and the historical arms
have different training seeds. Likewise, R′−R = 0.60 is not a training-significance
test, and D−B0 is not a pure estimate of pseudo-label noise: filtering, excluded
classes and KL calibration also differ.

**C is an observed GT reference under this training policy.** Its outside score
is 2.58 points above R′. C still applies KL outside the original camera frustum,
where GT supervision is also active; it is not an unconstrained theoretical
upper bound.

R′ ran on the second host. The frozen-model cross-host calibration passed:
outside mIoU-9 is 65.0125 on A and 64.9965 on B, a −0.0160 difference. This checks
the inference instrument; it does not establish equal training behaviour on
the two CPU architectures.

## Completed no-KL diagnostic: D_noKL and Rprime_noKL

The paired diagnostic removed the KL term while retaining each supervision selector,
the original initialization, batch size, augmentation and ten-epoch budget. Both arms
used the explicit training seed **20260923**. The
[protocol](../results/v04/nokl_20260923/protocol.md),
[paired preflight gate](../results/v04/nokl_20260923/pair_gate.json),
[final status snapshot](../results/v04/nokl_20260923/status_snapshot.json), and
[paired comparison](../results/v04/nokl_20260923/comparison.json) are included.
The [final artifact manifest](../results/v04/nokl_20260923/final_manifest.json)
records the hashes of the scores, audits and status files.

| Arm | Selected epoch | Outside mIoU-9 ± SD | Outside mIoU-8 ± SD |
|---|---:|---:|---:|
| D_noKL | 4 | **81.44 ± 0.55** | 82.15 ± 0.42 |
| Rprime_noKL | 2 | **84.90 ± 0.12** | 85.21 ± 0.09 |

The new no-KL gap is **3.46 points**, compared with the historical KL-on gap of
**8.68 points**. The gap reduction is **5.22 points**. D_noKL changes by +6.76
points from historical D, while Rprime_noKL changes by +1.54 points from historical
R′. These historical-versus-new changes are exploratory because the historical arms
used different seeds and the new A process aligned CPU dispatch to B.

The whole-stream supervision-count gate passes exactly: Rprime_noKL and D_noKL have
the same supervised-voxel total in all ten epochs, with a mean ratio of **1.000000**.
The final R′ files and checkpoint were copied back to host 134 before host 133 was
retired; future work can run from host 134.

Preflight found that identical NumPy versions could select different voxel
representatives on Intel AVX512 and AMD AVX2. Disabling NumPy AVX512 dispatch for
the new A process made all four probed coordinate hashes and supervised counts
match B exactly. The original preflight records and the amendment are preserved.
This is a four-sample check; the whole-stream supervision-count gate is still
required after training.

The runners audited all ten epochs, the selected checkpoints, zero KL points and
finite weights, then scored three full inference draws. Historical KL-on versus new
KL-off comparisons remain exploratory because the historical seeds differ and A's
runtime dispatch was aligned for the new pair. A matched-seed, matched-environment
2×2 replication is needed for a causal claim about the KL interaction.

## Next milestones

1. Use the completed no-KL result to improve **B0**: prioritise the preservation loss if it
   contributes to the gap, or the coverage and quality of valid camera
   pseudo-labels if the gap persists. Keep oracle GT arms as diagnostics.
2. Repeat the main method and relevant baseline with at least **three independent
   training seeds**. If making a KL mechanism claim, repeat all four cells under
   matched seeds and runtime settings.
3. Freeze the protocol and evaluate on independent data unseen during method
   development. Do not reuse a training sequence as a new test set.
4. Complete the two engineering gates below: compare the existing B0 checkpoint
   against the original model in fixed-trajectory mapping replay, then integrate
   live pose input and measure concurrent FAST-LIVO2/PTv3 operation. The first
   gate can proceed before further method development is complete.
5. Package code, configuration, model artifacts and a reproducible demonstration.
   Livox and other non-repetitive scanners remain the subsequent v0.5 phase.

The existing mapping throughput results concern v0.2/v0.3 with a precomputed TUM
trajectory. The fine-tuned v0.4 student has not yet passed either engineering gate.

## Engineering acceptance: replay, then live pose input

**Gate 1 — B0 in the existing mapper.** Load the trained B0 student through the
mapping inference path, preserving the original 16-way head and common-9 scoring
contract. Check its output against the experiment inference path before judging
map quality. Compare B0 and the original nuScenes model on the same bag, saved
FAST-LIVO2 trajectory, frame set, mapping parameters and hardware. Measure map
semantics, dynamic trails, scan throughput, per-scan processing latency, dropped
scans, GPU memory and host memory. Keep scan-output and full-map-output rates
separate. This establishes whether the adaptation gain survives deployment into
the mapper; the old model's timings do not establish B0's performance.

**Gate 2 — online pose arrival and concurrent execution.** Add a timestamped pose
buffer fed by live odometry. Specify the bounded waiting and missing-pose policy
needed for scan deskew and image projection, then run FAST-LIVO2 and the semantic
mapper concurrently. Measure pose arrival delay, waiting time, queue backlog,
resource contention and sensor-to-output latency in addition to Gate 1's metrics.
Distinguish a bag-based concurrent replay from subsequent live-sensor acceptance.

The current implementation loads `TrajInterp(args.traj)` at startup and
interpolates a complete saved trajectory. `opt/run.sh` supplies `--traj` and
launches bag replay. `/semantic_scan` is published per processed sweep unless
disabled; `/semantic_map` defaults to `--map-rate 1.0` and reduces its cadence
when snapshot generation is expensive. Thus the recorded 10 Hz scan throughput
and the map publication cadence describe different parts of the system.

## Reproduce the reported numbers

```bash
python3 tools/summarize_v04.py --check
python3 tools/summarize_v04.py                 # print the complete table
```

This requires only Python 3.9+ and reads the archived JSON outputs. It checks
source/artifact hashes, the no-KL preflight hashes, all completed epoch audits,
the evaluated-point counts, per-draw metrics and the R′ supervision-count gate.
It does not rerun GPU inference.

The [source snapshot](../experiments/v04/README.md) records the training,
supervision, inference and scoring code without changing the deployed v0.3
mapping path. Datasets, model checkpoints and per-point caches remain external.
The [snapshot manifest](../results/v04/snapshot_manifest.json) maps each published
artifact to its original relative path, host alias and SHA-256 hash.

The original preregistration remains unchanged. The R/R′ notes and no-KL protocol
are separate follow-ups, with their own timing. R′'s note explicitly records that
its second section was appended after training started from an already-frozen
preflight JSON; this update does not relabel it as an earlier GitHub publication.
