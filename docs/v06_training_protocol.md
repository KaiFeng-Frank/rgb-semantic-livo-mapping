# v0.6 training protocol: a held-out sequence, three seeds, the cost of the KL anchor

Updated 2026-09-26. **Training and scoring are complete: [v0.6 results](v06_results.md).**
This page records the protocol and the code as they were fixed before any v0.6 number
existed. Code: [`experiments/v06/`](../experiments/v06/README.md).

## Why seq 09 is held out

Every v0.4/v0.5 arm trained on sequences 00, 01, 02, 04, 05, 06, 09 and 10, selected
checkpoints on 08, and was tested on 07. Sequence 07 never entered training and never
selected a checkpoint, but every design decision of v0.4 and v0.5 — whether to keep the
anti-forgetting KL anchor, the confidence-gate threshold, the 0.20 m map voxel — was made
while looking at seq-07 numbers. That is decision contamination: weaker than training
contamination, invisible to a train/test audit, and biased in the optimistic direction.
v0.6 withholds seq 09 so that one number exists which no design decision has seen.

| split | sequences |
|---|---|
| train | 00, 01, 02, 04, 05, 06, 10 |
| val (checkpoint selection) | 08 |
| test | 07, 09 |

Seq 10 stays in training so the new absolute numbers remain comparable to v0.4/v0.5;
seq 09 alone holds 1591 frames, more than seq 07's 1101. Seq 03 is absent for the same data
reason as in v0.4. The v0.4 modules are not modified: `src/split_v2.py` only subclasses
them, so every v0.4/v0.5 arm stays reproducible from its original configuration.

## Jobs

Strictly serial, one GPU; each configuration inherits the v0.4 arm and changes only the
seed, the split and, for the no-KL arm, `kl_enabled=False`.

| job | configuration | seed | role |
|---|---|---|---|
| armB0_s1, s2, s3 | `arm_B0_v2_s{1,2,3}.py` | 20260924, 20260925, 20260926 | the deployable method, three training seeds |
| armB0_noKL_s1, s2, s3 | `arm_B0_noKL_v2_s{1,2,3}.py` | 20260924, 20260925, 20260926 | the same three seeds without the KL anchor |
| armRprime_noKL_s1 | `arm_Rprime_noKL_v2_s1.py` | 20260924 | random target-GT voxels, reference, one seed |

The KL anchor pins out-of-frustum predictions to the frozen zero-shot model. On the GT arm
D, removing it moved outside-frustum mIoU-9 from 74.68 to 81.44
([v0.4](v04_results.md#completed-no-kl-diagnostic-d_nokl-and-rprime_nokl)). B0 is the only
deployable arm, so the size of that penalty is measured on B0 itself, one factor apart and
on identical seeds, instead of being extrapolated from D.

## Guards against training on the held-out sequence

- **Static check at import.** The split table must not put 07 or 09 in train or val, and
  the test split must be exactly {07, 09}.
- **Path guard.** Every training and validation data list is read back from the paths
  actually loaded; a held-out sequence id in any path raises. This catches a stale `.pyc`,
  a configuration naming the v0.4 dataset class, or a subclass that forgot the split.
- **Self-check before the queue** (`opt/train_queue_v06.sh`): the table, the sequences each
  split really loads, a deliberate seq-09 path that the guard must reject, and the v0.4
  split unchanged. A failed self-check aborts the queue before anything trains; so does a
  failure of the first job.

## Scoring plan, fixed before training finished

`opt/score_v06.sh` runs unattended once the queue reaches a terminal state:

1. Extract every student with the v0.5 checks, each against a different student as the
   negative control.
2. Cache predictions for 8 models × 2 test sequences × 2 forward passes (32 caches). Two
   passes per checkpoint separate the seed-to-seed spread from the model's own
   pass-to-pass non-determinism.
3. Map-level replay of every cache through `tools/map_eval.py` on each sequence's
   FAST-LIVO2 trajectory. `opt/prep_seq09.sh` built the seq-09 rig: bags, the 100 Hz IMU
   stream, and a FAST-LIVO2 trajectory.
4. `tools/v06_report.py` answers four questions, in order of how much they can hurt:
   - **Decision contamination**, as a difference-in-differences against the zero-shot arm,
     which no decision was tuned against: [B0(07) − B0(09)] − [ZS(07) − ZS(09)]. The raw
     seq-07 − seq-09 gap also contains the two scenes being differently hard, so it is
     printed beside it.
   - **The cost of the KL anchor**: B0 against B0 without KL, same seeds.
   - **Seed spread against pass spread**: if they are the same size, the arm's variance is
     the model's own non-determinism.
   - **Do v0.5's three map findings survive on a clean sequence?** The map beats the scan;
     the margin shrinks as the classifier strengthens; the strongest arm loses the planar
     classes at map level.

## Recorded deviation

Job 5 (armB0_noKL_s2) was killed by a host out-of-memory event 34 minutes before it would
have finished. The queue was restarted with `opt/train_queue_v06_resume.sh`, which skips
completed jobs and resumes an interrupted one from its last checkpoint. Job 5's epoch 10
therefore ran from the epoch-9 checkpoint; Pointcept re-seeds at start, so that epoch's
data order and augmentation differ from an uninterrupted run. This is the same class of
perturbation as the measured run-to-run non-determinism and touches one epoch of one seed.

**Correction, after scoring.** The paragraph above is kept as it was written before
scoring. It was not the same class of perturbation as run-to-run non-determinism: the
resume changed the weights, not only the data order. On one GPU, Pointcept v1.5.1's
`CheckpointLoader` loaded the frozen KL anchor, which holds the released weights, into the
student, and did not restore the student head. Epoch 10 therefore restarted from the
released model, with epoch 9's optimizer state and final learning rate. The armB0_noKL_s2
checkpoint first scored was that one epoch, not a ten-epoch no-KL run. It has since been
withdrawn and archived, and seed 2 retrained from scratch without resuming; every
B0-without-KL result is the rerun's. See
[deviation 1 of the results](v06_results.md#protocol-deviations-and-disclosures).

## Results

Complete: [v0.6 results](v06_results.md). The files are:

- [`REPORT.txt`](../results/v06/scoring/REPORT.txt), the report;
- [`runs_summary.json`](../results/v06/scoring/runs_summary.json), the per-run readouts;
- [`val_curves.json`](../results/v06/training/val_curves.json), the validation curves, the
  selected epochs and the resume audit.

## Note on the propagation premise check and seq 09

After this protocol was written, the map-propagated pseudo-label premise check was
replicated on seq 09 using its ground truth to measure label precision (see
docs/v06_evaluation_unit.md). No design choice depends on that replication: the operating
point (k ≥ 2 votes, majority ≥ 2/3) and the stuff-only class policy were fixed on seq 07
before it ran. The trained models still never see seq 09. The stronger statement above,
that no design decision has looked at seq 09, no longer holds for the propagation premise
and is recorded here.
