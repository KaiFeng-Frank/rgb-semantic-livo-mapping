# v0.6 results: a held-out sequence, three training seeds, the KL anchor

Updated 2026-09-26. Training and scoring are complete. Seven students were trained with
seq 09 held out: B0 and B0 without the KL anchor on three seeds each, and R′ without KL on
one. With the released zero-shot model, each was scored on seq 07 and seq 09 in two forward
passes and replayed through the map: 32 runs. The questions and the report generator were
fixed before any v0.6 number existed ([protocol](v06_training_protocol.md)). Numbers are
[`REPORT.txt`](../results/v06/scoring/REPORT.txt) verbatim unless another file is named.

- **No decision contamination of B0 is measurable.** Difference-in-differences against the
  zero-shot model: −0.78, inside B0's 0.91 seed spread.
- **The KL anchor is necessary for camera pseudo-label distillation.** B0 scores 71.50 with
  it and 50.64 without on seq 07, 68.31 and 54.15 on seq 09. On GT supervision v0.4
  measured the opposite.
- **Seed spread is real.** 0.913 between training seeds against 0.064 between forward
  passes (B0, seq 07).
- **v0.5's map findings (a) and (b) hold on seq 09. (c) does not generalise.**
- **B0 on the clean sequence:** 68.31 per scan, 74.12 on the map per point, 61.94 per cell.

One deviation changes an arm. After the host's out-of-memory killer ended armB0_noKL_s2,
its resume restarted the last epoch from the released weights
([deviation 1](#protocol-deviations-and-disclosures)). Every B0-without-KL aggregate
includes it; no ZS, B0 or R′ number does.

## Evaluation contract

- **Split.** Train on 00, 01, 02, 04, 05, 06 and 10 (15,637 frames per epoch); select
  checkpoints on 08 (4,071 frames); test on 07 and 09. Seq 07 (1,101 frames) is the
  sequence every v0.4/v0.5 design decision was made on. Seq 09 (1,591 frames) was withheld
  from training, from checkpoint selection and from every decision the trained arms depend
  on. The one later use of seq-09 ground truth, the propagation replication, is
  [recorded in the protocol](v06_training_protocol.md#note-on-the-propagation-premise-check-and-seq-09).
  The queue's self-check passed before anything trained, including a deliberate seq-09
  path that the guard had to reject ([`selfcheck.log`](../results/v06/training/selfcheck.log),
  [`queue.log`](../results/v06/training/queue.log)).
- **Arms.**
  - B0: camera pseudo-labels (filter E), full fine-tuning, KL anchor on. This is the
    deployable method.
  - B0 without KL: the same configuration with `kl_enabled=False`, on the same three seeds
    (20260924, 20260925, 20260926).
  - R′ without KL: random target-GT voxels, the reference, seed 20260924.
  - ZS: the released nuScenes weights, never trained.

  Each trained arm runs ten epochs. The
  [realised configurations](../results/v06/training/configs/) differ from their v0.4 arm in
  seed, split (the `…V2` dataset classes) and save path, and the no-KL arms also in
  `kl_enabled`.
- **Scoring.** All seven students passed the v0.5 extraction checks
  (`results/v06/scoring/extract_*.json`). Each of the eight models ran twice over each test
  sequence. Each of the 32 prediction sets was replayed through `tools/map_eval.py` on that
  sequence's FAST-LIVO2 trajectory, with 0.20 m voxels and confidence gate 0.5.
- **Metric: out-of-frustum mIoU-9 (common-9), in three readings.**
  - *Per scan* (REPORT.txt: offline) is the network's per-scan predictions at every
    evaluated point, the v0.4 metric.
  - *Map* is the fused map looked up at every evaluated point, per point. A point that
    lands in no voxel is left out, the convention of the evaluation-unit tables. v0.5's
    headline map number counted such a point wrong;
    [`runs_summary.json`](../results/v06/scoring/runs_summary.json) carries both.
  - *Per cell* scores every map cell once (section 5).
- **Evaluated points.** Seq 07: 125,886,114, of which 105,671,053 are out of frustum.
  Seq 09: 188,141,271, of which 158,407,085.
- **± is the population standard deviation (divisor n) over all runs of an arm.** For B0
  and B0 without KL that is three training seeds × two forward passes, six runs, so it
  covers training-seed variation and pass-to-pass non-determinism together. For ZS and R′
  it is one checkpoint × two passes, so it covers pass-to-pass variation only. It is not a
  confidence interval. It is also not v0.4's ±, which was the sample SD of one training
  run's inference draws.

## Results

| arm | training supervision | seq 07 per scan | seq 07 map | seq 09 per scan | seq 09 map |
|---|---|---:|---:|---:|---:|
| ZS | none, released nuScenes model | 64.96 ± 0.11 | 71.07 ± 0.16 | 60.99 ± 0.11 | 67.25 ± 0.07 |
| **B0** | **camera pseudo-labels, KL anchor** | **71.50 ± 0.92** | **77.51 ± 1.87** | **68.31 ± 0.82** | **74.12 ± 0.89** |
| B0 without KL | camera pseudo-labels | 50.64 ± 6.90 | 55.53 ± 7.96 | 54.15 ± 3.04 | 60.96 ± 3.74 |
| R′ without KL | random target-GT voxels, reference | 85.37 ± 0.13 | 86.12 ± 0.07 | 83.57 ± 0.12 | 83.86 ± 0.08 |

*Out-of-frustum mIoU-9. Seq 07 is design-seen; seq 09 is clean.*

The next table gives both forward passes for every checkpoint, from
[`runs_summary.json`](../results/v06/scoring/runs_summary.json). The selected epoch and its
seq-08 validation mIoU come from
[`val_curves.json`](../results/v06/training/val_curves.json).

| checkpoint | selected epoch (val mIoU) | seq 07 per scan | seq 07 map | seq 09 per scan | seq 09 map |
|---|---|---|---|---|---|
| ZS | — | 64.84 / 65.07 | 71.23 / 70.92 | 61.09 / 60.88 | 67.18 / 67.32 |
| B0 s1 | 3 (0.6703) | 71.91 / 71.55 | 77.32 / 77.31 | 69.38 / 69.46 | 75.25 / 75.45 |
| B0 s2 | 4 (0.6731) | 70.29 / 70.28 | 75.75 / 74.95 | 67.44 / 67.49 | 73.34 / 73.31 |
| B0 s3 | 1 (0.6707) | 72.48 / 72.49 | 79.83 / 79.93 | 67.99 / 68.09 | 73.79 / 73.56 |
| B0 without KL s1 | 2 (0.5326) | 45.82 / 45.99 | 50.15 / 50.18 | 53.69 / 53.94 | 59.82 / 59.91 |
| B0 without KL s2, resumed | 10 (0.6180) | 60.54 / 60.25 | 67.03 / 66.52 | 57.98 / 58.07 | 65.66 / 66.30 |
| B0 without KL s3 | 7 (0.5268) | 45.78 / 45.46 | 49.71 / 49.59 | 50.78 / 50.44 | 56.97 / 57.07 |
| R′ without KL | 7 (0.8697) | 85.50 / 85.23 | 86.19 / 86.05 | 83.45 / 83.68 | 83.78 / 83.94 |

## 1. Decision contamination

Every v0.4/v0.5 design decision was taken while looking at seq-07 numbers. That includes
keeping the KL anchor, the confidence gate and the 0.20 m voxel. Seq 09 was withheld from
them. The report reads contamination as a difference-in-differences on the per-scan score,
against the zero-shot model, which no decision was tuned on:
[arm(07) − arm(09)] − [ZS(07) − ZS(09)].

| arm | seq 07 | seq 09 | seq 07 − seq 09 | minus the zero-shot gap |
|---|---:|---:|---:|---:|
| ZS | 64.96 | 60.99 | +3.97 | — |
| **B0** | 71.50 | 68.31 | +3.19 | **−0.78** |
| B0 without KL | 50.64 | 54.15 | −3.51 | −7.48 |
| R′ without KL | 85.37 | 83.57 | +1.80 | −2.17 |

**B0: −0.78, inside its seed spread** (0.913 on seq 07, 0.820 on seq 09). The sign is the
opposite of contamination: the tuned method loses less between the two sequences than the
untuned model. Read alone, B0's raw gap of +3.19 would look like optimism on seq 07, but the
zero-shot model, which no decision touched, shows +3.97. No decision contamination of B0 is
measurable at this resolution.

The assumption, as the report states it: the two scenes are equally hard for a tuned arm as
for an untuned one. If a tuned arm is differentially better at seq 07's particular scene
content, some of that lands in this number as contamination.

B0 without KL's −7.48 includes the resumed seed (deviation 1). R′'s −2.17 comes from a
single training seed.

## 2. The KL anchor

This section compares B0 with B0 without KL on the same three seeds. The two configurations
differ only in `kl_enabled` and the save path.

| reading | with KL | without | with − without |
|---|---:|---:|---:|
| seq 07, per scan | 71.50 | 50.64 | +20.86 |
| seq 07, map | 77.51 | 55.53 | +21.98 |
| seq 09, per scan | 68.31 | 54.15 | +14.15 |
| seq 09, map | 74.12 | 60.96 | +13.16 |

REPORT.txt titles this section "what the anti-forgetting KL anchor cost" and the last column
"cost of the anchor". That column is with minus without, so it is what *removing* the anchor
costs. The report's "per-point" rows are the per-scan reading here.

**For camera pseudo-label distillation the anchor is necessary.** Every B0 run scores above
every B0-without-KL run on both sequences, resumed seed included
([`runs_summary.json`](../results/v06/scoring/runs_summary.json)). That holds per scan, on
the map per point and per cell. Without the anchor the three seeds spread 6.900 per scan on
seq 07 and 3.037 on seq 09. On seq 07 that spread is the resumed seed: s1 reads
45.82 / 45.99, s3 45.78 / 45.46 and s2 60.54 / 60.25. On seq 09 the three read
53.69 / 53.94, 57.98 / 58.07 and 50.78 / 50.44.

**For GT supervision v0.4 measured the opposite.** Removing the anchor raised D from 74.68
to 81.44, and R′ from 83.36 to 84.90, outside the frustum on seq 07. That was one training
run each, and the KL-on and KL-off runs used different seeds
([v0.4](v04_results.md#completed-no-kl-diagnostic-d_nokl-and-rprime_nokl)). Both are
measurements; neither is explained here.

## 3. Seed spread against pass spread

All values are per scan, out-of-frustum. The between-seed figure is the population SD of the
three per-seed means. The between-pass figure is the mean, over seeds, of the population SD
of each seed's two forward passes. The report calls seed spread real when it exceeds twice
the pass spread.

| arm | seq | between seeds | between passes | report |
|---|---|---:|---:|---|
| B0 | 07 | 0.913 | 0.064 | seed spread is real |
| B0 | 09 | 0.820 | 0.037 | seed spread is real |
| B0 without KL | 07 | 6.900 | 0.131 | seed spread is real |
| B0 without KL | 09 | 3.037 | 0.113 | seed spread is real |

**Three seeds measure what repeated passes of one seed cannot.** B0's seed-to-seed SD is an
order of magnitude above its pass-to-pass SD on both sequences. v0.4's ± for B0, 0.17, was
the SD of one training run's inference draws, so it measured the instrument rather than the
method. The seeds also select different epochs: 3, 4 and 1 for B0. The B0-without-KL spread
includes the resumed seed.

## 4. v0.5's three map findings on a clean sequence

v0.5 measured three structural findings on seq 07 only. Here are the same readings on both
sequences, per scan → map:

| arm | seq 07 | seq 09 |
|---|---|---|
| ZS | 64.96 → 71.07 | 60.99 → 67.25 |
| B0 | 71.50 → 77.51 | 68.31 → 74.12 |
| B0 without KL | 50.64 → 55.53 | 54.15 → 60.96 |
| R′ without KL | 85.37 → 86.12 | 83.57 → 83.86 |

- **(a) The map beats the per-scan predictions it is built from. Holds on seq 09,** for all
  four arms, and in every one of the 32 runs.
- **(b) The margin shrinks as the classifier strengthens. Holds on seq 09:** from +6.81 for
  the weakest arm (B0 without KL) to +0.29 for the strongest (R′), against +4.89 → +0.75 on
  seq 07. The report compares the weakest and the strongest arm; the weakest arm's mean
  includes the resumed seed.
- **(c) The strongest arm loses the planar classes at map level. Not a general finding.** It
  reproduces on seq 07 with the new training run: R′'s road goes 97.29 → 96.03 and its sidewalk
  93.97 → 91.23. On seq 09 road does not lose (92.80 → 93.00), and sidewalk loses by 0.20
  (85.79 → 85.59).

(c) was the finding this baseline was built to produce. The report's rule for it was
archived with the source snapshot before scoring ran: if (c) does not reproduce on seq 09,
it was a seq-07 artefact and must not be claimed. It is not claimed. v0.5's statement
stands as a measurement on seq 07.

## 5. Per cell beside per point

As set by the [evaluation unit](v06_evaluation_unit.md), map results carry both units. The
values are means over each arm's runs. Out-of-frustum cells are those whose scored points
are majority out-of-frustum. The per-cell sparse-oracle relabels every sparse cell
(n_obs ≤ 4) to its GT majority.

| arm | seq 07 map, per point | per cell | per-cell sparse-oracle | wrong cells | seq 09 map, per point | per cell | per-cell sparse-oracle | wrong cells |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ZS | 71.07 | 64.47 | 79.02 | 12.74 % | 67.25 | 55.26 | 71.60 | 16.91 % |
| **B0** | 77.51 | **70.29** | 83.00 | 11.43 % | 74.12 | **61.94** | 78.34 | 15.28 % |
| B0 without KL | 55.53 | 52.44 | 70.48 | 34.14 % | 60.96 | 49.27 | 70.80 | 28.17 % |
| R′ without KL | 86.12 | 79.58 | 89.63 | 6.63 % | 83.86 | 78.76 | 89.22 | 8.29 % |

**Per cell, the gap between the sequences is wider than per point for ZS and B0.** Per
point, B0 moves from 77.51 on seq 07 to 74.12 on seq 09; per cell it moves from 70.29 to
61.94. The zero-shot model moves from 64.47 to 55.26 per cell. R′ barely moves, from 79.58
to 78.76. Relabelling the sparse cells perfectly would lift B0 from 61.94 to 78.34 per cell on
seq 09, and from 70.29 to 83.00 on seq 07.

The next table compares the map with the per-scan predictions per cell. Each cell gets
weight 1, spread over its points; both sides use the same points and the same GT.

| arm | seq 07 per scan → map | seq 09 per scan → map |
|---|---|---|
| ZS | 61.49 → 63.84 | 52.51 → 54.90 |
| B0 | 67.02 → 69.47 | 58.49 → 61.44 |
| B0 without KL | 50.16 → 51.73 | 45.81 → 48.90 |
| R′ without KL | 78.59 → **78.34** | 77.09 → 77.98 |

Per cell, (a) holds for every arm on seq 09. On seq 07 it holds for every arm but R′, whose
map scores below its per-scan predictions in both passes.

## Protocol deviations and disclosures

**1. Job 5 (armB0_noKL_s2) was killed, and its resume did not restore the student.** At
epoch 10, step 687 of 7818, the machine's out-of-memory killer ended the job. The memory was
held by another workload on the shared host. The queue was restarted with
`opt/train_queue_v06_resume.sh`, which skips completed jobs and resumes an interrupted one
from its last checkpoint. Here that was the epoch-9 `model_last.pth`, loaded with
Pointcept's `resume=True`. Pointcept re-seeds at start, so epoch 10's data order and
augmentation differ from an uninterrupted run. That much is the deviation the protocol
recorded before scoring.

The resume also replaced the student's weights, which the protocol did not know.

- **The loader strips every checkpoint key.** Pointcept v1.5.1's `CheckpointLoader` (the file
  on the host is byte-identical to the tagged release) removes the first seven characters of
  every key on one GPU, although no `module.` prefix was ever added.
- **The anchor lands in the student.** The checkpoint's `frozen_backbone.*` is the frozen KL
  anchor: the released model, stored in fp16. Stripped, it becomes `backbone.*` and loads
  into the student.
- **The student's own weights are dropped.** Its `backbone.*` and `seg_head.*` match
  nothing, and `strict=False` drops them silently.
- **The log lists 477 missing keys,** all of them the anchor's or the student head's.
  Simulating the stripping on the checkpoint's key set reproduces that list exactly.

Epoch 10 therefore trained the released model, starting from epoch 9's optimizer state, at
the schedule's final learning rate. The log prints 0.00001 at the epoch's first step and
0.00000 from step 1000. The training log shows the restart in the loss:

| step | loss |
|---|---:|
| first step of the resumed epoch | 0.8021 |
| first step of the whole job | 0.8018 |
| first step of the interrupted epoch 10 | 0.0208 |

The mean logged loss was 0.3868 over the resumed epoch, against 0.1004 over the 687
interrupted steps.

That epoch's checkpoint is the one selected and scored. Its val mIoU is 0.6180, against the
job's own best of 0.5204 (epoch 5) before the interruption and 0.5326 / 0.5268 for the
other two seeds. **armB0_noKL_s2 is not a ten-epoch no-KL run.** The full audit is in
[`val_curves.json`](../results/v06/training/val_curves.json) under
`jobs.armB0_noKL_s2.resume_audit`. That job has not been re-run.

- **Affected:** every B0-without-KL aggregate. That is the arm's row in the main table, the
  gaps in section 2, its spread in section 3, its figure in section 1 and the weakest-arm
  gain in (b).
- **Not affected:** ZS, B0 and R′, and every reading built from them, including B0's
  contamination and seed-spread figures.
- **Direction unchanged:** every B0 run scores above every B0-without-KL run, s2 included,
  so the direction of section 2 does not depend on it.

**2. Training set.** Each epoch draws 15,637 frames from seven sequences, against v0.4's
17,228 from eight, which included seq 09. The v0.6 arms therefore differ from the v0.4/v0.5
arms in training data as well as seed. The seq-07 numbers here are not a re-measurement of
v0.4's B0.

**3. Checkpoint selection** used Pointcept's point-weighted seq-08 val mIoU, as in v0.4: the
epoch with the best val mIoU of ten is kept. It sees neither test sequence and weights
points, not cells. The selected epochs are:

| arm | selected epochs |
|---|---|
| B0 | 3, 4, 1 |
| B0 without KL | 2, 10 (the resumed epoch), 7 |
| R′ | 7 |

**4. Scoring code.** The replays ran `tools/map_eval.py` and `opt/score_v06.sh` with two
edits made after the [source snapshot](../experiments/v06/README.md) and before scoring:

- the voxel-map capacity constants doubled (`cap0` and `CAP` 2^22 → 2^23, `hash_cap`
  2^24 → 2^25). Seq 09's maps hold 6.40–6.48 M voxels and seq 07's 2.69–2.73 M;
- replay parallelism went from 4 to 2.

Nothing else changed apart from the matching comment and log line. The as-run hashes are in
the manifest's `updates` record; `tools/v06_report.py` ran as archived.

## Files and reproduction

| path | content |
|---|---|
| `results/v06/scoring/REPORT.txt` | the report, as written by `tools/v06_report.py` |
| `results/v06/scoring/runs_summary.json` | per-run out-of-frustum readouts of the 32 replays (per scan, map per point in both abstention conventions, per cell, per class, voxel counts), each with its source's SHA-256 |
| `results/v06/scoring/extract_*.json` | the seven student extractions, all PASS |
| `results/v06/scoring/score.log` | the scoring run |
| `results/v06/training/queue.log`, `selfcheck.log` | the training queue and its split self-check |
| `results/v06/training/val_curves.json` | per-epoch seq-08 validation, selected epochs, and the armB0_noKL_s2 resume audit |
| `results/v06/training/configs/` | the seven realised training configurations |

The 32 map-level outputs, the training logs and the checkpoints are not published.
`tools/v06_report.py`, re-run over the 32 outputs, reproduces REPORT.txt byte for byte. The
main table can be recomputed from the per-run file without a GPU or dataset:

```bash
python3 - <<'PY'
import json, statistics as st
runs = json.load(open("results/v06/scoring/runs_summary.json"))["runs"]
for arm in ("ZS", "B0", "B0_noKL", "Rprime_noKL"):
    row = []
    for seq in ("07", "09"):
        for key in ("offline_all", "map_all_lookup"):
            v = [r["outside"][key]["miou9"] for r in runs if r["arm"] == arm and r["seq"] == seq]
            row.append("%6.2f±%.2f" % (st.mean(v), st.pstdev(v)))
    print("%-13s" % arm, *row)
PY
```

Every published file is hashed in the
[manifest](../results/v06/snapshot_manifest.json); the check in
[`experiments/v06/README.md`](../experiments/v06/README.md#environment-and-replay) verifies
them.
