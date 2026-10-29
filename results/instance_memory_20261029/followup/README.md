# Bounded follow-up diagnosis, no additional training

The [preregistered follow-up](../../../config/instance_memory_followup_20261029.json)
kept the architecture frozen and required evidence availability across at least
30 points, five distinct GT instances and two collection logs before more training.
Metadata-only time separation selected eight added development scenes (322 frames)
inside the original four development logs. The original frozen extractor was used;
all six source hashes and the backbone hash exactly match the first export. Original
train/dev caches are inherited only after exact scene metadata equality and file
hash checks. New test logs were preselected but their annotations/cache were not opened.

## Result: the data gate failed

There are 262 new object observations and 1,323 foreground errors from the fixed
seed-0 current head. GT-filtered past evidence could correct 798 point errors, but
all are on **two trucks in one collection log**, scene-0398. This privileged count
is an evidence-availability diagnostic, not a deployable algorithm score. It does
not justify repeating more training against a narrow story. No added training was
launched and no new holdout was unsealed.

The expanded 12-scene development audit of the existing memory head finds 40
corrections and 26 new errors compared with the SAME head with history disabled.
That is not evidence of a gain over the separately trained matched current control,
and this expanded development has already been inspected. The first held-out
failure remains the reported algorithm result.

## Where coverage is lost

| Stage, eight added scenes | GT instance points |
|---|---:|
| All scanned instances | 881,514 |
| Classes supported by the existing detector | 860,404 |
| Inside front camera field of view and depth limits | 62,467 |
| Supported classes inside field of view | 60,850 |
| Supported classes passing sparse depth visibility | 60,802 |
| Associated with visual instance masks | 36,797 |
| Remaining after connected-component selection | 21,884 |

The component stage rejects 249 fragmented groups containing 10,971 GT instance
points, plus 64 groups with insufficient points containing 157 GT instance points.
Trimming accepted groups removes further points. **No oversized-object rejection
occurred.** These numbers identify where evidence is lost; they do not establish
that relaxing geometric guards is safe. GT is used only by this observer, never
by the detector or association rules. Replayed final component indices exactly
match every original cached frame (symmetric difference zero).

With fixed coverage, visual-tracker point IDF1 is 4.5143%; even perfect assignment
of all currently associated GT points has an optimistic upper bound of 4.8436%.
Thus the original +1 pp whole-scan identity gate is unattainable on these clips.
The bound is diagnostic and was added after the first pilot failed; it does not
retroactively pass the first candidate. A subsequent protocol needs an explicitly
fixed camera-observable tracking domain while retaining whole-scan coverage and
semantic errors, rather than silently changing the old score denominator.

The practical next work is to measure purity of rejected fragments and find diverse
real correction examples in TRAINING logs, before changing geometry or training.
The current evidence does not support adding model capacity. The follow-up's
preselected test logs remain untouched for a separately preregistered evaluation.

`opportunity.json` contains the frozen-head availability diagnostic; `coverage.json`
contains stage/frame counts; `dev-audit-seed0.json` contains history-read diagnostics.
Raw scans, images, per-point annotations and caches remain outside Git.

## Rejected-fragment purity audit

An observer-only replay examined the existing largest component when the 60%
whole-mask fraction veto is removed; live gating and every cached prediction
remain unchanged. Of 249 rejected groups, 168 contain a component with at least
six points. They would add 6,595 points, of which 6,457 have nonignored panoptic
labels; 6,022 belong to each group's dominant true instance (93.26% weighted
purity). There are 149 groups with purity at least 90%, but 19 below it. Thus a
blanket relaxation also introduces mixed/background evidence. Raw PTv3 makes
304 semantic errors over the newly considered valid points; this is not a count
of errors a memory model has corrected.

This supports investigating *track-confirmed admission of candidate fragments*,
using previously established appearance/motion state without semantic-class
agreement or GT input. It does not justify removing the ambiguity veto globally,
and it is not an implementation or validated gain of that next candidate. Any
such change needs a same-geometry single-frame control and separate attribution
for recovered coverage versus semantic memory readout. The current candidate
still fails its original gates.
