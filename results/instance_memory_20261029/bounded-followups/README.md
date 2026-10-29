# Bounded follow-ups: none admitted for deployment

These are development-only experiments on 12 previously inspected scenes (483
keyframes, four collection logs). The new follow-up holdout remains closed. None
changes the first pilot's failed outcome, and none is enabled by default or merged
to main. These comparisons use official native16 labels and panoptic point GT;
they are not no-3D-label experiments or leaderboard scores.

## Track-confirmed fragment admission

Sources: `853cf15`, `d2a4824`; protocol in
`config/instance_fragments_20261029.json`. All six first-round heads remain frozen.
One raw PTv3/YOLO forward exports both original components and weak candidates.
Two past trusted observations must confirm a candidate; candidates cannot update
their own confirmation state. Current-only and unconditionally admitted candidates
are controls. The foreground domain is the same candidate union for every arm.

| Seed | Confirmed candidate observations | Added points | New semantic fixes / harms vs trusted current | Camera point IDF1, confirmed | Camera point IDF1, all fragments |
|---|---:|---:|---:|---:|---:|
| 0 | 9 | 667 | 0 / 0 | 47.56% | 55.81% |
| 1 | 14 | 1,015 | 0 / 0 | 46.11% | 53.02% |
| 2 | 14 | 1,030 | 0 / 0 | 47.94% | 54.35% |

The unchanged visual tracker with all fragments reaches 56.20% camera point IDF1.
The learned confirmation path does not beat this simpler comparator and adds no
semantic corrections in any seed. Geometry-only confirmation fixes 17 points for
seed 0 but harms 16 and 6 for seeds 1 and 2. **The gate failed; no threshold sweep
or online promotion followed.** Full-scan/foreground counts, all controls and
scene-level outcomes are retained in the compressed seed reports.

## Frozen raw-point temporal context

Source: `77a9b9e`; protocol in `config/raw_context_20261029.json`. Each input keeps
all current points first and unchanged, then appends at most 12,000 supported,
nonduplicate past points from one previous scan. The instance translation variant
uses causal visual IDs/centers, not ground-truth object motion. Supplied ego poses
are used, as in the original semantic pilot.

| Input | Full-scan mIoU | Repaired / newly wrong points vs current | Median / p95 ms |
|---|---:|---:|---:|
| Current only | 69.8580% | 0 / 0 | 34.01 / 40.55 |
| Unwarped raw history | 68.5452% | 142,400 / 213,859 | 43.04 / 50.25 |
| Ego-aligned history | 68.6350% | 151,624 / 196,678 | 44.52 / 52.26 |
| Ego + instance translation | 68.5866% | 151,590 / 197,152 | 44.86 / 52.32 |

All four logs lose accuracy. Five predetermined duplicate-current checks change
zero labels. Thus duplicate rows/current voxel replacement do not explain the
failure, but feeding a frozen single-frame backbone historical geometry does not
demonstrate useful temporal inference. **The gate failed.** The slight reduction
in GT-track class changes for ego alignment accompanies *more* wrong predictions;
it is not an identity or semantic improvement.

Timing includes context preparation and its corresponding PTv3 forward, excludes
live detection, IPC and map publication, and is not an end-to-end real-time claim.
The current-only maximum was 807.66 ms on cold execution; full timing is retained.

## Native BoT-SORT feature memory

Source: `f2d3301`; protocol in `config/native_reid_20261029.json`. Both trackers
receive exactly the same YOLO detections and camera-motion estimate. The custom
segmentation predictor now transports the official detector features through NMS,
empty-mask removal and tracker indexing. A local subclass fixes an initialization
order bug in pinned Ultralytics 8.3.228; installed packages are unchanged. The ReID
arm uses the pinned official native-feature path and appearance threshold, with
no additional model, download or training. The original baseline is repaired in
the same interface.

| Geometry | Baseline camera point IDF1 | ReID camera point IDF1 | Baseline / ReID full-scan point IDF1 |
|---|---:|---:|---:|
| Projected mask | 60.8134% | 58.3342% | 7.5888% / 7.1492% |
| Trusted component | 48.9389% | 46.5863% | 4.9891% / 4.6907% |
| Largest component | 56.1960% | 53.5412% | 6.1016% / 5.7304% |

All four logs worsen under every geometric policy. Trusted-component camera
identity switches increase from 17 to 22, with supported transitions declining
from 230 to 213. The native feature memory was populated in 2,209 track-frame
instances; this was not a dormant switch. Baseline component coverage differs from
the original cache at **zero points**, under both component policies.

**The gate failed; the existing tracker remains the default.** Detector features
are not identity-trained descriptors, and this test uses 2 Hz keyframes; it does
not rule out other trained descriptors or full-rate tracking. Those qualifications
are not evidence of improvement in this experiment.

Shared detector forward median/p95: 8.52/12.39 ms; shared GMC: 13.48/24.96 ms.
Baseline tracker: 1.12/2.62 ms; ReID tracker: 1.05/2.29 ms. These component timings
exclude PTv3, IPC and map publication and do not establish deployment latency.

## Raw-evidence availability, without learned association

Source `0eae955` checks causal history on original visual IDs, using the original
16 training scenes and all 12 development scenes. It reads only trusted
components; all controls use the same cached raw point probabilities. GT judges
availability after prediction and never supplies a runtime association.

| Split | Points recoverable from same-object raw history beyond current pooling | Distinct objects | Distinct collection logs |
|---|---:|---:|---:|
| Training | 787 | 11 | 7 |
| Development | 273 | 3 | 1 |

These are privileged availability counts, not achieved corrections. A fixed 50/50
mixture of each current point posterior and the raw historical mean actually
fixes/harms 510/202 training points and 168/53 development points. Of the 115 net
development corrections, 116 come from one log, with -2/+3/-2 elsewhere. Hard
current-frame pooling itself harms more points than it fixes. Thus repeated
observations and a favorable pooled aggregate do not provide broad development
evidence. No further training was launched from this audit. It motivates a
bounded metadata-selected extension to independent development logs, preserving
the still-closed follow-up holdout and reporting every selected log.

## Verification and artifacts

35 tests passed across feature transport, tracker feature initialization,
association, lifecycle, semantic state, raw-context invariants and fragment
admission. An initial test invocation named a nonexistent `test_geometry` module;
the corrected invocation used the existing `test_association` module and passed.
No test failure is hidden as an algorithm result.

The folders contain full reports, per-scene records, frozen cache/model/source
hashes and run logs, without raw images, scans or annotation caches. Large JSON
reports are gzip-compressed. `SHA256SUMS.json` records archived artifact hashes.
The experiments use existing local data and isolated low-priority work; unrelated
CT-VIO CPU experiments were left running. No new training occurred in these three
follow-ups.
