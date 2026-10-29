# Independent development coverage audit

**The preregistered training-admission gate failed. No new training was launched.**

Eight collection logs were selected by a fixed SHA256 ordering from the remaining
unused official-training logs; the first scene name in each selected log was used.
No label, description, prediction or score selected these scenes. Selection was
committed in `7d4c107` before feature extraction. All eight scenes and all 321
keyframes are reported; no scene was replaced after inspection. Old train/dev
clips and both earlier and new holdout logs were excluded from selection. The
new follow-up holdout remains unopened. Released-backbone pretraining exposure
still applies, so this is not a whole-system blind benchmark.

| New scene | Trusted observations | Privileged recoverable points beyond current pooling | Actual 50/50-history net corrections |
|---|---:|---:|---:|
| 0149 | 8 | 0 | 0 |
| 0803 | 32 | 0 | +5 |
| 0868 | 7 | 0 | 0 |
| 0383 | 2 | 0 | 0 |
| 0218 | 154 | 13 | -57 |
| 0347 | 20 | 0 | +1 |
| 0420 | 65 | 9 | -22 |
| 1074 | 18 | 0 | 0 |

The 22 privileged opportunities belong to two objects in two logs, below the
fixed minimum of 30 points on five objects. The frozen history control improves
only two of eight logs with observations, rather than a majority. It corrects
27 valid points but newly harms 100; these are actual causal outputs. The
privileged counts instead use GT after prediction to ask whether a same-object
past class contains a missing answer, and are not policy results.

All 20 inspected development scenes combined have 295 privileged opportunities
on five objects in three logs, but reporting that pooled count alone would hide
the failed independent extension. The old training set remains unchanged: 787
opportunities on 11 objects across seven logs. It does not justify selecting a
new model on the uninformative or adverse independent development outcomes.

The exact selected metadata, cache/source/model hashes, inherited-file hashes,
raw-evidence report, extraction log and gate are archived here. The 12 older
development scene files are reused by immutable-path symlinks, not duplicated.
Only 321 selected panoptic annotation files were recovered from the remote
official archive; no images, raw scans or annotation payloads are committed.

The existing visual source contains 2 Hz keyframes only. nuScenes records camera
images at 12 Hz and annotates synchronized keyframes at 2 Hz ([official data
format](https://www.nuscenes.org/data-format)). A separate, fixed full-rate tracking
comparison is therefore a concrete remaining test of temporal identity recovery.
This is a limitation of the current input stream, not a demonstrated explanation
for the failures above. No full-rate result is claimed here.
