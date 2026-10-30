# ST-Occ paired history diagnostic — protocol frozen, results pending

The official pretrained full-validation reproduction is complete. This next
experiment compares native memory with the same weights receiving native
cold-history inputs on every frame. It is an input intervention, not an
independently trained single-frame baseline, and does not establish a PTv3 gain.

[protocol.json](protocol.json) fixes 16 complete scenes / 642 frames from 16
collection logs, selected by metadata hash before evaluating this contrast.
The pool is the previously evaluated official validation set, not a fresh blind
test. No training, checkpoint selection or output policy fitting is performed.

Only inference histories are reset. The native attention and uncertainty heads
remain, and official mSTCV bookkeeping continues across frames. Real inference
must verify current-feature attention, absent previous predicted motion,
unchanged metric state, matching scene-first predictions, and native rerun
parity with the earlier full validation.

Point evaluation uses the same current LiDAR queries inside the ego grid and
camera mask for both arms. Panoptic GT identities are evaluation-only. Report
point mIoU, corrections and harms, first observation/reappearance/error-recovery
events, car/pedestrian subgroups, temporal class changes, and measured cost.
A missing observation is not automatically an occlusion, and an object recovery
delay at labeled keyframes is not network inference latency.

Bulk data acquisition and inference stay on the remote host. The local GPU is
not used. Raw panoptic targets must pass per-file checks and semantic/point-count
agreement with the separately downloaded lidarseg labels before object scoring.
