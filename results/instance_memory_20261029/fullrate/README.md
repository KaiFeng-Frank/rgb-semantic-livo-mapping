# Full-rate causal images: identity improvement on development

**Projected-mask identity passes the frozen continuation gate. Semantic accuracy
and end-to-end online delivery are not demonstrated by this experiment.**

The same eight metadata-selected independent development logs and the same 321
scored LiDAR keyframes are used. We recovered 1,536 intervening CAM_FRONT images,
giving 1,857 images within exactly the original keyframe time bounds. Their
timestamps come from official sample-data metadata. [nuScenes records cameras
at 12 Hz and annotates synchronized keyframes at 2 Hz](https://www.nuscenes.org/data-format).
Actual selected sequences have median intervals of 100 ms and maximum intervals
of 150–250 ms. No interpolated/generated images or future images are used.

Every image runs the same frozen YOLO11n-seg. At scored keyframes the two trackers
share the exact same current detections; one updates only at keyframes, the other
also uses the intervening past images. Both use the original non-ReID BoT-SORT
matching settings and separate causal camera-motion histories. The nominal lost
buffers are 15 keyframes and 90 images, with the same additional 7.5-second real
timestamp expiry. This is **additional temporal evidence for a mature tracker**,
not a new Transformer or a new network contribution. No new training occurred.

## Frozen results

| Geometry | Keyframe camera point IDF1 | Full-rate camera point IDF1 | Gain | Positive logs | Gate |
|---|---:|---:|---:|---:|---|
| Projected mask | 55.4713% | 60.5425% | +5.0712 pp | 5 / 8 | Pass |
| Trusted component | 48.1352% | 49.6105% | +1.4753 pp | 4 / 8 | Fail |
| Largest component | 56.2071% | 61.5709% | +5.3638 pp | 4 / 8 | Fail |

The gate requires at least +1 pp camera point IDF1, no full-scan point IDF1 loss,
and improvement in a strict majority of the eight logs. All three prespecified
geometry policies are reported; the failed policies are not renamed successes.
The new follow-up holdout remains unopened.

For projected masks, full-scan point IDF1 rises from 7.2850% to 8.2646%; camera
association quality rises from 0.08444 to 0.14838. Camera identity switches fall
from 49 to 42 while supported transitions rise from 479 to 537 and associated
GT points rise from 24,330 to 26,295. Thus the switch reduction does not come from
tracking fewer points. These are defined point metrics, not official nuScenes PAT.

Improvements are uneven: 0149, 0868 and 1074 gain substantially; 0383 and 0218
also improve, while 0803, 0347 and 0420 worsen. This is development evidence for
retaining full-rate causal input, not a claim that every scene improves.

## Attribution and parity

Before scoring, identical three-frame inputs produced identical pixel-index
masks in both arms (11 instances in each frame). During scoring, the new
keyframe comparator differs from the older extraction cache at 24 component
points, all in scene 0218. A same-forward diagnostic compared the timestamp-capped
tracker with the original tracker: the **original tracker matches the old cache
at every point**, while the capped tracker reproduces the same 24 differences.
This identifies the shared timestamp-expiry change as the cause. The table above
compares the two equally timestamp-capped arms; it is explicitly not a claim of
bit-identical native-versus-modified tracking. The diagnostic changed no thresholds
or scored policies. Its source and full per-frame output are retained.

## Runtime and acquisition

Measured detector plus full-rate tracking/GMC time is median 17.32 ms, p95 21.65 ms,
maximum 36.86 ms over 1,857 images. These are paired per-image sums, not sums of
separate percentiles. First model setup took 1.83 s. Camera-to-point association,
mask-message packaging, PTv3, IPC and map publication are excluded. This is an
offline causal evaluation, **not an end-to-end real-time ROS result**.

The source ZIP is 24.38 GB, but only selected members were fetched remotely using
validated HTTP ranges: 232.20 MB of images and 288.98 MB transferred by the
successful fetch. An initial mirror timeout was incorrectly wrapped by Python's
ZIP parser as a corrupt archive; the ZIP64 end record and directory were verified,
and reuse of the resolved CDN address completed acquisition. No full ZIP was
downloaded. Each selected JPEG was verified by member size, CRC32 and SHA256;
the evaluator rechecks SHA256 and official metadata dimensions. Temporary signed
download URLs are never stored in the manifest. Images remain in runtime storage,
not GitHub; the pinned source revision, member hashes and request list are archived.

Source/protocol commits: `ed40033`, `40a38c4`, `21f2912`, `5c1de24`, `e3da7d2`.
Full reports, per-scene frame timings, parity audit, acquisition manifest and
logs accompany this file. The range-reader tests passed, including refusing a
full-archive response and unbounded reads. The next required work is actual
online delivery and independently verified semantic benefit; this result alone
does not complete the semantic-and-instance objective or authorize a default merge.
