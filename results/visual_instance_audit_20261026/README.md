# Identity changes and semantic changes are separate failures

This audit reproduces the prior six-scene / 242-frame diagnostic with the same
detector, tracker, calibration and thresholds, adding per-instance logging only.
**Every frame's count dictionary matches the archived original.** It explains the
22/180 identity-proxy changes; it does not add semantic fusion or change inference.

The present online implementation attaches a track ID and a current-frame visual
class to a point. It does not maintain a shared semantic posterior for each track,
and it leaves PTv3 point predictions unchanged. A stable ID therefore does not
force a stable class, or make every point in the instance share a PTv3 class.

## What the recorded errors mean

| Checked adjacent pair | Count | Interpretation |
|---|---:|---|
| Same ID, changed majority GT object | 22 / 180 | Identity proxy changes, not semantic flips |
| Of those 22, unchanged visual class | **22 / 22** | Class consistency does not guarantee identity |
| Of those 22, both visual predictions agree with their respective majority GT classes | **19 / 22** | Both frames can be semantically correct while the ID moves to another object |
| Of those 22, a tied GT-object vote on either side | **3 / 22** | Inconclusive identity evidence; original counter depended on tie ordering |
| Same ID and same majority GT object, both visual classes correct | **137 / 158** | Stable correct observations |
| Same ID and same majority GT object, both visual classes wrong | **14 / 158** | Stable error; consistency is not correctness |
| Same ID and same majority GT object, correct → wrong | **4 / 158** | Current-frame class overwrites a previously correct class |
| Same ID and same majority GT object, wrong → correct | **3 / 158** | New observation corrects the current-frame class |

All seven class changes are car/truck confusions. These are **visual hypotheses**,
not changes caused by an instance-conditioned PTv3 semantic smoother: that
smoother does not exist in the current implementation.

GT identity comes from majority votes among assigned points that fall inside a GT
box and have that box's semantic class. It is not official panoptic identity GT.
The original rule only required three total box-supported points; it did not
require a strict majority or reject ties. For example, scene-0092 frame 5 has
three different cars with one vote each. Its chosen token follows encounter order.
Describing all 22 as proven tracker ID switches was too strong.

As an additional support diagnostic, only three changed-identity pairs have at
least ten box-supported points and at least 80% dominant support relative to both
supported and all assigned points **on both sides**. This does not prove the other
19 are correct, and it is not a revised score. Sparse support, mixed masks and
projection may affect their GT majority. The original 180-pair denominator and
22-change count are retained.

## Concrete cases

Frames below use zero-based indices. Full tokens, point votes, source image hashes,
detector boxes and mask bounds are preserved in `cases.json` and the compressed
instance/pair ledgers. Original image paths refer to the existing local dataset;
raw images and point clouds are not copied into Git.

**ID 16 switches between two parked white SUVs, while both classes stay car.**
In scene-0092 frame 13, 26 of its 27 assigned points support instance
`233e77a8…`. In frame 14, all 26 assigned points support a different instance,
`c10945c3…`. Their world centers are approximately 2.8 m apart. Both visual
scores are approximately 0.72, and both labels are car. The original images
were inspected: the detection moves to the neighboring similar white SUV as
the ego vehicle passes the parked row. This case supports a real association
switch, rather than a semantic class inconsistency.

**ID 2 keeps the same vehicle but flips car → truck → car.** In scene-0092
frames 11, 12 and 13, majority identity remains `84fd04db…` (GT class car).
In frames 11 and 12, all 22 and 27 assigned points respectively belong to this
one object. Visual classes are car (0.352), truck (0.490), then car (0.451).
The ID is unchanged; current-frame classification varies. No instance-level
semantic evidence is accumulated to stabilize this decision.

## Why this implementation permits these outcomes

The installed Ultralytics 8.3.228 tracker has two relevant behaviors, checked
against its local source and tagged upstream source:

1. `STrack.update` / `re_activate` assign the newly matched detection's class to
   the track. There is no semantic class history or class-posterior fusion in
   that assignment. Our detector reads this current class into each frame.
2. With appearance ReID disabled, BoT-SORT's first matching stage uses predicted
   boxes, IoU and detection score after global camera-motion compensation.
   It has no class-equality gate in that cost. Our 3D projection checks depth
   locally but does not feed 3D object identity/motion back into this 2D tracker.

Sources: [class update](https://github.com/ultralytics/ultralytics/blob/v8.3.228/ultralytics/trackers/byte_tracker.py),
[association cost](https://github.com/ultralytics/ultralytics/blob/v8.3.228/ultralytics/trackers/bot_sort.py).
Local source hashes and relevant line ranges are in `source-audit.json`.

This diagnostic feeds 2 Hz camera keyframes, about 0.5 seconds apart. Similar
adjacent vehicles and appreciable image displacement are a plausible contributor
to the observed identity switching under the configured matching method. No
frame-rate/ReID ablation was run, so this is not a measured causal attribution
or a claim about accuracy at 10 Hz. The prior HKU 10 Hz test lacks dynamic-object
association evidence and cannot settle this question.

A shared instance label would make observations agree, but could preserve an
initial error or pass evidence between two same-class objects after an ID switch.
The missing design step is a track-level semantic state that can be revised by
new evidence, with identity reliability checked separately from class agreement.
Do not freeze the first class just to obtain zero label flips. Identity quality
and semantic correction/harm need separate measurements.

## Reproduction and checks

```bash
python -m tools.visual_instances.evaluate --index /path/to/index.json \
  --weights /path/to/yolo11n-seg.pt --out /path/to/new-audit --audit-instances
python -m unittest tools.visual_instances.test_association \
  tools.visual_instances.test_lifecycle tools.visual_instances.test_audit -v
```

Use the same optional runtime and existing index as the
[original diagnostic](../visual_instances_20261026/README.md). The audit verifies
its adjacent-pair and changed-identity totals against the evaluator. Thirteen
association/lifecycle/audit tests pass, including independent checks that a class
flip is not an identity switch and a tied GT proxy is not strong identity evidence.
`original-count-parity.json` records exact per-frame count parity with the original.
All audit GT access remains after inference; no GT enters online output.
