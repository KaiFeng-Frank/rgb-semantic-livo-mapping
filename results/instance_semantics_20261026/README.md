# Shared instance semantics: implementation and measured limits

This change makes visual instances affect **published semantic labels**, using
bounded, revisable instance state. It retains PTv3 and requires no new training.
`INSTANCE_SEMANTICS=active` enables it in the online launcher; association-only
remains the compatible default. See [the interface](../../tools/visual_instances/README.md).

The implementation is functional, but **additional semantic accuracy from temporal
history is not established**. In these development clips, single-frame instance
pooling obtains exactly the same final labels as the complete temporal method.

## Same-forward real-data comparison

Six already inspected nuScenes scenes, 242 keyframes at 2 Hz, front camera only;
8,402,272 scan points, 6,290,255 semantically labelled points. No GT labels or
object boxes enter prediction, association or shared state. All arms use the
same PTv3 and YOLO forwards; only postprocessing differs. The method was revised
on these same development scenes, so this is not a held-out effectiveness claim.
No numerical parameter sweep was performed.

| Method | Native16 mIoU (%) | Corrected points | Newly wrong points |
|---|---:|---:|---:|
| Raw PTv3 | 58.017030 | 0 | 0 |
| Single-frame instance pooling | 58.035945 | 84 | 0 |
| Temporal pooling, without 3D identity veto | 58.035945 | 84 | 0 |
| Temporal + identity veto, without contradiction veto | 58.035686 | 84 | 2 |
| Complete shared state | 58.035945 | 84 | 0 |

The 84 corrections are 35 truck, 20 car and 29 bus points; there is no measured
person improvement here. Aggregate improvement is only **0.018915 percentage
points**. Complete state unifies 6,943 points across 125 component observations;
63 observations consult history. Associated GT thing coverage in the front
camera frustum is 57.82%, but associated points are only 0.175% of the full scan
total. Do not confuse front-camera coverage with whole-scene coverage.

[Full confusion matrices and per-scene metrics](paired/semantic-summary.json)
· [configuration/model/source hashes](paired/semantic-protocol.json)
· [compact derived analysis](analysis.json).
Compressed per-frame evidence includes raw labels/confidences, GT used only in
evaluation, point indices and each arm's result.

## Why consistency was wrong, and what changed

`scene-0092`, frame 18, visual track 12: all six evaluated points are GT **truck**.
Raw PTv3 predicts four car and two truck points. Vision also predicts car, score
0.545. History lifts the car state to about 0.674 and incorrectly turns the two
correct truck points into car. Those two points already support truck at about
0.699 and 0.689. This is an erroneous shared class on the same object, not an ID
swap. The complete rule abstains on that component and discards its semantic
history; it does not pretend that the four remaining raw errors were corrected.

The contradiction gate applies when history alone makes a decision acceptable.
Current-instance pooling remains usable when its own posterior and margin support
the decision. A rejected earlier blanket veto suppressed 81 of the 84 useful
corrections; [that result is retained](development/blanket-veto-rejected/semantic-summary.json).
[The initial 84-correction/2-harm result](development/initial/semantic-summary.json)
and its detailed failure diagnosis are also retained. Raw GPU predictions differ
slightly between separate runs; comparisons above are strictly within one forward
pass, not between those different baselines.

## Identity is checked separately

There are 22 GT-box-majority changes in 180 adjacent visual-ID pairs, including
three tied votes. For those 22 candidate changes, 15 have no credible 3D
component, four start fresh state, and three fail the motion gate. None inherits
the previous semantic state. This does **not** prove all tracker switches are
solved: box-majority labels are a proxy, 16 same-majority pairs also lose continuity
to the motion gate, and gaps/occlusions remain difficult.

State is limited to 128 tracks × four observations with a two-second expiry;
observed peaks here are eight tracks and eleven retained observations. Each
accepted component has one label/confidence; an abstained component is explicitly
invalid and retains raw point labels. Session restart, suspected identity change,
strong new semantic evidence, and received pose-graph revision are handled
separately. Only raw observations enter history. The current dynamic-object layer
does not recycle enhanced labels into the persistent static semantic map.

## Actual ROS delivery

Real scene-0519 camera images and raw LiDAR feed independent YOLO and live PTv3
workers at original 2 Hz. Ego poses are supplied; this is not a live odometry
accuracy experiment. GT is read only by the observer after predictions arrive.
The ROS path interpolates already received ego poses and enforces a causal 60 ms
camera gap. This differs from the offline exact acquisition-pose projection, so
the point counts need not match the offline run.

All **41/41** outputs arrive, including 4,826 associated points in 40 frames.
**34 points are corrected, zero newly wrong**; mIoU is 45.993117 → 46.001225%.
1,434 points receive a valid shared label; both output topics agree exactly.
LiDAR-publish-to-instance-receipt latency: median **86.05 ms**, P95 **115.34 ms**,
maximum **121.15 ms**. No semantic queue drops; final raw-map rebuild parity passes.
[Subscriber report](real-ros/report.json) · [synthetic actual-ROS contracts](ros-fixture.json).
Shared-state CPU processing inside that actual ROS run is median 3.60 ms,
P95 4.47 ms and maximum 8.67 ms; these are component times, not total latency.

The synthetic ROS test separately exercises wrong-initial-class correction,
actual history-induced label change, same-ID replacement, a +100 m pose revision,
session restart and missing-camera fallback. It is a behavior test, not accuracy
evidence. Development harness failures are retained: missing initial DDS discovery,
an incorrectly strong synthetic expectation at a one-second gap, the blanket
veto producing zero real changes, and a Python 3.11 test accidentally loading
Python 3.12 SciPy through ROS PYTHONPATH. Each was corrected and rerun.

## Full LIVO runtime

Two separate 1× replays of the same 127.696 s HKU recording use local RTX 5070 Ti
Laptop GPU, CycloneDDS and the same explicit CPU isolation. Vision is enabled
in **both** arms; only shared instance semantics is active/off. No other GPU
experiment runs concurrently. Both process all 1,277 camera images, produce all
1,274 initialized frontend/corrected poses and all 122 semantic keyframes, with
zero post-initialization missing images and zero visual queue drops.

| Shared state | Corrected pose median | P95 | Maximum |
|---|---:|---:|---:|
| Off | 38.67 ms | 51.29 ms | 87.85 ms |
| Active | 38.08 ms | 51.32 ms | 73.62 ms |

Semantic-input-to-instance-output latency is measured separately: active median
132.12 ms / P95 **471.35 ms** / maximum 602.35 ms, versus off 136.55 / **454.34** /
589.39 ms. These include the existing semantic/map pipeline and begin at the
frontend semantic input subscriber, not the raw sensor. [Timing and boundaries](semantic-delivery-timing.json).
The 51 ms numbers above are pose latency, not semantic latency; semantic output
is still about 1 Hz in the integrated pipeline.

[Active record](hku-active/assessment.json) · [off record](hku-off/assessment.json)
· [exact local launch script](run-hku-local.sh).
The active/off P95 difference is about 0.04 ms in these two runs, not a measured
hard upper bound or a claim of a speedup. Full posterior export and the optional
processing path were enabled, but this HKU clip has **zero associated instance
points**. It validates integration and empty-instance overhead at 10 Hz camera /
about 1 Hz semantic keyframes; it cannot validate populated-instance runtime or
dynamic traffic accuracy. Populated-instance behavior is separately exercised by
the real 2 Hz nuScenes ROS run above. Existing prior control stalls remain in
[the earlier runtime records](../visual_instances_20261026/README.md).

## Verification

24 visual/instance unit contracts pass, including confident-current corrections,
contradictory history rejection, strong new evidence, geometry competition,
appearance veto, pose revision and no enhanced-output feedback. The existing
semantic-memory suite reports 199 passed, one skipped and 128 subtests; 79
graph/map/fusion tests run with two skipped. The native19 contracts also pass.
Logs are retained in [tests](tests/). GitHub runs both integration contract jobs
on the PR. Synthetic data establish behavior; the paired and subscriber reports
provide the real-data accuracy evidence.

## Reproduction and scope

Use the environment and commands in [the implementation guide](../../tools/visual_instances/README.md).
`evaluate.py --semantic-fusion --audit-instances` runs paired inference;
`real_ros.py --instance-semantics active` verifies real subscriber changes;
`validate_ros.py --instance-semantics` runs the synthetic ROS behavior test.
The existing consistency index is pinned by hash. Data, images, weights and full
posterior caches are not copied into Git.

Recompute the saved evidence summary without GPU inference:

```bash
python -m tools.visual_instances.summarize_semantics results/instance_semantics_20261026/paired
```

Fast motion can exceed the conservative bootstrap gate, sparse objects can fail
the component gate, and HSV appearance is not learned ReID. Consistently wrong
current evidence can still produce a consistently wrong accepted class. A moving
traffic sequence with live odometry and denser camera/semantic cadence is still
needed to establish that case; the present evidence does not claim it.
