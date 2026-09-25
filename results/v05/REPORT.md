# v0.5 re-qualification: trained semantic models in the live mapping pipeline

Machine: RTX 4090, Ubuntu 24.04, ROS 2 Jazzy; Pointcept v1.5.1; fp16, shuffle_orders=False, intensity x0.2, grid 0.05; seq07 held-out (read at scoring time only). Files: out/v05/.

## Checkpoint extraction and verification (tools/extract_student.py)

| tag | source epoch / seq08 val | tensors (student / anchor / other) | S1 strict 488/488 | S2 tensors equal fp32 / fp16 | S3 tree | S4 cross vs within agreement (all; margin>2) | NEG control (all) | GT acc distil / plain / worker | verdict |
|---|---|---|---|---|---|---|---|---|---|
| ZS | 48 / 0.7831 | 488 / 0 / 0 | 0 missing, 0 unexpected | 488 / 488 | equal | 0.95847 vs 0.95963; 0.99411 vs 0.99610 | - | 87.99 / 87.82 / 87.76 | PASS |
| B0 | 1 / 0.6831 | 488 / 488 / 0 | 0 missing, 0 unexpected | 488 / 488 | equal | 0.97575 vs 0.97665; 0.99967 vs 0.99970 | 0.89770 | 90.07 / 90.12 / 90.18 | PASS |
| Rprime_noKL | 2 / 0.8678 | 488 / 488 / 0 | 0 missing, 0 unexpected | 488 / 488 | equal | 0.99089 vs 0.99086; 0.99890 vs 0.99913 | 0.89665 | 95.81 / 95.78 / 95.75 | PASS |

## M1 -- performance (re-measured; identical node arguments, only --ptv3-ckpt differs)

Saturated runs: bag rate 2.0, RELIABLE, conf-gate 0.5, expect-voxels 4M, three reps each (a, b, c), interleaved ZS/B0/RP (a,b first, then c). Frame period = node-intrinsic time between consecutive stage-B completions (in-callback perf_counter), i.e. the throughput ceiling, not bag delivery.

| model | rep | period mean / p95 / max (ms) | ceiling Hz | PTv3 worker mean / p95 (ms) | stage A | gate | de-skew | proj | map | pub | stage B | recv / proc / bp-drop |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ZS | a | 57.83 / 65.84 / 119.70 | 17.3 | 57.53 / 61.91 | 2.77 | 3.03 | 5.72 | 4.75 | 25.39 | 4.77 | 43.77 | 1093 / 973 / 120 |
| ZS | b | 56.48 / 62.25 / 126.60 | 17.7 | 56.33 / 58.11 | 2.73 | 2.94 | 5.52 | 4.48 | 23.24 | 4.44 | 40.74 | 1053 / 960 / 93 |
| ZS | c | 58.07 / 62.52 / 371.80 | 17.2 | 57.54 / 59.38 | 2.70 | 2.93 | 5.63 | 4.42 | 23.54 | 4.36 | 41.00 | 1091 / 967 / 124 |
| ZS | a ticks | worker ms first 15 s 57.3, after 57.1 (74 ticks) | | | | | | | | | | |
| ZS | b ticks | worker ms first 15 s 56.1, after 56.6 (73 ticks) | | | | | | | | | | |
| ZS | c ticks | worker ms first 15 s 57.5, after 57.4 (74 ticks) | | | | | | | | | | |
| **ZS** | **mean of 3 / median** | **57.46 / 63.54 (median 57.83 / 62.52)** | **17.3** | **57.13 (median 57.53)** | | | | | | | **41.84** | |
| B0 | a | 57.68 / 62.99 / 156.51 | 17.3 | 57.43 / 59.41 | 2.71 | 3.01 | 5.57 | 4.62 | 23.10 | 4.29 | 40.72 | 1078 / 971 / 107 |
| B0 | b | 58.56 / 66.02 / 357.64 | 17.1 | 58.06 / 59.96 | 2.61 | 3.01 | 5.46 | 4.35 | 22.98 | 4.35 | 40.26 | 1072 / 942 / 130 |
| B0 | c | 57.61 / 61.90 / 130.73 | 17.4 | 57.46 / 59.35 | 2.64 | 2.81 | 5.47 | 4.43 | 23.31 | 4.25 | 40.40 | 1080 / 965 / 115 |
| B0 | a ticks | worker ms first 15 s 57.1, after 57.4 (74 ticks) | | | | | | | | | | |
| B0 | b ticks | worker ms first 15 s 57.4, after 58.1 (73 ticks) | | | | | | | | | | |
| B0 | c ticks | worker ms first 15 s 57.5, after 57.4 (73 ticks) | | | | | | | | | | |
| **B0** | **mean of 3 / median** | **57.95 / 63.63 (median 57.68 / 62.99)** | **17.3** | **57.65 (median 57.46)** | | | | | | | **40.46** | |
| RP | a | 65.53 / 72.50 / 137.73 | 15.3 | 65.41 / 70.22 | 2.54 | 2.96 | 5.77 | 4.49 | 23.71 | 4.26 | 41.32 | 1069 / 840 / 229 |
| RP | b | 58.10 / 66.13 / 121.91 | 17.2 | 57.96 / 61.22 | 2.71 | 3.04 | 5.62 | 4.45 | 23.63 | 4.46 | 41.33 | 1079 / 965 / 114 |
| RP | c | 60.39 / 64.59 / 415.05 | 16.6 | 58.86 / 61.35 | 2.67 | 3.06 | 5.68 | 4.52 | 23.65 | 4.36 | 41.39 | 1097 / 943 / 153 |
| RP | a ticks | worker ms first 15 s 58.3, after 68.2 (73 ticks) | | | | | | | | | | |
| RP | b ticks | worker ms first 15 s 56.8, after 58.3 (74 ticks) | | | | | | | | | | |
| RP | c ticks | worker ms first 15 s 58.3, after 59.5 (74 ticks) | | | | | | | | | | |
| **RP** | **mean of 3 / median** | **61.34 / 67.74 (median 60.39 / 66.13)** | **16.6** | **60.75 (median 58.86)** | | | | | | | **41.35** | |

v0.2 reference (opt/out/stats_cap.json, zero-shot): period 60.98 / 65.99 ms; stage breakdown (stats_after3.json) A 2.71 / gate 2.79 / de-skew 5.07 / proj 4.14 / map 22.19 / pub 4.12; worker 56.43 / 58.58.

Full bag at rate 1.0 (the qualification run; the map .npz used for M2 live scoring and M3):

| model | recv / processed / bp-drop / no-pose | published est. | map voxels | peak VRAM (MiB) | node peak RSS (GB) | worker peak RSS (GB) | wall (s) | PTv3 mean (ms) | period mean (ms) | checkpoint | drop timing |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ZS | 1081 / 1075 / 6 / 0 | 1081 | 2667737 | 1459 | 1.232 | 1.631 | 128.1 | 57.31 | 103.79 | released | drops: first tick with drops at recv 11 (bp-drop 6), final bp-drop 6 over 128 ticks; qdepth mean 0.0019 |
| B0 | 1091 / 1084 / 7 / 0 | 1091 | 2677268 | 1459 | 1.091 | 1.611 | 125.5 | 57.88 | 104.03 | B0_student.pth | drops: first tick with drops at recv 14 (bp-drop 7), final bp-drop 7 over 125 ticks; qdepth mean 0.0046 |
| RP | 1092 / 1086 / 6 / 0 | 1092 | 2708416 | 1459 | 1.173 | 1.621 | 127.2 | 57.89 | 103.79 | Rprime_noKL_student.pth | drops: first tick with drops at recv 13 (bp-drop 6), final bp-drop 6 over 127 ticks; qdepth mean 0.0018 |

v0.2 reference: 1092/1092 processed, 0 dropped, peak VRAM 1459 MiB, node RSS 1.171 GB, 2700844 voxels.

## M2 -- map-level semantic quality (common-9, SemanticKITTI seq07 GT)

Replay = the pipeline's stage B run on the CPU from the SAME cached per-point predictions the offline numbers were scored from (`opt/replay_v05.py`, extends `opt/replay_v03.py`'s map_accuracy). Per subset: out-of-frustum / in-frustum / global. mIoU-9 conventions: `offline_*` and `map_*_inserted` use abstain-excluded (no abstention exists there); `map_all_lookup` and `live` use abstain-WRONG (a point with no voxel is a miss for a map consumer).

### zero-shot (released nuScenes PTv3-m1)  (6 replay draws: ZS_r1, ZS_r2, ZS_r3, ZS_r4, ZS_r5, ZS_r6)

frames 1101 (no-pose 5), map voxels 2701475/2701678/2701778/2701553/2701449/2701372, inserted fraction of evaluated points 0.9741, gated-out points found in a voxel 0.9878

| reading | out mIoU-9 | out acc | in mIoU-9 | in acc | global mIoU-9 | global acc |
|---|---|---|---|---|---|---|
| offline_all (== frozen harness) | 65.01 +- 0.25 | 87.45 +- 0.05 | 65.35 +- 0.19 | 89.38 +- 0.02 | 65.06 +- 0.21 | 87.76 +- 0.04 |
| offline_inserted (pose ok, conf>=0.5) | 67.27 +- 0.27 | 88.38 +- 0.06 | 67.55 +- 0.23 | 90.51 +- 0.03 | 67.34 +- 0.24 | 88.72 +- 0.05 |
| map, per-point GT, inserted pts | 71.63 +- 0.36 | 90.35 +- 0.06 | 71.29 +- 0.48 | 91.59 +- 0.05 | 71.62 +- 0.33 | 90.55 +- 0.05 |
| map, voxel-majority GT (v0.3 def.) | 73.05 +- 0.36 | 91.03 +- 0.06 | 72.30 +- 0.48 | 92.06 +- 0.06 | 72.96 +- 0.33 | 91.19 +- 0.05 |
| map, ALL evaluated pts (lookup) | 70.37 +- 0.37 | 89.49 +- 0.06 | 69.72 +- 0.50 | 90.72 +- 0.06 | 70.28 +- 0.33 | 89.69 +- 0.05 |
| LIVE map (rate-1.0 ROS run, lookup; one draw) | 72.00 | 89.68 | 70.58 | 90.82 | 71.75 | 89.86 |

live map: 2667737 voxels, lookup hit fraction 0.9988

**Out-of-frustum GAP, map-level minus offline (positive = the map is better than the per-scan prediction):** end-to-end +5.36 (all points: 65.01 -> 70.37); of which gate selection +2.26 (65.01 -> 67.27), fusion on the inserted points +4.36 (67.27 -> 71.63), coverage of gated-out/no-pose points -1.26 (71.63 -> 70.37).
LIVE map vs offline (out-of-frustum): +6.99 mIoU-9 (65.01 -> 72.00).

per-class IoU, out-of-frustum, offline -> map (all points): car 94.4 -> 95.7; large_vehicle 38.1 -> 61.1; two_wheeler 38.6 -> 42.0; person 38.2 -> 39.8; road 81.3 -> 83.3; sidewalk 69.4 -> 73.7; terrain 74.3 -> 79.6; vegetation 68.0 -> 72.5; manmade 82.9 -> 85.6

classes the map scores LOWER than the per-scan prediction (out-of-frustum): none

map, ALL evaluated pts, abstain-EXCLUDED convention (out-of-frustum): mIoU-9 70.81, acc 89.94

in-frustum minus out-of-frustum point accuracy: offline +1.93, map (all points) +1.23 -- every voxel is voted on from many viewpoints, so the camera-frustum distinction largely dissolves at map level

### B0 -- pseudo-labels from the 2D teacher, zero human 3D labels (DEPLOYABLE)  (3 replay draws: B0_r1, B0_r2, B0_r3)

frames 1101 (no-pose 5), map voxels 2693408/2693312/2693083, inserted fraction of evaluated points 0.9733, gated-out points found in a voxel 0.9841

| reading | out mIoU-9 | out acc | in mIoU-9 | in acc | global mIoU-9 | global acc |
|---|---|---|---|---|---|---|
| offline_all (== frozen harness) | 72.98 +- 0.17 | 90.14 +- 0.03 | 70.61 +- 0.34 | 87.49 +- 0.07 | 72.61 +- 0.19 | 89.72 +- 0.03 |
| offline_inserted (pose ok, conf>=0.5) | 75.60 +- 0.14 | 91.09 +- 0.03 | 72.07 +- 0.28 | 88.26 +- 0.07 | 75.04 +- 0.14 | 90.63 +- 0.03 |
| map, per-point GT, inserted pts | 79.06 +- 0.22 | 92.11 +- 0.05 | 78.58 +- 0.31 | 92.21 +- 0.03 | 79.05 +- 0.23 | 92.13 +- 0.04 |
| map, voxel-majority GT (v0.3 def.) | 80.83 +- 0.26 | 92.80 +- 0.04 | 79.85 +- 0.31 | 92.71 +- 0.03 | 80.71 +- 0.26 | 92.79 +- 0.04 |
| map, ALL evaluated pts (lookup) | 77.49 +- 0.19 | 91.13 +- 0.05 | 77.28 +- 0.41 | 91.57 +- 0.02 | 77.49 +- 0.23 | 91.20 +- 0.04 |
| LIVE map (rate-1.0 ROS run, lookup; one draw) | 77.82 | 91.54 | 77.69 | 91.94 | 77.85 | 91.60 |

live map: 2677268 voxels, lookup hit fraction 0.9994

**Out-of-frustum GAP, map-level minus offline (positive = the map is better than the per-scan prediction):** end-to-end +4.51 (all points: 72.98 -> 77.49); of which gate selection +2.62 (72.98 -> 75.60), fusion on the inserted points +3.46 (75.60 -> 79.06), coverage of gated-out/no-pose points -1.57 (79.06 -> 77.49).
LIVE map vs offline (out-of-frustum): +4.84 mIoU-9 (72.98 -> 77.82).

per-class IoU, out-of-frustum, offline -> map (all points): car 95.6 -> 96.6; large_vehicle 55.5 -> 68.8; two_wheeler 45.4 -> 53.0; person 60.6 -> 67.7; road 85.0 -> 86.7; sidewalk 76.3 -> 78.8; terrain 79.9 -> 83.2; vegetation 72.7 -> 75.7; manmade 85.7 -> 86.8

classes the map scores LOWER than the per-scan prediction (out-of-frustum): none

map, ALL evaluated pts, abstain-EXCLUDED convention (out-of-frustum): mIoU-9 78.03, acc 91.60

in-frustum minus out-of-frustum point accuracy: offline -2.65, map (all points) +0.44 -- every voxel is voted on from many viewpoints, so the camera-frustum distinction largely dissolves at map level

### Rprime_noKL -- randomly-scattered GT supervision (UPPER BOUND ONLY, not deployable)  (3 replay draws: RP_r1, RP_r2, RP_r3)

frames 1101 (no-pose 5), map voxels 2721905/2722199/2722034, inserted fraction of evaluated points 0.9926, gated-out points found in a voxel 0.9623

| reading | out mIoU-9 | out acc | in mIoU-9 | in acc | global mIoU-9 | global acc |
|---|---|---|---|---|---|---|
| offline_all (== frozen harness) | 84.89 +- 0.24 | 95.94 +- 0.01 | 86.99 +- 0.07 | 95.72 +- 0.01 | 85.37 +- 0.20 | 95.90 +- 0.01 |
| offline_inserted (pose ok, conf>=0.5) | 85.25 +- 0.24 | 96.06 +- 0.01 | 87.25 +- 0.08 | 95.84 +- 0.01 | 85.71 +- 0.21 | 96.02 +- 0.01 |
| map, per-point GT, inserted pts | 85.99 +- 0.15 | 95.73 +- 0.00 | 87.69 +- 0.03 | 95.46 +- 0.01 | 86.40 +- 0.11 | 95.69 +- 0.01 |
| map, voxel-majority GT (v0.3 def.) | 88.22 +- 0.16 | 96.65 +- 0.01 | 89.29 +- 0.03 | 96.07 +- 0.01 | 88.51 +- 0.12 | 96.56 +- 0.01 |
| map, ALL evaluated pts (lookup) | 85.25 +- 0.14 | 95.18 +- 0.00 | 86.49 +- 0.03 | 94.99 +- 0.01 | 85.55 +- 0.11 | 95.15 +- 0.01 |
| LIVE map (rate-1.0 ROS run, lookup; one draw) | 85.54 | 95.59 | 87.39 | 95.37 | 85.98 | 95.56 |

live map: 2708416 voxels, lookup hit fraction 0.9997

**Out-of-frustum GAP, map-level minus offline (positive = the map is better than the per-scan prediction):** end-to-end +0.35 (all points: 84.89 -> 85.25); of which gate selection +0.35 (84.89 -> 85.25), fusion on the inserted points +0.74 (85.25 -> 85.99), coverage of gated-out/no-pose points -0.74 (85.99 -> 85.25).
LIVE map vs offline (out-of-frustum): +0.65 mIoU-9 (84.89 -> 85.54).

per-class IoU, out-of-frustum, offline -> map (all points): car 98.2 -> 97.3; large_vehicle 61.8 -> 64.4; two_wheeler 83.0 -> 84.9; person 67.7 -> 72.7; road 96.9 -> 95.3; sidewalk 93.5 -> 90.5; terrain 83.9 -> 83.4; vegetation 85.7 -> 86.3; manmade 93.2 -> 92.4

classes the map scores LOWER than the per-scan prediction (out-of-frustum): car (98.2 -> 97.3), road (96.9 -> 95.3), sidewalk (93.5 -> 90.5), terrain (83.9 -> 83.4), manmade (93.2 -> 92.4)

map, ALL evaluated pts, abstain-EXCLUDED convention (out-of-frustum): mIoU-9 85.79, acc 95.64

in-frustum minus out-of-frustum point accuracy: offline -0.22, map (all points) -0.19 -- every voxel is voted on from many viewpoints, so the camera-frustum distinction largely dissolves at map level


## M3 -- RViz2 screenshots (DISPLAY=:0, captured from the latched TRANSIENT_LOCAL map after each run)

Files: out/v05/rviz_<model>_<view>_<rgb|class>.png with model in {ZS, B0, RP}, view in {top (overview, focal 20,93,-3, distance 360, pitch 1.50), obl (street view, focal 4.6,21.7,-0.5, distance 55, pitch 0.42), zA, zB (10 m tiles below)}. Class view = RViz Intensity transformer on the `class` channel, bounds 0..15, rainbow (legend: out/v05/rviz_class_legend_rainbow0-15.png; car yellow, driveable_surface cyan-blue, sidewalk blue, terrain violet, manmade purple, vegetation magenta, truck cyan).

Live-map disagreement ZS vs B0: 2636599 voxels matched by key, 247948 differ (9.40 %). Top 10 m tiles (centre; voxels; differing; accuracy of A / B on GT-labelled voxels; dominant changes):
- (-115.1, 143.8, -1.1): 6963 voxels, 3361 differ (48.3 %), acc A 87.8 / B 43.5, manmade->vegetation x2475, road->sidewalk x589, manmade->sidewalk x149
- (-95.6, 155.5, -1.5): 10170 voxels, 2693 differ (26.5 %), acc A 67.0 / B 93.7, road->sidewalk x1477, manmade->vegetation x622, terrain->vegetation x255
- (-95.9, 4.4, 4.0): 10248 voxels, 2579 differ (25.2 %), acc A 42.6 / B 84.9, road->sidewalk x1711, manmade->vegetation x630, manmade->person x62
- (-15.3, 145.3, -3.8): 8025 voxels, 2559 differ (31.9 %), acc A 69.5 / B 81.9, manmade->vegetation x1511, road->sidewalk x824, manmade->sidewalk x85

Live-map disagreement B0 vs Rprime_noKL: 2667129 voxels matched by key, 288122 differ (10.80 %). Top 10 m tiles (centre; voxels; differing; accuracy of A / B on GT-labelled voxels; dominant changes):
- (-74.7, 156.1, -1.9): 11850 voxels, 2799 differ (23.6 %), acc A 62.1 / B 97.7, manmade->vegetation x1661, road->sidewalk x353, vegetation->manmade x304
- (-35.3, 155.4, -3.8): 7660 voxels, 2798 differ (36.5 %), acc A 72.5 / B 96.2, road->sidewalk x1023, large_vehicle->car x620, manmade->car x373
- (4.3, 14.3, -0.3): 17128 voxels, 2559 differ (14.9 %), acc A 90.6 / B 89.2, large_vehicle->manmade x1183, road->sidewalk x596, manmade->large_vehicle x179
- (13.5, -14.2, 1.0): 8058 voxels, 2550 differ (31.6 %), acc A 89.6 / B 86.9, vegetation->manmade x1503, manmade->vegetation x865, manmade->sidewalk x52

captured PNGs (25): rviz_B0_obl_class.png, rviz_B0_obl_rgb.png, rviz_B0_top_class.png, rviz_B0_top_rgb.png, rviz_B0_zA_class.png, rviz_B0_zA_rgb.png, rviz_B0_zB_class.png, rviz_B0_zB_rgb.png, rviz_RP_obl_class.png, rviz_RP_obl_rgb.png, rviz_RP_top_class.png, rviz_RP_top_rgb.png, rviz_RP_zA_class.png, rviz_RP_zA_rgb.png, rviz_RP_zB_class.png, rviz_RP_zB_rgb.png, rviz_ZS_obl_class.png, rviz_ZS_obl_rgb.png, rviz_ZS_top_class.png, rviz_ZS_top_rgb.png, rviz_ZS_zA_class.png, rviz_ZS_zA_rgb.png, rviz_ZS_zB_class.png, rviz_ZS_zB_rgb.png, rviz_class_legend_rainbow0-15.png

