# Instance memory: first frozen experiment, 2026-10-29

后续状态：连续图像已有[八日志实例收益](fullrate/README.md)，并完成
[真实 ROS 交付与语义对照](fullrate-delivery/README.md)。实例收益在一个在线片段
的两个重复中保留；新增语义平滑均未通过门槛。下面第一轮训练结果保持不变。
进一步的[概率记忆对照](point-probability/README.md)获得整体 mIoU 收益，但前景
退化使准入失败；[SAM 2 同步特征记忆](sam2-feasibility/README.md)的两次成本检查
也未通过，没有启动完整精度回放或默认部署。
随后[补齐完整调用计时并检查官方编译](sam2-compile/README.md)：等价 CPU 优化使
32 帧组件中位耗时从 85.51 降至 66.84 ms，但仍超预算；编译的掩码等价性与
重复输出检查失败。没有新增语义精度声明。
另一项[DEVA 原生在线记忆检查](deva-feasibility/README.md)完成四次同输入回放，
输出一致但组件中位耗时约 275–280 ms；分阶段确认主要成本在多目标神经网络。
该同步配置停止，没有启动完整精度评价。

**Result: failed the advancement gates.** The implementation and actual ROS delivery work; the memory has not demonstrated a reliable semantic or identity improvement over the matched controls. It remains opt-in.

## Frozen holdout (161 frames, four collection logs)

The pretrained B0 PTv3 / YOLO11n-seg backbones and every input observation are shared. Official training labels supervise 49,264-parameter heads (26,512 semantic trainable parameters). The encoder/identity descriptor is frozen and shared. Each semantic arm has equal capacity and optimization steps. All six selected checkpoints were SHA-sealed before holdout extraction.

This is a head-level project holdout from the official training pool, already seen by the released backbone during pretraining; it is not an official benchmark or whole-system blind test. Native16 scores are not comparable to earlier common9 KITTI scores.

| Seed | Current / memory selected epoch | Current mIoU % | Memory mIoU % | Foreground accuracy gain, pp | Point IDF1 gain vs visual tracker, pp |
|---|---|---:|---:|---:|---:|
| 0 | 30 / 2 | 43.05683 | 43.02074 | -1.1715 | -0.0487 |
| 1 | 4 / 6 | 43.01160 | 42.98476 | -0.8916 | +0.1147 |
| 2 | 13 / 11 | 42.97805 | 42.99340 | +0.5391 | +0.0228 |

Mean foreground gain is -0.5080 pp, below the prespecified +1 pp gate; mean identity gain is +0.0296 pp, also below +1 pp. Neither is positive for all three seeds. Raw full-scan mIoU is 43.01980%. The same memory head with history disabled outperforms its history-enabled output in all three seeds. A positive comparison with raw alone would misattribute current-frame learning to memory.

The original score file reports accuracy over all associated valid points (including a few background points). `holdout-metric-supplement.json` additionally supplies the preregistered strict foreground metric (`0 <= native16 GT < 10`). This is a reporting correction only: every prior confusion-matrix score and count was asserted identical. No checkpoint was reselected.

The identity comparator is the existing visual tracker on the **same selected components**. Only 9,189 / 245,423 GT instance points are associated (3.74% coverage). Misses remain in the denominator. Point IDF1 is 6.099% for the visual tracker and 6.050%, 6.213%, 6.121% for memory. Identity switches drop from 5 to 3/4/3, but that alone is not evidence of better identity accuracy. The current and temporal semantic arms share the same identity module, so their ID gains cannot be attributed to semantic attention. These diagnostics are explicitly defined in code; they are not official PAT.

## Diagnosis on training/development only

The selected seed-0 memory head with no history already gets 9,640/9,641 associated development points correct. Enabling history changes no development point label. There are 31 mismatched-instance records out of 184 identity-scorable records on development. More importantly, on training, the subset with verified same-instance history gains 54 correct points but harms 134: association is not the sole failure. The audit uses GT only after causal matching and prediction. It does not teacher-force tracks.

Training coverage is 474 object observations from 89 strongly supported instances; only one bus and one motorcycle are represented. The failure calls for checking evidence/coverage and the utility of retrieved observations, not a larger attention network. The first split and checkpoints remain closed. Any follow-up uses additional preselected train/dev scenes and a new uninspected head-level holdout if development gates justify evaluation.

## Actual online delivery

Four real ROS runs (current/memory before and after a no-op pose-update fix) feed the original camera and LiDAR at 2 Hz, run both frozen perception frontends live, and publish actual semantics/IDs. All 41 scans and images are processed per run, with no stale/full/order drops, pending work or lost scan output; map replay parity passes. They use supplied ego poses, **not** a running LIVO front end. A separate CTVIO CPU job was active on this shared workstation; these are not isolated hardware benchmarks.

| Final run | Startup s | Median ms | p95 ms | Maximum ms |
|---|---:|---:|---:|---:|
| ros-current-final1 | 14.72 | 92.14 | 126.64 | 246.28 |
| ros-memory-final1 | 15.71 | 105.24 | 142.35 | 304.26 |

Cached CPU memory p95 is 3.66–4.17 ms; it excludes frontends and IPC. Cold execution was up to 529 ms for the first evaluated arm, so it is not hidden in a warm p95. The ROS timings include actual delivery and show sustained 2 Hz operation for this clip, **not 10 Hz semantic SLAM**. Raw PTv3 predictions and asynchronously available associations differ across separate live processes, so the ROS accuracy numbers are not used as paired causal evidence; the frozen same-forward holdout is the accuracy comparison.

The final online-only correctness fix prevents a growing trajectory with unchanged historical poses from resetting instance velocity on every message. It does not execute in cached training/evaluation. Original online runs are retained. No part of this fix is counted as learned improvement.

## Evidence and reproduction

- `seed-*/`: all selected weights, training traces, original holdout scores and metric supplements.
- `holdout-selection-seal.json`: checkpoint hashes sealed before holdout.
- `source-training.tar.gz`, `extractor-source.tar.gz`: exact first-round code snapshots; later metric/online fixes are separately documented.
- `split.json`, `cache-protocol.json`: split/provenance without distributing dataset annotations or images.
- `train-audit-seed0.json`, `dev-audit-seed0.json`: label-sidecar diagnostics.
- `ros-*/`: actual live output reports, events and final states.
- `regression-online-final.log`: 25 causal, identity, serialization and worker-lifecycle tests passed.
- `run-ros-local.sh`: exact local live invocation; data/runtime paths require the documented existing installation.

See [design and reproduction](../../docs/instance_memory.md) and [pre-training protocol](../../config/instance_memory_20261029.json). No new official validation/test claims, arXiv result, leaderboard submission or default merge is made.
