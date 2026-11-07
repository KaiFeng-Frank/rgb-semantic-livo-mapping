# Local RTX 5070 Ti inference and input admission

**Strict model/kernel validation passed. Direct GPU inference passed. Online input admission did not pass.**

The local RTX 5070 Ti Laptop GPU runs the frozen native 16-class B0 PTv3 checkpoint. Strict loading matches all 488 state tensors (46,159,312 parameters). CUDA scatter, sparse convolution and variable-length FlashAttention agree with separate reference operations within the recorded tolerances. The recovered Pointcept source matches the archived per-file hashes; the weight SHA256 matches the GitHub release.

Six fixed real KITTI seq07 scans warm each direct timing run; frames 120–139 are measured. Results include the client transport, voxelization and model execution. These are three short timing runs, not a frame-rate guarantee or a full online LIO test.

| Attempt | Frames | Median, ms | P95, ms | Maximum, ms | Over 80 ms |
|---|---:|---:|---:|---:|---:|
| admission-20261025-retry2 | 20 | 70.20 | 80.97 | 82.18 | 2 |
| admission-20261025-complete | 20 | 69.49 | 75.95 | 77.48 | 0 |
| admission-20261025-ready | 20 | 63.22 | 66.72 | 68.46 | 0 |

**Input delivery prevents a valid deadline conclusion.** The earlier geometry probes receive 9/26, 17/26 and 5/26 published clouds. The final probe explicitly waits for worker initialization and DDS matching and isolates geometry/online domains; both paths still receive 0/26. Therefore the frozen requirement of complete ordered coverage and at least 95% timely semantic scans fails. Empty log predicates for byte preservation or timely inference are vacuous and are not treated as successful verification. The cause of this large-cloud delivery failure remains unresolved; it cannot be attributed to model inference because the geometry control also fails.

The initialized-worker wait fixes a benchmark weakness: DDS discovery alone could let a probe send before the GPU worker finished starting. Every prior failed attempt is preserved, and none is replaced with a selected passing timing run. Deadline (80 ms), point samples, model/configuration, and admission threshold were unchanged.

The isolated runtime is `/home/frank/semantic_livo_runtime/venv`, reusing the existing CUDA-12.8 PyTorch installation. FlashAttention 2.8.3 is built for sm120 with the upstream switches disabling backward, dropout, ALiBi, softcap and local-window attention, all unused by this frozen inference configuration. Kernel sources are unchanged; [build patch](flash-setup.patch), [options](flash-build-options.json), [wheel hashes](wheel-manifest.json), and [runtime checks](runtime-ready.json) preserve the exact setup. This runtime is for inference, not training.

Original scans stayed in RAM and were released when each probe exited. Only sample hashes, class histograms, protocol/source hashes, logs and timing records are preserved. No raw point files, images, model weights or binary wheels are added to Git. Task-generated compilation intermediates and the interrupted archive download were cleaned after installation.

[Decision and all attempts](assessment.json), [final protocol](admission-20261025-ready/protocol.json), [final complete result](admission-20261025-ready/result.json), [model and source recovery](recovery.json). Before claiming online performance, fix the PointCloud2 delivery path and repeat the same protocol. The separate seq09 accuracy admission also failed.
