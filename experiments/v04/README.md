# v0.4 experiment source snapshot

This directory archives the remote experiment implementation captured on
2026-09-24. It accompanies the [results and interpretation](../../docs/v04_results.md).
Files under `source/` are copied byte-for-byte; their original relative paths and
SHA-256 hashes are recorded in
[`results/v04/snapshot_manifest.json`](../../results/v04/snapshot_manifest.json).
The repository's deployed mapping modules are unchanged by this snapshot.

## Contents

| Source path | Role |
|---|---|
| `src/distil_ext.py` | Original 16-way student, marginalised common-9 loss, frozen KL anchor and epoch audit |
| `src/pointcept_ext.py` | Pointcept v1.5.1 setup and SemanticKITTI common-9 dataset |
| `src/filter_e.py` | Pseudo-label filtering used by B0/B1 |
| `src/random_supervise.py` | R's frozen random raw-point selector |
| `src/voxel_random_supervise.py`, `src/distil_ext_rprime.py` | R′'s count-matched selection after voxelisation |
| `src/seqreg.py`, `tools/align_seq.py` | Sequence/calibration registration and sequence 08 frame alignment |
| `tools/make_pseudo.py`, `tools/cache_seg2d_conf.py`, `tools/pseudo_inventory.py` | Camera pseudo-label preparation and rare-class frame weights |
| `tools/train_distil*.py` | Single-GPU training entry points |
| `tools/cache_trained.py`, `src/score_2d_vs_3d.py` | Full-sequence inference and the frozen scoring contract |
| `tools/smoke*.py`, `tools/verify_arm_R*.py` | Original gradient and supervision checks |
| `tools/nokl_20260923/prepare.py` | Preflight/configuration preparation for the no-KL pair |
| `src/Pointcept_v151/configs/semantic_kitti/` | Base experiment configuration and arm overrides |

Final resolved configurations, including the actual random seeds, are in
[`results/v04/training/`](../../results/v04/training/). Completed arms also have all
ten epoch audit rows. The rare-class weights file is the exact input pinned by
the no-KL preflight.

## Environment and replay

This is a research source snapshot with the original absolute paths retained;
it is not a standalone installer. The experiment root was
`/data/wuyou/livo_sem`, with Pointcept v1.5.1 under `src/Pointcept_v151`, the released
nuScenes checkpoint under `weights/`, and prepared data under `data/pointcept_sk`.
The upstream Pointcept checkout, licensed datasets, teacher weights, generated
pseudo-label caches and trained checkpoints are not bundled here.

The no-KL preflight records Python 3.10.21, NumPy 2.2.6 and PyTorch 2.5.1+cu124 on
RTX 4090 GPUs. Do not substitute Pointcept HEAD for v1.5.1. Read the root
[`CRITICAL_CONSTRAINTS.md`](../../CRITICAL_CONSTRAINTS.md) before rebuilding the GPU
environment. Recreate the dataset layout and adjust absolute paths in a separate
working copy if replaying elsewhere; retain this archive unchanged for provenance.

Once the original layout and prerequisites exist, the original entry-point shape
is:

```bash
python tools/train_distil.py --config-file /path/to/resolved/config.py \
    --options save_path=/path/to/new/experiment

# For Rprime or Rprime_noKL, use tools/train_distil_rprime.py instead.
python tools/cache_trained.py --ckpt /path/to/new/experiment/model/model_best.pth \
    --seq 07 --frames all --out /path/to/new/inference_draw_1
```

Run these in the reconstructed experiment root, where the archived modules have
their recorded relative paths. Use new output locations; do not overwrite the
historical runs. Score three independent inference draws with the archived
`src/score_2d_vs_3d.py`. Inference repeats do not substitute for training seeds.
The no-KL protocol's CPU-dispatch setting and hash/count gates are part of its
replay conditions.

The host-specific systemd/SSH driver is not included. Its training, checkpoint
audit, inference and comparison sequence is described by the
[no-KL protocol](../../results/v04/nokl_20260923/protocol.md).

## Host handoff

The Rprime_noKL run was trained on the second host (133). Before that host was
retired, its `model_best.pth`, `model_last.pth`, resolved configuration, ten-epoch
audit and Rprime source modules were copied to the primary host (134) under the
same `/data/wuyou/livo_sem` layout. The final score files and paired comparison in
the repository are the preserved copies. Future training, inference and mapping
integration should use host 134; host 133 is not required for the reported result.

## Historical output interpretation

`tools/arm_verdict.py` is retained as the code that produced the archived verdict
JSON. Its aggregate `PROPAGATION` label does not enforce the additional P3 sign
gate. Some original comments also call C a ceiling or describe D/R contrasts as
geometry-only. The [results document](../../docs/v04_results.md) gives the
qualified interpretation: C retains KL, class mix and KL support differ, and
the historical training seeds are not paired.

To verify the published numbers without the GPU environment, run from the
repository root:

```bash
python3 tools/summarize_v04.py --check
```
