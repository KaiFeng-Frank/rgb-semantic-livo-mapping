# Recover the trained models and experiment evidence

The source repository alone does not contain trained weights. The accompanying
[v0.6 artefact release](https://github.com/KaiFeng-Frank/rgb-semantic-livo-mapping/releases/tag/v0.6-artifacts-20260928)
preserves the experiment outputs that would otherwise depend on the original host.
The experiment source commit is `fb36c69337dce37cb1195defafcbf0723d3d7693`.

Download `artifact-manifest.json`, `SHA256SUMS` and `RESTORE.md` first. The manifest
gives each asset's exact size, SHA-256, category and original relative destination.
For a downloaded asset, verify its checksum before loading it. Each tar archive
has a separate `.index.json` with checksums for every member.

## What is preserved

- Seven v0.6 inference students: B0 and B0 without KL, three seeds each; R′ without
  KL, one seed. `v06-armB0_s1-student.pth` is a fixed seed-1 example for inference,
  not a claim that it is the best seed.
- Both the selected and last full training checkpoint for those seven runs,
  including the training state they originally contained.
- The v0.5 ZS/B0/R′ inference models used for the original online measurements
  and screenshots, and available v0.4 component-reference selected checkpoints.
- Online pose, per-frame timing, subscriber-delivery and map records; v0.5 maps
  and figures; runtime source, configurations, training logs and environment
  package versions.
- Full sequence-07 and sequence-09 prediction caches for ZS, B0 seed 1 and B0
  without KL seed 1, pass 1. These preserve paired examples for later videos and
  component illustrations. The other prediction caches can be regenerated, with
  the documented run-to-run variation.

The no-KL seed-2 assets come from `armB0_noKL_s2_clean`. The defective resumed
checkpoint is not an inference or training asset in this release; its withdrawal
and provenance remain documented in the source repository.

## Restore

For inference, download only the desired `*-student.pth` first (about 185 MB).
Use the corresponding extraction report in `results/v06/scoring/` to cross-check
its hash. Restore it under the manifest's `restore_to` path, or pass an explicit
checkpoint path to the versioned worker. Full `*-best.pth` and `*-last.pth`
files are training checkpoints, not extracted inference students.

Extract evidence/runtime tar files into an empty experiment directory; they keep
paths such as `out/v06/runs/` and `src/`. Use the archived Pointcept v1.5.1 source
and the existing verified loader. Follow `CRITICAL_CONSTRAINTS.md` and
`experiments/v06/README.md`; never fix a model mismatch by dropping state keys.

Raw KITTI/SemanticKITTI data and ROS bags are not duplicated in this release.
Restore those from their original dataset distributions, using the project's
frame-alignment and calibration instructions. Predictions are not RGB images
or raw point coordinates. The caches alone therefore cannot render a driving
video. Upstream datasets and pretrained models retain their original terms.

An offline video rendered from cached predictions shows segmentation output,
not new inference or display timing. The saved timing evidence belongs to the
original RTX 4090 runs; new hardware needs a fresh measurement.
