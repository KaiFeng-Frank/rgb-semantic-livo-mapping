Preserves trained semantic models and the evidence needed to recover inference,
training state and component comparisons after the experiment host is unavailable.

The experiment source is `fb36c69337dce37cb1195defafcbf0723d3d7693`.
The source repository already contains the clean-seed reports; the Release adds
the binary artefacts that Git did not contain.

Start with **RESTORE.md**, **artifact-manifest.json**, and **SHA256SUMS**.
The manifest maps asset names to their original paths and hashes. Uploads are
verified against GitHub's SHA-256 digest before this release is published.

Includes the seven v0.6 inference students and their selected/last training
checkpoints, the v0.5 models used in online tests, available v0.4 component models,
online timing/pose/map evidence, runtime/configuration metadata, and paired
ZS/B0/no-KL prediction caches for sequences 07 and 09.

The no-KL seed-2 model is the verified **from-scratch clean rerun**. The withdrawn
resumed model is excluded. Dataset bags/raw images/point clouds are not bundled;
restore them from their original distributions. A cached-prediction video is not
a new latency measurement. Existing latency results refer to the original RTX 4090.
