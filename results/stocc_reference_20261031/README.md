# ST-Occ first-step status: preflight passed, inference not yet run

**No ST-Occ accuracy or temporal-consistency result has been obtained.** The
server was shut down by the user during preparation and is expected to be
restarted later. Do not treat installation or strict weight loading as model
reproduction success.

## Verified before loss of SSH

| Item | Observed result |
|---|---|
| Official source | Pinned at `1633f62e2e6677a5fa474905977acfeca4e7819e` |
| Official checkpoint | 275,743,785 bytes; SHA-256 matches the release digest |
| Strict checkpoint load | Passed, 68,794,628 parameters, no missing/unexpected keys |
| Dedicated runtime | Python 3.8.20, PyTorch 1.12.1+cu113, torchvision 0.13.1+cu113 |
| Compiled dependency | Official MMCV 1.6.0 CUDA wheel; GPU NMS executed successfully |
| ST-Occ custom extension | Official source installed successfully with CUDA 11.3/GCC 9 |
| Occupancy archive | Pinned public mirror archive and annotations verified by LFS SHA-256 |
| Extracted labels | 6,019 label files from 150 validation scenes; one inspected file had the expected `(200,200,16)` arrays and class range 0–17 |
| Camera files | First two official archives verified/extracted: 5,484 + 1,902 = 7,386 validation JPEGs; later downloads had started |
| Metadata generation | Initial missing `lidarseg` symlink corrected; original generator restarted, completion not yet observed |
| Actual inference | **0 verified frames** |
| Local GPU | Not used |

These are observations captured from command output before disconnection,
summarized here; the original raw manifests/logs remain on the remote disk and
have **not** all been copied back. The final post-shutdown disk contents and
background-job state must be checked after restart. TASeg was running when
preparation began; the agent did not stop it. Its survival/resumption after the
user's shutdown is not known.

The Occ3D transfer mirror is not an independently authenticated copy of the
author's Google Drive archive. Integrity checks establish identity with the
pinned mirror objects. Full token/shape/coverage validation remains required.

## Saved locations

- Remote root: `/home/wuyou/stocc_reference`.
- Original source: `upstream/`.
- Weight: `downloads/stocc-r50-256x704.pth`.
- Runtime: `env/`; matching CUDA compiler/runtime: `cuda113/`; `env.sh`.
- Camera manifests: `camera-acquisition/*/{state.json,files.json}`;
  `camera-plan.json`; overall completion marker `camera-acquisition/complete.json`.
- Occupancy provenance: `occ3d-acquisition.json`, `occ3d-label-manifest.json`.
- Metadata log: `metadata.log`.
- Strict-load record: `strict-checkpoint.json`.
- Local source archive:
  `/home/frank/semantic_livo_runtime/stocc-reference-20261031/source.tar.gz`.
- Reproduction tooling/protocol: [../../tools/stocc_reference](../../tools/stocc_reference).

Large images, label archives and model weights are not added to Git. The official
checkpoint is recoverable from its public release URL with the saved digest.

## Resume sequence

1. Reconnect to the existing server and inspect GPU, RAM, disk and TASeg state.
   Do not start a second training run. No local-GPU fallback is authorized.
2. Confirm which files survived shutdown and whether any acquisition process
   still exists. Sync the latest tools from this branch: the local evaluator was
   refined after SSH disconnected, so the remote copy may be older.
3. Resume missing camera archives only. Retain the completed manifests and
   verify all 36,114 validation images and 6,019 label tokens. Reuse cached
   weights and environment after checking their digests/versions.
4. Inspect `metadata.log` and the original generated
   `bevdetv4-nuscenes_infos_global_val.pkl`; complete the original generator if
   needed. Keep its original scene bounds and pose protocol.
5. Run a complete available scene with the official `_flicker_eval.py` config.
   Confirm the runtime metadata-container adapter, sequence order, memory
   population, predicted flow, class mapping, memory use and finite outputs.
   The custom evaluation wrapper has only passed syntax checks so far.
6. Run the short evaluation-label intervention and compare prediction hashes;
   then run all 150 scenes with official metrics. Report actual scores and any
   discrepancy with the published 42.13% mIoU / 8.68% mSTCV.

The full task remains incomplete until actual inference/evaluation succeeds.
No result is attributed to our memory mechanism, PTv3 or instance association.
