# ST-Occ official pretrained validation

**The 40-frame smoke test and input-isolation checks pass. Full 6,019-frame
validation is running; no full-set accuracy claim is made yet.** This uses the
authors' six-camera occupancy model, not PTv3 or our semantic repair head. No
model training or validation-driven parameter tuning is performed.

| Check | Observed evidence |
|---|---|
| Official source | `1633f62e2e6677a5fa474905977acfeca4e7819e` |
| Released R50 weight | SHA-256 `68491f55a2208fad884e3b664d0e34b8311688376380f7722e30cac6bde5ffa0` |
| Strict loading | All keys match; 68,794,628 parameters |
| Validation data | 150 scenes, 6,019 labels, 36,114 six-camera JPEGs; complete hash/shape/token audit |
| Metadata | Original transformation functions; all scene indices contiguous and all tokens unique |
| Smoke | Complete scene-0003, 40 frames; mIoU 29.066489%, mSTCV 6.650970% |
| Independent smoke mIoU | Exported prediction hashes verified; CPU confusion-matrix result matches official evaluator |
| GT isolation | First 5 predictions unchanged when evaluation occupancy is constant and GT velocity is zero |
| Full evaluation | Active remote run `runs/full-validation-workers8` |
| Training | None; TASeg stopped during the user's shutdown, epoch-33 checkpoint preserved |
| Local GPU | Unused |

The smoke scene is not representative of the full validation set. Its scores
must not be compared with the authors' full-set **42.13% mIoU / 8.68% mSTCV**.
The published reference and metric configuration are in the
[official model zoo](https://github.com/matthew-leng/ST-Occ/tree/1633f62e2e6677a5fa474905977acfeca4e7819e).

## Runtime and data qualifications

Python 3.8.20, torch 1.12.1+cu113 and the released weights run on the remote
RTX 4090. MMCV 1.6.0 replaces the documented 1.5.2; the input wrapper preserves
the CPU metadata container expected by the original inference code. One exact
integer-shape operation is patched from `prod` to height × width to avoid the
old NVRTC/Ada failure; equality passed on 10,003 integer pairs. The complete
[patch](evidence/compatibility.patch), [digest record](evidence/compatibility.json)
and [environment](evidence/runtime-versions.txt) are saved. Learned arithmetic,
weights, geometry, memory and class mapping are unchanged.

Original nuScenes camera archives were authenticated against current official
S3 ETags. The Occ3D labels came from a pinned public transfer mirror, with
archive SHA-256 and complete local label audits; the mirror was **not separately
matched against the authors' inaccessible Google Drive archive**. Source URLs,
archive and per-file manifests are retained in [evidence/](evidence/).

The first full run used two CPU loading workers, originally selected to share
resources with TASeg. After reboot, TASeg was no longer running. The run was
superseded after 564 frames to use eight CPU workers on the otherwise idle
machine; partial evidence is preserved in [full-validation/](full-validation/).
This is a loader-throughput adjustment, not a model or metric change. All
564 overlapping frames have identical tokens and prediction hashes
([parity record](evidence/loader-parity.json)).

## What this step establishes

Full-set success requires 6,019 unique saved predictions in 150 complete scenes,
official occupancy mIoU and mSTCV, independent exported-prediction/mIoU checks,
and measured runtime. The full evaluation remains incomplete until those
artifacts exist.

Even a successful official-system validation is not an ablation proving the
memory contribution, an instance-tracking result, or an improvement to our
PTv3 pipeline. mSTCV is the official aligned-voxel variability measure, with
GT-speed exclusion and GT-change correction; it is not object identity or
object recovery. Official supplied poses and full-scene memory bounds are
retained. Model-call timing includes official metric bookkeeping and excludes
data loading; total throughput also includes prediction export and evaluation.

## Reproducibility and saved evidence

- [Protocol and tools](../../tools/stocc_reference/README.md).
- [Smoke outputs and evaluator snapshot](smoke-0003/).
- [Evaluation-label intervention](label-control-0003/) and
  [label-plus-speed intervention](label-and-speed-control-0003/).
- [Data audit](evidence/data-audit.json), [metadata audit](evidence/metadata-audit.json),
  [source provenance](evidence/source.json), and [checkpoint provenance](evidence/model-download.json).
- [Pre-shutdown observations](observed_preflight.json) are historical, superseded
  by the post-restart audits above.

Remote working root: `/home/wuyou/stocc_reference`. Images, labels, weights and
full voxel prediction files stay on the remote disk. Git contains tooling,
configuration, digests, controls and aggregate evidence only.
