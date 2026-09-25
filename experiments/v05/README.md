# v0.5 experiment source snapshot

This directory archives the code of the v0.5 re-qualification, captured on 2026-09-26. It
accompanies the [results and interpretation](../../docs/v05_results.md). Files under
`source/` keep their original relative paths; each one's original path and SHA-256, and
the published SHA-256 where the two differ, are recorded in
[`results/v05/snapshot_manifest.json`](../../results/v05/snapshot_manifest.json). The only
edits are path sanitising: the host-specific data root is rewritten to `/data/`.

## Contents

| source path | role |
|---|---|
| `src/ptv3_loader_verified.py`, `src/ptv3_worker.py`, `src/ptv3_client.py`, `src/semantic_map_node.py` | the four mapper modules at the end of v0.5, with the `--ptv3-ckpt` selector threaded node → client → worker → loader |
| `tools/v05_patch.py` | the selector edit as applied: exact-string substitutions, each required to match once |
| [`checkpoint_selector.diff`](checkpoint_selector.diff) | the same change as a unified diff against the pre-v0.5 modules |
| `tools/extract_student.py` | student extraction from a trained checkpoint and the four checks S1–S4 with a negative control |
| `opt/runs_v05.sh` | timing runs: saturated (bag rate 2.0) and full bag at rate 1.0; identical arguments except `--ptv3-ckpt` |
| `opt/replay_v05.py` | map-level scorer: CPU replay of stage B from cached predictions; offline, gated, map and live-map readings |
| `opt/rp_cache_v05.sh`, `opt/replay_batch_v05.sh` | the RP inference draws and the replay batch |
| `opt/v05_orchestrate.sh`, `opt/v05_orchestrate2.sh`, `opt/v05_post.sh` | sequencing: GPU jobs never overlap, timing runs only on a quiet machine, CPU post-work after timing |
| `tools/v05_zoom_region.py` | picks the RViz zoom tiles where two live maps disagree most |
| `opt/capture_v05.sh`, `opt/v05_captures.sh`, `opt/v05_captures2.sh` | RViz2 captures from the latched map |
| `tools/v05_report.py` | writes `REPORT.md` and `summary.json` from the measured files |
| `out/v05/chk_b0_rp.py` | checks that the replay's `offline_all` reading reproduces the frozen v0.4 scorer per draw |

The repository's `src/` still holds the pre-v0.5 modules; apart from the interpreter-path
sanitising in `ptv3_client.py` and `ptv3_worker.py`, they equal the `a/` side of
`checkpoint_selector.diff`. Without `--ptv3-ckpt` the v0.5 modules load the released
checkpoint from the same constant, so their default behaviour is the v0.3 behaviour.

## Environment and replay

A research source snapshot, not an installer. The experiment root was `/data/livo_sem`
(after sanitising) with Pointcept v1.5.1 under `src/Pointcept_v151`, the released nuScenes
checkpoint under `weights/`, trained checkpoints under `exp/sk/`, and extracted students
under `weights/v05/`. Datasets, checkpoints, prediction caches, bags and map `.npz` files
are not bundled. GPU scripts ran in the `ptv3` conda environment (Python 3.10, PyTorch
2.5.1+cu124); CPU replays in a separate environment. Read the root
[`CRITICAL_CONSTRAINTS.md`](../../CRITICAL_CONSTRAINTS.md) (C1, R2, V1–V3) before rebuilding.

The original entry points, once the layout exists:

```bash
python tools/extract_student.py --ckpt exp/sk/armB0/model/model_best.pth --tag B0 \
    --out weights/v05/B0_student.pth --report out/v05/extract_B0.json \
    --neg weights/v05/ZS_student.pth
bash opt/runs_v05.sh rt B0            # full bag, rate 1.0, writes out/v05/map_B0.npz
python opt/replay_v05.py --pred <cache dir> --pred-order raw --tag B0_r1 \
    --json-out out/v05/map_B0_r1.json
```

Verify the published files against the manifest without the GPU environment:

```bash
python3 - <<'PY'
import hashlib, json, pathlib
for e in json.load(open("results/v05/snapshot_manifest.json"))["files"]:
    d = pathlib.Path(e["published"]).read_bytes()
    assert len(d) == e["bytes"] and hashlib.sha256(d).hexdigest() == e["sha256"], e["published"]
print("ok")
PY
```
