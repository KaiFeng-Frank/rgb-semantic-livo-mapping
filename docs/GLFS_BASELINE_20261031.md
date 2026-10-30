# Official GLFS on the existing causal evidence

This experiment tests the released GLFS fusion/calibration implementation before
considering another custom memory head. It is an outdoor LiDAR adapter, not a
replication of the paper's ScanNet scores and not a replacement for its own
reconstruction system.

The fixed protocol is [config/glfs_baseline_20261031.json](../config/glfs_baseline_20261031.json).
The official repository is
[joaomcm/Semantic-3D-Mapping-Uncertainty-Calibration](https://github.com/joaomcm/Semantic-3D-Mapping-Uncertainty-Calibration/tree/ed5c59facbed7d05a90a3ed20e952b9dbba97bd6),
commit `ed5c59facbed7d05a90a3ed20e952b9dbba97bd6` (MIT).
Its pinned calibration-loss submodule is `joaomcm/meta-calibration`,
commit `a0baa1738be91cdf42dd23a7f38f399bab2dc480`.
The GLFS copyright and permission notice is preserved in
[third_party/glfs/LICENSE](../third_party/glfs/LICENSE).

## What stays fixed

- Frozen B0 PTv3, Pointcept v1.5.1, raw nuScenes keyframes and supplied ego poses,
  original intensity normalization, all serialization shuffling disabled.
- The original query selector and causal previous-frame retrieval: at most
  4,096 queries and four historical candidates per query. No ground-truth
  identity filtering or fused-prediction feedback.
- Independently trained `balanced_current` checkpoints from seeds 0, 1 and 2.
  These provide the current prediction. Each historical observation retains
  the original raw B0 logits. This asymmetry is part of the existing interface.
- All arms receive exactly the same current logits, individual historical
  observations, source ranges, source incidence angles and validity masks.
  Nonquery predictions are identical for all arms.

The October 30 cache stores pooled history and cannot reconstruct the original
individual logits or their geometry. It is therefore insufficient for this
baseline. One shared extraction on the remote reconstructs all 963 frames;
PTv3 is not trained. Actual logits are retained before softmax rather than
recovering them from probabilities, which would change vector temperature scaling.

## Released method and explicit adaptations

The released training script fixes `L=epsilon=0`. We reproduce that configuration:
per-class exponential temperature scaling, Laplace smoothing `1e-4`, positive
class/range/incidence lookup weights and normalized arithmetic fusion. There is
no new gate, MLP or attention module. The adapter has 400 fitted parameters.

Changes required by the input domain are declared before scoring:

- 16 real classes, ignored label -1. Class zero is **barrier**, not ScanNet void.
- Source LiDAR radial range replaces RGB-D depth; fixed range edges are
  `[0,10,20,30,40,50]` metres. Incidence edges remain `[0,30,60,90]` degrees.
- Source normals use covariance eigenvectors of the 16 nearest source scan
  voxel representatives. This additional geometry cost is reported, not hidden
  in a fusion-only latency number. Sparse and degenerate neighborhoods are counted.
- Explicit masks distinguish padding from a valid uniform/zero-logit observation.
- The actual authors' mDECE loss is loaded from the pinned submodule, with its
  hard-coded accumulator device corrected and class-zero exclusion removed.
  NLL class weights use fit-only frequencies; hard-coded ScanNet weights are
  not transferable. Formula: `-log(frequency)`. This rule is reconstructed from
  the released arrays: every log weight equals twice the logarithm of its
  corresponding square-root inverse-frequency weight, within rounding error.
- CPU float32 fitting replaces the original Lightning mixed-precision runner.
  Lookup parameters remain float64 as in the released code. No external logging
  service or credential in the original training script is executed.

The paired arms are current only, fixed uniform probability average, and GLFS.
GLFS with the same learned parameters but history disabled is also reported,
to distinguish historical benefit from calibration of the current classifier.

## Fitting, selection and interpretation

Four of the original 16 training logs are selected by a predeclared hash of log
identity for calibration checkpoint selection; the remaining 12 fit GLFS.
Selection uses full-point fixed-16 mIoU. The eight previously inspected development
logs are evaluated after selection. Four existing held-out logs remain closed.
The frozen current classifiers were trained under the older train/dev protocol;
this new split controls **fusion fitting**, not the independence of those existing
classifiers. No development result is called a project-level blind test.

Optimizer, initialization, loss weights, batch size, maximum epochs and early
stopping follow the released training defaults. Three seeds accompany the three
already-frozen current classifiers. No architecture or hyperparameter search is
triggered by a disappointing score.

Report full-point mIoU, per-log/class outcomes, fixes and harms relative to current,
object recovery using both fixed raw-baseline and current-defined episodes,
camera foreground accuracy, ECE/NLL/Brier, and runtime. These GT-object diagnostics
do not measure a newly produced tracker identity: this adapter does not produce one.
Accuracy and deployment admission remain distinct. Cached fusion timing excludes
normal construction, retrieval and the backbone; it cannot establish online SLAM
throughput or the existing 5 ms module target.

## Verification and execution

`test_glfs_author_parity.py` executes the official weight estimator, calibrator and
forward function extracted from the pinned source without executing top-level
services. It compares outputs, temperature gradients, lookup gradients, bin edges,
padding, and mDECE values/gradients. The first CPU run passed with maximum output
error `2.98e-8`; the same test also passed in the remote environment.

Remote working directory: `/home/wuyou/glfs_reference`, isolated source/environment.
Local project: `/home/frank/semantic_livo_instance_memory`, branch
`feat/instance-memory-20261029`. The local GPU is reserved for the user.

```bash
source /home/wuyou/taseg_reference/runtime/env.sh
source /home/wuyou/glfs_reference/venv/bin/activate
cd /home/wuyou/glfs_reference/project
export PTV3_POINTCEPT=/home/wuyou/glfs_reference/Pointcept_v151
export PTV3_CONFIG=/home/wuyou/glfs_reference/inputs/ptv3-config.py
export PTV3_CHECKPOINT=/home/wuyou/glfs_reference/backbone.pth
python -m tools.semantic_memory.test_glfs_author_parity --upstream ../upstream --loss ../dece.py
python -m tools.semantic_memory.glfs_extract --root .. --data /home/wuyou/taseg_reference/data/nuscenes --limit 3
python -m tools.semantic_memory.glfs_extract --root .. --data /home/wuyou/taseg_reference/data/nuscenes
CUDA_VISIBLE_DEVICES='' python -m tools.semantic_memory.glfs_experiment --root ..
```

The extraction requires the pinned PTv3 environment variables and matched CUDA
driver user libraries from the remote runtime. Outputs are created without
overwriting existing runs. No accuracy result is available at protocol creation.
