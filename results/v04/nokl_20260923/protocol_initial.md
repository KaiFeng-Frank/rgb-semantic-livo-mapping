# Paired no-KL diagnostic, frozen before training
Created 2026-09-23 after the existing D/R/Rprime/C results were known.
This is a new prospective diagnostic, not a preregistration of the earlier results.

## Question and fixed intervention
Compare D_noKL (camera-frustum GT) with Rprime_noKL (voxel-random GT with the same post-grid count).
Disable model.kl_enabled. In both arms set the now-inactive kl_lambda=None to prevent calibration-log side effects.
Keep released nuScenes PTv3 initialization, 16-way head, common-9 supervised loss, augmentations,
optimizer, batch size 2, 8 workers, rare-class frame resampling, and 10 epochs unchanged.
Use the same explicit training seed 20260923 for the two NEW arms.
The saved historical seeds were D=55331836 and Rprime=59599306.
Therefore the historical KL-on vs new KL-off interaction is exploratory and includes training-seed variation.
Three inference draws measure inference variation, not independent training variation.

## Splits and model selection
Train: 00,01,02,04,05,06,09,10; 17228 resampled frames per epoch.
Seq03 stays absent; do not add data to one arm.
Select model_best only by seq08 GT validation under the unchanged evaluator.
Seq07 is accessed for final scoring only, never checkpoint/hyperparameter selection.
Score all 1101 test frames through the unchanged cache_trained.py and score_2d_vs_3d.py:
three independent forward passes, intensity*0.2, grid 0.05, fp16, shuffle_orders=False, no TTA.
Primary: seq07 out-of-frustum mIoU-9. Also report mIoU-8 without two_wheeler,
per-class, in-frustum, and global metrics. No claim of inference-repeat significance as training significance.

## Gates
Before starting either training run, require equal initial-student hashes, training-frame order hashes,
normalized training config hashes, core-code hashes, and paired sample coordinates/counts across machines.
A real batch must produce finite nonzero student gradients, KL=0, n_kl=0, loss=CE+Lovasz,
and must never invoke the anchor forward. Strict released-weight load remains 488 tensors.
Every completed epoch must have kl_points=0 and positive supervised counts.
Require exactly epochs 1..10, a final checkpoint at epoch10, and model_best selected within 1..10.
Do not infer successful training from the existence of an early checkpoint.
Rprime_noKL must pass the full-stream voxel-count check against D_noKL:
each epoch within D_noKL mean +/-5 sample SD; report ratio and every epoch.
Any check failure is a failed diagnostic gate, even if scores exist; retain all outputs.
Never alter historical checkpoints, helpers, results, or prior preregistrations.

## Reading the paired outcome
Historical gap = Rprime(83.3566851028)-D(74.6775498130)=8.6791352898.
Report new gap = Rprime_noKL-D_noKL, each arm's change from its own historical reference,
and gap reduction = historical gap-new gap, including signs rather than selecting a favorable narrative.
A persistent gap supports the role of supervision distribution INCLUDING its class mix.
A smaller gap indicates that the existing KL policy may contribute; the different historical training seeds
prevent a definitive mechanistic attribution without independent paired training replications.
Class mix remains unmatched. Random GT supervision is a diagnostic oracle, not a deployable camera-pseudo-label method.
Rprime_noKL uses the SAME frozen selector salt as historical Rprime.
Do not retrofit the existing +/-0.60 inference-noise band into a training-seed confidence interval.

## Operations
A runs D_noKL; B runs Rprime_noKL. Each GPU has one training job.
Independent systemd user services survive this interactive session.
Each runner checks exit codes, audits, then performs three tests and scoring.
After A finishes scoring it waits, without GPU use, for B and writes the paired comparison on A.
New artifacts are confined to exp/sk/arm{D_noKL,Rprime_noKL}, out/v04/nokl_20260923,
the two new config files, and tools/nokl_20260923.
