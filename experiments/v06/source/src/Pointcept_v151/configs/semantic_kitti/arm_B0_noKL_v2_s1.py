"""v0.6 arm B0_noKL -- arm B0 with the anti-forgetting KL anchor removed.  seed 20260924.

WHY THIS ARM EXISTS
-------------------
The KL anchor pins out-of-frustum predictions to the frozen zero-shot model, which
structurally suppresses the very phenomenon the distillation is meant to produce.
Measured on arm D: removing it changed the result from +9.67 to +16.43 mIoU.  B0 is the
only deployable arm, so its headline number must not carry that penalty, and the size
of the penalty must be measured on B0 itself rather than extrapolated from D.

Identical to arm_B0_v2_s1.py in every other respect -- same seed, same split, same
supervision -- so the pair isolates exactly one factor.
"""
_base_ = ["./arm_B0.py"]

seed = 20260924

model = dict(kl_enabled=False)

dataset_type = "DistilSemanticKITTIDatasetV2"
data = dict(
    train=dict(type="DistilSemanticKITTIDatasetV2"),
    val=dict(type="SemanticKITTICommon9DatasetV2"),
    test=dict(type="SemanticKITTICommon9DatasetV2"),
)
