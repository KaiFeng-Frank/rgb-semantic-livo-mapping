"""v0.6 arm Rprime_noKL -- voxel-level random GT supervision.  seed 20260924.

The UPPER BOUND, not a deployable method: it trains on randomly scattered ground truth,
which no real deployment can obtain.  It exists to bound what the pipeline could reach
if supervision were free, so a single seed is enough -- the variance that matters is on
the deployable arm.
"""
_base_ = ["./arm_Rprime_noKL.py"]

seed = 20260924

dataset_type = "DistilSemanticKITTIDatasetVRV2"
data = dict(
    train=dict(type="DistilSemanticKITTIDatasetVRV2"),
    val=dict(type="SemanticKITTICommon9DatasetV2"),
    test=dict(type="SemanticKITTICommon9DatasetV2"),
)
