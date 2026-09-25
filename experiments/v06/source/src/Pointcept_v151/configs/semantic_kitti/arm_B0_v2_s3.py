"""v0.6 arm B0 -- pseudo-label distillation under filter E.  seed 20260926.

The DEPLOYABLE arm: its supervision (a 2D teacher, filtered) is something a real
deployment can actually produce.  Three seeds so the reported number carries a
variance rather than resting on one training run.

v0.6 split: seq 09 is held out alongside seq 07, so one test sequence exists that no
design decision of v0.4 / v0.5 has ever been looked at.  See src/split_v2.py.
"""
_base_ = ["./arm_B0.py"]

seed = 20260926

dataset_type = "DistilSemanticKITTIDatasetV2"
data = dict(
    train=dict(type="DistilSemanticKITTIDatasetV2"),
    val=dict(type="SemanticKITTICommon9DatasetV2"),
    test=dict(type="SemanticKITTICommon9DatasetV2"),
)
