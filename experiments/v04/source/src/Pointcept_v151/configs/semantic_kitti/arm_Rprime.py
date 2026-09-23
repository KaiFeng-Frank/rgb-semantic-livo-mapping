"""DERIVED BY HAND FROM arm_D.py -- see out/v04/armRprime/config_diff.txt for the diff.

Arm R': GT supervision on a per-frame random subset of the SURVIVING VOXELS, matched
voxel-for-voxel to the number of voxels arm D contributes to the loss in that same frame
and that same voxelisation (separates supervision COUNT-AFTER-VOXELISATION from frustum
GEOMETRY; identical to arm D except for the supervision selector and the pinned lambda)
filter E boundaries re-derived on seq 08: {"require_visible": true, "range_lt50": false, "teachers_agree": true, "conf_min": 0.9, "drop_depth_edge": false}

WHY NOT write_arm_configs.py:  that generator emits arms whose ONLY per-arm fields are
`model` and `data.train` scalars.  Arm R' also has to reorder the train transform
pipeline (one extra GridSample key, one extra transform after it), which the generator
cannot express.  Written by hand, and diffed against arm_D.py line by line instead.
"""
_base_ = ["./semseg-pt-v3m1-distil-common9.py"]

filter_spec = {'require_visible': True, 'range_lt50': False, 'teachers_agree': True, 'conf_min': 0.9, 'drop_depth_edge': False}

# kl_lambda is PINNED to arm D's calibrated value, not calibrated at
# step 0 like every other arm.  Arm R''s only meaningful comparison is
# against arm D, and the step-0 calibration is a one-batch gradient-norm
# ratio that spans 0.441 / 0.969 / 1.531 across three runs of the SAME
# full-fine-tune configuration (out/v04/lambda_calibration.jsonl) -- a
# 3.5x range on the anti-forgetting strength, against a +-0.60 mIoU
# decision band.  Left free, arm R' would differ from arm D in TWO
# respects, supervision geometry AND anchor strength, and a two-factor
# contrast attributes nothing.  This exact error already cost this
# experiment one conclusion: B1 vs D differ in both freezing and label
# source, so their 11x ratio supports nothing.  Same value, same reason,
# same wording as arm R.

model = dict(
    freeze_backbone=False,
    exclude_classes=(),
    kl_lambda=1.5305520007241327,
)

data = dict(train=dict(
    # DistilSemanticKITTIDataset + two additions and nothing else: it accepts the new
    # mode name, and it emits `prio`, the frame's frozen per-point rank, so the ranking
    # survives GridSample.  segment and frustum leave get_data UNMASKED -- arm R' masks
    # AFTER the voxelisation, which is the entire point of the arm.
    type='DistilSemanticKITTIDatasetVR',
    label_source='gt',
    supervise='voxel_random',
    filter_spec=filter_spec,
    # The transform list is restated in full because mmcv-style config merging REPLACES
    # a list rather than merging into it.  It is the base list verbatim, with exactly
    # two edits, both marked below.  Everything else -- the augmentations, their
    # parameters and their order, the grid size, the hash, the mode, ToTensor, Collect
    # and the feat keys -- is byte-identical to what arms B1 / B0 / C / D / R ran.
    transform=[
        dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
        dict(type="RandomScale", scale=[0.9, 1.1]),
        dict(type="RandomFlip", p=0.5),
        dict(type="RandomJitter", sigma=0.005, clip=0.02),
        # EDIT 1 of 2: "prio" added to keys, so the frame's frozen rank is subsampled by
        # the SAME idx_unique as coord / strength / segment / frustum and each surviving
        # voxel inherits the rank of the representative point the sampler kept.
        dict(type="GridSample", grid_size=0.05, hash_type="fnv", mode="train",
             keys=("coord", "strength", "segment", "frustum", "prio"),
             return_grid_coord=True),
        # EDIT 2 of 2: the selector, AFTER the voxelisation.  It counts arm D's post-grid
        # supervised voxels in this frame, draws that many label-valid survivors by
        # lowest rank, masks `segment` to them, and overloads `frustum` with the
        # selection so the KL anchor covers exactly the non-supervised voxels.
        dict(type="VoxelRandomSupervise", ignore_index=-1,
             audit_path="/data/wuyou/livo_sem/out/v04/armRprime/voxel_audit.jsonl"),
        dict(type="ToTensor"),
        dict(type="Collect", keys=("coord", "grid_coord", "segment", "frustum"),
             feat_keys=("coord", "strength")),
    ],
))
