"""
v0.4 distillation config -- ONE file, four arms, selected with --options.

Differences from semseg-pt-v3m1-0-common9.py, and why (the frozen design):
  * the head stays 16-WAY.  That config built a 9-way head and loaded a head-stripped
    checkpoint with strict=False; the frozen design forbids it ("keep the original
    16-way nuScenes head.  Do NOT build a 9-way head").  DistilSegmentorMiB loads the
    released checkpoint into BOTH the student and the frozen anchor with strict=True
    and adds ZERO new parameters; the supervised loss is MiB marginalisation
    p9(c) = sum_{k in g(c)} p16(k), with nuScenes other_flat(11) in no group.
  * an anti-forgetting KL to the frozen anchor on every OUT-OF-FRUSTUM point, on the
    full 16-dim simplex.
  * no PointClip / SphereCrop anywhere: the input stays the full 360 deg sweep.

ARMS (only these four lines change):
  B1  model.freeze_backbone=True   data.train.label_source=pseudo  supervise=filterE
      model.exclude_classes="(terrain,manmade)"
  B0  as B1 with freeze_backbone=False
  D   label_source=gt  supervise=frustum   exclude_classes=()
  C   label_source=gt  supervise=all       exclude_classes=()
"""
_base_ = ["../_base_/default_runtime.py"]

batch_size = 2              # MEASURED 13.00 GB / 0.251 s-iter on the 4090; bs4 OOMs
batch_size_val = 1
num_worker = 8
mix_prob = 0.8
empty_cache = False
enable_amp = True
evaluate = True

# DistilSegmentorMiB loads the released checkpoint itself, with strict=True, into the
# student AND the anchor.  CheckpointLoader must therefore load nothing.
weight = None

shuffle_orders = True       # as trained.  The deployed/eval path pins it False.

model = dict(
    type="DistilSegmentorMiB",
    backbone_out_channels=64,
    freeze_backbone=False,
    exclude_classes=(),
    kl_enabled=True,
    kl_lambda=None,          # None -> calibrated at step 0 and LOGGED
    use_lovasz=True,
    backbone=dict(
        type="PT-v3m1",
        in_channels=4,
        order=["z", "z-trans", "hilbert", "hilbert-trans"],
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=shuffle_orders,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        cls_mode=False,                  # NEVER rename to enc_mode (that is HEAD; C1)
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("nuScenes", "SemanticKITTI", "Waymo"),
    ),
)

epoch = 10
eval_epoch = 10
optimizer = dict(type="AdamW", lr=0.0002, weight_decay=0.005)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.0002, 0.00002],            # head, backbone -- backbone LR <= 1e-4
    pct_start=0.04,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=100.0,
)
param_dicts = [dict(keyword="block", lr=0.00002)]

dataset_type = "DistilSemanticKITTIDataset"
data_root = "/data/wuyou/livo_sem/data/pointcept_sk"
pseudo_root = "/data/wuyou/livo_sem/out/pseudo"
ignore_index = -1
names = ["car", "large_vehicle", "two_wheeler", "person",
         "road", "sidewalk", "terrain", "vegetation", "manmade"]

# filter E.  The CRITERION is frozen; these BOUNDARIES are re-derived on seq 08 (val)
# by tools/distil_strata.py and written here by tools/write_filterE.py.  Never tuned.
filter_spec = dict(require_visible=True, range_lt50=True, teachers_agree=True,
                   conf_min=0.90, drop_depth_edge=True)

sample_weights_json = "/data/wuyou/livo_sem/out/pseudo/rare_weights.json"

data = dict(
    num_classes=9,
    ignore_index=ignore_index,
    names=names,
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        pseudo_root=pseudo_root,
        filter_spec=filter_spec,
        label_source="pseudo",
        supervise="filterE",
        sample_weights_json=sample_weights_json,
        transform=[
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(type="GridSample", grid_size=0.05, hash_type="fnv", mode="train",
                 keys=("coord", "strength", "segment", "frustum"),
                 return_grid_coord=True),
            dict(type="ToTensor"),
            dict(type="Collect", keys=("coord", "grid_coord", "segment", "frustum"),
                 feat_keys=("coord", "strength")),
        ],
        test_mode=False,
        ignore_index=ignore_index,
    ),
    # VALIDATION = seq 08 GROUND TRUTH, every point, no pseudo-label anywhere near it.
    # P6: checkpoint selection must never see a pseudo-label.  P5: never seq 07.
    val=dict(
        type="SemanticKITTICommon9Dataset",
        split="val",
        data_root=data_root,
        transform=[
            dict(type="GridSample", grid_size=0.05, hash_type="fnv", mode="train",
                 keys=("coord", "strength", "segment"), return_grid_coord=True),
            dict(type="ToTensor"),
            dict(type="Collect", keys=("coord", "grid_coord", "segment"),
                 feat_keys=("coord", "strength")),
        ],
        test_mode=False,
        ignore_index=ignore_index,
    ),
    test=dict(
        type="SemanticKITTICommon9Dataset",
        split="test",
        data_root=data_root,
        transform=[],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(type="GridSample", grid_size=0.05, hash_type="fnv",
                          mode="test", return_grid_coord=True,
                          keys=("coord", "strength")),
            crop=None,
            post_transform=[
                dict(type="ToTensor"),
                dict(type="Collect", keys=("coord", "grid_coord", "index"),
                     feat_keys=("coord", "strength")),
            ],
            aug_transform=[[dict(type="RandomScale", scale=[1, 1])]],
        ),
        ignore_index=ignore_index,
    ),
)

# PreciseEvaluator stays OUT (it needs real pointops; ours is stubbed).  Final scoring
# goes through this project's own harness on the frozen score_2d_vs_3d.py.
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="SemSegEvaluator"),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="RareClassAudit"),
]
