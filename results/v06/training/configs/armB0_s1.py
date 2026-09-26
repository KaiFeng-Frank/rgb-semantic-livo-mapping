weight = None
resume = False
evaluate = True
test_only = False
seed = 20260924
save_path = '/data/livo_sem/exp/sk2/armB0_s1'
num_worker = 8
batch_size = 2
batch_size_val = 1
batch_size_test = None
epoch = 10
eval_epoch = 10
sync_bn = False
enable_amp = True
empty_cache = False
find_unused_parameters = False
mix_prob = 0.8
param_dicts = [dict(keyword='block', lr=2e-05)]
hooks = [
    dict(type='CheckpointLoader'),
    dict(type='IterationTimer', warmup_iter=2),
    dict(type='InformationWriter'),
    dict(type='SemSegEvaluator'),
    dict(type='CheckpointSaver', save_freq=None),
    dict(type='RareClassAudit')
]
train = dict(type='DefaultTrainer')
test = dict(type='SemSegTester', verbose=True)
shuffle_orders = True
model = dict(
    type='DistilSegmentorMiB',
    backbone_out_channels=64,
    freeze_backbone=False,
    exclude_classes=('terrain', 'manmade'),
    kl_enabled=True,
    kl_lambda=None,
    use_lovasz=True,
    backbone=dict(
        type='PT-v3m1',
        in_channels=4,
        order=['z', 'z-trans', 'hilbert', 'hilbert-trans'],
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
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        cls_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=('nuScenes', 'SemanticKITTI', 'Waymo')))
optimizer = dict(type='AdamW', lr=0.0002, weight_decay=0.005)
scheduler = dict(
    type='OneCycleLR',
    max_lr=[0.0002, 2e-05],
    pct_start=0.04,
    anneal_strategy='cos',
    div_factor=10.0,
    final_div_factor=100.0)
dataset_type = 'DistilSemanticKITTIDatasetV2'
data_root = '/data/livo_sem/data/pointcept_sk'
pseudo_root = '/data/livo_sem/out/pseudo'
ignore_index = -1
names = [
    'car', 'large_vehicle', 'two_wheeler', 'person', 'road', 'sidewalk',
    'terrain', 'vegetation', 'manmade'
]
filter_spec = dict(
    require_visible=True,
    range_lt50=False,
    teachers_agree=True,
    conf_min=0.9,
    drop_depth_edge=False)
sample_weights_json = '/data/livo_sem/out/pseudo/rare_weights.json'
data = dict(
    num_classes=9,
    ignore_index=-1,
    names=[
        'car', 'large_vehicle', 'two_wheeler', 'person', 'road', 'sidewalk',
        'terrain', 'vegetation', 'manmade'
    ],
    train=dict(
        type='DistilSemanticKITTIDatasetV2',
        split='train',
        data_root='/data/livo_sem/data/pointcept_sk',
        pseudo_root='/data/livo_sem/out/pseudo',
        filter_spec=dict(
            require_visible=True,
            range_lt50=False,
            teachers_agree=True,
            conf_min=0.9,
            drop_depth_edge=False),
        label_source='pseudo',
        supervise='filterE',
        sample_weights_json='/data/livo_sem/out/pseudo/rare_weights.json',
        transform=[
            dict(
                type='RandomRotate',
                angle=[-1, 1],
                axis='z',
                center=[0, 0, 0],
                p=0.5),
            dict(type='RandomScale', scale=[0.9, 1.1]),
            dict(type='RandomFlip', p=0.5),
            dict(type='RandomJitter', sigma=0.005, clip=0.02),
            dict(
                type='GridSample',
                grid_size=0.05,
                hash_type='fnv',
                mode='train',
                keys=('coord', 'strength', 'segment', 'frustum'),
                return_grid_coord=True),
            dict(type='ToTensor'),
            dict(
                type='Collect',
                keys=('coord', 'grid_coord', 'segment', 'frustum'),
                feat_keys=('coord', 'strength'))
        ],
        test_mode=False,
        ignore_index=-1,
        loop=1),
    val=dict(
        type='SemanticKITTICommon9DatasetV2',
        split='val',
        data_root='/data/livo_sem/data/pointcept_sk',
        transform=[
            dict(
                type='GridSample',
                grid_size=0.05,
                hash_type='fnv',
                mode='train',
                keys=('coord', 'strength', 'segment'),
                return_grid_coord=True),
            dict(type='ToTensor'),
            dict(
                type='Collect',
                keys=('coord', 'grid_coord', 'segment'),
                feat_keys=('coord', 'strength'))
        ],
        test_mode=False,
        ignore_index=-1),
    test=dict(
        type='SemanticKITTICommon9DatasetV2',
        split='test',
        data_root='/data/livo_sem/data/pointcept_sk',
        transform=[],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type='GridSample',
                grid_size=0.05,
                hash_type='fnv',
                mode='test',
                return_grid_coord=True,
                keys=('coord', 'strength')),
            crop=None,
            post_transform=[
                dict(type='ToTensor'),
                dict(
                    type='Collect',
                    keys=('coord', 'grid_coord', 'index'),
                    feat_keys=('coord', 'strength'))
            ],
            aug_transform=[[{
                'type': 'RandomScale',
                'scale': [1, 1]
            }]]),
        ignore_index=-1))
