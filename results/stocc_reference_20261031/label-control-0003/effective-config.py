point_cloud_range = [-40, -40, -1.0, 40, 40, 5.4]
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
dataset_type = 'NuScenesDataset'
data_root = 'data/nuscenes/'
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)
file_client_args = dict(backend='disk')
train_pipeline = [
    dict(
        type='PrepareImageInputs',
        is_train=True,
        data_config=dict(
            cams=[
                'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'
            ],
            Ncams=6,
            input_size=(256, 704),
            src_size=(900, 1600),
            resize=(-0.06, 0.11),
            rot=(-5.4, 5.4),
            flip=True,
            crop_h=(0.0, 0.0),
            resize_test=0.0)),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=dict(
            rot_lim=(-22.5, 22.5),
            scale_lim=(1.0, 1.0),
            flip_dx_ratio=0.5,
            flip_dy_ratio=0.5),
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=dict(backend='disk')),
    dict(
        type='PointToMultiViewDepth',
        downsample=1,
        grid_config=dict(
            x=[-40, 40, 0.8],
            y=[-40, 40, 0.8],
            z=[-1, 5.4, 0.8],
            depth=[2.0, 42.0, 0.5])),
    dict(
        type='ObjectRangeFilter',
        point_cloud_range=[-40, -40, -1.0, 40, 40, 5.4]),
    dict(
        type='ObjectNameFilter',
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]),
    dict(
        type='LoadOccupancy',
        ignore_nonvisible=True,
        fix_void=True,
        occupancy_path='data/nuscenes/gts'),
    dict(type='LoadFlowFromVelocity'),
    dict(
        type='DefaultFormatBundle3D',
        class_names=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]),
    dict(
        type='Collect3D',
        keys=[
            'img_inputs', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_occupancy',
            'gt_depth', 'gt_flow'
        ])
]
test_pipeline = [
    dict(
        type='CustomDistMultiScaleFlipAug3D',
        tta=False,
        transforms=[
            dict(
                type='PrepareImageInputs',
                data_config=dict(
                    cams=[
                        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                        'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'
                    ],
                    Ncams=6,
                    input_size=(256, 704),
                    src_size=(900, 1600),
                    resize=(-0.06, 0.11),
                    rot=(-5.4, 5.4),
                    flip=True,
                    crop_h=(0.0, 0.0),
                    resize_test=0.0)),
            dict(
                type='LoadAnnotationsBEVDepth',
                bda_aug_conf=dict(
                    rot_lim=(-22.5, 22.5),
                    scale_lim=(1.0, 1.0),
                    flip_dx_ratio=0.5,
                    flip_dy_ratio=0.5),
                classes=[
                    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                    'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                    'traffic_cone'
                ],
                is_train=False),
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5,
                file_client_args=dict(backend='disk')),
            dict(
                type='LoadOccupancy',
                ignore_nonvisible=True,
                fix_void=True,
                occupancy_path='data/nuscenes/gts'),
            dict(type='LoadOccVelocityMap', enlarge=1.4, z_enlarge_ratio=0.8),
            dict(
                type='DefaultFormatBundle3D',
                class_names=[
                    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                    'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                    'traffic_cone'
                ],
                with_label=False),
            dict(
                type='Collect3D',
                keys=[
                    'points', 'img_inputs', 'gt_occupancy', 'visible_mask',
                    'gt_speed_map'
                ])
        ])
]
eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=dict(backend='disk')),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        file_client_args=dict(backend='disk')),
    dict(
        type='DefaultFormatBundle3D',
        class_names=[
            'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
            'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
        ],
        with_label=False),
    dict(type='Collect3D', keys=['points'])
]
data = dict(
    samples_per_gpu=6,
    workers_per_gpu=6,
    test_dataloader=dict(runner_type='IterBasedRunnerEval'),
    train=dict(
        type='NuScenesDataset',
        data_root='data/nuscenes/',
        ann_file='data/nuscenes/bevdetv4-nuscenes_infos_global_train.pkl',
        pipeline=[
            dict(
                type='PrepareImageInputs',
                is_train=True,
                data_config=dict(
                    cams=[
                        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                        'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'
                    ],
                    Ncams=6,
                    input_size=(256, 704),
                    src_size=(900, 1600),
                    resize=(-0.06, 0.11),
                    rot=(-5.4, 5.4),
                    flip=True,
                    crop_h=(0.0, 0.0),
                    resize_test=0.0)),
            dict(
                type='LoadAnnotationsBEVDepth',
                bda_aug_conf=dict(
                    rot_lim=(-22.5, 22.5),
                    scale_lim=(1.0, 1.0),
                    flip_dx_ratio=0.5,
                    flip_dy_ratio=0.5),
                classes=[
                    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                    'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                    'traffic_cone'
                ]),
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5,
                file_client_args=dict(backend='disk')),
            dict(
                type='PointToMultiViewDepth',
                downsample=1,
                grid_config=dict(
                    x=[-40, 40, 0.8],
                    y=[-40, 40, 0.8],
                    z=[-1, 5.4, 0.8],
                    depth=[2.0, 42.0, 0.5])),
            dict(
                type='ObjectRangeFilter',
                point_cloud_range=[-40, -40, -1.0, 40, 40, 5.4]),
            dict(
                type='ObjectNameFilter',
                classes=[
                    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                    'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                    'traffic_cone'
                ]),
            dict(
                type='LoadOccupancy',
                ignore_nonvisible=True,
                fix_void=True,
                occupancy_path='data/nuscenes/gts'),
            dict(type='LoadFlowFromVelocity'),
            dict(
                type='DefaultFormatBundle3D',
                class_names=[
                    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                    'barrier', 'motorcycle', 'bicycle', 'pedestrian',
                    'traffic_cone'
                ]),
            dict(
                type='Collect3D',
                keys=[
                    'img_inputs', 'gt_bboxes_3d', 'gt_labels_3d',
                    'gt_occupancy', 'gt_depth', 'gt_flow'
                ])
        ],
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=False,
        box_type_3d='LiDAR',
        use_valid_flag=True,
        img_info_prototype='bevdet',
        sequences_split_num=2,
        use_sequence_group_flag=True,
        use_global_info=True,
        filter_empty_gt=False),
    val=dict(
        type='NuScenesDataset',
        data_root='data/nuscenes/',
        ann_file='data/nuscenes/bevdetv4-nuscenes_infos_global_val.pkl',
        pipeline=[
            dict(
                type='CustomDistMultiScaleFlipAug3D',
                tta=False,
                transforms=[
                    dict(
                        type='PrepareImageInputs',
                        data_config=dict(
                            cams=[
                                'CAM_FRONT_LEFT', 'CAM_FRONT',
                                'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK',
                                'CAM_BACK_RIGHT'
                            ],
                            Ncams=6,
                            input_size=(256, 704),
                            src_size=(900, 1600),
                            resize=(-0.06, 0.11),
                            rot=(-5.4, 5.4),
                            flip=True,
                            crop_h=(0.0, 0.0),
                            resize_test=0.0)),
                    dict(
                        type='LoadAnnotationsBEVDepth',
                        bda_aug_conf=dict(
                            rot_lim=(-22.5, 22.5),
                            scale_lim=(1.0, 1.0),
                            flip_dx_ratio=0.5,
                            flip_dy_ratio=0.5),
                        classes=[
                            'car', 'truck', 'construction_vehicle', 'bus',
                            'trailer', 'barrier', 'motorcycle', 'bicycle',
                            'pedestrian', 'traffic_cone'
                        ],
                        is_train=False),
                    dict(
                        type='LoadPointsFromFile',
                        coord_type='LIDAR',
                        load_dim=5,
                        use_dim=5,
                        file_client_args=dict(backend='disk')),
                    dict(
                        type='LoadOccupancy',
                        ignore_nonvisible=True,
                        fix_void=True,
                        occupancy_path='data/nuscenes/gts'),
                    dict(
                        type='LoadOccVelocityMap',
                        enlarge=1.4,
                        z_enlarge_ratio=0.8),
                    dict(
                        type='DefaultFormatBundle3D',
                        class_names=[
                            'car', 'truck', 'construction_vehicle', 'bus',
                            'trailer', 'barrier', 'motorcycle', 'bicycle',
                            'pedestrian', 'traffic_cone'
                        ],
                        with_label=False),
                    dict(
                        type='Collect3D',
                        keys=[
                            'points', 'img_inputs', 'gt_occupancy',
                            'visible_mask', 'gt_speed_map'
                        ])
                ])
        ],
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=True,
        box_type_3d='LiDAR',
        sequences_split_num=1,
        use_global_info=True,
        img_info_prototype='bevdet',
        occupancy_path='data/nuscenes/gts',
        use_sequence_group_flag=True),
    test=dict(
        type='NuScenesDataset',
        data_root='data/nuscenes/',
        ann_file='data/nuscenes/stocc-smoke-nuscenes_infos_global_val.pkl',
        pipeline=[
            dict(
                type='CustomDistMultiScaleFlipAug3D',
                tta=False,
                transforms=[
                    dict(
                        type='PrepareImageInputs',
                        data_config=dict(
                            cams=[
                                'CAM_FRONT_LEFT', 'CAM_FRONT',
                                'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK',
                                'CAM_BACK_RIGHT'
                            ],
                            Ncams=6,
                            input_size=(256, 704),
                            src_size=(900, 1600),
                            resize=(-0.06, 0.11),
                            rot=(-5.4, 5.4),
                            flip=True,
                            crop_h=(0.0, 0.0),
                            resize_test=0.0)),
                    dict(
                        type='LoadAnnotationsBEVDepth',
                        bda_aug_conf=dict(
                            rot_lim=(-22.5, 22.5),
                            scale_lim=(1.0, 1.0),
                            flip_dx_ratio=0.5,
                            flip_dy_ratio=0.5),
                        classes=[
                            'car', 'truck', 'construction_vehicle', 'bus',
                            'trailer', 'barrier', 'motorcycle', 'bicycle',
                            'pedestrian', 'traffic_cone'
                        ],
                        is_train=False),
                    dict(
                        type='LoadPointsFromFile',
                        coord_type='LIDAR',
                        load_dim=5,
                        use_dim=5,
                        file_client_args=dict(backend='disk')),
                    dict(
                        type='LoadOccupancy',
                        ignore_nonvisible=True,
                        fix_void=True,
                        occupancy_path='data/nuscenes/gts'),
                    dict(
                        type='LoadOccVelocityMap',
                        enlarge=1.4,
                        z_enlarge_ratio=0.8),
                    dict(
                        type='DefaultFormatBundle3D',
                        class_names=[
                            'car', 'truck', 'construction_vehicle', 'bus',
                            'trailer', 'barrier', 'motorcycle', 'bicycle',
                            'pedestrian', 'traffic_cone'
                        ],
                        with_label=False),
                    dict(
                        type='Collect3D',
                        keys=[
                            'points', 'img_inputs', 'gt_occupancy',
                            'visible_mask', 'gt_speed_map'
                        ])
                ])
        ],
        classes=[
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        modality=dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=False),
        test_mode=True,
        box_type_3d='LiDAR',
        sequences_split_num=1,
        use_global_info=True,
        img_info_prototype='bevdet',
        occupancy_path='data/nuscenes/gts',
        use_sequence_group_flag=True))
evaluation = dict(
    interval=53360,
    pipeline=[
        dict(
            type='CustomDistMultiScaleFlipAug3D',
            tta=False,
            transforms=[
                dict(
                    type='PrepareImageInputs',
                    data_config=dict(
                        cams=[
                            'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                            'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'
                        ],
                        Ncams=6,
                        input_size=(256, 704),
                        src_size=(900, 1600),
                        resize=(-0.06, 0.11),
                        rot=(-5.4, 5.4),
                        flip=True,
                        crop_h=(0.0, 0.0),
                        resize_test=0.0)),
                dict(
                    type='LoadAnnotationsBEVDepth',
                    bda_aug_conf=dict(
                        rot_lim=(-22.5, 22.5),
                        scale_lim=(1.0, 1.0),
                        flip_dx_ratio=0.5,
                        flip_dy_ratio=0.5),
                    classes=[
                        'car', 'truck', 'construction_vehicle', 'bus',
                        'trailer', 'barrier', 'motorcycle', 'bicycle',
                        'pedestrian', 'traffic_cone'
                    ],
                    is_train=False),
                dict(
                    type='LoadPointsFromFile',
                    coord_type='LIDAR',
                    load_dim=5,
                    use_dim=5,
                    file_client_args=dict(backend='disk')),
                dict(
                    type='LoadOccupancy',
                    ignore_nonvisible=True,
                    fix_void=True,
                    occupancy_path='data/nuscenes/gts'),
                dict(
                    type='LoadOccVelocityMap',
                    enlarge=1.4,
                    z_enlarge_ratio=0.8),
                dict(
                    type='DefaultFormatBundle3D',
                    class_names=[
                        'car', 'truck', 'construction_vehicle', 'bus',
                        'trailer', 'barrier', 'motorcycle', 'bicycle',
                        'pedestrian', 'traffic_cone'
                    ],
                    with_label=False),
                dict(
                    type='Collect3D',
                    keys=[
                        'points', 'img_inputs', 'gt_occupancy', 'visible_mask',
                        'gt_speed_map'
                    ])
            ])
    ])
checkpoint_config = dict(interval=2668)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = None
load_from = 'ckpts/r50_256x705_depth_pretrain.pth'
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
num_gpus = 8
samples_per_gpu = 6
num_iters_per_epoch = 2668
num_epochs = 36
checkpoint_epoch_interval = 1
use_custom_eval_hook = True
train_sequences_split_num = 2
test_sequences_split_num = 1
filter_empty_gt = False
do_history = False
history_cat_num = 16
history_cat_conv_out_channels = 160
find_unused_parameters = True
data_config = dict(
    cams=[
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT'
    ],
    Ncams=6,
    input_size=(256, 704),
    src_size=(900, 1600),
    resize=(-0.06, 0.11),
    rot=(-5.4, 5.4),
    flip=True,
    crop_h=(0.0, 0.0),
    resize_test=0.0)
bda_aug_conf = dict(
    rot_lim=(-22.5, 22.5),
    scale_lim=(1.0, 1.0),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5)
use_checkpoint = False
sync_bn = True
grid_config = dict(
    x=[-40, 40, 0.8],
    y=[-40, 40, 0.8],
    z=[-1, 5.4, 0.8],
    depth=[2.0, 42.0, 0.5])
depth_categories = 80
grid_config_bevformer = dict(
    x=[-40, 40, 0.8], y=[-40, 40, 0.8], z=[-1, 5.4, 1.6])
bev_h_ = 100
bev_w_ = 100
bev_z_ = 8
numC_Trans = 80
_dim_ = 256
_pos_dim_ = 40
_ffn_dim_ = 320
_num_levels_ = 1
empty_idx = 18
num_cls = 19
fix_void = True
img_norm_cfg = None
occ_size = [200, 200, 16]
voxel_out_indices = (0, 1, 2)
voxel_out_channel = 256
voxel_channels = [64, 128, 256]
num_tsa_layers = 3
model = dict(
    type='STOcc',
    use_depth_supervision=True,
    fix_void=True,
    do_history=False,
    history_cat_num=16,
    single_bev_num_channels=80,
    readd=True,
    uncertainty_config=dict(
        global_vis_cnt=True,
        global_vis_cnt_alpha=1,
        global_distance=True,
        global_distance_alpha=20,
        global_distance_max=60,
        global_prev_distribution=True,
        global_distribu_decay=2),
    relative_scale=True,
    implicit_uncertainty=True,
    embed_feature_uncertainty=True,
    w_au=True,
    au_ef=True,
    disable_flow_init=True,
    temporal_encoder=dict(
        type='VoxelFormerEncoder',
        uncertainty_head=dict(
            type='UncertaintyEstimator_w_feature_au',
            input_size=22,
            hidden_size=64),
        num_layers=3,
        pc_range=[-40, -40, -1, 40, 40, 5.4],
        return_intermediate=False,
        transformerlayers=dict(
            type='VoxelFormerLayer',
            attn_cfgs=[
                dict(
                    type='VoxelTemporalSelfAttention_w_Uncertainty',
                    embed_dims=80,
                    num_heads=8,
                    num_levels=1)
            ],
            ffn_cfgs=dict(
                type='FFN',
                embed_dims=80,
                feedforward_channels=512,
                num_fcs=2,
                ffn_drop=0.1,
                act_cfg=dict(type='ReLU', inplace=True)),
            feedforward_channels=512,
            ffn_dropout=0.1,
            operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
    temporal_pe=dict(
        type='VoxelLearnedPositionalEncoding',
        num_feats=40,
        row_num_embed=100,
        col_num_embed=100,
        z_num_embed=8),
    img_backbone=dict(
        pretrained='ckpts/resnet50-0676ba61.pth',
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        with_cp=False,
        style='pytorch'),
    img_neck=dict(
        type='CustomFPN',
        in_channels=[1024, 2048],
        out_channels=256,
        num_outs=1,
        start_level=0,
        with_cp=False,
        out_ids=[0]),
    depth_net=dict(
        type='CM_DepthNet',
        in_channels=256,
        context_channels=80,
        downsample=16,
        grid_config=dict(
            x=[-40, 40, 0.8],
            y=[-40, 40, 0.8],
            z=[-1, 5.4, 0.8],
            depth=[2.0, 42.0, 0.5]),
        depth_channels=80,
        with_cp=False,
        loss_depth_weight=1.0,
        use_dcn=False),
    forward_projection=dict(
        type='LSSViewTransformerFunction3D',
        grid_config=dict(
            x=[-40, 40, 0.8],
            y=[-40, 40, 0.8],
            z=[-1, 5.4, 0.8],
            depth=[2.0, 42.0, 0.5]),
        input_size=(256, 704),
        downsample=16),
    backward_projection=dict(
        type='BackwardProjection',
        bev_h=100,
        bev_w=100,
        in_channels=80,
        out_channels=80,
        pc_range=[-40, -40, -1.0, 40, 40, 5.4],
        transformer=dict(
            type='BEVFormer',
            use_cams_embeds=False,
            embed_dims=80,
            encoder=dict(
                type='bevformer_encoder',
                num_layers=1,
                pc_range=[-40, -40, -1.0, 40, 40, 5.4],
                grid_config=dict(
                    x=[-40, 40, 0.8], y=[-40, 40, 0.8], z=[-1, 5.4, 1.6]),
                data_config=dict(
                    cams=[
                        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                        'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'
                    ],
                    Ncams=6,
                    input_size=(256, 704),
                    src_size=(900, 1600),
                    resize=(-0.06, 0.11),
                    rot=(-5.4, 5.4),
                    flip=True,
                    crop_h=(0.0, 0.0),
                    resize_test=0.0),
                return_intermediate=False,
                transformerlayers=dict(
                    type='BEVFormerEncoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiScaleDeformableAttention',
                            embed_dims=80,
                            dropout=0.0,
                            num_levels=1),
                        dict(
                            type='DA_SpatialCrossAttention',
                            pc_range=[-40, -40, -1.0, 40, 40, 5.4],
                            dbound=[2.0, 42.0, 0.5],
                            dropout=0.0,
                            deformable_attention=dict(
                                type='DA_MSDeformableAttention',
                                embed_dims=80,
                                num_points=8,
                                num_levels=1),
                            embed_dims=80)
                    ],
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=80,
                        feedforward_channels=320,
                        ffn_drop=0.0,
                        act_cfg=dict(type='ReLU', inplace=True)),
                    feedforward_channels=320,
                    ffn_dropout=0.0,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        positional_encoding=dict(
            type='CustormLearnedPositionalEncoding',
            num_feats=40,
            row_num_embed=100,
            col_num_embed=100)),
    img_bev_encoder_backbone=dict(
        type='CustomResNet3D',
        depth=18,
        with_cp=False,
        block_strides=[1, 2, 2],
        n_input_channels=80,
        block_inplanes=[64, 128, 256],
        out_indices=(0, 1, 2),
        norm_cfg=dict(type='SyncBN', requires_grad=True)),
    img_bev_encoder_neck=dict(
        type='FPN3D',
        with_cp=False,
        in_channels=[64, 128, 256],
        out_channels=256,
        norm_cfg=dict(type='SyncBN', requires_grad=True)),
    occupancy_head=dict(
        type='OccHead_w_AU',
        with_cp=False,
        use_focal_loss=True,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        soft_weights=True,
        final_occ_size=[200, 200, 16],
        empty_idx=18,
        num_level=3,
        in_channels=[256, 256, 256],
        out_channel=19,
        point_cloud_range=[-40, -40, -1.0, 40, 40, 5.4],
        loss_weight_cfg=dict(
            loss_voxel_ce_weight=1.0,
            loss_voxel_sem_scal_weight=1.0,
            loss_voxel_geo_scal_weight=1.0,
            loss_voxel_lovasz_weight=1.0),
        loss_flow=dict(type='L1Loss', loss_weight=1.0),
        return_occ=True,
        detach_flow_head=True),
    pts_bbox_head=None,
    flickering_cnt=True,
    precise_upscale=True,
    pretrained=None,
    train_cfg=None)
occupancy_path = 'data/nuscenes/gts'
share_data_config = dict(
    type='NuScenesDataset',
    classes=[
        'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
        'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
    ],
    modality=dict(
        use_lidar=False,
        use_camera=True,
        use_radar=False,
        use_map=False,
        use_external=False),
    img_info_prototype='bevdet',
    occupancy_path='data/nuscenes/gts',
    use_sequence_group_flag=True)
test_data_config = dict(
    pipeline=[
        dict(
            type='CustomDistMultiScaleFlipAug3D',
            tta=False,
            transforms=[
                dict(
                    type='PrepareImageInputs',
                    data_config=dict(
                        cams=[
                            'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                            'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'
                        ],
                        Ncams=6,
                        input_size=(256, 704),
                        src_size=(900, 1600),
                        resize=(-0.06, 0.11),
                        rot=(-5.4, 5.4),
                        flip=True,
                        crop_h=(0.0, 0.0),
                        resize_test=0.0)),
                dict(
                    type='LoadAnnotationsBEVDepth',
                    bda_aug_conf=dict(
                        rot_lim=(-22.5, 22.5),
                        scale_lim=(1.0, 1.0),
                        flip_dx_ratio=0.5,
                        flip_dy_ratio=0.5),
                    classes=[
                        'car', 'truck', 'construction_vehicle', 'bus',
                        'trailer', 'barrier', 'motorcycle', 'bicycle',
                        'pedestrian', 'traffic_cone'
                    ],
                    is_train=False),
                dict(
                    type='LoadPointsFromFile',
                    coord_type='LIDAR',
                    load_dim=5,
                    use_dim=5,
                    file_client_args=dict(backend='disk')),
                dict(
                    type='LoadOccupancy',
                    ignore_nonvisible=True,
                    fix_void=True,
                    occupancy_path='data/nuscenes/gts'),
                dict(
                    type='DefaultFormatBundle3D',
                    class_names=[
                        'car', 'truck', 'construction_vehicle', 'bus',
                        'trailer', 'barrier', 'motorcycle', 'bicycle',
                        'pedestrian', 'traffic_cone'
                    ],
                    with_label=False),
                dict(
                    type='Collect3D',
                    keys=[
                        'points', 'img_inputs', 'gt_occupancy', 'visible_mask'
                    ])
            ])
    ],
    sequences_split_num=1,
    ann_file='data/nuscenes/bevdetv4-nuscenes_infos_global_val.pkl',
    use_global_info=True,
    type='NuScenesDataset',
    classes=[
        'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
        'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
    ],
    modality=dict(
        use_lidar=False,
        use_camera=True,
        use_radar=False,
        use_map=False,
        use_external=False),
    img_info_prototype='bevdet',
    occupancy_path='data/nuscenes/gts',
    use_sequence_group_flag=True)
key = 'test'
lr = 0.0002
optimizer = dict(type='AdamW', lr=0.0002, weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.001,
    step=[96048])
runner = dict(type='IterBasedRunner', max_iters=96048)
custom_hooks = [
    dict(
        type='MEGVIIEMAHook',
        init_updates=10560,
        priority='NORMAL',
        interval=5336),
    dict(type='SequentialControlHook', temporal_start_iter=8004)
]
fp16 = dict(loss_scale='dynamic')
