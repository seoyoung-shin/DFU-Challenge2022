# dataset settings
dataset_type = 'DFUCDataset'
data_root = 'data/DFUC2022'

img_norm_cfg = dict(
    mean=[159.68, 146.26, 140.54], std=[39.81, 46.39, 49.42], to_rgb=True)
img_norm_cfg = dict(
    mean=[159.68, 146.26, 140.54], std=[39.81, 46.39, 49.42], to_rgb=True)
image_scale = (480, 640)
crop_size = (256, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    #dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='Resize', img_scale=image_scale, ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=image_scale,
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=4, #batch_size = samper_per_gpu * gpu_num
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='DFUC2022_train/DFUC2022_train_images',
        ann_dir='DFUC2022_train/DFUC2022_train_masks',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='DFUC2022_val/DFUC2022_val_images',
        ann_dir='DFUC2022_val/DFUC2022_val_masks',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='DFUC2022_test/DFUC2022_test_images',
        ann_dir='DFUC2022_test/DFUC2022_test_masks',
        pipeline=test_pipeline))
