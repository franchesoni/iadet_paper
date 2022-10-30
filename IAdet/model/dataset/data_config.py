# define this as in loop_config.py !
dataset_type = "CustomDataset"
IMG_PREFIX = "data/VOCdevkit/"
TRAIN_ANN_FILE = "something_is_wrong_in_data_config_train"
TEST_ANN_FILE = "something_is_wrong_in_data_config_test"

TRAIN_REPEATS = 1

# dataset settings, based on PASCAL VOC
data_root = IMG_PREFIX
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1000, 600), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1000, 600),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='RepeatDataset',
        times=TRAIN_REPEATS,
        dataset=dict(
            type=dataset_type,
            ann_file=TRAIN_ANN_FILE,
            img_prefix=IMG_PREFIX,
            pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        ann_file=TEST_ANN_FILE,
        img_prefix=IMG_PREFIX,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=TEST_ANN_FILE,
        img_prefix=IMG_PREFIX,
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='mAP')
