dataset_type = 'CocoDataset'
data_root = '/content/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(800, 800), keep_ratio=True),
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
        img_scale=(800, 800),
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
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        #ann_file=data_root + 'gdrive/MyDrive/ssdd_coco/train1/train.json',
        ann_file=data_root + 'gdrive/MyDrive/HRSID_JPG/annotations/train2017.json',
        img_prefix=data_root + 'gdrive/MyDrive/HRSID_JPG/train_image/',
        #img_prefix=data_root + 'gdrive/MyDrive/ssdd_coco/train1/after_train_image/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'gdrive/MyDrive/HRSID_JPG/annotations/test2017.json',
        img_prefix=data_root + 'gdrive/MyDrive/HRSID_JPG/test_image/',
        #img_prefix=data_root + 'gdrive/MyDrive/ssdd_coco/val1/val_image/',
        #gdrive/MyDrive/ssdd_coco/val1/val.json
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'gdrive/MyDrive/HRSID_JPG/annotations/test2017.json',
        #'gdrive/MyDrive/ssdd_coco/test1/test.json'
        img_prefix=data_root + 'gdrive/MyDrive/HRSID_JPG/test_image/',
        #img_prefix=data_root + 'gdrive/MyDrive/ssdd_coco/test1/after_test_image/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')
