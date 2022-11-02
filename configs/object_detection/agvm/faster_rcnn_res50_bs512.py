_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn_align.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', 
    '../_base_/default_runtime.py'
]

model = dict(
    backbone=dict(
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=False))

use_agvm = True
optimizer = dict(
    constructor='GroupOptimizerConstructor', 
    type='SGD',
    lr=0.38,
    momentum=0.9, 
    weight_decay=0.0001)
    
warmup_iters = 460

optimizer_config = dict(
    _delete_=True,
    type='AgvmOptimizerHook',
    momentum=0.97,
    lr_update_interval=10,
    lr_update_start_iter=560,
    warmup_iters=warmup_iters,
    grad_clip=dict(max_norm=1.0, norm_type=2),
    grad_clip_iter_range=[360, 560])

runner = dict(type='EpochBasedRunner', max_epochs=16)

lr_config = dict(
    policy='step',
    warmup='exp',
    warmup_iters=warmup_iters,
    warmup_ratio=0.001,
    step=[12, 14])

evaluation = dict(interval=1)
checkpoint_config = dict(interval=1)

log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])

data = dict(
    samples_per_gpu=16, workers_per_gpu=10)