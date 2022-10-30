_base_ = [
    '/home/franchesoni/projects/current/detection_loop/mmdetection/configs/_base_/models/faster_rcnn_r50_fpn.py',
    'data_config.py',
    '/home/franchesoni/projects/current/detection_loop/mmdetection/configs/_base_/schedules/schedule_1x.py',
    '/home/franchesoni/projects/current/detection_loop/mmdetection/configs/_base_/default_runtime.py'
]
# fp16 settings
fp16 = dict(loss_scale=512.)

CHECKPOINT = "IAdet/ablations/models/fasterrcnn/faster_rcnn_r50_fpn_fp16_1x_coco_20200204-d4dc1471.pth"
# define some variables in data_config too!
dataset_type = "CustomDataset"
IMG_PREFIX = "data/VOCdevkit/"
TRAIN_ANN_FILE = "something_is_wrong_in_full_config_train"
TEST_ANN_FILE = "something_is_wrong_in_full_config_test"
TRAIN_REPEATS = 1

# overwrite config
log_level = 'DEBUG'
data_root = IMG_PREFIX
model = dict(roi_head=dict(bbox_head=dict(num_classes=1)))
data = dict(train=dict(type="RepeatDataset", times=TRAIN_REPEATS, dataset=dict(classes=('instance',), filter_empty_gt=False)))
runner = dict(type="CustomRunner", max_epochs=1000)
load_from = CHECKPOINT
auto_resume = False
checkpoint_config = dict(max_keep_ckpts = 1)

