_base_ = "ssd300_voc0712.py"

# define some variables in data_config too!
dataset_type = "CustomDataset"
IMG_PREFIX = "data/VOCdevkit/"
TRAIN_ANN_FILE = "something_is_wrong_in_full_config_train"
TEST_ANN_FILE = "something_is_wrong_in_full_config_test"

TRAIN_REPEATS = 1
CHECKPOINT = "IAdet/model/coco_pretrained_ckpt.pth"

# overwrite config
log_level = 'DEBUG'
data_root = IMG_PREFIX
model = dict(bbox_head=dict(num_classes=1))
data = dict(train=dict(type="RepeatDataset", times=TRAIN_REPEATS, dataset=dict(classes=('instance',), filter_empty_gt=False)))
runner = dict(type="EpochBasedRunner", max_epochs=1000)
load_from = CHECKPOINT
auto_resume = False

# # trash
# checkpoint_config = dict(max_keep_ckpts = 1)
