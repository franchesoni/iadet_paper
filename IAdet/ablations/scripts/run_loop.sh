LABEL=$1
SEED=$2
ABLATION=$3
export MM=/home/franchesoni/projects/current/detection_loop/mmdetection
cd $MM

wget -nc -O $MM/IAdet/model/coco_pretrained_ckpt.pth "https://download.openmmlab.com/mmdetection/v2.0/ssd/ssd300_coco/ssd300_coco_20210803_015428-d231a06e.pth"
# launch training
python $MM/IAdet/ablations/scripts/train.py $MM/IAdet/ablations/models/$ABLATION/full_config.py --work-dir $MM/data/results/ablations/hill_$ABLATION/run/ --seed $SEED --label $LABEL --ablation $ABLATION --no-validate
