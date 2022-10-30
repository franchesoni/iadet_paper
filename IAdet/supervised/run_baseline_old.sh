export MM=/home/franchesoni/projects/current/detection_loop/mmdetection
cd $MM

# train with time budget
timeout 2046s python $MM/tools/train.py $MM/IAdet/model/dataset/full_config.py --work-dir $MM/IAdet/results/supervised/baseline --seed=0  --no-validate  #--auto-scale-lr
# test
python $MM/tools/test.py $MM/IAdet/model/dataset/full_config.py $MM/IAdet/results/supervised/baseline/latest.pth --out IAdet/results/supervised/baseline/baseline.pkl --eval mAP 
