export MM=/home/franchesoni/projects/current/detection_loop/mmdetection
cd $MM

rm -rf $MM/data/results/supervised
mkdir $MM/data/results/supervised

# train traditonally and supervisedly
for LABEL in 0 1 2 #3 4 5 6 7 8 9
do
  rm -rf $MM/data/results/supervised/label_$LABEL
  mkdir $MM/data/results/supervised/label_$LABEL
  timeout 1500s python $MM/IAdet/supervised/train.py $MM/IAdet/model/dataset/full_config.py --work-dir $MM/data/results/supervised/label_$LABEL --seed=0  --no-validate  --label $LABEL  #--auto-scale-lr
  # python $MM/tools/test.py $MM/IAdet/model/dataset/full_config.py $MM/data/results/supervised/label_$LABEL/latest.pth --out IAdet/results/supervised/baseline/baseline.pkl --eval mAP 
done

# train from last checkpoint and supervisedly
for LABEL in 0 1 2 #3 4 5 6 7 8 9
do
  rm -rf $MM/data/results/supervised/IA_label_${LABEL}
  mkdir $MM/data/results/supervised/IA_label_${LABEL}
  timeout 1500s python $MM/IAdet/supervised/train.py $MM/IAdet/model/dataset/full_config.py --work-dir $MM/data/results/supervised/IA_label_$LABEL --seed=0  --no-validate  --label $LABEL --changeloadfrom True #--auto-scale-lr
  # python $MM/tools/test.py $MM/IAdet/model/dataset/full_config.py $MM/data/results/supervised/label_$LABEL/latest.pth --out IAdet/results/supervised/baseline/baseline.pkl --eval mAP 
done

