export MM=/home/franchesoni/projects/current/detection_loop/mmdetection
cd $MM

for LABEL in {0..9}
do
  echo "Starting evaluation of label $LABEL"
  for EPOCHTEN in {0..50}
  do
    EPOCH=$((10*EPOCHTEN+1))
    if [ -f data/results/hill/label_${LABEL}/run/epoch_${EPOCH}.pth ]; then 
      echo "Label $LABEL"
      echo "Epoch $EPOCH"
      python $MM/IAdet/curve/test.py $MM/IAdet/model/dataset/full_config.py $MM/data/results/hill/label_${LABEL}/run/epoch_${EPOCH}.pth --eval mAP --gpu-id 1 --label $LABEL
      python $MM/data/results/curve/clean_curve_run.py
    fi
  done
done
  # python $MM/tools/test.py $MM/IAdet/hill/model/loop_config.py $MM/IAdet/hill/model/run/epoch_$i.pth --out $MM/IAdet/results/curve/sheep_results_epoch_$i.pkl --eval mAP 