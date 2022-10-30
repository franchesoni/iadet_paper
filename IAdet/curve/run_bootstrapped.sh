export MM=/home/franchesoni/projects/current/detection_loop/mmdetection
cd $MM


for LABEL in {0..2}
do
  echo "Starting evaluation of label $LABEL"
  for EPOCHTEN in {0..50}
  do
    EPOCH=$((5*EPOCHTEN+1))
    if [ -f data/results/supervised/IA_label_${LABEL}/epoch_${EPOCH}.pth ]; then 
      echo "IA_label $LABEL"
      echo "Epoch $EPOCH"
      python $MM/IAdet/curve/test.py $MM/IAdet/model/dataset/full_config.py $MM/data/results/supervised/IA_label_${LABEL}/epoch_${EPOCH}.pth --eval mAP --gpu-id 0 --label $LABEL
      python $MM/data/results/curve/clean_curve_run.py bootstrapped/curve_run.log
    fi
  done
done