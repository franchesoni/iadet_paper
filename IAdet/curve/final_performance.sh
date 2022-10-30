export MM=/home/franchesoni/projects/current/detection_loop/mmdetection
cd $MM

rm -rf data/results/final_performance
mkdir data/results/final_performance
for LABEL in {0..10}
do
  python $MM/tools/test.py $MM/IAdet/model/dataset/full_config.py $MM/IAdet/results/hill/run/latest.pth --eval mAP --gpu-id 1 2>&1 | tee IAdet/results/final_performance/label_${LABEL}.log
done
# python $MM/tools/test.py $MM/IAdet/hill/model/loop_config.py $MM/IAdet/hill/model/run/epoch_$i.pth --out $MM/IAdet/results/curve/sheep_results_epoch_$i.pkl --eval mAP 