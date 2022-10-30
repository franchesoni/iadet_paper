export MM=/home/franchesoni/projects/current/detection_loop/mmdetection
cd $MM

### PART 1 ###
# simulate annotation without assistance
bash $MM/IAdet/no_assistance/no_assistance.sh

# annotation experiment
echo "cleaning up result dir..."
rm -rf data/results/hill
mkdir data/results/hill
echo "starting loop"
for SEED in 0  # seed is not doing much
do
  for LABEL in 2
  do
    rm -rf data/results/hill/label_${LABEL}
    mkdir data/results/hill/label_${LABEL}
    echo "Label $LABEL"
    # run the robot, save the terminal output to robot.log and the process number for later
    python -u $MM/IAdet/hill/robot.py $LABEL 2>&1 | tee data/results/hill/label_$LABEL/robot.log & robot_pid=$!;
    echo "started robot ${robot_pid}"
    echo "waiting for annotation file..."
    while [ ! -f data/results/hill/label_${LABEL}/run/annotations.pkl ]; do sleep 1; done  # wait for an annotation file to be generated
    # run the training loop, save the terminal output to loop.log and the process number for later
    bash $MM/IAdet/hill/run_loop.sh $LABEL $SEED 2>&1 | tee data/results/hill/label_$LABEL/loop.log & loop_pid=$!;
    echo "started loop ${loop_pid}"
    # # to debug:
    # # python -u $MM/IAdet/hill/robot_fake.py $LABEL 2>&1 | tee $MM/data/results/hill/label_${LABEL}/robot_fake.log & robot_pid=$!;
    # # bash $MM/IAdet/hill/run_loop_fake.sh $LABEL $SEED 2>&1 | tee $MM/data/results/hill/label_${LABEL}/loop_fake.log & loop_pid=$!;
    wait $robot_pid
    echo "waited ${robot_pid}"
    kill $loop_pid
    echo "killed ${loop_pid}"
  done
done

# ### PART 2 ###
# # compute training curves for each label

# # evaluate checkpoints of traditional run
# rm -rf $MM/data/results/curve
# mkdir $MM/data/results/curve
# bash $MM/IAdet/curve/run.sh 2>&1 | tee $MM/data/results/curve/curve_run.log

# # evaluate supervised checkpoints
# rm -rf $MM/data/results/curve/supervised
# mkdir $MM/data/results/curve/supervised
# bash $MM/IAdet/curve/run_baseline.sh 2>&1 | tee $MM/data/results/curve/supervised/curve_run.log

# # evaluate model trained from last hill checkpoint
# rm -rf $MM/data/results/curve/bootstrapped
# mkdir $MM/data/results/curve/bootstrapped
# bash $MM/IAdet/curve/run_bootstrapped.sh 2>&1 | tee $MM/data/results/curve/bootstrapped/curve_run.log

