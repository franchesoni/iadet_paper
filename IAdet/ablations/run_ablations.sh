# ABLATIONS for class 16 (sheeps)
export MM=/home/franchesoni/projects/current/detection_loop/mmdetection
export DATADIR=$MM/data/
export RESULTSDIR=$DATADIR/results/ablations
export SCRIPTS=$MM/IAdet/ablations/scripts
LABEL=16
SEED=0
cd $MM

echo "Cleaning all results..."
rm -rf $RESULTSDIR
mkdir $RESULTSDIR

##############################################################################################
# baseline without assistance
for ABLATION in baseline faster slower
do
  echo "Cleaning no_assistance results..."
  rm -rf $RESULTSDIR/no_assistance_$ABLATION
  mkdir $RESULTSDIR/no_assistance_$ABLATION

  echo "Start simulation of annotation for label $LABEL"
  python -u $SCRIPTS/no_assistance_robot.py $LABEL $ABLATION 2>&1 | tee $RESULTSDIR/no_assistance_$ABLATION/label_${LABEL}_no_assistance.log
done

#############################################################################################
for ABLATION in baseline fasterrcnn repeated faster slower random smallerlr frozen
do
  echo "Cleaning hill $ABLATION results..."
  rm -rf $RESULTSDIR/hill_$ABLATION
  mkdir $RESULTSDIR/hill_$ABLATION

  python -u $SCRIPTS/robot.py $LABEL $ABLATION 2>&1 | tee $RESULTSDIR/hill_$ABLATION/robot.log & robot_pid=$!;
  echo "started robot ${robot_pid}"
  while [ ! -f $RESULTSDIR/hill_$ABLATION/run/annotations.pkl ];
  do
    sleep 1
    echo "waiting for annotation file..."
  done
  bash $SCRIPTS/run_loop.sh $LABEL $SEED $ABLATION 2>&1 | tee $RESULTSDIR/hill_$ABLATION/loop.log & loop_pid=$!;
  echo "started loop ${loop_pid}"
  wait $robot_pid
  echo "waited ${robot_pid}"
  kill $loop_pid
  echo "killed ${loop_pid}"
done
