export MM=/home/franchesoni/projects/current/detection_loop/mmdetection
export DATADIR=$MM/data/
export RESULTSDIR=$DATADIR/results
cd $MM

rm -rf $RESULTSDIR/no_assistance
mkdir $RESULTSDIR/no_assistance
for LABEL in {0..19}
do
  echo "Start simulation of annotation for label $LABEL"
  python -u $MM/IAdet/no_assistance/no_assistance_robot.py $LABEL 2>&1 | tee $RESULTSDIR/no_assistance/label_${LABEL}_no_assistance.log
done

