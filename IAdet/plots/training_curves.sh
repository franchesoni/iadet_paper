for LABEL in {0..9}
do
  LOGFILENAME=$(find data/results/hill/label_${LABEL}/run -path "*.log.json")
  python tools/analysis_tools/analyze_logs.py plot_curve $LOGFILENAME --keys  loss --out IAdet/plots/figs/label_${LABEL}_training_curve.png
done