
- add clear_curve_run.py to repo
# IAdet paper code


The steps below assume you have cloned the repository in your own machine.

## Understanding the code
The IAdet paper evaluates a simulated human in the loop to create bounding box annotations for single classes. The code here refers to the different runs made, which roughly involve:
  - dataset preparation
  - training traditional supervised models
  - running the loop while saving all intermediate weights and counting the number of annotations
  - evaluating the performance for each one of the intermediate weights
  - retraining the model over the full dataset using the last saved weights
  - running ablation experiments (running the loop for different configs)

We first split the PASCAL VOC dataset into single classes.

## Data preparation 
1. download the PASCAL VOC dataset (both 2007 and 2012) into `data/VOCdevkit`
2. run the following 
  ```
  python tools/dataset_converters/pascal_voc.py data/VOCdevkit -o data/
  ```
3. separate data by class running
  ```
  python IAdet/prepare_single_label_dataset.py
  ```


## Experiments
0. Add the path of this repo in place of the $MM variable originally set to `MM=/home/franchesoni/projects/current/detection_loop/mmdetection` in every script (suggestion: use vscode find / replace) 
1. Comment out PART 2 of `IAdet/run_experiments.sh` and run
   ```
   bash IAdet/run_experiments.sh
   ```


### Time comparison
Now we can do the plots by calling `python IAdet/plots/n_vs_t.py` and obtain something like the following:

```
t_A = 2046.4797689914703
t_N = 2558.0
t_A / t_N = 0.8000311841248907
```

### Supervised baseline

Once the annotation experiment is run, one can see how much time it took. This number of seconds is the time budget for the supervised baseline.

- Add this number in `IAdet/supervised/run_baseline.sh` 
- Run: 
  ```
  bash IAdet/supervised/run_baseline.sh 2>&1 | tee IAdet/results/supervised/run_logs.log
  ```

### Performance curve computation

#### Assisted case
- See maximum number of epochs in `IAdet/results/hill/run` and set that number in `IAdet/curve/run.sh`, `run_baseline.sh` and `run_bootstrapped.sh`.
- Comment PART 1 and uncomment PART 2 of `IAdet/run_experiments.sh` and run
   ```
   bash IAdet/run_experiments.sh
   ```

#### Baseline

- See maximum number of epochs in `IAdet/results/supervised/baseline` and set that number in `IAdet/curve/run_baseline.sh`. Run that script.
- Run `bash IAdet/curve/run_baseline.sh 2>&1 | tee IAdet/results/supervised/curve.log`


### Ablation studies

Run
```
bash IAdet/ablations/run_ablations.sh
```




