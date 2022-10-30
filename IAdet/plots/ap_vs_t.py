import datetime

import numpy as np
import matplotlib.pyplot as plt

classes = {
0 : 'aeroplane', 
1 : 'bicycle', 
2 : 'bird', 
3 : 'boat', 
4 : 'bottle', 
5 : 'bus', 
6 : 'car', 
7 : 'cat', 
8 : 'chair', 
9 : 'cow', 
16 : 'sheep', 
}

t_N= {0:3247, 1:2920, 2:4302}

def load_aps(filename, many=False):
  with open(filename, "r") as f:
    lines = f.readlines()

  aps = []
  labels = []
  epochs = []
  for line in lines:
    if "OrderedDict" in line:
      ap = float(line.split("('AP50', ")[-1].split('), (')[0])
      aps.append(ap)
    if many:
      if "Label" in line:
        labels.append(int(line.split('Label ')[-1]))
      elif "IA_label " in line:
        labels.append(int(line.split('IA_label ')[-1]))
      elif "Epoch" in line:
        epochs.append(int(line.split('Epoch ')[-1]))
  if many:
    return aps, labels, epochs
  return aps

def load_times(filename):
  """ 
  e.g. 2022-09-20 02:07:17,491 - mmdet - INFO - Saving checkpoint at 1 epochs
  """
  with open(filename, "r") as f:
    lines = f.readlines()

  times, epochs = [], []
  for line in lines:
    if "Saving checkpoint" in line:
      time = datetime.datetime.strptime(line.split(",")[0], "%Y-%m-%d %H:%M:%S")
      times.append(time)
      epoch = int(line.split('at ')[-1].split(' epochs')[0])
      epochs.append(epoch)
  st = times[0]
  times = [(time - st).total_seconds() for time in times]
  return times, epochs

# plot 1, performance vs time for each label
import glob
plt.rcParams.update({'font.size': 16})
plt.figure()
N = 3
for label in range(N):
  aps, labels, ap_epochs = map(np.array, load_aps("data/results/curve/curve_run.log", many=True))
  labels, ap_epochs = labels[:len(aps)], ap_epochs[:len(aps)]
  aps = aps[labels==label]
  ap_epochs = ap_epochs[labels==label]
  times, epochs = map(np.array, load_times(f"data/results/hill/label_{label}/loop.log"))
  times = np.array([times[i] for i in range(len(times)) if epochs[i] in ap_epochs])
  epochs = np.array([epoch for epoch in epochs if epoch in ap_epochs])

  aps_sup, labels_sup, ap_epochs_sup = map(np.array, load_aps("data/results/curve/supervised/curve_run.log", many=True))
  aps_sup = aps_sup[labels_sup==label]
  ap_epochs_sup = ap_epochs_sup[labels_sup==label]
  times_sup, epochs_sup = map(np.array, load_times(glob.glob(f"data/results/supervised/label_{label}/*.log.json")[0][:-5]))
  times_sup = np.array([times_sup[i] for i in range(len(times_sup)) if epochs_sup[i] in ap_epochs_sup])
  epochs_sup = np.array([epoch for epoch in epochs_sup if epoch in ap_epochs_sup])

  aps_bst, labels_bst, ap_epochs_bst = map(np.array, load_aps("data/results/curve/bootstrapped/curve_run.log", many=True))
  aps_bst = aps_bst[labels_bst==label]
  ap_epochs_bst = ap_epochs_bst[labels_bst==label]
  times_bst, epochs_bst = map(np.array, load_times(glob.glob(f"data/results/supervised/IA_label_{label}/*.log.json")[0][:-5]))
  times_bst = np.array([times_bst[i] for i in range(len(times_bst)) if epochs_bst[i] in ap_epochs_bst])
  epochs_bst = np.array([epoch for epoch in epochs_bst if epoch in ap_epochs_bst])

  # valid = times_sup < times[-1]
  # times_sup = times_sup[valid]
  # epochs_sup = epochs_sup[valid]
  # aps_sup = aps_sup[:len(times_sup)] if len(times_sup) <= len(aps_sup) else aps_sup


  plt.plot(times[:len(aps)], aps, ['-', '--', ':'][label], label=f"{label} A", color=plt.cm.Set1(0))
  plt.plot(t_N[label] + times_sup[:len(aps_sup)], aps_sup, ['-', '--', ':'][label], label=f"{label} N",color=plt.cm.Set1(0.25))
  plt.plot(times[-1] + times_sup[:len(aps_sup)], aps_sup, ['-', '--', ':'][label], label=f"{label} M", color=plt.cm.Set1(0.5))
  plt.plot(times[-1] + times_bst[:len(aps_bst)], aps_bst, ['-', '--', ':'][label], label=f"{label} B", color=plt.cm.Set1(0.75))

  print('-'*80)
  print(f'{label}')
  print(f'A: {aps[-1]}')
  print(f'N: {aps_sup[-1]}')
  print(f'B: {aps_bst[-1]}')
  print(f'A/N: {aps[-1] / aps_sup[-1]}')

  # plt.plot(times_sup, aps_sup, '-', label="All annotations available")
plt.ylabel('Average Precision @ IoU = 0.5')
plt.xlabel('time (s)')
plt.legend(loc='lower right', fontsize=13, ncol=3)
plt.tight_layout()
plt.savefig(f'IAdet/plots/figs/apvst.png')

