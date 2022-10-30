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

def print_metrics(assisted_t, default_t, label, extra=""):
    t_A, t_N = assisted_t[-1], default_t[-1]
    print('-'*60)
    print(f"{label}: {classes[label]} {extra}")
    print(f"t_A = {t_A}")
    print(f"t_N = {t_N}")
    print(f"t_A / t_N = {t_A / t_N}")
    print(f"100 * (1-t_A / t_N) = {100 * (1-t_A / t_N)}")


def load_times_ks(filename):
  with open(filename, "r") as f:
    lines = f.readlines()

  ks, times = [], []
  for line in lines:
    if "k=" in line:
      ks.append(int(line.split('k=')[1].split(',')[0]))
      times.append(float(line.split('time=')[-1][:-2]))
  times = np.array(times) - times[0]
  return ks, times

speeds = {'faster': 5, 'baseline': 1, 'slower':0.2}

plt.rcParams.update({'font.size': 16})
plt.figure()
for label in [16]:
  for a, ablation in enumerate(['slower', 'baseline', 'faster']):
    default_k, default_t = load_times_ks(f"data/results/ablations/no_assistance_{ablation}/label_{label}_no_assistance.log")
    assisted_k, assisted_t = load_times_ks(f"data/results/ablations/hill_{ablation}/robot.log")
    plt.plot(assisted_t / default_t[-1], assisted_k, [':', '--', '-.', '-'][a%4], label=f"$IAdet$, $R={speeds[ablation]}$")
    print_metrics(assisted_t, default_t, label, extra=ablation)
  plt.plot(default_t / default_t[-1], default_k, '-', label=f"No assistance")#: {ablation}")
  # plt.plot(default_t, default_k[-1]*np.ones(len(default_t)), label="Dataset size")
  plt.ylabel('Number of annotations')
  plt.xlabel('t / $t_N$')
  plt.legend(fontsize=13, loc='lower right')
  plt.tight_layout()
  plt.savefig(f'IAdet/plots/figs/nvst_label_{label}_speed_ablation.png')



plt.figure()
for label in [16]:
  default_k, default_t = load_times_ks(f"data/results/ablations/no_assistance_baseline/label_{label}_no_assistance.log")
  # plt.plot(default_t, default_k[-1]*np.ones(len(default_t)), label="dataset size")
  for a, ablation in enumerate(['fasterrcnn', 'repeated', 'frozen', 'baseline', 'smallerlr', 'random']):
    assisted_k, assisted_t = load_times_ks(f"data/results/ablations/hill_{ablation}/robot.log")
    plt.plot(assisted_t, assisted_k, [':', '--', '-.', '-'][a%4], label=f"$IAdet$ {ablation}")
    print_metrics(assisted_t, default_t, label, extra=ablation)
  plt.plot(default_t, default_k, '-', label="No assistance")

plt.ylabel('Number of annotations')
plt.xlabel('time (s)')
plt.legend(loc='lower right', 
# bbox_to_anchor=(0.5, 0.5), 
fontsize=13, ncol=1)
plt.tight_layout()
plt.savefig(f'IAdet/plots/figs/nvst_label_{label}_ablations.png')


aks, ats = [], []
dks, dts = [], []
for label in range(10):
  assisted_k, assisted_t = load_times_ks(f"data/results/hill/label_{label}/robot.log")
  aks.append(assisted_k)
  ats.append(assisted_t)
  default_k, default_t = load_times_ks(f"data/results/no_assistance/label_{label}_no_assistance.log")
  dks.append(default_k)
  dts.append(default_t)

  # plt.rcParams.update({'font.size': 16})

  # plt.figure()
  # plt.plot(default_t, default_k, '-', label="No assistance")
  # plt.plot(assisted_t, assisted_k, '-', label="Assisted")
  # plt.plot(default_t, default_k[-1]*np.ones(len(default_t)), label="Dataset size")
  # plt.ylabel('Number of annotations')
  # plt.xlabel('time (s)')
  # plt.legend(fontsize=10)
  # plt.tight_layout()
  # plt.savefig(f'IAdet/plots/figs/nvst_label_{label}.png')

  print_metrics(assisted_t, default_t, label)

def find_value(time_index, values, t):
  assert all(sorted(time_index) == time_index)
  assert len(values) == len(time_index)

  if t <= time_index[0]:
    return values[0]

  if time_index[-1] <= t:
    return values[-1]

  i = 0
  while i < len(time_index) - 1 and time_index[i+1] < t:  
    i += 1
  # now ti[i] < t < ti[i+1]
  out = values[i] + (values[i+1] - values[i]) / (time_index[i+1] - time_index[i]) * (t - time_index[i])
  return out
  

ratioss = []
for i, at in enumerate(ats):
  ratioss.append([])
  for j, t in enumerate(at):
    dks_it = find_value(dts[i], dks[i], t)
    ratio = aks[i][j] / dks_it if dks_it > 0 else np.nan
    ratioss[i].append(ratio)

plt.figure()
box_size = 40
all_ts, all_ratios = np.concatenate(ats), np.concatenate(ratioss)
sorted_ts, sorted_ratios = zip(*sorted(zip(all_ts, all_ratios)))
smooth_ratios = np.array([np.nanmean(sorted_ratios[max(0, i-box_size):min(len(all_ts)-1, i+box_size)]) for i in range(len(all_ts))])

for i, ratios in enumerate(ratioss):
  plt.plot(ats[i], 100*(np.array(ratios) - 1), label=f"{i}: {classes[i]}")
plt.plot(sorted_ts, 100*(smooth_ratios-1), label="mean", color="black")

plt.ylabel('$IAdet$ advantage (%)')
plt.xlabel('time (s)')
plt.legend(fontsize=13, ncol=1, loc='upper left', bbox_to_anchor=(0.9, .9), fancybox=True)
plt.tight_layout()
plt.savefig('IAdet/plots/figs/ratios.png')
    
    



    # N = 150
    # plt.figure()
    # plt.plot(default_t[:N], default_k[:N], '-', label="No assistance")
    # plt.plot(assisted_t[:N], assisted_k[:N], '-', label="Assisted")
    # plt.ylabel('Number of annotations')
    # plt.xlabel('time (s)')
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig('IAdet/plots/figs/nvst_zoom.png')

