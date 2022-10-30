import os

hill_results_dir = "/home/franchesoni/projects/current/detection_loop/mmdetection/data/results/hill"

for label_dir in sorted(os.listdir(hill_results_dir)):
  with open(os.path.join(hill_results_dir, label_dir, "robot.log"), "r") as f:
    lines = f.readlines()
  times = [float(line.split('=')[-1]) for line in lines if "time" in line]
  print(label_dir)
  print(times[-1] - times[0])


