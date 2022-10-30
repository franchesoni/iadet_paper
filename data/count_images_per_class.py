import os

import mmcv

dirpath = "/home/franchesoni/projects/current/detection_loop/mmdetection/data/by_label/"
for label in range(20):
  # f1 = os.path.join(dirpath, f"label_{label}_voc07_test.pkl")
  f2 = os.path.join(dirpath, f"label_{label}_voc0712_trainval.pkl")

  # a1 = mmcv.load(f1)
  a2 = mmcv.load(f2)
  print(f'class {label} has {len(a2)} files')