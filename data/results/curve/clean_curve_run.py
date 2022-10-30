import os
import argparse

dirpath = '/home/franchesoni/projects/current/detection_loop/mmdetection/data/results/curve/'

parser = argparse.ArgumentParser()
parser.add_argument('filename', default='curve_run.log')
args = parser.parse_args()
filepath = os.path.join(dirpath, args.filename)

with open(filepath, 'r') as f:
  lines = f.readlines()

lines = [line for line in lines if 'task' not in line]
with open(filepath, 'w') as f:
  f.writelines(lines)
