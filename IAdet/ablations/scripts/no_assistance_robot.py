import argparse
import time
from pathlib import Path

import mmcv

parser = argparse.ArgumentParser()
parser.add_argument("label")
parser.add_argument("ablation")
args = parser.parse_args()
label = args.label
ablation = args.ablation

MM_path = Path(
    "/home/franchesoni/projects/current/detection_loop/mmdetection/"
)

trainval = mmcv.load(MM_path / f"data/by_label/label_{label}_voc0712_trainval.pkl")

print("label =", label)
if ablation == 'faster':
    R = 5
elif ablation == 'slower':
    R = 0.2
else:
    R = 1
st = time.time()
cumtime = st
for k in range(len(trainval)):
    print("-"*30)
    print(f"k={k}, time={cumtime}")
    B = len(trainval[k]["ann"]["labels"])
    I = 2 * B + 1
    cumtime += I / R
