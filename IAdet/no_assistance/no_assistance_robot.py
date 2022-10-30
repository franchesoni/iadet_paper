import argparse
import time
from pathlib import Path

import mmcv

parser = argparse.ArgumentParser()
parser.add_argument("label")
args = parser.parse_args()
label = args.label

MM_path = Path(
    "/home/franchesoni/projects/current/detection_loop/mmdetection/"
)

trainval = mmcv.load(MM_path / f"data/by_label/label_{label}_voc0712_trainval.pkl")

print("label =", label)
R = 1
st = time.time()
cumtime = st
for k in range(len(trainval)):
    print("-"*30)
    print(f"k={k}, time={cumtime}")
    B = len(trainval[k]["ann"]["labels"])
    I = 2 * B + 1
    cumtime += I / R
