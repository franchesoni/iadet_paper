import argparse
import shutil
import time
from pathlib import Path
import numpy as np

import mmcv
from mmdet.apis import init_detector, inference_detector
from mmdet.core.evaluation.mean_ap import tpfp_default

parser = argparse.ArgumentParser()
parser.add_argument("label")
parser.add_argument("ablation")
args = parser.parse_args()
label = args.label
ablation = args.ablation

IOU_THR = 0.5
SCORE_THR = 0.7
MM_path = Path(
    "/home/franchesoni/projects/current/detection_loop/mmdetection/"
)

trainval = mmcv.load(MM_path / f"data/by_label/label_{label}_voc0712_trainval.pkl")

config_file = f"/home/franchesoni/projects/current/detection_loop/mmdetection/IAdet/ablations/models/{ablation}/full_config.py"
checkpoint_file = f"/home/franchesoni/projects/current/detection_loop/mmdetection/data/results/ablations/hill_{ablation}/run/latest.pth"
dest_dir = Path(f"data/results/ablations/hill_{ablation}/run")
if dest_dir.exists():
    shutil.rmtree(dest_dir)
dest_dir.mkdir(exist_ok=True, parents=True)

if ablation == 'faster':
    R = 5
elif ablation == 'slower':
    R = 0.2
else:
    R = 1

for k in range(len(trainval)):
    print("-"*30)
    print(f"k={k}, time={time.time()}")
    st = time.time()
    B = len(trainval[k]["ann"]["labels"])

    if Path(checkpoint_file).exists():
        model = init_detector(config_file, checkpoint_file, device='cuda:0')
        img = MM_path / "data/VOCdevkit" / trainval[k]["filename"]
        result = inference_detector(model, img)[0]
        result = result[np.argsort(-result[:, -1])]
        score_thr = min(SCORE_THR, result[0, -1]-1e-5)
        
        model.show_result(img, [result], out_file='result.jpg', score_thr=score_thr)
        det_bboxes = result[result[:, -1] > score_thr]
        if len(det_bboxes) > 0:
            TP, FP = tpfp_default(
                det_bboxes=det_bboxes,
                gt_bboxes=trainval[k]["ann"]["bboxes"],
                gt_bboxes_ignore=trainval[k]["ann"]["bboxes_ignore"],
                iou_thr=IOU_THR,
            )
            TP, FP = TP.sum(), FP.sum()
            FN = B - TP
            I = 1 + min(FN * 2 + FP, 1 + 2 * B)
            print("Performance:", f"TP = {TP}, FP = {FP}, FN={FN}")
            print("Interaction change", f"{I / (1 + 2 * B)}={I}/{1 + 2*B}")
    else:
        I = 2 * B + 1
        print("No predictions")

    annotated, to_annotate = trainval[:k], trainval[k:]
    time.sleep(I / R)
    mmcv.dump(annotated, dest_dir / "annotations.pkl")
    mmcv.dump(to_annotate, dest_dir / "to_annotate.pkl")