

import os
import copy
import numpy as np
import mmcv

def filter_by_label(data, label_idx=14):
    out = []
    for dat in data:
        if (dat["ann"]["labels"] == label_idx).any():
            out_dat = copy.deepcopy(dat)
            bboxes, labels = out_dat["ann"]["bboxes"], out_dat["ann"]["labels"]
            out_dat["ann"]["bboxes"] = np.array(
                [
                    bboxes[i]
                    for i, label in enumerate(labels)
                    if label == label_idx
                ]
            )
            out_dat["ann"]["labels"] = np.array(
                [0 for label in labels if label == label_idx]
            )
            out.append(out_dat)
    return out

def save_one_label(filename, label_idx):
    data = mmcv.load(os.path.join("data", filename))
    new_data = filter_by_label(data, label_idx)
    mmcv.dump(new_data, os.path.join("data/by_label", f"label_{label_idx}_" + filename))

def visualize(filename, bboxes):
    mmcv.imshow_bboxes(img=os.path.join('data/VOCdevkit', filename), bboxes=bboxes, show=False, out_file='IAdet/results/gtdet.png')

if __name__ == '__main__':
    for label in range(20):
        save_one_label("voc0712_trainval.pkl", label)
        save_one_label("voc07_test.pkl", label)