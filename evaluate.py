import torch
import numpy as np

def compute_mIoU(pred, label, num_classes):
    ious = []
    pred = pred.cpu().numpy()
    label = label.cpu().numpy()
    for cls in range(num_classes):
        pred_inds = (pred == cls)
        label_inds = (label == cls)
        intersection = np.logical_and(pred_inds, label_inds).sum()
        union = np.logical_or(pred_inds, label_inds).sum()
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append(intersection / union)
    return np.nanmean(ious)
