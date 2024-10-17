import torch
import torch.nn.functional as F

def dice_loss(pred, target):
    smooth = 1.0
    intersection = (pred * target).sum()
    return 1 - (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def smooth_l1_loss(pred_bbox, target_bbox):
    return F.smooth_l1_loss(pred_bbox, target_bbox)

