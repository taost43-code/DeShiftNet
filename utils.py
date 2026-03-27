import argparse
import numpy as np
import torch

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def str2bool(v):
    if v.lower() in ['true', 1]:
        return True
    elif v.lower() in ['false', 0]:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    
    # Thresholding
    output_ = output > 0.5
    target_ = target > 0.5
    
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()
    
    iou = (intersection + smooth) / (union + smooth)
    dice = (2 * intersection + smooth) / (output_.sum() + target_.sum() + smooth)
    
    return iou, dice

def calculate_metrics(output, target):
    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    
    output = (output >= 0.5).astype(int)
    target = (target >= 0.5).astype(int)
    
    precision = precision_score(target.flatten(), output.flatten(), zero_division=0)
    recall = recall_score(target.flatten(), output.flatten(), zero_division=0)
    f1 = f1_score(target.flatten(), output.flatten(), zero_division=0)
    accuracy = accuracy_score(target.flatten(), output.flatten())
    
    return precision, recall, f1, accuracy
