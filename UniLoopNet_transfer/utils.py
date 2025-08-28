#!/usr/bin/env python
import torch
import torch.nn as nn
import numpy as np


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class Regularization(nn.Module):
    """L1 and L2 regularization"""
    def __init__(self, model, weight_decay1=0.0, weight_decay2=0.0):
        super(Regularization, self).__init__()
        if weight_decay1 <= 0 and weight_decay2 <= 0:
            print("param weight_decay can not <= 0")
            exit(0)
        self.model = model
        self.weight_decay1 = weight_decay1
        self.weight_decay2 = weight_decay2

    def to(self, device):
        self.device = device
        super().to(device)
        return self

    def forward(self, model):
        self.weight_list = self.get_weight(model)
        reg_loss = self.regularization_loss(self.weight_list, self.weight_decay1, self.weight_decay2)
        return reg_loss

    def get_weight(self, model):
        weight_list = []
        for name, param in model.named_parameters():
            if 'weight' in name:
                weight = (name, param)
                weight_list.append(weight)
        return weight_list

    def regularization_loss(self, weight_list, weight_decay1, weight_decay2):
        reg_loss = 0
        for name, w in weight_list:
            l1_reg = torch.norm(w, p=1)
            l2_reg = torch.norm(w, p=2)
            reg_loss = reg_loss + weight_decay1 * l1_reg + weight_decay2 * l2_reg
        return reg_loss


def get_mean_score(labels, scores, aug=2, cls=5):
    """
    Get mean score for augmented data
    
    Parameters:
    -----------
    labels: list, original labels
    scores: numpy array, prediction scores
    aug: int, augmentation factor
    cls: int, number of classes
    
    Returns:
    --------
    mean_scores: numpy array, averaged scores
    """
    n_samples = len(labels)
    mean_scores = []
    
    for i in range(n_samples):
        start_idx = i * aug
        end_idx = (i + 1) * aug
        sample_scores = scores[start_idx:end_idx]
        mean_score = np.mean(sample_scores, axis=0)
        mean_scores.append(mean_score)
    
    return np.array(mean_scores)