# --------------------------------------------------------
# Source code for Anchor3DLane
# Copyright (c) 2023 TuSimple
# @Time    : 2023/04/05
# @Author  : Shaofei Huang
# nowherespyfly@gmail.com
# --------------------------------------------------------

import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES, build_assigner
from .kornia_focal import FocalLoss
from .utils import get_class_weight, weight_reduce_loss

@LOSSES.register_module()
class DepthLoss(nn.Module):
    def __init__(self,
                 focal_alpha=0.25,
                 focal_gamma=2.,
                 anchor_len=10,
                 gt_anchor_len=200,
                 anchor_steps=[],
                 weighted_ce=False,
                 use_sigmoid=False,
                 loss_weights=None,
                 anchor_assign=True,
                 ssim = None):
        super(DepthLoss, self).__init__()
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.anchor_len = anchor_len
        self.anchor_steps = np.array(anchor_steps) - 1
        self.gt_anchor_len = gt_anchor_len
        self.use_sigmoid = use_sigmoid

        self.weighted_ce = weighted_ce
        self.loss_weights = loss_weights
        self.anchor_assign = anchor_assign
        self.fp16_enabled = False
        self.ssim = ssim
        
    def forward(self, depth_preds, depth_labels):
        
        valid_mask = (depth_labels > 1e-6).float()#表示深度标签中有效的像素点的掩码,真的可能有全0的哈
        
        # L1损失
        l1_loss = F.l1_loss(depth_preds*valid_mask, depth_labels*valid_mask, reduction='none') #直接放进去
        
        # SSIM损失（需要实现）
        if self.ssim is not None:
            ssim_loss = 1 - self.ssim(depth_preds, depth_labels)  # 示例伪代码
        else:
            ssim_loss = 0
        
        # 混合损失
        combined_loss = self.loss_weights['l1_loss']*l1_loss + self.loss_weights['ssim_loss']*ssim_loss
        
        combined_loss = (combined_loss * valid_mask).sum() / (valid_mask.sum() + 1e-6) 
        # combined_loss = combined_loss.sum()
        
        
        losses = {'depth_loss': combined_loss}

        # for k in losses.keys():
        #     losses[k] = losses[k] * self.loss_weights[k]

        return {'losses':losses}