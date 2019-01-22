"""
Smooth L1 loss node
Author: Josh McGrath
"""
from torch import nn
import torch

X = 0
Y = 1
W = 2
H = 3

class SmoothL1Loss(nn.Module):
    def __init__(self, lamb, reduce=False):
        """
        initialize a smooth L1 loss
        :param lamb: the lambda to be used
        """
        super(SmoothL1Loss, self).__init__()
        self.lamb = lamb
        self.reduce = reduce

    """
    all tensors are indexed by [L, x, y, h, w]
    """
    def forward(self, out, gt, roi, norm):
        """
        compute the smooth L1 loss for a set of gt boxes
        we assume all examples are to be counted in the loss
        (i.e) all background examples have been filtered out
        :param out: the output of the network bbox_deltas [K x 4]
        :param gt: the associated ground truth boxes for each box in out [K x 4]
        :param norm: normalization factor, usually the number of anchors
        :return: [N] losses or if reduce is on, the mean of these losses
        """
        t_x = (out[ :, X] - roi[ :, X])/roi[ :, W]
        t_y = (out[:, Y] - roi[:, Y])/roi[:, Y]
        t_w = torch.log(out[:, W]/roi[:,W])
        t_h = torch.log(out[:, H]/roi[:,H])
        gt_x = (gt[:, X] - roi[:, X])/roi[:, W]
        gt_y = (gt[:, Y] - roi[:, Y])/roi[:, Y]
        gt_w = torch.log(gt[:, W]/roi[:, W])
        gt_h = torch.log(gt[:, H]/roi[:, H])
        t = torch.stack((t_x, t_y, t_h, t_w))
        gt = torch.stack((gt_x, gt_y, gt_h, gt_w))
        box_diff = t - gt
        box_diff = torch.abs(box_diff)
        # print(f"box_diff: {box_diff}")
        signs = (box_diff < 1).detach().float()
        signs_inv = (box_diff >= 1).detach().float()
        smooth_l1 = (torch.pow(box_diff,2) * 0.5 * signs) + ((box_diff - 0.5) * signs_inv)
        # sum along 3rd dim
        smooth_l1 = smooth_l1.sum(2)
        smooth_l1 = smooth_l1.sum(1)
        smooth_l1 = smooth_l1.sum()/norm
        return self.lamb*smooth_l1




