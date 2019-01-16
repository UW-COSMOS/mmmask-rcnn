"""
Smooth L1 loss node
Author: Josh McGrath
"""
from torch import nn
import torch



class SmoothL1Loss(nn.Module):
    def __init__(self, lamb, reduce=False):
        """
        initialize a smooth L1 loss
        :param lamb: the lambda to be used
        """
        super(SmoothL1Loss, self).__init__()
        self.lamb = lamb
        self.reduce = reduce


    def forward(self, out, gt, norm):
        """
        compute the smooth L1 loss for a set of gt boxes
        we assume all examples are to be counted in the loss
        (i.e) all background examples have been filtered out
        :param out: the output of the network bbox_deltas [N x K x 4]
        :param gt: the associated ground truth boxes for each box in out [N x K x 4]
        :param norm: normalization factor, usually the number of anchors
        :return: [N] losses or if reduce is on, the mean of these losses
        """
        box_diff = out - gt
        box_diff = torch.abs(box_diff)
        # print(f"box_diff: {box_diff}")
        signs = (box_diff < 1).detach().float()
        signs_inv = (box_diff >= 1).detach().float()
        # print(signs)
        smooth_l1 = (torch.pow(box_diff,2) * 0.5 * signs) + ((box_diff - 0.5) * signs_inv)
        # print(f"post cases: {smooth_l1}")
        # sum along 3rd dim
        smooth_l1 = smooth_l1.sum(2)
        smooth_l1 = smooth_l1.sum(1)
        # print(f"pre flattening smooth_l1: {smooth_l1}")
        smooth_l1 = smooth_l1.sum()/norm
        return smooth_l1




