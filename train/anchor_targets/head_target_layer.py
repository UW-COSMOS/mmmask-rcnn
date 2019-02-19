"""
Module which takes the RPN output
and Generates Region Of Interest Proposals
Author: Josh McGrath
"""
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from utils.generate_anchors import generate_anchors
from utils.matcher import match



class HeadTargetLayer(nn.Module):
    def __init__(self, ncls=1):
        super(HeadTargetLayer, self).__init__()
        self.ncls = ncls
        print(f"there are {ncls} classes")
        self.cls_loss = CrossEntropyLoss(reduction="mean")

    def forward(self, rois, cls_scores, gt_boxes, gt_clses,device):
        """
        process proposals from the RPN
        :param rois : [N x L x 4]
        :param cls_scores: [N x L X C ] of scores not probabilities for C classes and L rois per image
        :param gt_boxes: [M x 4] [x1, y1, x2, y2]
        :param gt_cls:[Mx1] cls_idx
        :return:
        """
        N, C = cls_scores.shape
        pred = rois
        cls_loss = 0
        for idx, (gt_cls, gt_box) in enumerate(zip(gt_clses, gt_boxes)):
            pred_batch = pred[idx]
            pred_batch = pred_batch.to(device)
            gt_box = gt_box.squeeze(0)
            matches = match(pred_batch, gt_box, device)
            pos_mask = matches >= 0
            pos_inds = pos_mask.nonzero()
            pos_inds = pos_inds.reshape(-1)
            # build the positive labels
            gt_indxs = matches[pos_inds].long()
            gt_labels = gt_cls[gt_indxs].long()
            pred_scores = cls_scores[idx,:, :]
            #get logging info for non-bg classes
            l = self.cls_loss(pred_scores, gt_labels)
            cls_loss = l + cls_loss 
        return cls_loss
