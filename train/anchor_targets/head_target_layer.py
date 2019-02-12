"""
Module which takes the RPN output
and Generates Region Of Interest Proposals
Author: Josh McGrath
"""
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from utils.matcher import match, NEGATIVE
from utils.generate_anchors import generate_anchors
from train.losses.smooth_l1_loss import SmoothL1Loss
from time import sleep
from utils.matcher import match


def balance(gt_labels, ncls):
    """
    downsample from classes 
    in order to do balanced training samples
    """
    ret = []
    for cls in range(ncls):
        mask = gt_labels == cls
        nex = mask.sum()
        idxs = mask.nonzero()
        if nex == 0:
            continue
        sample_num = max(1, float(gt_labels.size(0)/ncls))
        sample_num = torch.ceil(torch.tensor(sample_num).float()).long()
        sample_idxs = list(idxs[:sample_num])
        ret.extend(sample_idxs)
    ret = torch.stack(ret).squeeze(1)
    return ret



class HeadTargetLayer(nn.Module):
    def __init__(self, ncls=1):
        super(HeadTargetLayer, self).__init__()
        self.ncls = ncls
        print(f"there are {ncls} classes")
        self.cls_loss = CrossEntropyLoss(reduction="mean")

    def forward(self, rois, cls_scores, bbox_deltas, gt_boxes, gt_clses,device):
        """
        process proposals from the RPN
        :param rois : [N x L x 4]
        :param bbox_deltas: [N x L x 4C ] per class bboxes
        :param cls_scores: [N x L X C ] of scores not probabilities for C classes and L rois per image
        :param gt_boxes: [M x 4] [x1, y1, x2, y2]
        :param gt_cls:[Mx1] cls_idx
        :return:
        """

        """
        Algorithm
        1) grab correct bbox_deltas according to class predictions
        2) apply bbox_deltas to the original RoIs
        3) calculate IoUs
        4)  
        
        """
        N, L, C = cls_scores.shape
        # ensure center and original anchors have been precomputed
        # drop objectness score
        max_scores, score_idxs = torch.max(cls_scores, dim=2)
        # reshape bbox deltas to be [N, L x C x 4] so we can index by score_idx
        bbox_deltas = bbox_deltas.reshape(N, L, C, 4)
        # filter to only the boxes we want
        bbox_lst = []
        for img_idx in range(N):
            for roi_idx in range(L):
                cls_idx = score_idxs[img_idx, roi_idx]
                bbox_lst.append(bbox_deltas[img_idx, roi_idx, cls_idx, :])
        bbox_deltas = torch.stack(bbox_lst)
        bbox_deltas = bbox_deltas.reshape(N, L, 4)
        # now we can apply the bbox deltas to the RoIs
        pred = rois#+bbox_deltas
        # Now produce matches [L x 1]
        cls_loss = 0
        for idx, (gt_cls, gt_box) in enumerate(zip(gt_clses, gt_boxes)):
            pred_batch = pred[idx]
            gt_box = gt_box.squeeze(0)
            matches = match(pred_batch, gt_box, device)
            pos_mask = matches >= 0
            pos_inds = pos_mask.nonzero()
            pos_inds = pos_inds.reshape(-1)
            # build the positive labels
            gt_indxs = matches[pos_inds].long()
            gt_labels = gt_cls[gt_indxs].long()
            balance_inds = balance(gt_labels, self.ncls)
            gt_labels = gt_labels[balance_inds]
            pred_scores = cls_scores[idx,balance_inds, :]
            #get logging info for non-bg classes
            l = self.cls_loss(pred_scores, gt_labels)
            cls_loss = l + cls_loss 
        return cls_loss
