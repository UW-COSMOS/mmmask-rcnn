"""
Module which takes the RPN output
and Generates Region Of Interest Proposals
Author: Josh McGrath
"""
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from utils.bbox_overlaps import bbox_overlaps
from utils.generate_anchors import generate_anchors
from train.losses.smooth_l1_loss import SmoothL1Loss
from time import sleep

NEITHER = -1
NEGATIVE = -2


def match(regions, gt_boxes, upper, lower,device):
    """
    Get positive region indexes for each box
    :param regions: predicted regions [KA x 4]
    :param gt_boxes: [ M x 4]
    :return: [KA x 4] index to gt box of boxes which have either
        a) an IOU of 0.7 or greater with a gt_box
        b) the highest IOU for a given gt_box


    """
    # get ious for each predicted box and gt target
    # get back an NxM tensor of IOUs
    overlaps = bbox_overlaps(regions, gt_boxes, device)
    # now get best prediction for each
    best_score_pred, match_idxs_pred = torch.max(overlaps, dim=1)
    # mask for the boxes with
    ret = NEITHER * torch.ones(regions.size(0),device=device).long()
    mask = best_score_pred >= upper
    # check for empty tensor
    if match_idxs_pred[mask].size(0) > 0:
        ret[mask] = match_idxs_pred[mask]
    # now we need the highest iou wrt to each
    # get back a vector indexed by gt_boxes which we need to
    # index back to targets
    best_score_gt, match_idxs_gt = torch.max(overlaps, dim=0)
    ret[match_idxs_gt] = torch.arange(0, gt_boxes.size(0)).to(device)
    # finally, for anything with max iou < lower add a negative value
    mask = best_score_pred < lower
    ret[mask] = NEGATIVE
    return ret


class HeadTargetLayer(nn.Module):
    def __init__(self, ratios, scales, image_size=1920, upper=0.4, lower=0.1, bg_ratio=1.0, ncls=1):
        super(HeadTargetLayer, self).__init__()
        self.feat_stride = None
        self.ratios = ratios
        self.scales = scales
        self.image_size = image_size
        self.upper = upper
        self.lower = lower
        self.bg_ratio = bg_ratio
        self.anchors = None
        self.BACKGROUND =ncls 
        self.cls_loss = CrossEntropyLoss(reduction="mean")
        self.bbox_loss = SmoothL1Loss(10)

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
        pred = rois + bbox_deltas
        # Now produce matches [L x 1]
        cls_loss = 0
        bbox_loss = 0
        
        rois = rois.reshape(1,-1,4)
        for idx, (gt_cls, gt_box) in enumerate(zip(gt_clses, gt_boxes)):
            pred_batch = pred[idx]
            gt_box = gt_box.squeeze(0)
            matches = match(pred_batch, gt_box, self.upper, self.lower,device)
            pos_mask = matches >= 0
            pos_inds = pos_mask.nonzero()
            neg_mask = matches == NEGATIVE
            neg_inds = neg_mask.nonzero()
            # sample down
            pos_inds = pos_inds.reshape(-1)
            bg_num = torch.round(torch.tensor(pos_inds.size(0) * self.bg_ratio)).long() +1
            perm = torch.randperm(neg_inds.size(0))
            sample_neg_inds = perm[:bg_num].to(device)
            # build the positive labels
            gt_indxs = matches[pos_inds].long()
            pos_labels = gt_cls[gt_indxs].long()
            neg_labels = self.BACKGROUND*torch.ones(sample_neg_inds.size(0)).long().to(device)
            gt_labels = torch.cat((pos_labels, neg_labels))
            pred_scores = torch.cat((cls_scores[idx,pos_inds, :], cls_scores[idx,sample_neg_inds, :]))
            l = self.cls_loss(pred_scores, gt_labels)
            cls_loss = l + cls_loss 
            # now we can compute the bbox loss
            sample_pred_bbox = pred_batch[pos_inds, :]
            sample_roi_bbox = rois[idx, pos_inds, :]
            gt_bbox = gt_box[gt_indxs, :]
            gt_bbox = gt_bbox.reshape(-1,4)
            # no normalization happens at the head
            bbox_loss = bbox_loss + self.bbox_loss(sample_pred_bbox, gt_bbox,sample_roi_bbox, N)
        return cls_loss, bbox_loss
