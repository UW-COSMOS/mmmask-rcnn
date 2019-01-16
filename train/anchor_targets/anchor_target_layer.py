"""
Module which takes the RPN output
and Generates Region Of Interest Proposals
Author: Josh McGrath
"""
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss
from .bbox_transform import bbox_overlaps
from utils.generate_anchors import generate_anchors
from train.losses.smooth_l1_loss import SmoothL1Loss

NEITHER = -1
NEGATIVE = -2



def match(regions, gt_boxes, upper, lower):
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
    overlaps = bbox_overlaps(regions, gt_boxes)
    # now get best prediction for each
    best_score_pred, match_idxs_pred = torch.max(overlaps, dim=1)
    # mask for the boxes with
    ret = NEITHER*torch.ones(regions.size(0))
    mask = best_score_pred >= upper
    # check for empty tensor
    if match_idxs_pred[mask].size(0) > 0:
        ret[mask] = match_idxs_pred[mask].float()
    # now we need the highest iou wrt to each
    # get back a vector indexed by gt_boxes which we need to
    # index back to targets
    best_score_gt, match_idxs_gt = torch.max(overlaps, dim=0)
    ret[match_idxs_gt] = torch.arange(0, gt_boxes.size(0))
    # finally, for anything with max iou < lower add a negative value
    mask = best_score_pred < lower
    ret[mask] = NEGATIVE
    return ret


class AnchorTargetLayer(nn.Module):
    def __init__(self, ratios, scales, image_size=1920, upper=0.4, lower=0.1, bg_ratio=1.0):
        super(AnchorTargetLayer, self).__init__()
        self.feat_stride = None
        self.ratios = ratios
        self.scales = scales
        self.image_size = image_size
        self.upper = upper
        self.lower = lower
        self.bg_ratio = bg_ratio
        self.anchors = None

        self.cls_loss = BCEWithLogitsLoss(reduction="mean")

        self.bbox_loss = SmoothL1Loss(10)

    def forward(self, cls_scores, bbox_deltas, gt_boxes, device):
        """
        process proposals from the RPN
        :param bbox_deltas: [N x 4K x H x W ]
        :param cls_scores: [N x 2K x H x W  ] of scores not probabilities
        :param gt_boxes: [M x 4] [x1, y1, x2, y2]
        :return:
        """

        """
        Algorithm
        1) get all center points
        2) make all anchors using center points
        3) apply bbox_deltas
        4) calculate IoUs
        5) find positive labels
        6) find negative labels
        7) sample down the negative labels
        8) calculate losses
        """
        # ensure center and original anchors have been precomputeds
        if self.feat_stride is None:
            self.feat_stride = round(self.image_size / float(cls_scores.size(3)))
        if self.anchors is None:
            self.anchors = generate_anchors(self.feat_stride,
                                            cls_scores.size(3),
                                            self.ratios,
                                            self.scales).to(device)

        H = cls_scores.size(2)
        W = cls_scores.size(3)
        batch_size = cls_scores.size(0)
        # TODO support batching
        cls_scores = cls_scores.squeeze()
        cls_scores = cls_scores.permute(1, 2, 0)

        # apply bbox deltas but first reshape to (batch,H,W,4K)
        bbox_deltas = bbox_deltas.permute(0, 2, 3, 1)
        # squeeze out batch dimension
        bbox_deltas = bbox_deltas.squeeze()
        # reshape again to match anchors (H,W,Kx4)
        bbox_deltas = bbox_deltas.reshape(bbox_deltas.shape[0], bbox_deltas.shape[1], -1, 4)
        _anchors = self.anchors.float()
        regions = _anchors + bbox_deltas
        # now we clip the boxes to the image
        regions = torch.clamp(regions, 0, self.image_size)
        # now we can start matching
        regions = regions.view(batch_size, -1, 4, H, W).permute(0, 3, 4, 1, 2)
        # reshaped to [batch x L x 4]
        regions = regions.reshape(batch_size, -1, 4)
        matches = match(regions.squeeze(0), gt_boxes[:, :4].squeeze(0), self.upper, self.lower)
        # filter out neither targets
        pos_mask = matches >= 0
        pos_inds = pos_mask.nonzero()
        neg_mask = matches == NEGATIVE
        neg_inds = neg_mask.nonzero()
        # now we downsample the negative targets
        pos_inds = pos_inds.reshape(-1)
        bg_num = torch.round(torch.tensor(pos_inds.size(0)*self.bg_ratio)).long()
        perm = torch.randperm(neg_inds.size(0))
        sample_neg_inds = perm[:bg_num]
        gt_cls = torch.cat((torch.ones(pos_inds.size(0)), torch.zeros(sample_neg_inds.size(0)))).to(device)
        # grab cls_scores from each point
        # first we need to reshape the cls_scores to match
        # the anchors
        cls_scores = cls_scores.reshape(batch_size, -1, 1)
        # reshape for pre batching
        cls_scores = cls_scores.squeeze()
        pred_cls = torch.cat((cls_scores[pos_inds], cls_scores[sample_neg_inds])).to(device)
        cls_loss = self.cls_loss(pred_cls, gt_cls)
        # we only do bbox regression on positive targets
        # get and reshape matches
        gt_indxs = matches[pos_inds].long()
        sample_gt_bbox = gt_boxes[gt_indxs, :]
        # TODO fix when implementing batches
        regions = regions.squeeze(0)
        sample_pred_bbox = regions[pos_inds, :]
        norm = torch.tensor(self.anchors.size(0)).float()
        bbox_loss = self.bbox_loss(sample_pred_bbox, sample_gt_bbox, norm)
        return cls_loss, bbox_loss











