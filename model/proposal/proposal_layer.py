"""
Module which takes the RPN output
and Generates Region Of Interest Proposals
Author: Josh McGrath
"""

import torch
from torch import nn
from model.nms.nms import nms
from utils.generate_anchors import generate_anchors


def filter_regions(regions, min_size):
    """
    remove regions which are too small
    :param regions: [ Hx Wx K x4]
    :param min_size: integer, minimum length of a side
    :return: indexes of regions to be removed
    """
    # TODO implement, not currently implemented in matterport, or
    # in pytorch faster rcnn, or the old facebook implementation
    return regions



class ProposalLayer(nn.Module):
    def __init__(self, ratios, scales, image_size=1920, NMS_PRE=3000, NMS_POST=300, min_size=64, threshold=0.6):
        super(ProposalLayer,self).__init__()
        self.feat_stride = None
        self.ratios = ratios
        self.scales = scales
        self.image_size = image_size
        self.NMS_PRE = NMS_PRE
        self.NMS_POST = NMS_POST
        self.threshold = threshold
        self.min_size = min_size
        self.cpu = torch.device("cpu")
        self.anchors = None

    def forward(self, cls_scores, bbox_deltas, device):
        """
        process proposals from the RPN
        :param bbox_deltas: [N x 4K x H x W ]
        :param cls_scores: [N x 2K x H x W  ] of scores not probabilities
        :return:
        """

        """
        Algorithm
        1) get all center points
        2) make all anchors using center points
        3) apply bbox_deltas
        4) clip boxes to image
        5) filter small boxes
        6) pre NMS fitering by score
        7) NMS filtering
        8) post NMS filtering by score
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
        # remove all negative class scores
        batch_size = cls_scores.size(0)
        # TODO support batching
        cls_scores = cls_scores.squeeze()
        cls_scores_pos = cls_scores.permute(1, 2, 0)
        # get only even idxs
        even_idxs = torch.arange(0, cls_scores_pos.size(2), 2)
        cls_scores_pos = cls_scores_pos[:, :, even_idxs]

        # apply bbox deltas but first reshape to (0,2,3,1) = (12)(23)
        bbox_deltas = bbox_deltas.permute(0, 2, 3, 1)
        # first squeeze out batch dimension
        bbox_deltas = bbox_deltas.squeeze()
        # reshape again to match anchors
        bbox_deltas = bbox_deltas.reshape(bbox_deltas.shape[0], bbox_deltas.shape[1], -1, 4)
        _anchors = self.anchors.float()
        regions = _anchors + bbox_deltas
        # now we clip the boxes to the image
        regions = torch.clamp(regions, 0, self.image_size)
        # TODO filter any boxes which are too small
        # now we can grab the pre NMS regions
        # first we reshape the tensors to be N x K, N x K x 4
        cls_scores_pos = cls_scores_pos.permute(2, 0, 1).reshape(batch_size, -1)
        regions = regions.view(batch_size, -1, 4, H, W).permute(0, 3, 4, 1, 2)
        regions = regions.reshape(batch_size, -1, 4)
        pre_nms = min(self.NMS_PRE, cls_scores_pos.size(1))
        _, sort_order = cls_scores_pos.topk(pre_nms, dim=1)
        cls_scores_pos = cls_scores_pos[0,sort_order].reshape(-1,1)
        regions = regions[0,sort_order, :].squeeze()
        keep_idx_i = nms(regions, cls_scores_pos.squeeze(), self.threshold)
        keep_idx_i = keep_idx_i.long().view(-1)
        keep_idx_i = keep_idx_i[:self.NMS_POST]
        proposals = regions[keep_idx_i, :]
        cls_scores_pos = cls_scores_pos[keep_idx_i, :]
        output = cls_scores.new(batch_size, self.NMS_POST, 5)
        # TODO change after batching
        num_proposals = proposals.size(0)
        output[0, :, 0] = cls_scores_pos.squeeze()
        output[0, :num_proposals, 1:] = proposals
        return output

    def backward(self, ctx, grad_output):
        pass







