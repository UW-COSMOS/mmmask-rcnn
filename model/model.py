"""
Multi Modal Faster-RCNN implementation
Author: Josh McGrath
"""
import torch
from torch import nn
from model.backbone.backbone import get_backbone
from model.rpn.rpn import RPN
from model.head.object_classifier import MultiModalClassifier
from model.proposal.proposal_layer import ProposalLayer
from model.roi.roi_pool import ROIPool
class MMFasterRCNN(nn.Module):
    def __init__(self, kwargs):
        """
        Initialize a MMFasterRCNN network
        :param kwargs: configuration for the network, see model configs
        """
        super(MMFasterRCNN, self).__init__()
        self.img_size = kwargs["IMG_SIZE"]
        self.backbone = get_backbone(kwargs["BACKBONE"])

        # size should be informed at least partially by the receptive field
        # ensure this is meant to be a free parameter
        self.RPN = RPN(
            self.backbone.output_depth,
            kwargs["RPN"]["DEPTH"],
            kwargs["RATIOS"],
            kwargs["SCALES"],
            kwargs["RPN"]["WINDOW_SIZE"]
        )
        #TODO calculate feature stride
        feat_stride = 32
        self.proposal_layer = ProposalLayer(
            feat_stride,
            kwargs["RATIOS"],
            kwargs["SCALES"],
            kwargs["IMG_SIZE"],
            kwargs["PROPOSAL"]["NMS_PRE"],
            kwargs["PROPOSAL"]["NMS_POST"],
            kwargs["PROPOSAL"]["MIN_SIZE"],
            kwargs["PROPOSAL"]["NMS_THRESHOLD"]
        )
        #TODO spatial scale?
        self.ROI_pooling = ROIPool(
            kwargs["ROI_POOL"]["OUTPUT_SIZE"],
            kwargs["ROI_POOL"]["SPATIAL_SCALE"]
        )
        self.head = MultiModalClassifier()
        #add code to default to using the index as the name
        self.cls_names = kwargs["CLASSES"]["NAMES"]

    def forward(self, img):
        """
        Process an Image through the network
        :param img: [SIZE x SIZE x 3] tensor
        :param mask: boolean mask of which anchors to process
            if None, all anchors are processed
        :return: [(cls_index,[x1, y1, x2, y2])] for each non bg-class
        """
        feature_map = self.backbone.forward(img)
        cls_branch_preds, cls_branch_scores, bbox_branch = self.RPN(feature_map)
        rois = self.proposal_layer(cls_branch_preds, bbox_branch)
        #TODO this will break batching without a reshape
        maps = torch.stack([self.ROI_pooling(roi) for roi in rois])





