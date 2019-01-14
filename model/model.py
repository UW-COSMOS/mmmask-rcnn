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
from model.roi.roi_align import ROIAlign
class MMFasterRCNN(nn.Module):
    def __init__(self, kwargs):
        """
        Initialize a MMFasterRCNN network
        :param kwargs: configuration for the network, see model configs
        """
        super(MMFasterRCNN, self).__init__()
        self.img_size = kwargs["IMG_SIZE"]
        self.scales = kwargs["SCALES"]
        self.ratios = kwargs["RATIOS"]
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
        self.proposal_layer = ProposalLayer(
            kwargs["RATIOS"],
            kwargs["SCALES"],
            kwargs["IMG_SIZE"],
            kwargs["PROPOSAL"]["NMS_PRE"],
            kwargs["PROPOSAL"]["NMS_POST"],
            kwargs["PROPOSAL"]["MIN_SIZE"],
            kwargs["PROPOSAL"]["NMS_THRESHOLD"]
        )
        #TODO spatial scale?
        output_size = (kwargs["ROI_POOL"]["OUTPUT_SIZE"],kwargs["ROI_POOL"]["OUTPUT_SIZE"])
        self.ROI_pooling = ROIAlign(
            output_size,
            kwargs["ROI_POOL"]["SPATIAL_SCALE"],
            kwargs["ROI_POOL"]["SAMPLING_RATIO"]
        )
        #add code to default to using the index as the name
        self.cls_names = kwargs["CLASSES"]["NAMES"]
        self.classification_head = MultiModalClassifier(kwargs["ROI_POOL"]["OUTPUT_SIZE"],
                                                        kwargs["ROI_POOL"]["OUTPUT_SIZE"],
                                                        self.backbone.output_depth,
                                                        kwargs["HEAD"]["INTERMEDIATE"],
                                                        len(self.cls_names))

    def forward(self, img):
        """
        Process an Image through the network
        :param img: [Nx3xSIZE x SIZE] tensor
        :param mask: boolean mask of which anchors to process
            if None, all anchors are processed
        :return: [(cls_index,[x1, y1, x2, y2])] for each non bg-class
        """
        feature_map = self.backbone.forward(img)
        print(f"feature map: {feature_map.shape}")
        rpn_cls_branch_preds, rpn_cls_branch_scores, rpn_bbox_branch =\
            self.RPN(feature_map)
        print(f"rpn_cls_banch: {rpn_cls_branch_preds.shape}")
        print(f"bbox branch:{rpn_bbox_branch.shape}")
        rois = self.proposal_layer(rpn_cls_branch_preds, rpn_bbox_branch)
        print(f"rois : {rois.shape}")
        #TODO this will break batching without a reshape
        maps = self.ROI_pooling(feature_map, rois)
        print(f"maps: {maps.shape}")
        cls_preds, cls_scores, bbox_deltas = self.classification_head(maps)
        return rpn_cls_branch_scores, rpn_bbox_branch,cls_preds, cls_scores, bbox_deltas





