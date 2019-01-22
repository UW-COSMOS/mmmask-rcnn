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
from model.roi.roi_align import ROIAlign
from utils.memory import get_gpu_mem
from model.connected_components import connected_components as cc
class MMFasterRCNN(nn.Module):
    def __init__(self, kwargs):
        """
        Initialize a MMFasterRCNN network
        :param kwargs: configuration for the network, see model configs
        """
        super(MMFasterRCNN, self).__init__()
        self.kwargs = kwargs
        self.img_size = kwargs["IMG_SIZE"]
        self.scales = kwargs["SCALES"]
        self.ratios = kwargs["RATIOS"]
        self.backbone = get_backbone(kwargs["BACKBONE"])

        # size should be informed at least partially by the receptive field
        # ensure this is meant to be a free parameter
        if kwargs["PROPOSAL"]["METHOD"] == "RPN":
            self.RPN = RPN(
                self.backbone.output_depth,
                kwargs["RPN"]["DEPTH"],
                kwargs["RATIOS"],
                kwargs["SCALES"],
                kwargs["RPN"]["WINDOW_SIZE"]
            )
            self.proposal_layer = ProposalLayer(
                kwargs["RATIOS"],
                kwargs["SCALES"],
                kwargs["IMG_SIZE"],
                kwargs["PROPOSAL"]["NMS_PRE"],
                kwargs["PROPOSAL"]["NMS_POST"],
                kwargs["PROPOSAL"]["MIN_SIZE"],
                kwargs["PROPOSAL"]["NMS_THRESHOLD"]
            )
        else:
            self.RPN = None
            self.proposal_layer = cc.get_proposals
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

    def forward(self, img, device):
        """
        Process an Image through the network
        :param img: [Nx3xSIZE x SIZE] tensor
				:param device: the device to process on         
				:return: [(cls_index,[x1, y1, x2, y2])] for each non bg-class
        """
        feature_map = self.backbone.forward(img)
        if self.RPN is not None:
            rpn_cls_branch_preds, rpn_cls_branch_scores, rpn_bbox_branch =\
                self.RPN(feature_map)
            rois = self.proposal_layer(rpn_cls_branch_preds, rpn_bbox_branch, device)
        else:
            rois = self.proposal_layer(img, verbose=True)
            rois.to(device)
        # TODO needs to be fixed for batching
        rois = rois.squeeze()
        maps = self.ROI_pooling(feature_map, rois)
        cls_preds, cls_scores, bbox_deltas = self.classification_head(maps)
        return rpn_cls_branch_scores, rpn_bbox_branch, rois, cls_preds, cls_scores, bbox_deltas





