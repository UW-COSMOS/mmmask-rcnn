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
from model.connected_components import connected_components as cc
class MMFasterRCNN(nn.Module):
    def __init__(self, kwargs):
        """
        Initialize a MMFasterRCNN network
        :param kwargs: configuration for the network, see model configs
        """
        super(MMFasterRCNN, self).__init__()
        self.kwargs = kwargs
        self.backbone,self.shared = get_backbone(kwargs["BACKBONE"])

        # size should be informed at least partially by the receptive field
        # ensure this is meant to be a free parameter
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

    def forward(self, img, device=torch.device("cpu"),proposals=None):
        """
        Process an Image through the network
        :param img: [Nx3xSIZE x SIZE] tensor
		:param device: the device to process on
		:return: [(cls_index,[x1, y1, x2, y2])] for each non bg-class
        """
        N = img.size(0)
        feature_map = self.backbone.forward(img)
        feature_map = self.shared(feature_map)
        maps = []
        for batch_el in range(N):
            rois = proposals[batch_el].to(device).float()
            # add fake scores
            rois = rois.reshape(-1, 4)
            L, _ = rois.shape
            fake_rois = torch.zeros(L,5).to(device)
            fake_rois[:, 1:] = rois
            rois = fake_rois
            map = self.ROI_pooling(feature_map, rois)
            maps.append(map)
        maps = torch.stack(maps)
        cls_preds, cls_scores, bbox_deltas = self.classification_head(maps,proposals)
        return proposals,cls_preds, cls_scores, bbox_deltas


    def set_weights(self,mean, std):
        for child in self.children():
            if child == self.shared:
                continue
            for parm in self.modules():
                if not hasattr(parm, "weight"):
                    continue
                w = parm.weight
                # skip convolutional layers
                if w.requires_grad:
                    nn.init.normal_(w,mean, std)
                    if hasattr(parm, "bias") and not (parm.bias is None) :
                        nn.init.constant_(parm.bias, 0)

                




