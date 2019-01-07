"""
Multi Modal Faster-RCNN implementation
Author: Josh McGrath
"""

from torch import nn
from model.roi.generate_anchors import generate_anchors
from model.backbone.backbone import get_backbone
from model.rpn.rpn import RPN
from model.head.object_classifier import MultiModalClassifier


class MMFasterRCNN(nn.Module):
    def __init__(self, **kwargs):
        """
        Initialize a MMFasterRCNN network
        :param kwargs: configuration for the network, see model configs
        """
        super(MMFasterRCNN).__init__(self)
        self.img_size = kwargs["img_size"]
        self.backbone = get_backbone(kwargs["backbone"])

        # size should be informed at least partially by the receptive field
        # ensure this is meant to be a free parameter
        self.RPN = RPN(self.backbone.output_depth, kwargs["size"])
        self.RPN_threshold = kwargs["rpn"]["threshold"]
        self.head = MultiModalClassifier()
        self.region_anchors = generate_anchors(self.img_size,
                                               kwargs["stride"],
                                               kwargs["ratios"],
                                               kwargs["scales"])
        #add code to default to using the index as the name
        self.cls_names = kwargs["cls_names"]

    def forward(self, img, mask=None):
        """
        Process an Image through the network
        :param img: [SIZE x SIZE x 3] tensor
        :param mask: boolean mask of which anchors to process
            if None, all anchors are processed
        :return: [(cls_index,[x1, y1, x2, y2])] for each non bg-class
        """
        feature_map = self.backbone.forward(img)
        proposals = []
        for anchor in self.region_anchors:
            sub_map = self.attention(anchor, feature_map)
            score, box_coords = self.RPN(sub_map)
            if score > self.RPN_threshold:
                proposals.append(box_coords)
        for proposal in proposals:
            # do I use the same feature map as the RPN did?
            pass


    def attention(self, anchor, feature_map):
        """
        get the convolutional feature map for a given anchor region
        :param anchor:
        :param feature_map:
        :return: [H x W x 3] feature map slice
        """
        pass