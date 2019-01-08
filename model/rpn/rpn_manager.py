"""
Module to manage the RPN sliding window and
Do Non-Max Suppression at test time
Author: Josh McGrath
"""

import torch
from model.rpn.rpn import RPN
from model.rpn.generate_anchors import generate_anchors

class RPNManager:
    def __init__(self, input_depth, output_depth=512,
                 window_size=3,
                 ratios=[1, 0.5,2],
                 scales=2**torch.arange(7,10),
                 threshold=0.7):
        """
        Initialize a RPN Manager
        :param input_depth: the depth of the convolutional map
        :param output_depth: the dimensionality after fully connected layer
        :param window_size: the size of convolutional window to use
        """
        self.input_depth = input_depth
        self.output_depth = output_depth
        self.window_size = window_size
        self.scales = scales
        self.ratios = ratios
        self.threshold = threshold
        self.rpn = RPN(input_depth, output_depth, window_size)

    def collect_proposals(self, feature_map, nms):
        """
        Collect proposals for a given feature map
        :param feature_map: [N x 3 x H x W] tensor
        :param nms: Boolean flag on whether or not to perform Non-
            Max Suppression
        :return: [(obj, slice, coords)]
        """
        pass
