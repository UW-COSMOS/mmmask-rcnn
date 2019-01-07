"""
Region Proposal Network component of
Faster R-CNN

Author: Josh McGrath
"""

from torch import nn


class RPN(nn.Module):
    def __init__(self,input_depth, size=3):
        """
        Initialize a region Proposal network
        :param input_depth: number of filters coming out of the backbone
        :param size: window size of RPN
        """
        pass
    def forward(self, feature_slice):
        """
        process a slice of the convolutional feature map
        :param feature_slice:
        :return: (objectness, [x1, y1, x2, y2])
        """
        pass
