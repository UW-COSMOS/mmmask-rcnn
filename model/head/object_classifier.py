"""
Multimodal Network Head Implementation
of Faster RCNN
Author: Josh McGrath
"""

from torch import nn


class MultiModalClassifier(nn.Module):
    def __init__(self):
        super(MultiModalClassifier).__init__(self)
