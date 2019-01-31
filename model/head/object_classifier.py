"""
Multimodal Network Head Implementation
of Faster RCNN
Author: Josh McGrath
"""

from torch import nn
from torch.nn.functional import relu, softmax

class MultiModalClassifier(nn.Module):
    def __init__(self, pool_height, pool_width, pool_depth, intermediate, ncls):
        """
        Initialize a Head object
        :param pool_height: The height of the output ROI pool
        :param pool_width: the width of the output ROI pool
        :param pool_depth: the depth of the output ROI pool
        :param intermediate: the dimensionality of the intermediate FC layer
        :param ncls: the number of classes
        """
        super(MultiModalClassifier, self).__init__()
        self.height = pool_height
        self.width = pool_width
        self.depth = pool_depth
        self.dropout = nn.Dropout(p=0.3)
        self.FC = nn.Linear(self.height*self.width*self.depth, intermediate)
        self.FC_2 = nn.Linear(intermediate, intermediate)
        # background class
        self.cls_branch = nn.Linear(intermediate, ncls)
        self.bbox_branch = nn.Linear(intermediate, 4*ncls)


    def forward(self, roi_maps, proposals=None):
        """

        :param roi_maps: [NxLxDHxW]
        :return:
        """
        x_addition = self.featurize_proposals(proposals)
        N, L, D, H, W = roi_maps.shape 
        x = roi_maps.view(N,-1, self.depth * self.width * self.height)
        x = self.FC(x)
        x = self.dropout(x)
        x = relu(x)
        x = self.FC_2(x)
        x = self.dropout(x)
        x = relu(x)
        cls_scores = self.cls_branch(x)
        bbox_scores = self.bbox_branch(x)
        return softmax(cls_scores, dim=2), cls_scores, bbox_scores

    def featurize_proposals(proposals):
        pass


