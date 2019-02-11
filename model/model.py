"""
Multi Modal Faster-RCNN implementation
Author: Josh McGrath
"""
import torch
from torch import nn
from model.layers.featurization import Featurizer
from model.head.object_classifier import MultiModalClassifier
from model.utils.config_manager import ConfigManager
class MMFasterRCNN(nn.Module):
    def __init__(self, cfg):
        """
        Initialize a MMFasterRCNN network
        :param cfg: configuration file path for the network, see model configs
        """
        super(MMFasterRCNN, self).__init__()
        cfg = ConfigManager(cfg)
        self.featurizer = Featurizer(cfg)
        self.head = MultiModalClassifier(cfg.HEAD_HEIGHT,
                                         cfg.HEAD_WIDTH,
                                         cfg.HEAD_DEPTH,
                                         cfg.HEAD_DIM,
                                         len(cfg.CLASSES))


    def forward(self, *inputs):
        """
        Process an Image through the network
        """
        maps, proposals = self.featurizer(*inputs)
        cls_preds, cls_scores, bbox_deltas = self.classification_head(maps)
        return proposals, cls_preds, cls_scores, bbox_deltas


    def set_weights(self,mean, std):
        for child in self.children():
            if child == self.shared:
                continue
            for parm in self.modules():
                if not hasattr(parm, "weight"):
                    continue
                w = parm.weight
                # skip pretrained layers
                if w.requires_grad:
                    nn.init.normal_(w,mean, std)
                    if hasattr(parm, "bias") and not (parm.bias is None) :
                        nn.init.constant_(parm.bias, 0)

                




