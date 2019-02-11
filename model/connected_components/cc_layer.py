import torch
from torch import nn
from model.connected_components.connected_components import get_components
from torchvision.transforms import Resize
class CCLayer(nn.Module):
    def __init__(self, cfg):
        super(CCLayer, self).__init__()
        self.warped_size = cfg.CC_LAYER.WARPED_SIZE

    def forward(self, img, proposals=None):
        """
        Delegation function
        :param imgs:
        :param proposals:
        :return:
        """
        if proposals is None:
            proposals = get_components(img)
        windows = []
        for proposal in proposals:
            window = self.warp(img, proposal)
            windows.append(window)
        windows = torch.stack(windows)
        return windows, proposals

    def warp(self,img, proposal):
        """
        warp an image to a fixed size
        :param img:
        :param proposal:
        :return:
        """
        # slice = img[]
        pass