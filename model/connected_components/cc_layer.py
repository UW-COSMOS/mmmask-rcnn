import torch
from torch import nn
from model.connected_components.connected_components import get_components
from torchvision.transforms import ToPILImage, ToTensor

class CCLayer(nn.Module):
    def __init__(self, cfg):
        super(CCLayer, self).__init__()
        self.warped_size = cfg.CC_LAYER.WARPED_SIZE
        self.pil = ToPILImage()
        self.ten = ToTensor()

    def forward(self, img, device, proposals=None):
        """
        Delegation function
        :param img:
        :param device:
        :param proposals:
        :return:
        """
        if proposals is None:
            proposals = get_components(img)
        windows = []
        for proposal in proposals:
            window = self.warp(img, proposal, device)
            windows.append(window)
        windows = torch.stack(windows)
        return windows, proposals

    def warp(self,img, proposal, device):
        """
        warp an image to a fixed size
        :param img:
        :param proposal:
        :return:
        """
        img = self.pil(img)
        img = img.crop(proposal)
        img = img.resize((self.warped_size, self.warped_size))
        tens = self.ten(img)
        return tens.to(device)

