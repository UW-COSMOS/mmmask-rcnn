#cls_branch [1x24x60x60]
#bbox_branch [1x48x 60x60]
import torch
from model.proposal.proposal_layer import ProposalLayer
img_size = 500
min_size = 16
ratios = [1, 0.5, 2]
scales = [64, 128, 256]

layer = ProposalLayer(ratios, scales, image_size=img_size, min_size=min_size)
rdata_cls = torch.rand(1, 18, 60, 60)
rdata_bbox = torch.rand(1, 36, 60, 60)
print(layer(rdata_cls, rdata_bbox))