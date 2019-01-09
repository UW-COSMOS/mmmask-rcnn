from model.model import MMFasterRCNN
import torch
import yaml

with open("config.yaml") as fh:
    document = fh.read()
    args = yaml.load(document)
    print(document)
    net = MMFasterRCNN(args)
    rdata = torch.rand(1,3,1920, 1920)
    net(rdata)
