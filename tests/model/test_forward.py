from model.model import MMFasterRCNN
import torch
import yaml

print('rpn test')
with open("model_config.yaml") as fh:
    document = fh.read()
    args = yaml.load(document)
    print(document)
    device = torch.device("cuda:0")
    net = MMFasterRCNN(args)
    net.to(device)
    rdata = torch.rand(1,3,1920, 1920, device=device)
    net(rdata, device)

print('cc test')
with open("model_config_cc.yaml") as fh:
    document = fh.read()
    args = yaml.load(document)
    print(document)
    device = torch.device("cuda:0")
    net = MMFasterRCNN(args)
    net.to(device)
    rdata = torch.rand(1,3,1920, 1920, device=device)
    net(rdata, device)

