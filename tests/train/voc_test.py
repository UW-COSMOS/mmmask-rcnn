"""
Perform testing with a sample of
VOC data w/ annotations
"""
from model.model import MMFasterRCNN
from train.data_layer.gt_dataset import GTDataset
from train.data_layer.xml_loader import XMLLoader
from train.train import TrainerHelper
import yaml
import torch

device = None
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
loader = XMLLoader("data/annotations", "data/images", "jpg")
dataset = GTDataset(loader)
train_config = yaml.load(open("train_config.yaml").read())
model_config = yaml.load(open("model_config.yaml").read())
print("------- Building Model --------")
model = MMFasterRCNN(model_config)
trainer = TrainerHelper(model, dataset, train_config,device)
trainer.train()
