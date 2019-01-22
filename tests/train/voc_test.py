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
from  argparse import ArgumentParser

parser = ArgumentParser(description="run on a PASCAL VOC dataset")
parser.add_argument('img_path', type=str)
parser.add_argument('anno_path', type=str)
args = parser.parse_args()
device = None
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
loader = XMLLoader(args.anno_path, args.img_path, "jpg")
dataset = GTDataset(loader)
train_config = yaml.load(open("train_config.yaml").read())
model_config = yaml.load(open("model_config.yaml").read())
print("------- Building Model --------")
model = MMFasterRCNN(model_config)
trainer = TrainerHelper(model, dataset, train_config,device)
trainer.train()
