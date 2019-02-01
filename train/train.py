"""
Training helper class
Takes a model, dataset, and training paramters
as arguments
"""
import torch
from  torch import nn
from os.path import join
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
from train.anchor_targets.anchor_target_layer import AnchorTargetLayer
from train.anchor_targets.head_target_layer import HeadTargetLayer
from functools import partial
import bitmath
from tensorboardX import SummaryWriter
from .data_layer.transforms import NormalizeWrapper
import torchvision.transforms as transform
from utils.boundary_utils import centers_size
from torch.utils.data import random_split
import numpy as np

def unpack_cls(cls_dict, gt_list):
    arr = map(lambda x: cls_dict[x], gt_list)
    return torch.tensor(list(arr))




def collate(batch, cls_dict):
    """
    collation function for GTDataset class
    :param batch:
    :return:
    """
    exs = [item[0] for item in batch]
    gt_box = [item[1][0] for item in batch]
    gt_cls = [unpack_cls(cls_dict, item[1][1]) for item in batch]
    proposals = [item[2] for item in batch]
    return torch.stack(exs).float(), gt_box, gt_cls, proposals

def format(bytes):
    return bitmath.Byte(bytes).to_GiB()


def prep_gt_boxes(boxes, device):
    boxes = [box.reshape(-1, 4).float().to(device) for box in boxes]
    boxes = [box.reshape(1,-1,4) for box in boxes]
    return boxes

class TrainerHelper:
    def __init__(self, model, dataset, params,device):
        """
        Initialize a trainer helper class
        :param model: a MMFasterRCNN model
        :param dataset: a GTDataset inheritor to load data from
        :param params: a dictionary of training specific parameters
        """
        self.model = model.to(device)
        val_size  = 100
        train_size = len(dataset) - val_size
        self.train_set, self.val_set = random_split(dataset, (train_size, val_size))
        self.params = params
        self.cls = dict([(val, idx) for (idx, val) in enumerate(model.cls_names)])
        self.device = device
        if params["USE_TENSORBOARD"]:
            self.writer = SummaryWriter()
        self.head_target_layer = HeadTargetLayer(
                                     ncls=len(model.cls_names)).to(device)
                                     


    def train(self):
        optimizer = optim.Adam(self.model.parameters(), 
                              lr=self.params["LEARNING_RATE"],
                              weight_decay=self.params["WEIGHT_DECAY"])
        train_loader = DataLoader(self.train_set,
                            batch_size=self.params["BATCH_SIZE"],
                            collate_fn=partial(collate,cls_dict=self.cls),
                            pin_memory=True,
                            num_workers=3)
                            
        self.model.train(mode=True)
        iter = 0
        tot_cls_loss = 0.0
        tot_bbox_loss = 0.0
        for epoch in tqdm(range(self.params["EPOCHS"]),desc="epochs"):
            for idx, batch in enumerate(tqdm(train_loader, desc="batches", leave=False)):
                optimizer.zero_grad()
                try:
                    ex, gt_box, gt_cls, proposals = batch
                    ex = ex.to(self.device)
                    gt_box = gt_box
                    gt_cls = [gt.to(self.device) for gt in gt_cls]
                    gt_box = prep_gt_boxes(gt_box, self.device)
                    rois, cls_preds, cls_scores, bbox_deltas = self.model(ex, self.device, proposals=proposals)
                    rois = centers_size(rois[0])
                    rois = rois.unsqueeze(0).to(self.device).float()
                    cls_loss = self.head_target_layer(rois,
                            cls_scores, bbox_deltas, gt_box, gt_cls, self.device)
                    loss = cls_loss 
                    tot_cls_loss += float(cls_loss)
                    loss.backward()
                    nn.utils.clip_grad_value_(self.model.parameters(), 5)
                    optimizer.step()
                except Exception as e:
                    print(e)
                if idx % self.params["PRINT_PERIOD"] == 0:
                    del loss 
                    del cls_loss
                    del cls_preds
                    del cls_scores
                    torch.cuda.empty_cache()
                    val_loader = DataLoader(self.val_set,
                            batch_size=self.params["BATCH_SIZE"],
                            collate_fn=partial(collate,cls_dict=self.cls),
                            pin_memory=True,
                            num_workers=3)


                    self.validate(val_loader, iter)
                    if not (idx == 0 and epoch ==0):
                        self.writer.add_scalar("train_cls_loss", tot_cls_loss, iter)
                        tot_cls_loss = 0.0
                        tot_bbox_loss = 0.0
                    iter += 1
                    del val_loader
            if epoch % self.params["CHECKPOINT_PERIOD"] == 0:
                name = f"model_{epoch}.pth"
                path = join(self.params["SAVE_DIR"], name)
                torch.save(self.model.state_dict(), path)

    def validate(self,loader,iter):
        self.model.eval()
        tot_cls_loss = 0.0
        torch.cuda.empty_cache()
        for batch in loader:
            ex, gt_box, gt_cls, proposals = batch
            ex = ex.to(self.device)
            gt_box = gt_box
            gt_cls = [gt.to(self.device) for gt in gt_cls]
            gt_box = prep_gt_boxes(gt_box, self.device)
            # forward pass
            rois, cls_preds, cls_scores, bbox_deltas = self.model(ex, self.device, proposals=proposals)
            # calculate losses
            cls_preds = cls_preds.squeeze(0)
            L ,ncls = cls_preds.shape
            preds, idxs = torch.max(cls_preds, dim=1)
            rois = centers_size(rois[0])
            rois = rois.unsqueeze(0).to(self.device).float()
            cls_loss = self.head_target_layer(rois,
                    cls_scores, bbox_deltas, gt_box, gt_cls, self.device)
            # update batch losses, cast as float so we don't keep gradient history
            tot_cls_loss += float(cls_loss)
        self.output_batch_losses(
                                 tot_cls_loss,
                                 iter) 

        self.model.train()
    def output_batch_losses(self,  cls_loss,iter ):
        """
        output either by priting or to tensorboard
        :param rpn_cls_loss:
        :param rpn_bbox_loss:
        :param cls_loss:
        :param bbox_loss:
        :return:
        """
        if self.params["USE_TENSORBOARD"]:
            vals = {
                "cls_loss": cls_loss,
            }
            for key in vals:
                self.writer.add_scalar(key, vals[key], iter)
        print(f"  head_cls_loss: {cls_loss}")


def check_grad(model):
    flag = False
    for param in model.parameters():
        if not(param.grad is None):
            if not(param.grad.data.sum() == 0):
                flag = True
    return flag


def save_weights(model):
    save = {}
    for key in model.state_dict():
        save[key] = model.state_dict()[key].clone()
    return save


def check_weight_update(old, new):
    flag = False
    for key in old.keys():
        if not (old[key] == new[key]).all():
            flag = True
    return flag
