"""
Training helper class
Takes a model, dataset, and training paramters
as arguments
"""
import torch
from os.path import join
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from train.anchor_targets.anchor_target_layer import AnchorTargetLayer
from train.anchor_targets.head_target_layer import HeadTargetLayer
from functools import partial
import bitmath


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
    return torch.stack(exs).float(), gt_box, gt_cls

def format(bytes):
    return bitmath.Byte(bytes).to_GiB()



class TrainerHelper:
    def __init__(self, model, dataset, params,device):
        """
        Initialize a trainer helper class
        :param model: a MMFasterRCNN model
        :param dataset: a GTDataset inheritor to load data from
        :param params: a dictionary of training specific parameters
        """
        self.model = model.to(device)
        self.dataset = dataset
        self.params = params
        self.cls = dict([(val, idx) for (idx, val) in enumerate(model.cls_names)])
        self.device = device
        self.anchor_target_layer = AnchorTargetLayer(model.ratios, model.scales,model.img_size).to(device)
        self.head_target_layer = HeadTargetLayer(model.ratios,
                                                 model.scales,
                                                 model.img_size,
                                                 ncls=len(model.cls_names)).to(device)


    def train(self):
        optimizer = optim.SGD(self.model.parameters(), lr=self.params["LEARNING_RATE"],weight_decay=self.params["WEIGHT_DECAY"])
        loader = DataLoader(self.dataset,
                            batch_size=self.params["BATCH_SIZE"],
                            collate_fn=partial(collate,cls_dict=self.cls),
                            pin_memory=True)
        for epoch in tqdm(range(self.params["EPOCHS"])):
            for batch in loader:
                # not currently supporting batching
                optimizer.zero_grad()
                old_weights = save_weights(self.model)
                ex, gt_box, gt_cls = batch
                ex = ex.to(self.device)
                gt_box = gt_box
                gt_cls = [gt.to(self.device) for gt in gt_cls]
                gt_box = [gt.reshape(1, -1,4).float().to(self.device) for gt in gt_box]
                # forward pass
                rpn_cls_scores, rpn_bbox_deltas, rois, cls_preds, cls_scores, bbox_deltas = self.model(ex, self.device)
                # calculate losses
                rpn_cls_loss, rpn_bbox_loss = self.anchor_target_layer(rpn_cls_scores, rpn_bbox_deltas,gt_box, self.device)
                # add gt classes to boxes
                cls_loss, bbox_loss = self.head_target_layer(rois, cls_scores, bbox_deltas, gt_box, gt_cls, self.device)
                print(f"  rpn_cls_loss: {rpn_cls_loss}, rpn_bbox_loss: {rpn_bbox_loss}")
                print(f"  head_cls_loss: {cls_loss}, bbox_loss: {bbox_loss}")
                loss = rpn_cls_loss + rpn_bbox_loss + cls_loss + bbox_loss
               	loss.backward() 
                print(check_grad(self.model))
                optimizer.step()
                new_weights = save_weights(self.model)
                print(f"weights updated? {check_weight_update(old_weights, new_weights)}")

            #anchor
            if epoch % self.params["CHECKPOINT_PERIOD"] == 0:
                name = f"model_{epoch}.pth"
                path = join(self.params["SAVE_DIR"], name)
                torch.save(self.model.state_dict(), path)

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

