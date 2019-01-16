"""
Training helper class
Takes a model, dataset, and training paramters
as arguments
"""
import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from train.anchor_targets.anchor_target_layer import AnchorTargetLayer
from train.anchor_targets.head_target_layer import HeadTargetLayer
from functools import partial


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
        self.anchor_target_layer = AnchorTargetLayer(model.scales, model.ratios,model.img_size).to(device)
        self.head_target_layer = HeadTargetLayer(model.scales,
                                                 model.ratios,
                                                 model.img_size,
                                                 ncls=len(model.cls_names)).to(device)


    def train(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.params["LEARNING_RATE"])
        loader = DataLoader(self.dataset,
                            batch_size=self.params["BATCH_SIZE"],
                            collate_fn=partial(collate,cls_dict=self.cls))
        for epoch in tqdm(range(self.params["EPOCHS"])):
            for batch in loader:
                optimizer.zero_grad()
                # not currently supporting batching
                optimizer.zero_grad()
                ex, gt_box, gt_cls = batch
                # fix for batching
                ex = ex.to(self.device)
                gt_box = gt_box[0]
                gt_cls = gt_cls[0].to(self.device)
                gt_box = gt_box.reshape(1, -1,4).float().to(self.device)
                # forward pass
                rpn_cls_scores, rpn_bbox_deltas, rois, cls_preds, cls_scores, bbox_deltas = self.model(ex, self.device)
                print("forward pass complete")
                # calculate losses
                rpn_cls_loss, rpn_bbox_loss = self.anchor_target_layer(rpn_cls_scores, rpn_bbox_deltas,gt_box, self.device)
                # add gt classes to boxes
                cls_loss, bbox_loss = self.head_target_layer(rois, cls_scores, bbox_deltas, gt_box, gt_cls, self.device)
                print(f"rpn_cls_loss: {rpn_bbox_loss}, rpn_bbox_loss: {rpn_bbox_loss}")
                print(f"head_cls_loss: {cls_loss}, bbox_loss: {bbox_loss}")
                rpn_loss = rpn_cls_loss + rpn_bbox_loss
                rpn_loss.backward()
                head_loss = cls_loss + bbox_loss
                head_loss.backward()
                optimizer.step()


if __name__ == '__main__':
    pass
