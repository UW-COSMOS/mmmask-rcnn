"""
Training helper class
Takes a model, dataset, and training paramters
as arguments
"""
import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from train.losses.smooth_l1_loss import SmoothL1Loss
from train.anchor_targets.anchor_target_layer import AnchorTargetLayer

def collate(batch):
    """

    :param batch:
    :return:
    """
    exs = [item[0] for item in batch]
    gts = [item[1] for item in batch]
    return torch.stack(exs).float(), gts


class TrainerHelper:
    def __init__(self, model, dataset, params):
        """
        Initialize a trainer helper class
        :param model: a MMFasterRCNN model
        :param dataset: a GTDataset inheritor to load data from
        :param params: a dictionary of training specific parameters
        """
        self.model = model
        self.dataset = dataset
        self.params = params
        self.anchor_target_layer = AnchorTargetLayer(model.scales, model.ratios,500)


    def train(self):
        print("----------")
        print("beginning training")
        optimizer = optim.Adam(self.model.parameters(), lr=self.params["LEARNING_RATE"])
        loader = DataLoader(self.dataset,
                            batch_size=self.params["BATCH_SIZE"],
                            collate_fn=collate)
        for epoch in tqdm(range(self.params["EPOCHS"])):
            for batch in loader:
                #not currently supporting batching
                ex, gt = batch
                gt_boxes,gt_cls = gt[0]
                #fix for batching
                gt_boxes = gt_boxes.reshape(1, -1,4).float()
                # forward pass
                rpn_cls_scores, rpn_bbox_deltas, cls_preds, cls_scores, bbox_deltas = self.model(ex)
                print("forward pass complete")
                # calculate losses
                rpn_cls_loss, rpn_bbox_loss = self.anchor_target_layer(rpn_cls_scores, rpn_bbox_deltas,gt_boxes)
                print(f"rpn_cls_loss: {rpn_bbox_loss.shape}, rpn_bbox_loss: {rpn_bbox_loss.shape}")




if __name__ == '__main__':
    pass
