"""
Training helper class
Takes a model, dataset, and training paramters
as arguments
"""
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
    pass
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
        self.anchor_target_layer = AnchorTargetLayer(model.feat_stride, model.scales, model.ratios, params)


    def train(self):
        optimizer = optim.Adam(lr=self.params["LEARNING_RATE"])
        loader = DataLoader(self.dataset,
                            batch_size=self.params["BATCH_SIZE"],
                            collate_fn=collate)
        for epoch in tqdm(range(self.params["EPOCHS"])):
            for batch in loader:
                ex, gt = batch
                gt_cls, gt_boxes = gt
                # forward pass
                rpn_cls_branch_scores, rpn_bbox_branch, cls_preds, cls_scores, bbox_deltas = self.model(batch)
                # calculate losses
                target_attributions = self.anchor_target_layer(rpn_cls_branch_scores,gt_boxes)




if __name__ == '__main__':
    pass
