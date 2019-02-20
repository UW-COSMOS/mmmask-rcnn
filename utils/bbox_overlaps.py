"""
compute the overlap matrix
Author: Josh McGrath
"""
import torch
from utils.boundary_utils import absolute_coords
X = 0
Y = 1
H = 2
W = 3
X2 = 2
Y2 = 3
def get_iou(bboxes, gt_box, device):
    """

    :param bboxes: [L x 4]
    :param gt_box: [1x4]
    :return: [L x 1]
    """
    L, _ = bboxes.shape
    gt_box = gt_box.expand(L, 4)
    # convert to x1, y1, x2, y2
    coords_bbox = absolute_coords(bboxes, device)

    # do the same for the ground truth
    coords_gt = absolute_coords(gt_box, device)
    # now use this to compute the aligned IoU
    i_boxes = torch.ones(L, 4).to(device)
    i_boxes[:, X] = torch.max(coords_bbox[:, X], coords_gt[:, X])
    i_boxes[:, Y] = torch.max(coords_bbox[:, Y], coords_gt[:, Y])
    i_boxes[:, X2] = torch.min(coords_bbox[:, X2], coords_gt[:, X2])
    i_boxes[:, Y2] = torch.min(coords_bbox[:, Y2], coords_gt[:, Y2])
    i_area = (i_boxes[:, X2] - i_boxes[:, X]) * (i_boxes[:, Y2] - i_boxes[:, Y])
    i_area[i_boxes[:, X2] < i_boxes[:, X]] = 0
    i_area[i_boxes[:, Y2] < i_boxes[:, Y]] = 0
    boxes_area = bboxes[:, W] * bboxes[:, H]
    gt_area = gt_box[:, W] * gt_box[:, H]
    return i_area / (boxes_area + gt_area - i_area)





def bbox_overlaps(bboxes, gt_boxes,device):
    """

    :param bboxes: [L x 4]
    :param gt_boxes: [K x 4]
    :return: [L x K]
    """
    L, _ = bboxes.shape
    K, _ = gt_boxes.shape
    overlaps = []
    for i in range(K):
        overlap = get_iou(bboxes, gt_boxes[i, :],device)
        overlaps.append(overlap)
    return torch.stack(overlaps,dim=1)
