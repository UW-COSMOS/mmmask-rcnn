import torch
X = 0
Y = 1
H = 2
W = 3
X2 = 2
Y2 = 3

def absolute_coords(bboxes, device):
    """
    convert bboxes [x,y, h,w] to [x1,y1,x2,y2]
    :param bboxes:
    :return:
    """
    bboxes = bboxes.clone()
    L, _ = bboxes.shape
    coords_bbox = torch.ones(L, 4).to(device)
    coords_bbox[:, X] = bboxes[:, X] - bboxes[:, W] / 2
    coords_bbox[:, Y] = bboxes[:, Y] - bboxes[:, H] / 2
    coords_bbox[:, X2] = bboxes[:, X] + bboxes[:, W] / 2
    coords_bbox[:, Y2] = bboxes[:, Y] + bboxes[:, H] / 2
    return coords_bbox

def centers_size(bbox_coords, device):
    """
    convert from absolute coords to (center_x, center_y,h, w)
    :param bboxes: [L x (X1, Y1, X2, Y2)]
    :return: [L x (X, Y, H, W)]
    """
    bbox_coords = bbox_coords.clone()
    L, _ = bbox_coords.shape
    bboxes = torch.ones(L, 4)
    bboxes[:, X] = (bbox_coords[:, X] + bbox_coords[:, X2])/2.0
    bboxes[:, Y] = (bbox_coords[:, Y] + bbox_coords[:, Y2])/2.0
    bboxes[:, W] = (bbox_coords[:, X2] - bbox_coords[:, X])
    bboxes[:, H] = (bbox_coords[:, Y2] - bbox_coords[:, Y])
    return bboxes



def cross_boundary(bboxes, img_size, device, remove=True):
    """
    get the indexes of boxes which cross the image boundary
    :param bboxes: [L x 4]
    :param img_size: W the side length of the square image
    :param device: the device to do computation on
    :return: [idx_1, idx_2] of indexes in the L dimension
    """
    # turn w,h coordinates into X2, Y2 coordinates
    bbox_coords = absolute_coords(bboxes, device)
    if remove:
        conforming = ((bbox_coords <= img_size) * (bbox_coords >= 0)).sum(dim=1)
        mask = conforming == 4
        return bboxes[mask, :]
    else:
        # clamp coords to the desired region
        bbox_coords = bbox_coords.clamp(0, img_size)
        return centers_size(bbox_coords, device)


