"""
Bounding box ridge regression module
"""

import torch
import os
from ..utils import xml2list
import argv


def coordinates_to_anchor(coordinates):
    tl_x, tl_y, br_x, br_y = coordinates[0], coordinates[1], coordinates[2], coordinates[3]
    x = int(tl_x + br_x / 2)
    y = int(tl_y + br_y / 2)
    width = br_x - tl_x
    height = br_y - tl_y
    return x, y, width, height

def anchor_to_coordinates(x, y, width, height):
    tl_x = int(x - (width / 2))
    tl_y = int(y - (height / 2))
    br_x = int(x + (width / 2))
    br_y = int(y + (height / 2))
    return (tl_x, tl_y, br_x, br_y)

def calculate_iou(box1, box2):
    # Shamelessly adapted from
    # https://stackoverflow.com/questions/25349178/calculating-percentage-of-bounding-box-overlap-for-image-detector-evaluation
    # determine the coordinates of the intersection rectangle
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    bb2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    return iou


def associate_proposed_to_gt(proposed_boxes, gt_boxes):
    # The algorithm here is to calculate which of the gt_boxes has the highest IoU with the proposed boxes and return the matching sets of that
    # If there isn't an overlapping gt box, throw away the proposed box
    final_output = []
    for pb in proposed_boxes:
        ious = [calculate_iou(pb, gt) for gt in gt_boxes]
        max_el = max(ious)
        if max_el == 0:
            continue
        ind = ious.index(max_el)
        gt = gt_boxes[ind]
        final_output.append((pb, gt))
    return final_output


def write_datasets_from_xmls(proposed_xmls_dir, gt_xmls_dir):
    cls_file_handles = {}
    for f in os.listdir(proposed_xmls_dir):
        proposed_file = os.path.join(proposed_xmls_dir, f)
        gt_file = os.path.join(gt_xmls_dir, f)
        if not os.path.isfile(gt_file):
            print(f'Proposed xml file: {f} does not have a corresponding ground truth')
            continue
        proposed_list = xml2list(proposed_file)
        gt_list = xml2list(gt_file)
        proposed_cls_dict = {}
        gt_cls_dict = {}
        for item in proposed_list:
            item_cls, coords = item
            if item_cls in proposed_cls_dict:
                proposed_cls_dict[item_cls].append(coords)
            else:
                proposed_cls_dict[item_cls] = [coords]
        for item in gt_list:
            item_cls, coords = item
            if item_cls in gt_cls_dict:
                gt_cls_dict[item_cls].append(coords)
            else:
                gt_cls_dict[item_cls] = [coords]

        for key in proposed_cls_dict:
            if key not in gt_cls_dict:
                continue
            proposed_coords = proposed_cls_dict[key]
            gt_coords = gt_cls_dict[key]
            results = associate_proposed_to_gt(proposed_coords, gt_coords)

            if key not in cls_file_handles:
                key_fh = open(f'{key}.csv', 'w')
                key_fh.write('input_center_x,input_center_y,input_width,input_height,target_center_x,target_center_y,target_width,target_height\n')
                cls_file_handles[key] = key_fh
            fh = cls_file_handles[key]
            for result in results:
                inp = result[0]
                target = result[1]
                inp_x, inp_y, inp_width, inp_height = coordinates_to_anchor(inp)
                target_x, target_y, target_width, target_height = coordinates_to_anchor(target)
                fh.write(f'{inp_x},{inp_y},{inp_width},{inp_height},{target_x},{target_y},{target_width},{target_height}\n')
    for fh_key in cls_file_handles:
        fh = cls_file_handles[fh_key]
        fh.close()
    
if __name__ == '__main__':
    proposal_xml_dir = sys.argv[1]
    target_xml_dir = sys.argv[2]
    write_datasets_from_xml(proposal_xml_dir, target_xml_dir)





        




