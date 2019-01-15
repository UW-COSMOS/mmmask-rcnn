"""
Connected components algorithm for region proposal
"""

import torch
from PIL import Image
import numpy as np
from skimage import io
from torchvision.transforms import ToTensor
from timeit import default_timer as timer


def convert_image_to_binary_map(img):
    """
    :param img: [3 x H x W ] tensor
    :return: 
    """
    if img.type() != torch.float:
        img = img.float()
    white_tensor = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float)
    binary_map = torch.tensor((), dtype=torch.uint8).new_ones(img.shape[1], img.shape[2])
    for x in range(img.shape[1]):
        for y in range(img.shape[2]):
            z = img[:, x, y]
            if z.eq(white_tensor).all():
                binary_map[x, y] = 0
    return binary_map


def test_convert_image_to_binary_map():
    img_rand = np.random.rand(3, 20, 20)
    img_rand[:, :5, :5] = 1.0
    img_data = torch.from_numpy(img_rand)
    expected = torch.tensor((), dtype=torch.uint8).new_ones(20, 20)
    expected[:5, :5] = 0
    result = convert_image_to_binary_map(img_data)
    if not result.eq(expected).all():
        raise Exception('Test 1 test_convert_image_to_binary_map failed')

    img_rand = np.random.rand(3, 20, 20)
    img_rand[:, 8, 8] = 1.0
    img_data = torch.from_numpy(img_rand)
    expected = torch.tensor((), dtype=torch.uint8).new_ones(20, 20)
    expected[8, 8] = 0
    result = convert_image_to_binary_map(img_data)
    if not result.eq(expected).all():
        raise Exception('Test 2 test_convert_image_to_binary_map failed')


# TODO: Can clean this up. Some redundancy here
def get_components(bmap):
    label_map = np.zeros(bmap.shape)
    current_label = 1
    label_dict = {}
    # pass 1
    for y in range(bmap.shape[1]):
        for x in range(bmap.shape[0]):
            # Background pixel, pass it
            if bmap[x, y].item() == 0:
                continue
            # Top left corner
            if x == 0 and y == 0:
                label_map[x, y] = current_label
                current_label += 1
                continue
            # top row pixel
            if y == 0:
                # Check only the west pixel
                west = label_map[x-1, y]
                if west == 0:
                    label_map[x, y] = current_label
                    current_label += 1
                    continue
                else:
                    label_map[x, y] = west
                    continue
            # right column pixel
            if x == bmap.shape[0]-1:
                # Check all but northeast (since we're on the right most pixel and that is out of range
                west = label_map[x-1, y]
                north_west = label_map[x-1, y-1]
                north = label_map[x, y-1].item()
                points = [west, north_west, north]
                filtered_points = [p for p in points if p != 0]
                if len(filtered_points) == 0:
                    label_map[x, y] = current_label
                    current_label += 1
                    continue
                min_val = min(filtered_points)
                for f in filtered_points:
                    if f != min_val:
                        if min_val in label_dict:
                            label_dict[f] = label_dict[min_val]
                        else:
                            label_dict[f] = min_val
                label_map[x, y] = min_val
                continue
            # left column pixel
            if x == 0:
                # Check north and northeast
                north = label_map[x, y-1]
                north_east = label_map[x+1, y-1]
                points = [north, north_east]
                filtered_points = [p for p in points if p != 0]
                if len(filtered_points) == 0:
                    label_map[x, y] = current_label
                    current_label += 1
                    continue
                min_val = min(filtered_points)
                for f in filtered_points:
                    if f != min_val:
                        if min_val in label_dict:
                            label_dict[f] = label_dict[min_val]
                        else:
                            label_dict[f] = min_val
                label_map[x, y] = min_val
                continue
            # Normal pixel
            # finally, do all the west, north west, north, and north east points
            west = label_map[x-1, y]
            north_west = label_map[x-1, y-1]
            north = label_map[x, y-1]
            north_east = label_map[x+1, y-1]
            points = [west, north_west, north, north_east]
            filtered_points = [p for p in points if p != 0]
            if len(filtered_points) == 0:
                label_map[x, y] = current_label
                current_label += 1
                continue
            min_val = min(filtered_points)
            for f in filtered_points:
                if f != min_val:
                    if min_val in label_dict:
                        label_dict[f] = label_dict[min_val]
                    else:
                        label_dict[f] = min_val
            label_map[x, y] = min_val

                
    # pass 2
    components_list = {}
    for y in range(label_map.shape[1]):
        for x in range(label_map.shape[0]):
            if label_map[x, y] == 0:
                continue
            val = label_map[x, y]
            if val in label_dict:
                val = label_dict[val]
            tl_x, tl_y, br_x, br_y = x, y, x, y
            if val in components_list:
                tl_x, tl_y, br_x, br_y = components_list[val]
                if x < tl_x:
                    tl_x = x
                if x > br_x:
                    br_x = x
                # don't really need this but whatever i think it makes the code clear
                if y < tl_y:
                    tl_y = y
                if y > br_y:
                    br_y = y
            components_list[val] = (tl_x, tl_y, br_x, br_y)
    return list(components_list.values())

def test_get_components():
    test1 = torch.tensor((), dtype=torch.uint8).new_ones(20, 20)
    components = get_components(test1)
    expected = [(0, 0, 19, 19)]
    result = get_components(test1)
    if result != expected:
        print(expected)
        print('-----')
        print(result)
        raise Exception('test 1 test_get_components failed')
            
    test2 = torch.tensor((), dtype=torch.uint8).new_ones(20, 20)
    test2[2, :] = 0
    expected = [(0, 0, 1, 19), (3, 0, 19, 19)]
    result = get_components(test2)
    for element in expected:
        if element not in result:
            print(expected)
            print('-----')
            print(result)
            raise Exception('test 2 test_get_components failed')
    for element in result:
        if element not in expected:
            print(expected)
            print('-----')
            print(result)
            raise Exception('test 3 test_get_components failed')

    test3 = torch.tensor((), dtype=torch.uint8).new_ones(20, 20)
    test3[2, :] = 0
    test3[:, 10] = 0
    expected = [(0, 0, 1, 9), (3, 0, 19, 9), (0, 11, 1, 19), (3, 11, 19, 19)]
    result = get_components(test3)
    for element in expected:
        if element not in result:
            print(expected)
            print('-----')
            print(result)
            raise Exception('test 4 test_get_components failed')
    for element in result:
        if element not in expected:
            print(expected)
            print('-----')
            print(result)
            raise Exception('test 5 test_get_components failed')


# THIS IS UNFINISHED AND UNTESTED, DO NOT USE
def grid_proposal(img, recurse_depth=4):
    """
    Rough proposals based off a grid algorithm
    :param img: [SIZE x SIZE] binary tensor map
    :return: list of proposals [coords]
    """
    # Pass 1: large object row division
    zero_row = torch.tensor((), dtype=uint8).new_zeros(1, img.shape[1])
    heights_map = {}
    start_flag = False
    start_y = 0
    end_y = 0
    ind = 0
    for row in img:
        if row.eq(zero_row).all():
            if start_flag:
                continue
            start_flag = True
            start_y = ind
        else:
            if not start_flag:
                continue
            end_y = ind - 1
            height = end_y - start_y
            if height in heights_map:
                heights_map[height].append((start_y, end_y))
            else:
                heights_map[height] = [(start_y, end_y)]
        ind += 1
    heights = list(heights_map.keys())
    heights.sort()
    oset = dict.fromkeys(heights)
    oset_keys = list(oset.keys())
    split_heights = oset_keys[:3]
    all_spl_coords = []
    for spl_h in split_heights:
        spl_coords = heights_map[spl_h]
        for coord in spl_coords:
            all_spl_coords.append(coord)
    # sort by start y
    all_spl_coords.sort(key=lambda x: x[0])
    # crops are the end of the last object and the beginning of the next object
    row_crop_imgs = []
    for i in range(len(all_spl_coords)-1):
        t, b = all_spl_coords[i][1], all_spl_coords[i+1][0]
        crop = img[t:b, :]
        row_crop_imgs.append()


def get_proposals(img, min_area=1000):
    """
    Get the proposals from the img tensors
    :param img: [N x 3 x HEIGHT x WIDTH] tensor
    :return: [N x M x 4], where M is the index of the connected component proposal
    """
    cc_list = []
    for i in img:
        bmap = convert_image_to_binary_map(i)
        components = get_components(bmap)
        components_set = set()
        for ind, component in enumerate(components):
            for next_component in components[ind+1:]:
                tl_x1, tl_y1, br_x1, br_y1 = component
                tl_x2, tl_y2, br_x2, br_y2 = next_component
                # Lets pretend the max/min functions don't exist, shhhhhhh
                tl_x = tl_x1 if tl_x1 < tl_x2 else tl_x2
                tl_y = tl_y1 if tl_y1 < tl_y2 else tl_y2
                br_x = br_x1 if br_x1 > br_x2 else br_x2
                br_y = br_y1 if br_y1 > br_y2 else br_y2
                area = (br_x - tl_x) * (br_y - tl_y)
                if area > min_area:
                    components_set.add((tl_x, tl_y, br_x, br_y))
        components_list = list(components_set)
        components_list = [list(x) for x in components_list]
        cc_list.append(components_list)
    np_cc = np.array(cc_list, dtype='int32')
    return torch.from_numpy(np_cc)


def test_get_proposals():
    img_rand = np.random.rand(1,3,20,20)
    img_rand[0,:,  2, :] = 1.0
    img_rand[0,:,  :, 10] = 1.0
    inp_rand = torch.from_numpy(img_rand)
    np_expected = [[[0, 0, 19, 9], [0, 0, 1, 19], [0, 0, 19, 19], [3, 0, 19, 19], [0, 11, 19, 19]]]
    np_expected = np.array(np_expected, dtype='int32')
    expected = torch.from_numpy(np_expected)
    expected, _ = torch.sort(expected)
    proposals = get_proposals(inp_rand, min_area=0)
    proposals, _ = torch.sort(proposals)
    if proposals.shape != expected.shape:
        print(proposals.shape)
        print('----')
        print(expected.shape)
        raise Exception('Test 1 test_get_proposals failed')
    p = proposals[0, :, :].numpy()
    e = expected[0, :, :].numpy()
    for pp in p:
        if pp not in e:
            raise Exception('Test 2 test_get_proposals failed')
    for ee in e:
        if ee not in p:
            raise Exception('Test 3 test_get_proposals failed')


if __name__ == '__main__':
    test_convert_image_to_binary_map()
    test_get_components()
    test_get_proposals()
    img = Image.open('1812.10437.pdf-0001.png')
    ts = ToTensor()
    tens = ts(img)
    tens = tens[:3, :, :]
    us = tens.unsqueeze(0)
    start = timer()
    proposals = get_proposals(us)
    end = timer()
    print(f'Proposals timer: {end - start}')
    print(f'Number of proposals: {proposals.shape[1]}')



