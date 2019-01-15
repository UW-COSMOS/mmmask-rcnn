"""
Connected components algorithm for region proposal
"""

import torch
from PIL import Image
import numpy as np


def convert_image_to_binary_map(img):
    """
    :param img: [3 x SIZE x SIZE] tensor
    :return: 
    """
    if img.type() != torch.float:
        img = img.float()
    white_tensor = torch.tensor([255, 255, 255], dtype=torch.float)
    binary_map = torch.tensor((), dtype=torch.uint8).new_ones(img.shape[1], img.shape[2])
    for x in range(img.shape[1]):
        for y in range(img.shape[2]):
            z = img[:, x, y]
            if z.eq(white_tensor).all():
                binary_map[x, y] = 0
    return binary_map


def test_convert_image_to_binary_map():
    img_rand = np.random.randint(0, high=255, size=(3, 20, 20))
    img_rand[:, :5, :5] = 255
    img_data = torch.from_numpy(img_rand)
    expected = torch.tensor((), dtype=torch.uint8).new_ones(20, 20)
    expected[:5, :5] = 0
    result = convert_image_to_binary_map(img_data)
    if not result.eq(expected).all():
        raise Exception('Test 1 test_convert_image_to_binary_map failed')

    img_rand = np.random.randint(0, high=255, size=(3, 20, 20))
    img_rand[:, 8, 8] = 255
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


def get_proposals(img):
    """
    Get the proposals from the img tensors
    :param img: [N x 3 x SIZE x SIZE] tensor
    :return: [N x M x 4], where M is the index of the connected component proposal
    """
    for i in img:
        bmap = convert_image_to_binary_map(i)


if __name__ == '__main__':
    test_convert_image_to_binary_map()
    test_get_components()
    img_rand = np.random.randint(0, high=255, size=(2,3,20,20))
    img_rand[0, :, 1, 1] = 255
    img_rand[0, :, 5, 5] = 255
    img_rand[1, :, 8, 8] = 255
    img_data = torch.from_numpy(img_rand)
    get_proposals(img_data)


