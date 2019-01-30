"""
Connected components algorithm for region proposal
"""

import multiprocessing as mp
import torch
from PIL import Image, ImageFilter
import numpy as np
np.set_printoptions(threshold=np.nan)
from skimage import io
from torchvision.transforms import ToTensor, ToPILImage
from timeit import default_timer as timer
import math
import numbers
from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import os


def convert_image_to_binary_map(img):
    """
    :param img: [3 x H x W ] tensor
    :return: [H x W] binary tensor
    """
    if img.type() != torch.float:
        img = img.float()
    white_tensor = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float)
    binary_map = torch.tensor((), dtype=torch.uint8).new_ones(img.shape[1], img.shape[2])
    for x in range(img.shape[1]):
        for y in range(img.shape[2]):
            z = img[:, x, y]
            if ((z-white_tensor).abs() < 0.0001).all():
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

# https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/8
class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups)

# TODO: Can clean this up. Some redundancy here
def get_components(bmap, numpy=False):
    if numpy:
        bmap = torch.from_numpy(bmap)
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


def get_proposals(img, verbose=False, min_area=0):
    """
    Get the proposals from the img tensors
    :param img: [N x 3 x HEIGHT x WIDTH] tensor
    :param min_area: minimum area of a connected component to consider
    :return: [N x M x 4], where M is the index of the connected component proposal
    """
    cc_list = []
    for i in img:
        start_bmap = timer()
        bmap = convert_image_to_binary_map(i)
        end_bmap = timer()
        start_comp = timer()
        components = get_components(bmap)
        end_cmp = timer()
        components_set = set()
        start_cross = timer()
        for ind, component in enumerate(components):
            # Note that we do want to consider (c1, c1) pairs
            for next_component in components[ind:]:
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
        end_cross = timer()
        if verbose:
            print('get_proposal Timers\n------')
            print(f'Image to binary map: {end_bmap - start_bmap} s')
            print(f'Get components: {end_cmp - start_cmp} s')
            print(f'Cross components: {end_cross - start_cross} s')

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
    proposals = get_proposals(inp_rand, min_area=0, use_blur=False)
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

#https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
# Malisiewicz et al.
def non_max_suppression_fast(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
 
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
 
    # initialize the list of picked indexes    
    pick = []
 
    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
 
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
 
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
 
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
 
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
 
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
 
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))
 
    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")

def write_proposals(img_p, output_dir='tmp/cc_proposals'):
    img = Image.open(img_p)
    thresh = 245
    fn = lambda x : 0 if x > thresh else 255
    img_np = np.array(img.convert('RGB'))
    bmap_np = np.array(img.convert('L').point(fn, mode='1')).astype(np.uint8)
    img_height = bmap_np.shape[0]
    zero_col = np.zeros(img_height)
    left_w, right_w = 0, 0
    stop_left, stop_right = False, False
    for i in range(1, bmap_np.shape[1]):
        left = bmap_np[:, i]
        right = bmap_np[:, bmap_np.shape[1]-i]
        if not (left == zero_col).all():
            stop_left = True
        if not (right == zero_col).all():
            stop_right = True
        if stop_left and stop_right:
            diff = abs(left_w - right_w)
            if left_w < right_w:
                img_np = img_np[:, :bmap_np.shape[1]-diff, :]
                bmap_np = bmap_np[:, :bmap_np.shape[1]-diff]
            else:
                img_np = img_np[:, diff:, :]
                bmap_np = bmap_np[:, diff:]
            break
        elif stop_left:
            right_w += 1
        elif stop_right:
            left_w += 1
        else:
            right_w += 1
            left_w += 1
    blank_row_height = 10
    num_sections = int(img_height / blank_row_height)
    blank_row = np.zeros((blank_row_height, bmap_np.shape[1]))
    curr_top = 0
    white_rows = []
    for section in range(num_sections):
        curr_bot = curr_top + blank_row_height
        sub_img = bmap_np[curr_top:curr_bot, :]
        if (sub_img == blank_row).all():
            if len(white_rows) == 0:
                white_rows.append(curr_bot)
                curr_top += blank_row_height
                continue
            last_white_bot = white_rows[len(white_rows)-1]
            if last_white_bot == curr_top:
                white_rows[len(white_rows)-1] = curr_bot
            else:
                white_rows.append(curr_bot)
        curr_top += blank_row_height
    rows = []
    for i in range(len(white_rows)-1):
        curr = white_rows[i]
        nxt = white_rows[i+1]
        rows.append((bmap_np[curr:nxt, :], curr, nxt))
    block_coords = set()
    block_coords2 = {}
    blocks_list = []
    line_width = 5
    for row, top_coord, bottom_coord in rows:
        num_cols = get_columns_for_row(row)
        blocks, coords, col_idx = divide_row_into_columns(row, num_cols)
        for ind, b in enumerate(blocks):
            c = coords[ind]
            column_index = col_idx[ind]
            blank_row_height = 10
            num_sections = int(b.shape[0] / blank_row_height)
            blank_row = np.zeros((blank_row_height, b.shape[1]))
            curr_top = 0
            curr_bot = blank_row_height
            white_rows = []
            while curr_bot < b.shape[0]-1:
                sub_img = b[curr_top:curr_bot, :]
                if (sub_img == blank_row).all():
                    if len(white_rows) == 0:
                        white_rows.append(curr_bot)
                        curr_top += 1
                        curr_bot = curr_top + blank_row_height
                        continue
                    last_white_bot = white_rows[len(white_rows)-1]
                    if last_white_bot == curr_bot-1:
                        white_rows[len(white_rows)-1] = curr_bot
                    else:
                        white_rows.append(curr_bot)
                elif curr_top == 0:
                    white_rows.append(0)
                curr_top += 1
                curr_bot = curr_top + blank_row_height
            rows2 = []
            for i in range(len(white_rows)-1):
                curr = white_rows[i]
                nxt = white_rows[i+1]
                rows2.append((b[curr:nxt, :], curr, nxt))
            for r, c2, n in rows2:
                components = get_components(r, numpy=True)
                x1 = min(components, key=lambda x: x[1])
                x1 = x1[1]
                y1 = min(components, key=lambda x: x[0])
                y1 = y1[0]
                x2 = max(components, key=lambda x: x[3])
                x2 = x2[3]
                y2 = max(components, key=lambda x: x[2])
                y2 = y2[2]

                key = (num_cols, column_index)
                val = (top_coord + c2 + y1, c[0] + x1, top_coord + c2 + y2, c[0]+x2)
                if key in block_coords2:
                    block_coords2[key].append(val)
                else:
                    block_coords2[key] = [val]
    for key in block_coords2:
        coords_list = block_coords2[key]
        for ind2, bc in enumerate(coords_list):
            tl_y1, tl_x1, br_y1, br_x1 = bc
            block_coords.add((tl_x1, tl_y1, br_x1, br_y1))
            #for bc2 in coords_list[ind2:]:
            #    tl_y1, tl_x1, br_y1, br_x1 = bc
            #    tl_y2, tl_x2, br_y2, br_x2 = bc2
            #    block_coords.add((min(tl_x1, tl_x2), min(tl_y1, tl_y2), max(br_x1, br_x2), max(br_y1, br_y2)))
    block_coords = list(block_coords)
    img_p = os.path.basename(img_p)
    write_p = os.path.join(output_dir, img_p[:-4] + '.csv')
    write_img_p = os.path.join(output_dir, img_p)
    with open(write_p, 'w') as wp:
        for coord in block_coords:
            wp.write(f'{coord[0]},{coord[1]},{coord[2]},{coord[3]}\n')
    draw_cc(img_np, block_coords, write_img_p=write_img_p)
    return 'hello'


        #cc_list = []
        #for coord, block in zip(block_coords, blocks_list):
        #    components = get_components(block, numpy=True)
        #    if len(components) == 0:
        #        continue
        #    # four  passes. left/right/bot/top components
        #    def calc_distance_y(component):
        #        c_tl_y, c_tl_x, c_br_y, c_br_x = component
        #        if c_tl_x <= curr_x <= c_br_x:
        #            ret = math.sqrt((c_tl_y - curr_y) ** 2)
        #            return ret
        #        return float('inf')

        #    def calc_distance_x(component):
        #        c_tl_y, c_tl_x, c_br_y, c_br_x = component
        #        if c_tl_y <= curr_y <= c_br_y:
        #            ret = math.sqrt((c_tl_x - curr_x) ** 2)
        #            return ret
        #        return float('inf')

        #    def filter_above_y(component):
        #        c_tl_y, c_tl_x, c_br_y, c_br_x = component
        #        return c_tl_y > curr_y

        #    def filter_less_x(component):
        #        c_tl_y, c_tl_x, c_br_y, c_br_x = component
        #        return c_tl_x > curr_x
        #    curr_y = 0
        #    curr_x = 0
        #    delta = 2
        #    curr_components = components
        #    left_components = set()
        #    while len(curr_components) > 0:
        #        dist_curr_components = [calc_distance_x(c) for c in curr_components]
        #        min_d = min(dist_curr_components)
        #        if min_d == float('inf'):
        #            curr_components = [c for c in curr_components if filter_above_y(c)]
        #            curr_y += delta
        #            continue
        #        i = next(ind for ind, d in enumerate(dist_curr_components) if d == min_d)
        #        c_tl_y, c_tl_x, c_br_y, c_br_x = curr_components[i]
        #        left_components.add(curr_components[i])
        #        curr_components = [c for c in curr_components if filter_above_y(c)]
        #        curr_y += delta

        #    left_components = list(left_components)


        #    curr_y = 0
        #    curr_x = block.shape[1]
        #    curr_components = components
        #    right_components = set()
        #    while len(curr_components) > 0:
        #        dist_curr_components = [calc_distance_x(c) for c in curr_components]
        #        min_d = min(dist_curr_components)
        #        if min_d == float('inf'):
        #            curr_components = [c for c in curr_components if filter_above_y(c)]
        #            curr_y += delta
        #            continue
        #        i = next(ind for ind, d in enumerate(dist_curr_components) if d == min_d)
        #        c_tl_y, c_tl_x, c_br_y, c_br_x = curr_components[i]
        #        right_components.add(curr_components[i])
        #        curr_components = [c for c in curr_components if filter_above_y(c)]
        #        curr_y += delta

        #    right_components = list(right_components)


        #    curr_y = 0
        #    curr_x = 0
        #    curr_components = components
        #    top_components = set()
        #    while len(curr_components) > 0:
        #        dist_curr_components = [calc_distance_y(c) for c in curr_components]
        #        min_d = min(dist_curr_components)
        #        if min_d == float('inf'):
        #            curr_components = [c for c in curr_components if filter_less_x(c)]
        #            curr_x += delta
        #            continue
        #        i = next(ind for ind, d in enumerate(dist_curr_components) if d == min_d)
        #        c_tl_y, c_tl_x, c_br_y, c_br_x = curr_components[i]
        #        top_components.add(curr_components[i])
        #        curr_components = [c for c in curr_components if filter_less_x(c)]
        #        curr_x += delta
        #        # Same as above, but switch x and y order
        #        #dist_curr_components_x = [calc_distance_x(c) for c in curr_components]
        #        #min_x_d = min(dist_curr_components_x)
        #        #x_comps = [curr_components[ind] for ind, d in enumerate(dist_curr_components_x) if d == min_x_d]
        #        #dist_x_comps = [calc_distance_x(c) for c in x_comps]
        #        #min_x_d = min(dist_x_comps)
        #        #i = next(ind for ind, d in enumerate(dist_x_comps) if d == min_x_d)
        #        #c_tl_y, c_tl_x, c_br_y, c_br_x = x_comps[i]
        #        #curr_x = c_br_x
        #        #top_components.append(x_comps[i])
        #        #curr_components = [c for c in curr_components if filter_less_x(c)]
        #    top_components = list(top_components)


        #    curr_y = block.shape[0]
        #    curr_x = 0
        #    curr_components = components
        #    bottom_components = set()
        #    while len(curr_components) > 0:
        #        dist_curr_components = [calc_distance_y(c) for c in curr_components]
        #        min_d = min(dist_curr_components)
        #        if min_d == float('inf'):
        #            curr_components = [c for c in curr_components if filter_less_x(c)]
        #            curr_x += delta
        #            continue
        #        i = next(ind for ind, d in enumerate(dist_curr_components) if d == min_d)
        #        c_tl_y, c_tl_x, c_br_y, c_br_x = curr_components[i]
        #        bottom_components.add(curr_components[i])
        #        curr_components = [c for c in curr_components if filter_less_x(c)]
        #        curr_x += delta

        #    bottom_components = list(bottom_components)
        #    all_components = [left_components, right_components, top_components, bottom_components]

        #    tl_y, tl_x, br_y, br_x = coord[0], coord[1], coord[2], coord[3]
        #    def translate_to_global(component):
        #        c_tl_y, c_tl_x, c_br_y, c_br_x = component
        #        return (c_tl_x + tl_x, c_tl_y + tl_y, c_br_x + tl_x, c_br_y + tl_y)
        #    translated_components = []
        #    for comp_list in all_components:
        #        translated = [translate_to_global(c) for c in comp_list]
        #        translated_components.append(translated)
        #    components_set = set()
        #    for ind, component_set in enumerate(translated_components):
        #        if ind == len(translated_components)-1:
        #            break
        #        for next_component_set in translated_components[ind+1:]:
        #            for component in component_set:
        #                for next_component in next_component_set:
        #                    tl_x1, tl_y1, br_x1, br_y1 = component
        #                    tl_x2, tl_y2, br_x2, br_y2 = next_component
        #                    # Lets pretend the max/min functions don't exist, shhhhhhh
        #                    tl_x = tl_x1 if tl_x1 < tl_x2 else tl_x2
        #                    tl_y = tl_y1 if tl_y1 < tl_y2 else tl_y2
        #                    br_x = br_x1 if br_x1 > br_x2 else br_x2
        #                    br_y = br_y1 if br_y1 > br_y2 else br_y2
        #                    components_set.add((tl_x, tl_y, br_x, br_y))
        #    components_list = list(components_set)
        #    components_list = [list(x) for x in components_list]
        #    cc_list.append(components_list)
        ## Flatten
        #cc_list = [c for c_list in cc_list for c in c_list]
        #cc_list_suppressed = non_max_suppression_fast(np.asarray(cc_list), 0.9999)
        #draw_cc(img_np, cc_list_suppressed)
        #print(len(cc_list_suppressed))
        #return cc_list


def draw_grid(img_np, block_coords):
    for coords in block_coords:
        print(coords)
        img_np[coords[0]:coords[2], coords[1]-1:coords[1]+1, :] = 50
        img_np[coords[0]:coords[2], coords[3]-1:coords[3]+1, :] = 50
        img_np[coords[0]-1:coords[0]+1, coords[1]:coords[3], :] = 50
        img_np[coords[2]-1:coords[2]+1, coords[1]:coords[3], :] = 50
    Image.fromarray(img_np).save('test.png')


def draw_cc(img_np, cc_list, write_img_p=None):
    for coords in cc_list:
#        if coords[1] > 110 and coords[3] < 620:
        img_np[coords[1]:coords[3], coords[0]-2:coords[0]+2, :] = 50
        img_np[coords[1]:coords[3], coords[2]-2:coords[2]+2, :] = 50
        img_np[coords[1]-2:coords[1]+2, coords[0]:coords[2], :] = 50
        img_np[coords[3]-2:coords[3]+2, coords[0]:coords[2], :] = 50
    write_p = 'test.png' if write_img_p is None else write_img_p
    Image.fromarray(img_np).save(write_p)


def get_columns_for_row(row):
    # 3/100 width = test width. We need half that for later
    test_width = int(math.ceil(row.shape[1] / 200))
    half_test_width = int(math.ceil(test_width / 2))
    curr_c = 1
    for c in range(2, 6):
        # Attempt to divide rows into c columns
        row_w = row.shape[1]
        # Check the row at the middle positions for column
        test_points = []
        for i in range(1, c):
            test_points.append(int(row_w / c * i))
        def mark_empty_block(p):
            block = row[:, p-half_test_width:p+half_test_width]
            test_col = np.zeros((block.shape[0], block.shape[1]))
            return (block == test_col).all()
        test_blocks = [mark_empty_block(p) for p in test_points]
        if False not in test_blocks:
            curr_c = c
    return curr_c
   


def divide_row_into_columns(row, n_columns):
    splits = []
    coords = []
    col_idx = []
    for c in range(1, n_columns):
        prev_row_div = int(row.shape[1] / n_columns * (c - 1))
        row_div = int(row.shape[1] / n_columns * c)
        coords.append((prev_row_div, row_div))
        splits.append(row[:, prev_row_div:row_div])
        col_idx.append(c)
    final_col = int(row.shape[1] / n_columns * (n_columns - 1))
    splits.append(row[:, final_col:])
    coords.append((final_col, row.shape[1]))
    col_idx.append(n_columns)
    return splits, coords, col_idx

def test_divide_row_into_columns():
    row = np.ones((1, 30))
    actual, _ = divide_row_into_columns(row, 2)
    expected = np.ones((1, 15))
    for a in actual:
        if a.shape != expected.shape:
            print(f'Test 1 failed test_divide_row_into_columns(): expected: {expected.shape} actual: {a.shape}')

    actual, _ = divide_row_into_columns(row, 3)
    expected = np.ones((1, 10))
    for a in actual:
        if a.shape != expected.shape:
            print(f'Test 2 failed test_divide_row_into_columns(): expected: {expected.shape} actual: {a.shape}')

def test_get_columns_for_row():
    row = np.ones((1, 10))
    expected = 1
    actual = get_columns_for_row(row)
    if expected != actual:
        print(f'Test 1 failed test_get_columns_for_row(): expected: {expected} actual: {actual}')

    row = np.ones((2, 20))
    row[:, 8:12] = 0
    expected = 2
    actual = get_columns_for_row(row)
    if expected != actual:
        print(f'Test 2 failed test_get_columns_for_row(): expected: {expected} actual: {actual}')

    row = np.ones((60, 50))
    row[:, 8:13] = 0
    row[:, 18:23] = 0
    row[:, 28:33] = 0
    row[:, 38:43] = 0
    expected = 5
    actual = get_columns_for_row(row)
    if expected != actual:
        print(f'Test 3 failed test_get_columns_for_row(): expected: {expected} actual: {actual}')




if __name__ == '__main__':
    pool = mp.Pool(processes=240)
    results = [pool.apply_async(write_proposals, args=(os.path.join('img',x),)) for x in os.listdir('img')]
    [r.get() for r in results]
    print(results)
    #test_divide_row_into_columns()
    #test_get_columns_for_row()
    #test_convert_image_to_binary_map()
    #test_get_components()
    #test_get_proposals()
    #img = Image.open('1812.10437.pdf-0001.png')
    #ts = ToTensor()
    #tens = ts(img)
    #tens = tens[:3, :, :]
    #us = tens.unsqueeze(0)
    #start = timer()
    #proposals = get_proposals(us, output_blur=True)
    #end = timer()
    #print(f'Proposals timer: {end - start}')
    #print(f'Number of proposals: {proposals.shape[1]}')



