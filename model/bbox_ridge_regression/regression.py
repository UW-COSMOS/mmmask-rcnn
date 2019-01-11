"""
ridge regression module
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd


class BoundingBoxDataset(Dataset):
    def __init__(self, csv_file, tensor_dir, transform=None):
        self.frame = pd.read_csv(csv_file, dtype='int')
        self.transform = transform
        # Directory where the CNN feature map tensors are stored
        self.tensor_dir

    def __len__(self):
        return len(self.frame)

    # THIS NEEDS TO BE UPDATED WHEN YOU FIGURE OUT HOW THE FEATURE MAP GETS SAVED
    def __getitem__(self, idx):
        full = self.frame.iloc[idx]
        inp = (full['input_center_x'], full['input_center_y'], full['input_width'], full['input_height'])
        target = (full['target_center_x'], full['target_center_y'], full['target_width'], full['target_height'])
        img_name = full['img_name']
        img_conv_map_name = img_name[:-4]
        img_conv_map = torch.load(os.path.join(self.tensor_dir, img_conv_map_name + '.pt'))
        sample = {'input': inp, 'target': target, 'conv_map': img_conv_map}
        if self.transform:
            sample = self.transform(sample)
        return sample


class ToTensor(object):

    def __call__(self, sample):
        inp, target, conv_map = sample['input'], sample['target'], sample['conv_map']
        return {'input': torch.tensor(inp), 'target': torch.tensor(target), 'conv_map': conv_map}

# TODO: Test and save weights, then test
def train(dataset, conv_map_D_in, lr=0.001, l2_regularization=100):
    w_x = torch.randn(conv_map_D_in, 1, requires_grad=True)
    w_y = torch.randn(conv_map_D_in, 1, requires_grad=True)
    w_w = torch.randn(conv_map_D_in, 1, requires_grad=True)
    w_h = torch.randn(conv_map_D_in, 1, requires_grad=True)
    optimizer = optim.Adam([w_x, w_y, w_width, w_height], lr=lr, weight_decay=l2_regularization)
    # batch descent
    for i in range(len(dataset)):
        sample = dataset[i]
        inp, target, conv_map = sample['input'], sample['target'], sample['conv_map']
        t_x = (target[0] - inp[0]) / inp[2]
        t_y = (target[1] - inp[1]) / inp[3]
        t_w = math.log(target[2]/inp[2])
        t_h = math.log(target[3]/inp[3])
        l_x = (torch.tensor(t_x).sub(conv_map.mm(w_x))).pow(2)
        l_y = (torch.tensor(t_y).sub(conv_map.mm(w_y))).pow(2)
        l_w = (torch.tensor(t_w).sub(conv_map.mm(w_w))).pow(2)
        l_h = (torch.tensor(t_h).sub(conv_map.mm(w_h))).pow(2)
        l_x.backward()
        l_y.backward()
        l_w.backward()
        l_h.backward()
    optimizer.step()





