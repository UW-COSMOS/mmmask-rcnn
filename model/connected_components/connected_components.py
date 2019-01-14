"""
Connected components algorithm for region proposal
"""

import torch
from PIL import Image


def get_proposals(img):
    """
    Get the proposals from the img tensors
    :param img: [N x 3 x SIZE x SIZE] tensor
    :return: [N x M x 4], where M is the index of the connected component proposal
    """
    for i in img:
        print(i)
        print(i.shape())


if __name__ == '__main__':
    img_rand = np.random.rand(1,3,1920,1920)
    img_data = torch.from_numpy(rdata)
    get_proposals(img_data)


