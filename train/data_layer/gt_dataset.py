"""
A dataset class template for
an image dataset with ground truth
an example for the expected loader outputs
"""
from torch.utils.data import Dataset

class GTDataset(Dataset):
    def __init__(self, loader):
        """
        Initialize a GT dataset class
        :param loader: an object which loads from an indexable
        """
        self.loader = loader
    def __len__(self):
        return self.loader.size()

    def __getitem__(self, item):
        """
        get an image, ground truth pair
        :param item: int
        :return: [3 x size x size], [k x 5] (label, x1, y1, x2, y2)
        """
        ex = self.loader.ex[item]
        gt = self.loader.gt[item]
        return ex, gt
