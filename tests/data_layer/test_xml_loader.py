from train.data_layer.xml_loader import XMLLoader
import unittest
from os.path import join
import torch
from utils.voc_utils import ICDAR_convert

class TestXMLLoader(unittest.TestCase):
    def setUp(self):
        data_dir = "../data"
        self.xml_dir = join(data_dir, "annotations")
        self.img_dir = join(data_dir, "images")
        self.proposal_dir = join(data_dir, "proposals")
        self.img_type = "jpg"

    def test_init(self):
        loader = XMLLoader(self.img_dir,
                           self.xml_dir,
                           self.proposal_dir,
                           self.img_type)

        self.assertIsNotNone(loader)

    def test_load_img_shape(self):
        loader = XMLLoader(self.img_dir,
                           self.xml_dir,
                           self.proposal_dir,
                           self.img_type)
        img, gt, proposals, id = loader[0]
        self.assertIsInstance(img, torch.Tensor)
        self.assertEqual(img.shape[2], 500)
        self.assertEqual(img.shape[0], 3)

    def test_load_gt(self):
        loader = XMLLoader(self.img_dir,
                           self.xml_dir,
                           self.proposal_dir,
                           self.img_type)
        img, gt, proposals, id = loader[0]
        tens, cls_names = gt
        self.assertIsInstance(tens, torch.Tensor)
        self.assertEqual(tens.shape[1], 4)
        self.assertEqual(tens.shape[0], len(cls_names))

    def load_no_gt(self):
        loader = XMLLoader(self.img_dir,
                           xml_dir=None,
                           proposal_dir=self.proposal_dir,
                           img_type=self.img_type)
        res = loader[0]
        self.assertEqual(len(res), 3)



