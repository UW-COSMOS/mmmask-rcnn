from model.model import MMFasterRCNN
from model.utils.config_manager import ConfigManager
from PIL import Image
from torchvision.transforms import ToTensor
import torch
import unittest


class TestModel(unittest.TestCase):

    def setUp(self):
        self.cfg = ConfigManager("../data/model_config.yaml")
        self.image = Image.open("../data/images/2009_001444.jpg")
        transform = ToTensor()
        self.image = transform(self.image)
        self.proposals = [[1, 1, 344, 388]]
        self.device = torch.device("cpu")

    def test_init(self):
        model = MMFasterRCNN(self.cfg)
        self.assertIsNotNone(model)

    def test_forward_precomputed(self):
        model = MMFasterRCNN(self.cfg)
        proposals,cls_preds, cls_scores, bbox_deltas = model(self.image, self.device, self.proposals)
        self.assertEqual(self.proposals, proposals)
        self.assertEqual(1, cls_scores.shape[1])
        self.assertEqual((1, 1, 4*len(self.cfg.CLASSES)), bbox_deltas.shape)
