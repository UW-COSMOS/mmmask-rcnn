from model.model import MMFasterRCNN
from model.utils.config_manager import ConfigManager
from train.data_layer.xml_loader import XMLLoader
from train.train import TrainerHelper
import torch
import yaml
import unittest


class TestTrainingIntegration(unittest.TestCase):
    def setUp(self):
        self.cfg = ConfigManager("../data/model_config.yaml")
        self.model = MMFasterRCNN(self.cfg)
        self.loader = XMLLoader("../data/images", "../data/annotations",
                                "../data/proposals")
        self.device = torch.device("cpu")
        self.params = yaml.load(open("../data/train_config.yaml"))

    def test_init(self):
        trainer = TrainerHelper(self.model,
                                self.loader,
                                self.params,
                                self.device)

        self.assertIsNotNone(trainer)

    def test_runs(self):
        trainer = TrainerHelper(self.model,
                                self.loader,
                                self.params,
                                self.device)
        trainer.train()


