"""
An XML and image repo loader
meant to work with the GTDataset class
Author: Josh McGrath
"""
import os


class XMLLoader:
    """
    Loads examples and ground truth from given directories
    it is expected that the directories have no files
    other than annotations
    """
    def __init__(self, xml_dir, img_dir, img_type):
        """
        Initialize a XML loader object
        :param xml_dir: directory to get XML from
        :param img_dir: directory to load PNGs from
        :param img_type: the image format to load in
        """
        self.xml_dir = xml_dir
        self.img_dir = img_dir
        self.img_type = img_type
        # TODO match these so we can index them jointly
        self.annnotations = os.listdir(xml_dir)
        self.images = os.listdir(img_dir)
        self.num_images = len(self.annotations)

    def size(self):
        return self.num_images

