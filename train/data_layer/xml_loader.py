"""
An XML and image repo loader
meant to work with the GTDataset class
Author: Josh McGrath
"""
import os
from os.path import splitext
from skimage import io
from numpy import genfromtxt
import torch
from xml.etree import ElementTree as ET


def mapper(obj):
    """
    map a single object to the list structure
    :param obj: an Etree node
    :return: (type, (x1, y1, x2, y2))
    """
    bnd = obj.find("bndbox")
    coords = ["xmin", "ymin", "xmax", "ymax"]
    pts = [int(float(bnd.find(coord).text)) for coord in coords]
    x1, y1, x2, y2 = pts
    w = x2 - x1
    h = y2 - y1
    # get centers
    x = x1 + w/2
    y = y1 + h/w
    return obj.find("name").text, (x, y, w, h)


def xml2list(fp):
    """
    convert VOC XML to a list
    :param fp: file path to VOC XML file
    :return: [(type,(x1, y1, x2, y2))]
    """
    tree = ET.parse(fp)
    root = tree.getroot()
    objects = root.findall("object")
    lst = [mapper(obj) for obj in objects]
    lst.sort(key=lambda x: x[1])
    return lst


def load_image(base_path, identifier, img_type):
    """
    load an image into memory
    :param base_path: base path to image
    :param identifier:
    :param img_type:
    :return: [3 x SIZE x SIZE] tensor
    """
    path = os.path.join(base_path, f"{identifier}.{img_type}")
    # reads in as [SIZE x SIZE x 3]
    img_data = io.imread(path)
    img_data = img_data.transpose((2, 0, 1))
    img_data = torch.from_numpy(img_data).float()
    # squash values into [0,1]
    img_data = img_data / 255.0
    return img_data

def load_proposal(base_path, identifier):
	"""
	Load a set of proposals into memory
	
	"""
	path = os.path.join(base_path, f"{identifier}.csv")
	np_arr = genfromtxt(path, delimiter=",")
	return torch.from_numpy(np_arr)

def load_gt(xml_dir, identifier):
    """
    Load an XML ground truth document
    :param xml_dir: base path to xml
    :param identifier: xml document identifier
    :return: [K x 4] Tensor, [cls_names]
    """
    path = os.path.join(xml_dir, f"{identifier}.xml")
    as_lst = xml2list(path)
    cls_list, tensor_list = zip(*as_lst)
    # convert to tensors
    gt_boxes = torch.tensor(tensor_list)
    return gt_boxes, cls_list

class XMLLoader:
    """
    Loads examples and ground truth from given directories
    it is expected that the directories have no files
    other than annotations
    """
    def __init__(self, xml_dir, img_dir,proposal_dir, img_type):
        """
        Initialize a XML loader object
        :param xml_dir: directory to get XML from
        :param img_dir: directory to load PNGs from
        :param img_type: the image format to load in
        """
        self.xml_dir = xml_dir
        self.img_dir = img_dir
        self.proposal_dir = proposal_dir
        self.img_type = img_type
        self.imgs = os.listdir(img_dir)
        self.identifiers = [splitext(img)[0] for img in self.imgs]
        self.num_images = len(self.imgs)
        print(f"Constructed a {self.num_images} image dataset")

    def size(self):
        return self.num_images

    def __getitem__(self, item):
        identifier = self.identifiers[item]
        img = load_image(self.img_dir, identifier, self.img_type)
        gt = None
        if self.xml_dir is not None:
            gt = load_gt(self.xml_dir, identifier)
        proposals = load_proposal(self.proposal_dir, identifier)
        if gt is not None:
            return img, gt, proposals, identifier
        return img, proposals, identifier
