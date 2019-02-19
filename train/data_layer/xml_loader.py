"""
An XML and image repo loader
meant to work with the GTDataset class
Author: Josh McGrath
"""
from torch.utils.data import Dataset
import os
from os.path import splitext
from PIL import Image
from torchvision.transforms import ToTensor
from numpy import genfromtxt
import redis
import torch
from xml.etree import ElementTree as ET
from .transforms import NormalizeWrapper
import pickle
from utils.matcher import match
from collections import namedtuple
from uuid import uuid4
from tqdm import tqdm
normalizer = NormalizeWrapper(mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])

tens = ToTensor()
Example = namedtuple('Example', ["ex_window", "ex_proposal", "gt_cls", "gt_box"])

def mapper(obj, preprocessor=None):
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
    y = y1 + h/2
    cls = obj.find("name").text
    if preprocessor is not None:
        cls = preprocessor[cls]
    return cls, (x, y, w, h)


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
    pil = Image.open(path)
    return pil

def load_proposal(base_path, identifier):
    """
    Load a set of proposals into memory

    """
    path = os.path.join(base_path, f"{identifier}.csv")
    np_arr = genfromtxt(path, delimiter=",")
    bbox_absolute = torch.from_numpy(np_arr).reshape(-1,4)
    return bbox_absolute

def load_gt(xml_dir, identifier):
    """
    Load an XML ground truth document
    :param xml_dir: base path to xml
    :param identifier: xml document identifier
    :return: [K x 4] Tensor, [cls_names]
    """
    path = os.path.join(xml_dir, f"{identifier}.xml")
    as_lst = xml2list(path)
    if len(as_lst) == 0:
        cls_list = [0]
        tensor_list = [[0,0,0,0]]
    else:
        cls_list, tensor_list = zip(*as_lst)
    # convert to tensors
    gt_boxes = torch.tensor(tensor_list)
    return gt_boxes, cls_list

class XMLLoader(Dataset):
    """
    Loads examples and ground truth from given directories
    it is expected that the directories have no files
    other than annotations
    """


    def __init__(self, img_dir,xml_dir=None, proposal_dir=None,warped_size=300, img_type="jpg", host="redis"):
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
        self.warped_size = warped_size
        self.imgs = os.listdir(img_dir)
        self.identifiers = [splitext(img)[0] for img in self.imgs]
        self.uuids = []
        self.num_images = len(self.imgs)
        print(f"Constructed a {self.num_images} image dataset, ingesting to redis server")
        self.db = redis.Redis(host=host)
        self.class_stats = {}

        self._ingest()

    def __len__(self):
        return len(self.uuids)

    def __getitem__(self, item):
        bytes_rep = self.db.get(self.uuids[item])
        lst = pickle.loads(bytes_rep)
        return lst

    def _ingest(self):
        for identifier in tqdm(self.identifiers):
            img = load_image(self.img_dir, identifier, self.img_type)
            gt = None
            proposals = None
            if self.xml_dir is not None:
                gt = load_gt(self.xml_dir, identifier)
            if self.proposal_dir is not None:
                proposals = load_proposal(self.proposal_dir, identifier)
            ret = [img, gt, proposals, identifier]
            pts = self._unpack_page(ret)
            for pt in pts:
                uuid = str(uuid4())
                self.uuids.append(uuid)
                label = pt.gt_cls
                if label in self.class_stats:
                    self.class_stats[label] +=1
                else:
                    self.class_stats[label] = 1
                obj = pickle.dumps(pt)
                self.db.set(uuid, obj)



    def _unpack_page(self, page):
        img, gt, proposals, identifier = page
        gt_boxes, gt_cls = gt
        matches = match(proposals.float(),gt_boxes)
        labels = [gt_cls[match] for match in matches]
        windows = []
        proposals_lst = proposals.tolist()
        gt_box_lst = gt_boxes.tolist()
        for proposal in proposals_lst:
            img_sub = img.crop(proposal)
            img_sub = img_sub.resize((self.warped_size, self.warped_size))
            img_data = tens(img_sub)
            img_data = normalizer(img_data)
            windows.append(img_data)
        #switch to list of tensors
        proposals_lst = [torch.tensor(prop) for prop in proposals_lst]
        gt_box_lst = [torch.tensor(gt_box) for gt_box in gt_box_lst]
        collected = list(zip(windows,proposals_lst, labels, gt_box_lst))
        ret = [Example(*pt) for pt in collected]
        return ret



