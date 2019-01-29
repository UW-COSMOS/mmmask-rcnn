#!/usr/bin/env python3
"""
Script to run an end to end pipeline
"""

from argparse import ArgumentParser
import os
import subprocess
from model.utils.model2xml import model2xml
from model.utils.xml2list import xml2list
from model.utils.list2html import list2html
from tqdm import tqdm
import shutil
import torch
import yaml
from torch.utils.data import DataLoader
import utils.preprocess as pp
import multiprocessing as mp
from  model.connected_components.connected_components import write_proposals
from model.model import MMFasterRCNN
from train.data_layer.xml_loader import XMLLoader
from train.data_layer.gt_dataset import GTDataset
from utils.voc_utils import ICDAR_convert

# PDF directory path

parser = ArgumentParser(description="Run the classifier")
parser.add_argument("pdfdir", type=str, help="Path to directory of PDFs")
parser.add_argument('-w', "--weights", default=None, type=str, help="Path to weights file")

args = parser.parse_args()
# Setup directories
if not os.path.exists('tmp'):
    os.makedirs('tmp')
    os.makedirs('tmp/images')
    os.makedirs('tmp/images2')
if not os.path.exists('xml'):
    os.makedirs('xml')

img_d = os.path.join('tmp', 'images2')
def preprocess_pdfs(pdf_path):
    subprocess.run(['convert', '-density', '200', '-trim', os.path.join(args.pdfdir, pdf_path), '-quality', '100',
                    '-sharpen', '0x1.0', os.path.join('tmp','images', f'{pdf_path}-%04d.png')])

def flatten_png(img_f):
    subprocess.run(['convert', '-flatten', os.path.join('tmp', 'images', img_f), os.path.join('tmp', 'images', img_f)])

def preprocess_pngs(img_f):
    pth, padded_img = pp.pad_image(os.path.join('tmp', 'images', img_f))
    padded_img.save(os.path.join(img_d, img_f))

def convert_to_html(xml_f):
    xpath = os.path.join('xml', xml_f)
    l = xml2list(xpath)
    list2html(l, f'{xml_f[:-4]}.png', 'tmp/images', 'html')

pool = mp.Pool(processes=240)
results = [pool.apply_async(preprocess_pdfs, args=(x,)) for x in os.listdir(args.pdfdir)]
[r.get() for r in results]

results = [pool.apply_async(flatten_png, args=(x,)) for x in os.listdir(os.path.join('tmp', 'images'))]
[r.get() for r in results]

results = [pool.apply_async(write_proposals, args=(os.path.join('tmp', 'images', x),)) for x in os.listdir(os.path.join('tmp', 'images'))]
[r.get() for r in results]

results = [pool.apply_async(preprocess_pngs, args=(x,)) for x in os.listdir(os.path.join('tmp', 'images'))]
[r.get() for r in results]

with open('test.txt', 'w') as wf:
    for f in os.listdir('tmp/images'):
        wf.write(f[:-4] + '\n')

shutil.move('test.txt', 'tmp/test.txt')

loader = XMLLoader(None, img_d, 'tmp/cc_proposals', 'png')
dataset = GTDataset(loader)
loader = DataLoader(dataset, batch_size=1)
device = None
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
model_config = yaml.load(open('model_config.yaml').read())
model = MMFasterRCNN(model_config)
if args.weights is not None:
    model.load_state_dict(torch.load(args.weights))
else:
    model.set_weights(0, 0.1)
model.to(device)

if not os.path.exists('xml'):
    os.makedirs('xml')
classes = list(ICDAR_convert.keys())
for idx, batch in enumerate(tqdm(loader, desc="batches", leave=False)):
    ex, proposals, idn = batch
    ex = ex.to(device)
    rois, cls_preds, cls_scores, bbox_deltas = model(ex, device, proposals=proposals)
    N, L, C = cls_scores.shape
    # ensure center and original anchors have been precomputed
    # drop objectness score
    max_scores, score_idxs = torch.max(cls_scores, dim=2)
    # reshape bbox deltas to be [N, L x C x 4] so we can index by score_idx
    bbox_deltas = bbox_deltas.reshape(N, L, C, 4)
    # filter to only the boxes we want
    bbox_lst = []
    clss = []
    for img_idx in range(N):
        for roi_idx in range(L):
            cls_idx = score_idxs[img_idx, roi_idx]
            clss.append(cls_idx.item())
            bbox_lst.append(bbox_deltas[img_idx, roi_idx, cls_idx, :])
    bbox_deltas = torch.stack(bbox_lst)
    bbox_deltas = bbox_deltas.reshape(N, L, 4)
    pred_box = rois.double().to(device) + bbox_deltas.double().to(device)
    zipped = zip(clss, pred_box[0].int().tolist())
    
    model2xml(idn[0], 'xml', [1920, 1920], zipped, classes, max_scores[0])

    
#[convert_to_html(x) for x in os.listdir('xml')]
results = [pool.apply_async(convert_to_html, args=(x,)) for x in os.listdir('xml')]
[r.get() for r in results]
#shutil.rmtree('xml')
#shutil.rmtree('tmp')


