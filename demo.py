from __future__ import absolute_import
import _init_path

import os
import json
import numpy as np
import pickle as pkl
import datetime
from PIL import Image, ImageDraw
import argparse

from model import LVRN as net

import torch
import torch.nn as nn
import torchvision.transforms as transforms

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='testing')

    parser.add_argument('--GPU', dest='GPUid',
                        help='usage: --GPU 0',
                        default=0, type=int)                      

    args = parser.parse_args()
    return args

def load_model(model_path):
    model = net.LVRN()
    model.load_pretrain_parameters(model_path)
    return model

def overlap_ratio(x1, y1, w1, h1, x2, y2, w2, h2):
    intersection = max(0, min(x1 + w1, x2 + w2) - max(x1, x2)) * max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
    union = (w1 * h1) + (w2 * h2) - intersection
    return float(intersection) / float(union)

def pytorch_transform(img):
    img = np.array(img, dtype = np.uint8)
    if img.ndim == 2:
        img = np.expand_dims(img, axis=-1)
        img = np.repeat(img, 3, 2)
    img = img.transpose((2, 0, 1))
    img = img / 255.0
    transform_test = transforms.Compose([
        transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
    ])
    img = transform_test(torch.from_numpy(img)).float()
    return img

def evaluate_single_image(img, pdefined_anchors, model):
    # get eval img 
    img = img.resize((224,224), Image.BILINEAR)
    img = pytorch_transform(img)
    img = img.view(1, 3, 224, 224).cuda()

    # get eval rois
    rois = np.zeros((len(pdefined_anchors), 5))
    rois[:, 0] = 0
    rois[:, 1] = pdefined_anchors[:, 0] * 224 
    rois[:, 2] = pdefined_anchors[:, 1] * 224
    rois[:, 3] = pdefined_anchors[:, 2] * 224 
    rois[:, 4] = pdefined_anchors[:, 3] * 224 
    rois = rois.astype(np.int)
    rois = torch.from_numpy(rois).float().cuda()
    
    # forward
    scores = model(img, rois)

    # get best crop
    scores = scores.cpu().detach().numpy()
    idx = np.argmax(scores[:len(scores)])
    return pdefined_anchors[idx]

if __name__=="__main__":
    args = parse_args()
    print('Called with args:')
    print(args)
    
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.GPUid)  

    root = './images'

    # get model
    model_path = 'model/model_parameters.pth'
    print ("model:", model_path)
    model = load_model(model_path)
    model = model.cuda()
    model.eval()

    # get predefined boxes(x1, y1, x2, y2)
    pkl_file = 'pdefined_anchors.pkl'
    pdefined_anchors = np.array(pkl.load(open(pkl_file, 'rb'), encoding='iso-8859-1'))
    print ('num of pre-defined anchors: ', len(pdefined_anchors))

    for img_file in os.listdir(root):
        img_file = os.path.join(root, img_file)
        print('input : %s'%(img_file))
        img = Image.open(img_file)
        width = img.width
        height = img.height
        best_anchor = evaluate_single_image(img, pdefined_anchors, model)
        best_x = int(best_anchor[0] * width)
        best_y = int(best_anchor[1] * height)
        best_w = int(best_anchor[2] * width) - best_x
        best_h = int(best_anchor[3] * height) - best_y
        draw = ImageDraw.Draw(img)
        draw.rectangle((best_x, best_y, best_x + best_w, best_y + best_h), outline='red')    #draw highest score box
        img.save(img_file)
        print('output: %s'%(img_file))
