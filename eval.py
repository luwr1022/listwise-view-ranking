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
from datasets import dataset
from utils import config as cfg

import torch
import torch.nn as nn
import torchvision.transforms as transforms

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='evaluate a LVRN-attention network')

    parser.add_argument('--GPU', dest='GPUid',      help='usage: --GPU gpu_id',     default = 0,     type = int)                       

    parser.add_argument('--checkpoint', dest = 'checkpoint', help = 'checkpoint to load model', default = 0, type = int)
    parser.add_argument('--checkepoch', dest = 'checkepoch', help = 'checkepoch to load model', default = 1, type = int)

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
    # 3-channels
    # C, H, W
    # [0, 1]
    # torch.float
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

def anchor2roi(pdefined_anchors):
    # params:
    #        pdefined_anchors : pre-defined anchors, ndarray, (x1, y1, x2, y2)
    #        ratio_w          : int
    #        ratio_h          : int
    # return:
    #        rois             : rois for evaluate, torch.tensor().float(), (batch_idx, x1, y1, x2, y2)
    #                           batch_idx = 0
    rois = np.zeros((len(pdefined_anchors), 5))
    rois[:, 0] = 0
    rois[:, 1] = pdefined_anchors[:, 0] * cfg.inp_size[0] 
    rois[:, 2] = pdefined_anchors[:, 1] * cfg.inp_size[1] 
    rois[:, 3] = pdefined_anchors[:, 2] * cfg.inp_size[0]  
    rois[:, 4] = pdefined_anchors[:, 3] * cfg.inp_size[1]  
    rois = rois.astype(np.int)
    return torch.from_numpy(rois).float()

def evaluate_single_image(img_file_name, pdefined_anchors, model):
    # params:
    #       img_file_name    : image's name, string
    #       pdefined_anchors : pre-defined anchors, ndarray, (x1, y1, x2, y2)
    #       model            : model, torch.nn.module
    #       image_size       : image's resize size ,int
    img = Image.open(img_file_name)
    width = img.width
    height = img.height
    # get eval img 
    img = img.resize((cfg.inp_size[0],cfg.inp_size[1]), Image.BILINEAR)
    img = pytorch_transform(img)
    img = img.cuda()
    img = img.view(1, 3, cfg.inp_size[0], cfg.inp_size[1])

    # get eval rois
    roi = anchor2roi(pdefined_anchors)
    roi = roi.cuda()
    
    # forward
    scores = model(img, roi)

    # get best crop
    scores = scores.cpu().detach().numpy()
    idx = np.argmax(scores[:len(scores)])
    best_x = int(pdefined_anchors[idx][0] * width)
    best_y = int(pdefined_anchors[idx][1] * height)
    best_w = int(pdefined_anchors[idx][2] * width) - best_x
    best_h = int(pdefined_anchors[idx][3] * height) - best_y
    return best_x, best_y, best_w, best_h, width, height  

def evaluate_FCDB(pdefined_anchors, model):
    print('-------------------- evaluate FCDB --------------------')
    crops_string = open(cfg.FCDB_cropdir, 'r').read()
    crops = json.loads(crops_string)

    # offset
    cnt = 0
    alpha = 0.75
    alpha_cnt = 0
    accum_boundary_displacement = 0
    accum_overlap_ratio = 0
    accum_time = 0
    crop_cnt = 0

    for item in crops:
        crop = item['crop']
        img_file_name = os.path.join(cfg.FCDB_datadir, item['image_file_name'])
        if(not os.path.exists(img_file_name)):
            print("no ",item['image_file_name'])
            continue

        start = datetime.datetime.now()

        # ground truth
        x = crop[0]
        y = crop[1]
        w = crop[2]
        h = crop[3]

        # get best box
        best_x, best_y, best_w, best_h, width, height = evaluate_single_image(img_file_name, pdefined_anchors, model)

        end = datetime.datetime.now()
        accum_time += ((end - start).seconds + (end - start).microseconds / 10e6)

        boundary_displacement = (abs(best_x - x) + abs(best_x + best_w - x - w))/float(width) + (abs(best_y - y) + abs(best_y + best_h - y - h))/float(height)
        accum_boundary_displacement += boundary_displacement
        ratio = overlap_ratio(x, y, w, h, best_x, best_y, best_w, best_h)
        if ratio >= alpha:
            alpha_cnt += 1
        accum_overlap_ratio += ratio
        cnt += 1
        crop_cnt += len(pdefined_anchors)
        #print(x, y, w, h, "|", best_x, best_y, best_w, best_h, "|", ratio)

    print ('Average overlap ratio: {:.4f}'.format(accum_overlap_ratio / cnt))
    print ('Average boundary displacement: {:.4f}'.format(accum_boundary_displacement / (cnt * 4.0)))
    print ('Alpha recall: {:.4f}'.format(100 * float(alpha_cnt) / cnt))
    print ('Total image evaluated:', cnt)
    print ('Average crops per image:', float(crop_cnt) / cnt)
    print ('Average time per image:', float(accum_time) / cnt)

def evaluate_ICDB(pdefined_anchors, model):
    print('-------------------- evaluate ICDB --------------------')
    crops_string = open(cfg.ICDB_cropdir, 'r').read()
    crops = json.loads(crops_string)

    # offset
    cnt = [0, 0, 0]
    alpha = 0.75
    alpha_cnt = [0, 0, 0]
    accum_boundary_displacement = [0, 0, 0]
    accum_overlap_ratio = [0, 0, 0]
    crop_cnt = [0, 0, 0]

    best_accum_boundary_displacement = 0
    best_accum_center_displacement = 0
    best_accum_overlap_ratio = 0
    for item in crops:
        crop = item['crop']
        img_filename = os.path.join(cfg.ICDB_datadir, item['image_file_name'])
        if(not os.path.exists(img_filename)):
            continue

        best_ratio = 0
        best_disp = 0
        best_cen = 0
        # get best box
        best_x, best_y, best_w, best_h, width, height = evaluate_single_image(img_filename, pdefined_anchors, model)

        for i in range(3):
            # ground truth
            x = crop[i][0]
            y = crop[i][1]
            w = crop[i][2]
            h = crop[i][3]

            boundary_displacement = (abs(best_x - x) + abs(best_x + best_w - x - w))/float(width) + (abs(best_y - y) + abs(best_y + best_h - y - h))/float(height)
            ratio = overlap_ratio(x, y, w, h, best_x, best_y, best_w, best_h)
            accum_boundary_displacement[i] += boundary_displacement
            accum_overlap_ratio[i] += ratio
            cnt[i] += 1
            crop_cnt[i] += len(pdefined_anchors)
            if(ratio > best_ratio):
                best_ratio = ratio
                best_disp = boundary_displacement
            elif(ratio == best_ratio and boundary_displacement > best_disp):
                best_ratio = ratio
                best_disp = boundary_displacement
        # top - 1
        best_accum_boundary_displacement += best_disp
        best_accum_overlap_ratio += best_ratio

    for i in range(3):
        print ('/////////////////////////////////////// annotation '+str(i)+'///////////////////////////////////////////')
        print ('Average overlap ratio: {:.4f}'.format(accum_overlap_ratio[i] / cnt[i]))
        print ('Average boundary displacement: {:.4f}'.format(accum_boundary_displacement[i] / (cnt[i] * 4.0)))
        print ('Alpha recall: {:.4f}'.format(100 * float(alpha_cnt[i]) / cnt[i]))
        print ('Total image evaluated:', cnt[i])
        print ('Average crops per image:', float(crop_cnt[i]) / cnt[i])

    print ('top-1 result:')
    print ('Average best overlap ratio: {:.4f}'.format(best_accum_overlap_ratio / cnt[0]))
    print ('Average best boundary displacement: {:.4f}'.format(best_accum_boundary_displacement / (cnt[0] * 4.0)))

def evaluate_FLMS(pdefined_anchors, model):
    print('-------------------- evaluate FLMS --------------------')
    crop_string = open(cfg.FLMS_cropdir, 'r').read()
    crops = json.loads(crop_string)

    cnt = 0
    alpha = 0.75
    alpha_cnt = 0
    accum_boundary_displacement = 0
    accum_overlap_ratio = 0
    crop_cnt = 0

    for item in crops:
        crop = item['crop']
        img_file_name = os.path.join(cfg.FLMS_datadir, item['image_file_name'])
        if(not os.path.exists(img_file_name)):
            continue

        best_ratio = 0
        best_disp = 0
        best_cen = 0
        # get best box
        best_x, best_y, best_w, best_h, width, height = evaluate_single_image(img_file_name, pdefined_anchors, model)
        for i in range(10):
            # ground truth
            x = crop[i][0]
            y = crop[i][1]
            w = crop[i][2]
            h = crop[i][3]

            # there are some error in ground truth
            if (x == -1):
                continue
            if (x + w > width or y + h > height):
                continue

            boundary_displacement = (abs(best_x - x) + abs(best_x + best_w - x - w))/float(width) + (abs(best_y - y) + abs(best_y + best_h - y - h))/float(height)
            ratio = overlap_ratio(x, y, w, h, best_x, best_y, best_w, best_h)
            if(ratio > best_ratio):
                best_ratio = ratio
                best_disp = boundary_displacement
            elif(ratio == best_ratio and boundary_displacement > best_disp):
                best_ratio = ratio
                best_disp = boundary_displacement

        # top - 1
        accum_boundary_displacement += best_disp
        accum_overlap_ratio += best_ratio
        if(best_ratio >= alpha):
            alpha_cnt += 1
        cnt += 1
        crop_cnt += len(pdefined_anchors)

    print ('Average overlap ratio: {:.4f}'.format(accum_overlap_ratio / cnt))
    print ('Average boundary displacement: {:.4f}'.format(accum_boundary_displacement / (cnt * 4.0)))
    print ('Alpha recall: {:.4f}'.format(100 * float(alpha_cnt) / cnt))
    print ('Average crops per image:', float(crop_cnt) / cnt)

if __name__=="__main__":
    args = parse_args()
    print('Called with args:')
    print(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.GPUid)  

    # get model
    model_path = 'model/model_parameters.pth'
    print ("model:", model_path)
    model = load_model(model_path)
    model = model.cuda()
    model.eval()

    # get predefined boxes(x1, y1, x2, y2)
    pdefined_anchors = np.array(pkl.load(open(cfg.pkl_file, 'rb'), encoding='iso-8859-1'))
    print ('num of pre-defined anchors: ', len(pdefined_anchors))

    evaluate_FCDB(pdefined_anchors, model)
    #evaluate_ICDB(pdefined_anchors, model)
    evaluate_FLMS(pdefined_anchors, model)
