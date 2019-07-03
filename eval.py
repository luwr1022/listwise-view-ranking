import os
import json
import numpy as np
import pickle as pkl
import datetime
from PIL import Image, ImageDraw
import argparse

import net

import torch
import torch.nn as nn
import torchvision.transforms as transforms

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='testing')

    parser.add_argument('--GPU', dest='GPU',
                        help='usage: --GPU 0',
                        default=0, type=int)                      

    args = parser.parse_args()
    return args

def load_model(model_path):
    model = net.LVRN()
    model.load_parameters(model_path)
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

def evaluate_single_image(img_file_name, pdefined_anchors, model):
    img = Image.open(img_file_name)
    width = img.width
    height = img.height

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
    best_x = int(pdefined_anchors[idx][0] * width)
    best_y = int(pdefined_anchors[idx][1] * height)
    best_w = int(pdefined_anchors[idx][2] * width) - best_x
    best_h = int(pdefined_anchors[idx][3] * height) - best_y
    return best_x, best_y, best_w, best_h, width, height

def evaluate_FCDB(datadir, cropdir, pdefined_anchors, model):
    print('-------------------- evaluate FCDB --------------------')
    crops_string = open(cropdir, 'r').read()
    crops = json.loads(crops_string)

    # offset
    cnt = 0
    accum_boundary_displacement = 0
    accum_overlap_ratio = 0
    accum_time = 0

    for item in crops:
        crop = item['crop']
        img_file_name = os.path.join(datadir, item['image_file_name'])
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
        accum_overlap_ratio += ratio
        cnt += 1
        #print(x, y, w, h, "|", best_x, best_y, best_w, best_h, "|", ratio)

    print ('Average overlap ratio: {:.4f}'.format(accum_overlap_ratio / cnt))
    print ('Average boundary displacement: {:.4f}'.format(accum_boundary_displacement / (cnt * 4.0)))
    print ('Total image evaluated:', cnt)
    print ('Average time per image:', float(accum_time) / cnt)

def evaluate_FLMS(datadir, cropdir, pdefined_anchors, model):
    print('-------------------- evaluate FLMS --------------------')
    crop_string = open(cropdir, 'r').read()
    crops = json.loads(crop_string)

    cnt = 0
    accum_boundary_displacement = 0
    accum_overlap_ratio = 0

    for item in crops:
        crop = item['crop']
        img_file_name = os.path.join(datadir, item['image_file_name'])
        if(not os.path.exists(img_file_name)):
            continue

        best_ratio = 0
        best_disp = 0
        # get best box
        best_x, best_y, best_w, best_h, width, height = evaluate_single_image(img_file_name, pdefined_anchors, model)
        for i in range(10):
            # ground truth
            x = crop[i][0]
            y = crop[i][1]
            w = crop[i][2]
            h = crop[i][3]

            # some error in ground truth
            if (x == -1):
                continue
            if (x + w > width or y + h > height):
                continue

            boundary_displacement = (abs(best_x - x) + abs(best_x + best_w - x - w))/float(width) + (abs(best_y - y) + abs(best_y + best_h - y - h))/float(height)
            center = abs(best_x + float(best_w)/2 - x - float(w)/2)/float(width) + abs(best_y + float(best_h)/2 - y - float(h)/2)/float(height)
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
        cnt += 1

    print ('Average overlap ratio: {:.4f}'.format(accum_overlap_ratio / cnt))
    print ('Average boundary displacement: {:.4f}'.format(accum_boundary_displacement / (cnt * 4.0)))

if __name__=="__main__":
    args = parse_args()
    print('Called with args:')
    print(args)
    
    FCDB_datadir = 'data/FCDB/flickr-cropping-dataset/data'
    FCDB_cropdir = 'data/FCDB/flickr-cropping-dataset/gt_crop.json'
    
    FLMS_datadir = 'data/FLMS/image'
    FLMS_cropdir = 'data/FLMS/gt_crop.json'

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

    evaluate_FCDB(FCDB_datadir, FCDB_cropdir, pdefined_anchors, model)
    evaluate_FLMS(FLMS_datadir, FLMS_cropdir, pdefined_anchors, model)
