from __future__ import absolute_import
import _init_path
import os
import sys
import argparse
import numpy as np
import pickle
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import LVRN as net
from datasets import dataset
from utils import config as cfg

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a LVRN network')

    parser.add_argument('--GPU', dest='GPUid',      help='usage: --GPU gpu_id',     default = 0,     type = int)                      
    parser.add_argument('--bs',  dest='batch_size', help='usage: --bs  batch size', default = 25,    type = int)
    parser.add_argument('--lr',  dest='lr',         help='usage: --bs  lr',         default = 0.001, type = float)
    
    # GPUs
    parser.add_argument('--multiGPU', dest = 'set_multi_GPU', help = 'set multi GPUs', default = False, type = bool)
    parser.add_argument('--GPUs', dest = 'GPUs', help = 'multi GPUs', default = "0,1", type = str)
    args = parser.parse_args()
    return args

def adjust_learning_rate(optim, epoch):
    for param_group in optim.param_groups: 
        if(epoch in cfg.lr_decay_epoch):
            param_group['lr'] = param_group['lr'] * cfg.lr_decay
        print("epoch:", epoch, " learning rate:", param_group['lr'])

def save_model(net, optim, epoch, iteration, args):
    save_name = os.path.join(cfg.save_model_root, 'LVRN_{}_{}.pth'.format(epoch, iteration))
    torch.save({
        'model': net.module.state_dict() if args.set_multi_GPU else net.state_dict(),
        'optimizer': optim.state_dict()
        }, save_name)
    #print("save model:%s"%(save_name))

def score_rank_loss(scores, labels):
    # top-1
    # score [m, n]
    # label [m, n]
    scores = nn.LogSoftmax(dim = 1)(scores)
    labels = F.softmax(labels, 1)

    loss = 0 - (scores * labels)
    loss = torch.sum(loss, dim = 1)
    loss = loss.sum()
    return loss

def score_rank_focal_loss(scores, labels, gama):
    # top-1
    # score [m, n]
    # label [m, n]
    Pz = F.softmax(scores, 1)
    Py = F.softmax(labels, 1)
    Pz = torch.pow(Py - Pz, gama) * torch.log(Pz)


    loss = 0 - (Py * Pz)
    loss = torch.sum(loss, dim = 1)
    loss = loss.mean()
    return loss

def evaluate(model, test_loader, pdefined_anchors, writer):
    def overlap_ratio(x1, y1, w1, h1, x2, y2, w2, h2):
        intersection = max(0, min(x1 + w1, x2 + w2) - max(x1, x2)) * max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
        union = (w1 * h1) + (w2 * h2) - intersection
        return float(intersection) / float(union)
    aver_ratio = 0.0
    model.eval()

    rois = np.zeros((len(pdefined_anchors), 5))
    rois[:, 0] = 0
    rois[:, 1] = pdefined_anchors[:, 0] * cfg.inp_size[0]
    rois[:, 2] = pdefined_anchors[:, 1] * cfg.inp_size[1]
    rois[:, 3] = pdefined_anchors[:, 2] * cfg.inp_size[0]
    rois[:, 4] = pdefined_anchors[:, 3] * cfg.inp_size[1]
    rois = torch.from_numpy(rois).float().cuda()
    for _, (img, boxes, width, height) in enumerate(test_loader):
        if cfg.fixed_size:
            img = img.view(1, 3, cfg.inp_size[0], cfg.inp_size[1]).cuda()
        else:
            pass
        scores = model(img, rois)
        scores = scores.cpu().detach().numpy()
        idx = np.argmax(scores[:len(scores)])
        best_x = int(pdefined_anchors[idx][0] * width)
        best_y = int(pdefined_anchors[idx][1] * height)
        best_w = int(pdefined_anchors[idx][2] * width) - best_x
        best_h = int(pdefined_anchors[idx][3] * height) - best_y

        boxes = boxes.view(4).cpu().detach().numpy()
        ratio = overlap_ratio(boxes[0], boxes[1], boxes[2], boxes[3], best_x, best_y, best_w, best_h)
        aver_ratio += ratio
    
    aver_ratio /= len(test_loader)
    model.train()
    return aver_ratio
        
if __name__=="__main__":
    
    args = parse_args()
    print('Called with args:')
    print(args)

    #1. dataset
    if cfg.fixed_size:
        train_image_dst = os.path.join(cfg.save_data_root, 'train_image_' + str(cfg.inp_size[0]) + "x" + str(cfg.inp_size[1]))
        train_boxes_dst = os.path.join(cfg.save_data_root, 'train_boxes_' + str(cfg.inp_size[0]) + "x" + str(cfg.inp_size[1]))
        test_image_dst = os.path.join(cfg.save_data_root, 'test_image_' + str(cfg.inp_size[0]) + "x" + str(cfg.inp_size[1]))
        test_boxes_dst = os.path.join(cfg.save_data_root, 'test_boxes_' + str(cfg.inp_size[0]) + "x" + str(cfg.inp_size[1]))
        if cfg.padding:
            train_image_dst = train_image_dst + '_padding'
            train_boxes_dst = train_boxes_dst + '_padding'
            test_image_dst  = test_image_dst + '_padding'
            test_boxes_dst  = test_boxes_dst + '_padding'
        train_boxes_dst = train_boxes_dst + ".pkl"
        test_boxes_dst  = test_boxes_dst + ".pkl"
    else:
        train_image_dst = os.path.join(cfg.save_data_root, 'train_image_' + str(cfg.scale))
        train_boxes_dst = os.path.join(cfg.save_data_root, 'train_boxes_' + str(cfg.scale) + ".pkl")
        test_image_dst  = os.path.join(cfg.save_data_root, 'test_image_' + str(cfg.scale))
        test_boxes_dst  = os.path.join(cfg.save_data_root, 'test_boxes_' + str(cfg.scale) + ".pkl")
    
    kwargs = {'num_workers': 4, 'pin_memory': True} 
    train_loader = torch.utils.data.DataLoader(
        dataset = dataset.CPCDataset(train_image_dst, train_boxes_dst, cfg.is_shuffle, cfg.inp_size,  cfg.scale, 'train'),
        batch_size = args.batch_size,
        drop_last = False,
        **kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        dataset = dataset.CPCDataset(test_image_dst, test_boxes_dst, False,  cfg.inp_size, cfg.scale, 'test'),
        batch_size = 1,
        drop_last = False,
        **kwargs
    )
    print ("train dataset size %s%s" % (train_image_dst, len(train_loader)))
    print ("test dataset size %s%s" % (test_image_dst, len(test_loader)))

    #2. model
    model = net.LVRN(cfg.vgg_model_path)
    if args.set_multi_GPU:
        model = nn.DataParallel(model)
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.GPUid)  
    model = model.cuda()

    #3. optim SGD
    optim = torch.optim.SGD(
        model.parameters(),
        weight_decay = cfg.weight_decay,
        lr = args.lr,
        momentum = cfg.momentum
    )
    #4. pre-defined anchors(x1, y1, x2, y2)
    pdefined_anchors = np.array(pickle.load(open(cfg.pkl_file, 'rb'), encoding='iso-8859-1'))
    print ('num of pre-defined anchors: ', len(pdefined_anchors))

    #5. training
    writer = SummaryWriter()
    for epoch in range(0, cfg.max_epoch):
        adjust_learning_rate(optim, epoch)
        # train epoch
        model.train()
        epoch_loss = 0
        for batch_idx, (imgs, rois, labels) in enumerate(train_loader):
            iteration = batch_idx + epoch * len(train_loader)

            if iteration % cfg.interval_display == 0 and iteration != 0 and batch_idx != 0:
                print("iteration:", iteration, "train loss:", epoch_loss / batch_idx)
                ratio = evaluate(model, test_loader, pdefined_anchors, writer)
                writer.add_scalar('data/test_ratio', ratio, iteration)
                save_model(model, optim, epoch, batch_idx, args)     

            ############################# forward #################################
            # to cuda
            imgs = imgs.cuda()         #[m, 3, inp_size, inp_size]
            rois = rois.cuda()         #[m, n, 5]
            labels = labels.cuda()     #[m, n, 1]
            n = labels.size(1)
            m = labels.size(0)

            assert model.training
            with torch.enable_grad():          
                optim.zero_grad()
                # forward(only batch size 1)
                scores = torch.zeros((m, n)).float().cuda()
                for i in range(m):
                    if cfg.fixed_size:
                        img = imgs[i].view(1, 3, cfg.inp_size[0], cfg.inp_size[1])
                    else:
                        pass
                    scores[i] = (model(img, rois[i])).view(1, n)

                # score loss
                scores = scores.view(m, -1)
                loss = score_rank_loss(scores, labels)
                loss.backward()
                optim.step()
            ############################# tensorboardX train-loss #############################
            epoch_loss += loss.item()
            if iteration:
                writer.add_scalar('data/train_loss', loss.item(), iteration)
        
        epoch_loss /= len(train_loader)
        print("epoch:", epoch," loss:",epoch_loss)