import numpy as np
import os 

fixed_size = True
padding = False
############################################### other ###############################################

if fixed_size:
    # for vgg
    inp_size     = [224, 224]
    feat_size    = [32, 32]
    oup_size     = [7, 7]
    scale_factor = [2, 2]
    scale        = [0, 0]
else:
    inp_size     = [-1, -1]
    scale        = [0.6, 0.6]
    feat_size    = [32, 32]
    oup_size     = [7, 7]
    scale_factor = [2, 2]


############################################### path configure ###############################################
CPC  = '...'
FCDB = '../../data/FCDB'
FLMS = '../../data/FLMS'
save_data_root = './'
save_model_root = './'

# model: pytroch-pretrain-model
# set pretrained = True, for example: model = vgg16(pretrained = True), the pytorch code download pytroch-pretrain-model automatic.
# In my code, I set it False, and download in advance.
vgg_model_path    = ''
resnet50_model_path = ''
alexnet_model_path = ''
squeezenet1_0_model_path = ''

# apre-defined anchors
pkl_file = './pdefined_anchor1745.pkl'

# FCDB
FCDB_datadir = os.path.join(FCDB, 'flickr-cropping-dataset/data')
FCDB_cropdir = os.path.join(FCDB, 'flickr-cropping-dataset/gt_crop.json')

FLMS_datadir = os.path.join(FLMS, 'image')
FLMS_cropdir = os.path.join(FLMS, 'gt_crop.json')


# training
max_epoch = 18
momentum = 0.9
weight_decay =  0.0005
interval_display = 100
is_shuffle = True
lr_decay = 0.1
lr_decay_epoch = {6, 12}