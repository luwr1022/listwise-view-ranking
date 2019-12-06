
import os
from PIL import Image
import numpy as np
import config as cfg
import shutil

def image_transform_fixed_size(image_file_name):
    '''
    [width, height] -> [oup_size, oup_size]
    [0, 255] -> [0, 1]
    (H ,W ,C) -> (C, H, W)
    3 channels
    np.uint8

    params:
        image_file_name: file name of input image
        oup_size       : output size
    return:
        img            : output data
        width          : width of original image 
        height         : height of original image
    '''
    img = Image.open(image_file_name)
    width = img.width
    height = img.height
    img = img.resize((cfg.inp_size[0], cfg.inp_size[1]), Image.BILINEAR)
    img = np.array(img, dtype = np.uint8)
    if img.ndim == 2:
        img = np.expand_dims(img, axis = -1)
        img = np.repeat(img, 3, 2)
        
    img = img.transpose((2, 0, 1))
    img = img.astype(np.float32) / 255.0
    return img, width, height

def image_transform_scale(image_file_name):
    '''
    [width, height] -> [oup_size, oup_size]
    [0, 255] -> [0, 1]
    (H ,W ,C) -> (C, H, W)
    3 channels
    np.uint8

    params:
        image_file_name: file name of input image
        oup_size       : output size
    return:
        img            : output data
        width          : width of original image 
        height         : height of original image
    '''
    img = Image.open(image_file_name)
    width = img.width
    height = img.height
    img = img.resize((int(width * cfg.scale[0]), int(height * cfg.scale[1])), Image.BILINEAR)
    img = np.array(img, dtype = np.uint8)
    if img.ndim == 2:
        img = np.expand_dims(img, axis = -1)
        img = np.repeat(img, 3, 2)
        
    img = img.transpose((2, 0, 1))
    img = img.astype(np.float32) / 255.0
    return img, width, height

def del_and_create(dname):
    if os.path.exists(dname):
        shutil.rmtree(dname)
        os.makedirs(dname)
    else:
        os.makedirs(dname)