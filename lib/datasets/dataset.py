import os
import numpy as np
import pickle
import lmdb
import datetime
import torch
import PIL.Image as Image
from torch.utils import data
import torchvision.transforms as transforms

class CPCDataset(data.Dataset):
    def __init__(self, image_lmdb, boxes_pkl, shuffle, inp_size = 224, scale = 0.4, split = 'train'):
        print("loading image lmdb...")
        image_db = lmdb.open(image_lmdb, readonly=True)
        self.image_txn = image_db.begin(write = False)

        print("loading boxes data...")
        self.data = pickle.load(open(boxes_pkl, 'rb'))
        print(len(self.data), " boxes")

        self.nSamples = len(self.data)
        print("dataset size:", self.nSamples)

        self.indices = np.arange(self.nSamples)
        if shuffle:
            np.random.shuffle(self.indices)

        self.transform_train = transforms.Compose([
            transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
        ])
        self.inp_size = inp_size
        self.scale = scale
        self.split = split

    def __len__(self):
        return self.nSamples
    
    def __getitem__(self, index):
        id = self.indices[index]
        image_id = int(self.data[id]['image_id'])

        key = '{:0>10d}'.format(image_id)
        imageBin = self.image_txn.get(key.encode())
        imageBuf = np.fromstring(imageBin, dtype = np.float32)
        if self.inp_size != (-1, -1):
            imageBuf = np.reshape(imageBuf, (3, self.inp_size[0], self.inp_size[1]))
        else:
            width = self.data[id]['width']
            height = self.data[id]['height']
            imageBuf = np.reshape(imageBuf, (3, int(width * self.scale[0]), int(height * self.scale[1])))

        img = self.pytorch_transform(imageBuf.copy())
        if self.split == 'train':
            boxes = self.data[id]['boxes']
            rois = np.zeros((len(boxes), 5))
            rois[0:len(boxes), 1:5] = boxes
            score = np.array(self.data[id]['score'])

            return img, torch.from_numpy(rois).float().view(24, 5), torch.from_numpy(score).float().view(24)
        else:
            boxes = self.data[id]['gt_crop']
            width = self.data[id]['width']
            height = self.data[id]['height']
            return img, boxes, width, height
    
    def pytorch_transform(self, img):
        img = self.transform_train(torch.from_numpy(img).float())
        return img

if __name__=="__main__":
    image_lmdb = '/157Dataset/data-lu.weirui/processed_data/LVRN/test_image_224x224'
    boxes_pkl= '/157Dataset/data-lu.weirui/processed_data/LVRN/test_boxes_224x224.pkl' 
    
    kwargs = {'pin_memory': False} 
    train_loader = torch.utils.data.DataLoader(
        dataset = CPCDataset(image_lmdb, boxes_pkl, False, inp_size=(224,224), scale = (0.75, 0.75), split = 'test'),
        batch_size = 1,
        drop_last = True,
        **kwargs
    )
    for batch_idx, (img, boxes, width, height) in enumerate(train_loader):
        print(img.size())
        print(boxes)
        print(width)

        break
