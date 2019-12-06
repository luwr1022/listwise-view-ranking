import os
import json
import pickle
import numpy as np
import random
from PIL import Image
import lmdb
import config as cfg
import tools
import annToView

def create_train_lmdb():
    dataset_dir = os.path.join(cfg.CPC, 'images')
    annotation_dir = os.path.join(cfg.CPC, 'CollectedAnnotationsRaw')

    rank_score = np.linspace(24, 1, 24)
    rank_score = rank_score.astype(np.float)
    if cfg.fixed_size:
        image_dst = os.path.join(cfg.save_data_root, 'train_image_' + str(cfg.inp_size[0]) + "x" + str(cfg.inp_size[1]))
        boxes_dst = os.path.join(cfg.save_data_root, 'train_boxes_' + str(cfg.inp_size[0]) + "x" + str(cfg.inp_size[1]))
        if cfg.padding:
            image_dst = image_dst + '_padding'
            boxes_dst = boxes_dst + '_padding'
        boxes_dst = boxes_dst + ".pkl"
    else:
        image_dst = os.path.join(cfg.save_data_root, 'train_image_' + str(cfg.scale[0]))
        boxes_dst = os.path.join(cfg.save_data_root, 'train_boxes_' + str(cfg.scale[0]) + ".pkl")
    
    image_cnt = 0
    boxes_cnt = 0
    boxes_list = []
    tools.del_and_create(image_dst)
    print ('start to process image and create lmdb-dataset...')
    image_db = lmdb.open(image_dst, map_size = 8e10)
    with image_db.begin(write = True) as image_txn:
        for annotation_file_name in os.listdir(annotation_dir):
            # get candidate views
            view = annToView.get_view(os.path.join(annotation_dir, annotation_file_name))
            # get image name
            img_file_name = annotation_file_name.split('.')
            img_file_name = img_file_name[0] + '.' + img_file_name[1]
            img_file_name = os.path.join(dataset_dir, img_file_name)

            # image transform
            if cfg.fixed_size:
                image, width, height = tools.image_transform_fixed_size(img_file_name) 
                ratio_w = float(cfg.inp_size[0]) / width
                ratio_h = float(cfg.inp_size[1]) / height  
            else:
                image, width, height = tools.image_transform_scale(img_file_name) 
                ratio_w = cfg.scale[0]
                ratio_h = cfg.scale[1]
            
            # write image to lmdb
            key = '{:0>10d}'.format(int(image_cnt)) 
            image_txn.put(key.encode(), image.copy(order='C'))

            # write boxes to list
            boxes = np.zeros((24, 4))
            score = np.zeros((24, 1))       
            for i in range(0, 24):
                boxes[i] = view[i]['box']
                score[i] = rank_score[i]
  
            boxes[:,0] = boxes[:,0] * ratio_w
            boxes[:,1] = boxes[:,1] * ratio_h
            boxes[:,2] = boxes[:,2] * ratio_w 
            boxes[:,3] = boxes[:,3] * ratio_h 
            boxes = boxes.astype(np.int)

            B_i = {'image_id': image_cnt, 'boxes': boxes,'score': score, 'width':width, 'height':height}    
            boxes_list.append(B_i)

            if(image_cnt % 500 == 0 and image_cnt != 0):
                print("processed ", image_cnt, " images") 

            image_cnt += 1
            boxes_cnt += 24
        image_txn.put('num-images'.encode(), np.array([image_cnt]).astype(np.int))
    image_db.close()
    with open(boxes_dst, "wb") as f:
        pickle.dump(boxes_list, f)

def create_test_lmdb():
    if cfg.fixed_size:
        image_dst = os.path.join(cfg.save_data_root, 'test_image_' + str(cfg.inp_size[0]) + "x" + str(cfg.inp_size[1]))
        boxes_dst = os.path.join(cfg.save_data_root, 'test_boxes_' + str(cfg.inp_size[0]) + "x" + str(cfg.inp_size[1]))
        if cfg.padding:
            image_dst = image_dst + '_padding'
            boxes_dst = boxes_dst + '_padding'
        boxes_dst = boxes_dst + ".pkl"
    else:
        image_dst = os.path.join(cfg.save_data_root, 'test_image_' + str(cfg.scale[0]))
        boxes_dst = os.path.join(cfg.save_data_root, 'test_boxes_' + str(cfg.scale[0]) + ".pkl")
    
    tools.del_and_create(image_dst)
    print ('start to process image and create test lmdb-dataset...')
    image_db = lmdb.open(image_dst, map_size = 8e10)
    key = 1
    boxes_list = []
    with image_db.begin(write = True) as image_txn:
        crops_string = open(cfg.FCDB_cropdir, 'r').read()
        crops = json.loads(crops_string)
        for item in crops:
            crop = np.array(item['crop'])
            crop = np.reshape(crop, (-1,4))
            img_file_name = os.path.join(cfg.FCDB_datadir, item['image_file_name'])
            if(not os.path.exists(img_file_name)):
                print("no ",item['image_file_name'])
                continue
            # image transform
            if cfg.fixed_size:
                image, width, height = tools.image_transform_fixed_size(img_file_name) 
            else:
                image, width, height = tools.image_transform_scale(img_file_name) 
    
            # write to lmdb
            image_txn.put('{:0>10d}'.format(key).encode(), image.copy(order='C'))
            #print(key, ' ', img_file_name)

            B_i = {'image_id': key, 'gt_crop': crop, 'width':width, 'height': height}
            boxes_list.append(B_i)
            key += 1
    image_db.close()
    with open(boxes_dst, "wb") as f:
        pickle.dump(boxes_list, f)
    print("create dataset successfully! view lists size:", len(boxes_list))

if __name__ == "__main__":
    create_train_lmdb()
    create_test_lmdb()
