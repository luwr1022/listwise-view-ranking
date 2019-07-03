import os
import json
import numpy as np
import PIL.Image as Image
import scipy.io as sio

def str2int(box):
	box = box.split(' ')
	return [int(box[0]), int(box[1]), int(box[2]), int(box[3])]

# FCDB
crop_path = 'data/FCDB/flickr-cropping-dataset/cropping_testing_set.json'
gt_path   = 'data/FCDB/flickr-cropping-dataset/gt_crop.json'
crop_string = open(crop_path, 'r').read()
test_set = json.loads(crop_string)
data=[]
with open(gt_path, "w") as f:
	for item in test_set:
		url = item['url']
		img_name = url.split('/')[-1]
		crop = item['crop']
		data_dict = {"image_file_name":img_name, "crop":crop}
		data.append(data_dict)
	f.write(json.dumps(data))

# FLMS
images_path = 'data/FLMS/image'
ann_path    = 'data/FLMS/500_image_dataset.mat'
gt_path      = 'data/FLMS/gt_crop.json'
scale = [0.5, 0.6, 0.7, 0.8,0.9 ]
grids = [5, 5]
ratios = [0, 1.0, 9.0/16, 16.0/9, 3.0/4, 4.0/3]
#ratios =[0]
ann = sio.loadmat(ann_path)
ann = ann['img_gt']

with open(gtgt_pathdir,"w") as f:
	data=[]
	for item in ann:
		filename = item['filename'][0][0]
		img_file = os.path.join(images_path, filename)
		bbox = item['bbox'][0] #(y1,x1,y2,x2)
		temp = bbox[:, 0].copy()
		bbox[:, 0] = bbox[:, 1]#(x1,x1,y2,x2)
		bbox[:, 1] = temp      #(x1,y1,y2,x2)
		temp = bbox[:, 2].copy()
		bbox[:, 2] = bbox[:, 3]
		bbox[:, 3] = temp
		bbox[:, 2] = bbox[:, 2] - bbox[:, 0] 
		bbox[:, 3] = bbox[:, 3] - bbox[:, 1] # (x1,y1,w,h)
		crop = bbox.astype(np.int).tolist()
		dict = {"image_file_name":filename, "crop":crop}
		data.append(dict)
	f.write(json.dumps(data))



