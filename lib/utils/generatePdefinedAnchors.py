# generate pre-defined boxes (x1, y1, x2, y2)
import numpy as np
import pickle
import config as cfg

stride = 20
scales = [0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60]
ratios = [4.0/3, 3/4.0, 16.0/9, 9.0/16, 1] # width / height
image_size = [
    [800, 600],
    [600, 800],
    [1600, 900],
    [900, 1600],
    [800, 800],
] # width height

def getSubCrop(image_width, image_height):
    subcrop = []
    for scale in scales:
        for ratio in ratios:
            w = int(image_width * scale)
            h = int(w / ratio)
            if h > image_height:
                continue
            ww = image_width - w
            hh = image_height - h
            step_x = int(max(5, ww / stride))
            step_y = int(max(5, hh / stride))
            stride_x = ww / step_x
            stride_y = hh / step_y
            for i in range(step_x):
                for j in range(max(5, int(hh/stride))):
                    x = int(stride_x * i)
                    y = int(stride_y * j)
                    subcrop.append([float(x)/image_width, float(y)/image_height, float(x+w)/image_width, float(y+h)/image_height])

            h = int(image_height * scale)
            w = int(h * ratio)
            if w > image_width:
                continue
            ww = image_width - w
            hh = image_height - h
            step_x = int(max(5, ww / stride))
            step_y = int(max(5, hh / stride))
            stride_x = ww / step_x
            stride_y = hh / step_y
            for i in range(step_x):
                for j in range(max(5, int(hh/stride))):
                    x = int(stride_x * i)
                    y = int(stride_y * j)
                    subcrop.append([float(x)/image_width, float(y)/image_height, float(x+w)/image_width, float(y+h)/image_height])
    return bbox_nms(subcrop, 0.90)

def bbox_nms(crops, nms_threshold):
    crops = np.array(crops)
    keep_bboxes = np.ones(len(crops), dtype=np.bool)
    for i in range(len(crops)-1):
        if keep_bboxes[i]:
            overlap = bboxes_jaccard(crops[i], crops[(i+1):])
            keep_overlap = (overlap < nms_threshold)
            keep_bboxes[(i+1):] = np.logical_and(keep_bboxes[(i+1):], keep_overlap)
    idxes = np.where(keep_bboxes)[0]
    
    return crops[idxes].tolist()

def bboxes_jaccard(bboxes1, bboxes2):
    if isinstance(bboxes1, (tuple, list)):
        bboxes1 = np.array(bboxes1)
    if isinstance(bboxes2, (tuple, list)):
        bboxes2 = np.array(bboxes2)

    bboxes1 = np.transpose(bboxes1)
    bboxes2 = np.transpose(bboxes2)
    # Intersection bbox and volume.
    int_ymin = np.maximum(bboxes1[0], bboxes2[0])
    int_xmin = np.maximum(bboxes1[1], bboxes2[1])
    int_ymax = np.minimum(bboxes1[2], bboxes2[2])
    int_xmax = np.minimum(bboxes1[3], bboxes2[3])

    int_h = np.maximum(int_ymax - int_ymin, 0.)
    int_w = np.maximum(int_xmax - int_xmin, 0.)
    int_vol = int_h * int_w
    # Union volume.
    vol1 = (bboxes1[2] - bboxes1[0]) * (bboxes1[3] - bboxes1[1])
    vol2 = (bboxes2[2] - bboxes2[0]) * (bboxes2[3] - bboxes2[1])
    jaccard = int_vol / (vol1 + vol2 - int_vol)
    return jaccard

if __name__ == "__main__":
    total_crop = np.array([])
    for size in image_size:
        crop = getSubCrop(size[0],size[1])
        if len(total_crop) == 0:
            total_crop = np.array(crop)
        else:
            total_crop = np.vstack((np.array(crop), total_crop))
    total_crop = bbox_nms(total_crop, 0.90)
    overlap = bboxes_jaccard([0, 0, 1, 1], total_crop[:])
    keep_overlap = (overlap > 0.15)
    idxes = np.where(keep_overlap == True)[0]
    total_crop = np.array(total_crop)[idxes].tolist()
    with open(cfg.pdefined_anchor, 'wb') as f:
        pickle.dump(total_crop, f)