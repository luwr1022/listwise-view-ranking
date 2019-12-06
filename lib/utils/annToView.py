#coding=utf-8
#annotation to view
import os
import numpy as np
import os.path as osp
        
def _str2int(s):
    s = s.split(',')
    r = []
    for c in s:
        r.append(int(c))
    return r

def mysort(v):
    return v['mean']

def get_view(annotation_file):
    with open(annotation_file) as f:
        data = f.read()
    data = data.split(']')
    processed_data_str = []
    for d in data:
        dd = d.split('[')
        for ddd in dd:
            if(len(ddd) < 12 or ddd[-1] == ' '):
                continue
            processed_data_str.append(ddd)
    
    num_annotator = len(processed_data_str) - 24

    processed_data_int = []
    for s in processed_data_str:
        processed_data_int.append(_str2int(s))
    score = processed_data_int[0:num_annotator]
    boxes = processed_data_int[num_annotator:]
 
    assert len(boxes) == 24
    view = []
    # to dict {score, mean, box}
    for i in range(len(boxes)):
        v = dict()
        v['box'] = boxes[i]
        score_arr = []
        for n in range(num_annotator):
            score_arr.append(score[n][i])
        score_arr.sort(reverse = True)
        v['score'] = score_arr
        v['mean'] = float(sum(v['score'])) / num_annotator
        view.append(v)

    view.sort(key = mysort, reverse = True)

    return view

        
