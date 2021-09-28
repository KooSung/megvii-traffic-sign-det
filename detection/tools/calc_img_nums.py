#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time    :2021/09/20 20:33:48
'''
import os
from tqdm import tqdm
from glob import glob
import pandas as pd
import json 


def load_annotations(anno_dir):
    '''
    :param anno_dir:
    :return: data -> 'annotations', 'categories', 'images'
    images: {file_name': 'pure_bg_images/bb_V0033_I0004040.jpg', 'height': 1080, 'width': 1920, 'id': 793}
    '''
    with open(anno_dir) as fp:
        data = json.load(fp)
    cat_df = pd.DataFrame(data['categories'])
    anno_df = pd.DataFrame(data['annotations'])
    img_df = pd.DataFrame(data['images'])

    return anno_df, img_df, cat_df

anno_dir = '/home/megstudio/workspace/megengine-traffic-sign-det/annoations/coco_train_crop_800x800_min05_over02_debug.json'
anno_df, img_df, cat_df = load_annotations(anno_dir)
img_ids = img_df['id'].unique()
print(len(img_ids))

print(len(glob(os.path.join("/home/megstudio/workspace/data/train_cropped_800x800_min05_over02_debug/*.jpg"))))