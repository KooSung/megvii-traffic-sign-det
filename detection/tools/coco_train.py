#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time    :2021/09/20 20:26:03
'''
import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
from PIL import Image


class NpEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def cat():
    categories = [
        {"supercategory": "none", "id": 0, "name": "red_tl"},
        {"supercategory": "none", "id": 1, "name": "arr_s"},
        {"supercategory": "none", "id": 2, "name": "arr_l"},
        {"supercategory": "none", "id": 3, "name": "no_driving_mark_allsort"},
        {"supercategory": "none", "id": 4, "name": "no_parking_mark"}
    ]
    return categories


def img_and_anno(img_dir, anno_dir):
    '''
    {'file_name': 'pure_bg_images/bb_V0033_I0004040.jpg',
   'height': 1080,
   'width': 1920,
   'id': 793}],
    '''
    images = []
    annotations = []
    img_id = 0
    anno_id = 0
    anno_df = pd.read_json(anno_dir)
    bool_dict = {True: 1, False: 0}
    img_list = glob(os.path.join(img_dir, '*.jpg'))
    for img_path in tqdm(img_list):
        file_name = os.path.basename(img_path)

        df = anno_df[anno_df.file_name == file_name]
        if df.shape[0] == 0:
            continue

        img = Image.open(img_path)
        width, height = img.size
        images.append({'file_name': file_name, 'height': height, 'width': width, 'id': img_id})

        for i in range(df.shape[0]):
            annotations.append({
                'segmentation': df['segmentation'].iloc[i],
                'area': df['area'].iloc[i],
                'iscrowd': 0,
                'ignore': 0,
                'image_id': img_id,
                'bbox': df['bbox'].iloc[i],
                'category_id': df['category_id'].iloc[i],
                'id': anno_id
            })
            anno_id += 1
        img_id += 1
    return images, annotations


def merge_json(img_dir, anno_dir, res_json_base_dir):
    categories = cat()
    images, annotations = img_and_anno(img_dir, anno_dir)
    result = {'type': 'instance', 'images': images, 'annotations': annotations, 'categories': categories}

    res_dir = os.path.join(res_json_base_dir, 'coco_train_crop_800x800_min05_over02_debug' + '.json')
    with open(res_dir, 'w') as fp:
        json.dump(result, fp, indent=4, separators=(',', ': '), cls=NpEncoder)


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

    return data, anno_df, img_df, cat_df


if __name__ == '__main__':
    # train
    anno_dir = '/home/megstudio/workspace/megengine-traffic-sign-det/annoations/train_crop_800x800_min05_over02_debug.json'
    img_dir = '/home/megstudio/workspace/data/train_cropped_800x800_min05_over02_debug/'
    res_json_base_dir = '/home/megstudio/workspace/megengine-traffic-sign-det/annoations/'
    merge_json(img_dir, anno_dir, res_json_base_dir)