#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time    :2021/09/20 20:16:01
'''
import os
import cv2
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
from matplotlib import pyplot as plt
from albumentations import (BboxParams, Crop, Compose)


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


def get_aug(aug, min_area=0., min_visibility=0.):
    return Compose(aug, bbox_params=BboxParams(format='coco', min_area=min_area,
                                               min_visibility=min_visibility, label_fields=['category_id']))


def dataframe2dict(df, image, bboxes):
    '''
    ['segmentation', 'bbox', 'category_id', 'area', 'iscrowd',
     'image_id', 'id', 'ignore', 'uncertain', 'logo', 'in_dense_image']
    :param df:
    :return: annotations
    '''
    columns = list(df)
    annotations = {}
    for column in columns:
        annotations[column] = df[column].to_list()
    annotations['bboxes'] = bboxes
    annotations['image'] = image  # array

    return annotations


def bbox2segm(bbox):
    '''
    convert bbox to segmentation
    :param bbox:
    :return:segm
    '''
    x1, y1, w, h = bbox
    x2 = x1 + w - 1
    y2 = y1 + h - 1
    segm = [[x1, y1, x2, y1, x2, y2, x1, y2]]
    area = w * h
    return segm, area


def dict2dataframe(annotations_cropped, res):
    # res = []
    for i in range(len(annotations_cropped['bboxes'])):
        bbox = annotations_cropped['bboxes'][i]
        segm, area = bbox2segm(bbox)
        res.append({
            'segmentation': segm,
            'area': area,
            'iscrowd': 0,
            'ignore': 0,
            'bbox': bbox,
            'category_id': annotations_cropped['category_id'][i],
            'file_name': annotations_cropped['file_name']
        })
    return res


BOX_COLOR = (255, 0, 0)


def visualize_bbox(img, bbox, class_id, class_idx_to_name, color=BOX_COLOR, thickness=1):
    BOX_COLOR = (255, 0, 0)
    TEXT_COLOR = (255, 255, 255)
    x_min, y_min, w, h = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
    class_name = class_idx_to_name[class_id]
    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
    cv2.putText(img, class_name, (x_min, y_min - int(0.3 * text_height)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, TEXT_COLOR,
                lineType=cv2.LINE_AA)
    return img


def visualize(annotations, category_id_to_name):
    img = annotations['image'].copy()
    for idx, bbox in enumerate(annotations['bboxes']):
        img = visualize_bbox(img, bbox, annotations['category_id'][idx], category_id_to_name)
    plt.figure(figsize=(12, 12))
    plt.imshow(img)
    plt.show()


def mkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def check_bbox(df, height, width):
    bboxes = np.array(df.bbox.to_list())
    bboxes[:, 0] = np.clip(bboxes[:, 0], 0, width)
    bboxes[:, 1] = np.clip(bboxes[:, 1], 0, height)
    bboxes[:, 2] += bboxes[:, 0]
    bboxes[:, 3] += bboxes[:, 1]
    bboxes[:, 2] = np.clip(bboxes[:, 2], 0, width)
    bboxes[:, 3] = np.clip(bboxes[:, 3], 0, height)
    bboxes[:, 2] -= bboxes[:, 0]
    bboxes[:, 3] -= bboxes[:, 1]

    return bboxes


def debug_crop2(img_df, img_base_dir, img_cropped_base_dir, res_json_base_dir, sw=640, sh=480, overlop=0.2, min_visibility=0.5):
    mkdir(img_cropped_base_dir)
    mkdir(res_json_base_dir)

    img_ids = img_df['id'].unique()
    res = []
    for img_id in tqdm(img_ids):
        df = img_df[img_df['id'] == img_id]
        if df.shape[0] == 0:
            continue
        file_name = df.file_name.iloc[0]
        height = df.height.iloc[0]
        width = df.width.iloc[0]
        img_path = os.path.join(img_base_dir, file_name)
        img = Image.open(img_path)
        x = y = 0
        index = 0
        anno_single_df = anno_df[anno_df.image_id == img_id]
        if anno_single_df.shape[0] == 0:
            continue
        bboxes = check_bbox(anno_single_df, height, width)
        annotations = dataframe2dict(anno_single_df, np.array(img), bboxes)
        category_id_to_name = {0: '0', 1:'1', 2: '2', 3:'3', 4:'4'}
        if width <= sw:
            x_max = width
            for j in range(0, height + 1, int(sh * (1 - overlop))):
                if j < height - sh:
                    y = j
                else:
                    y = height - sh
                aug = get_aug([Crop(x_min=0, y_min=y,
                                    x_max=x_max, y_max=y + sh,
                                    always_apply=False, p=1.0)],
                              min_visibility=min_visibility)
                augmented = aug(**annotations)
                bboxes = augmented['bboxes']
                # print(bboxes)
                if len(bboxes) == 0:
                    continue
                img_cropped = Image.fromarray(augmented['image'])
                img_cropped_name = os.path.basename(file_name)[:-4] + '_' + str(index) + '_' + str(x) + '_' + str(
                    y) + '.jpg'
                # save the cropped image
                img_cropped.save(img_cropped_base_dir + img_cropped_name)
                # save the cropped annotations
                augmented['file_name'] = img_cropped_name
                res = dict2dataframe(augmented, res)
                # visualize the cropped image and bbox
                # visualize(augmented, category_id_to_name)
                index += 1
        elif height <= sh:
            y_max = height
            for i in range(0, width + 1, int(sw * (1 - overlop))):
                if i < width - sw:
                    x = i
                else:
                    x = width - sw
                aug = get_aug([Crop(x_min=x, y_min=0,
                                    x_max=x + sw, y_max=y_max,
                                    always_apply=False, p=1.0)],
                              min_visibility=min_visibility)
                augmented = aug(**annotations)
                bboxes = augmented['bboxes']
                # print(bboxes)
                if len(bboxes) == 0:
                    continue
                img_cropped = Image.fromarray(augmented['image'])
                img_cropped_name = os.path.basename(file_name)[:-4] + '_' + str(index) + '_' + str(x) + '_' + str(
                    y) + '.jpg'
                # save the cropped image
                img_cropped.save(img_cropped_base_dir + img_cropped_name)
                # save the cropped annotations
                augmented['file_name'] = img_cropped_name
                res = dict2dataframe(augmented, res)
                # visualize the cropped image and bbox
                # visualize(augmented, category_id_to_name)
                index += 1
        else:
            for j in range(0, height + 1, int(sh * (1 - overlop))):
                if j < height - sh:
                    y = j
                else:
                    y = height - sh
                for i in range(0, width + 1, int(sw * (1 - overlop))):
                    if i < width - sw:
                        x = i
                    else:
                        x = width - sw
                    aug = get_aug([Crop(x_min=x, y_min=y,
                                        x_max=x + sw, y_max=y + sh,
                                        always_apply=False, p=1.0)],
                                  min_visibility=min_visibility)
                    augmented = aug(**annotations)
                    bboxes = augmented['bboxes']
                    if len(bboxes) == 0:
                        continue
                    img_cropped = Image.fromarray(augmented['image'])
                    img_cropped_name = os.path.basename(file_name)[:-4] + '_' + str(index) + '_' + str(x) + '_' + str(
                        y) + '.jpg'
                    # save the cropped image
                    img_cropped.save(img_cropped_base_dir + img_cropped_name)
                    # save the cropped annotations
                    augmented['file_name'] = img_cropped_name
                    res = dict2dataframe(augmented, res)
                    # visualize the cropped image and bbox
                    # visualize(augmented, category_id_to_name)
                    index += 1
    res_dir = os.path.join(res_json_base_dir, 'train_crop_800x800_min05_over02_debug' + '.json')
    with open(res_dir, 'w') as fp:
        json.dump(res, fp, indent=4, separators=(',', ': '), cls=NpEncoder)


if __name__ == '__main__':
    # train
    anno_dir = '/home/megstudio/dataset/dataset-2805/annotations/train.json'
    anno_df, img_df, cat_df = load_annotations(anno_dir)
    img_ids = img_df['id'].unique()
    print(len(img_ids))

    img_base_dir = '/home/megstudio/dataset/dataset-2805/images'
    img_cropped_base_dir = '/home/megstudio/workspace/data/train_cropped_800x800_min05_over02_debug/'
    res_json_base_dir = '/home/megstudio/workspace/megengine-traffic-sign-det/annoations'
    sw = 800
    sh = 800
    debug_crop2(img_df, img_base_dir, img_cropped_base_dir, res_json_base_dir, sw=sw, sh=sh, overlop=0.2)