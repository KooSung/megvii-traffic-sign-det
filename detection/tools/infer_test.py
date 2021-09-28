#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time    :2021/09/21 14:40:27
'''
import os
import json
import cv2
import numpy as np 
from tqdm import tqdm
import pandas as pd 
from glob import glob
import megengine as mge
from tools.data_mapper import data_mapper
from tools.utils import DetEvaluator, import_from_file
from ensemble_boxes import *

logger = mge.get_logger(__name__)
logger.setLevel("INFO")


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

def ensemble_wbf_full_img(anno_dir, test_org_anno_dir, iou_thr, final_json_file):
    # 读取merge后的json文件
    anno_df = pd.read_json(anno_dir)
    _, img_org_df, _ = load_annotations(test_org_anno_dir)
    annotations = []
    for i in range(img_org_df.shape[0]):
        img_id =  img_org_df.id.iloc[i]
        img_width = img_org_df.width.iloc[i]
        img_height = img_org_df.height.iloc[i]
        df = anno_df[anno_df.image_id == img_id]
        if df.shape[0] == 0:
            continue
        merged_bboxes = np.array(df.bbox.to_list()) 
        merged_bboxes[:, 2] +=  merged_bboxes[:, 0]
        merged_bboxes[:, 3] +=  merged_bboxes[:, 1]
        merged_bboxes[:, 0] /= img_width
        merged_bboxes[:, 1] /= img_height
        merged_bboxes[:, 2] /= img_width
        merged_bboxes[:, 3] /= img_height
        merged_bboxes = list(merged_bboxes)
        merged_scores = df.score.to_list()
        merged_categories = df.category_id.to_list()
        norm_bboxes, scores, labels = weighted_boxes_fusion(
            [merged_bboxes],
            [merged_scores],
            [merged_categories],
            weights=None,
            allows_overflow=False,
            iou_thr=iou_thr,
            skip_box_thr=0.0)
        dets_bboxes = np.array(norm_bboxes)
        dets_bboxes[:, 0] *= img_width
        dets_bboxes[:, 1] *= img_height
        dets_bboxes[:, 2] *= img_width
        dets_bboxes[:, 3] *= img_height
        dets_bboxes[:, 2] -=  dets_bboxes[:, 0]
        dets_bboxes[:, 3] -=  dets_bboxes[:, 1]
        dets_scores = np.array(scores)
        dets_categories = np.array(labels)
        for j in range(dets_bboxes.shape[0]):
            annotations.append({'image_id':img_id,
                                'bbox':dets_bboxes[j],
                                'score':dets_scores[j], 
                                'category_id':int(dets_categories[j])})
    with open(final_json_file, 'w') as fp:
        json.dump(annotations, fp, indent=4, separators=(',', ': '), cls=NpEncoder)


def infer_crop_imgs():
    # 双阈值，待测试
    double_thresh = 0.3
    # 检测结果json保存路径
    res_json_path = "/home/megstudio/workspace/megengine-traffic-sign-det/detection/test_json/test_v1.json"
    # 模型config路径
    config = "/home/megstudio/workspace/megengine-traffic-sign-det/detection/configs/fr_r50_2x_bs_crop_800x800_min05_over02_freeze-0_recrop.py"
    # 最终模型权重路径
    weight = "/home/megstudio/workspace/megengine-traffic-sign-det/detection/workdirs/log-of-fr_r50_2x_bs_crop_800x800_min05_over02_freeze-0_recrop/epoch_23.pkl"
    # 验证集/测试集切片图片根目录
    img_root = "/home/megstudio/dataset/dataset-2805/images"
    # 验证集/测试集所有切片list
    img_listdir = glob(os.path.join(img_root, "*.jpg"))
    # 加载模型
    current_network = import_from_file(config)
    cfg = current_network.Cfg()
    cfg.backbone_pretrained = False
    model = current_network.Net(cfg)
    model.eval()

    state_dict = mge.load(weight)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict)

    evaluator = DetEvaluator(model)
    results = []
    
    val_anno_ori_file = "/home/megstudio/dataset/dataset-2805/annotations/test.json"
    _, img_df, _ = load_annotations(val_anno_ori_file)
    # 检测所有图片
    for i in tqdm(range(img_df.shape[0])):
        img_id = img_df.id.iloc[i]
        img_h = img_df.height.iloc[i]
        img_w = img_df.width.iloc[i]
        img_name = img_df.file_name.iloc[i]
        img_path = os.path.join(img_root, img_name)
        ori_img = cv2.imread(img_path)
        test_scales = [
            [img_h, img_w], [img_h*1.1, img_w*1.1], [img_h*1.2, img_w*1.2], [img_h*1.3, img_w*1.3], [img_h*1.4, img_w*1.4],
            [img_h*1.5, img_w*1.5], [img_h*1.6, img_w*1.6], [img_h*1.7, img_w*1.7], [img_h*1.8, img_w*1.8], [img_h*1.9, img_w*1.9],
            [img_h*2.0, img_w*2.0], [img_h*0.9, img_w*0.9]
        ]
        for test_scale in test_scales:
            image, im_info = DetEvaluator.process_inputs(
                ori_img.copy(), test_scale[0], test_scale[1],
            )
            # pred_res [[x1, y1, x2, y2, score, label]]
            pred_res = evaluator.predict(
                image=mge.tensor(image),
                im_info=mge.tensor(im_info)
            )
            # 若没有预测结果，则跳过
            if len(pred_res) == 0:
                continue
            # 双阈值
            if pred_res[:, -2].max() <= double_thresh:
                continue
            for pred in pred_res:
                bbox = pred[:4]
                x1, y1, x2, y2 = bbox
                w = x2 -x1
                h = y2 - y1 
                score = pred[4]
                label = pred[5]
                results.append({
                    'image_id': img_id,
                    'img_name': img_name,
                    'bbox': [x1, y1, w, h],
                    'score': score,
                    'category_id': int(label)
                })
            # ------------------------------HorizontalFlip--------------------------------------
            img_flip = cv2.flip(ori_img, 1)
            image, im_info = DetEvaluator.process_inputs(
                img_flip, test_scale[0], test_scale[1],
            )
            # pred_res [[x1, y1, x2, y2, score, label]]
            pred_res = evaluator.predict(
                image=mge.tensor(image),
                im_info=mge.tensor(im_info)
            )
            # 若没有预测结果，则跳过
            if len(pred_res) == 0:
                continue
            # 双阈值
            if pred_res[:, -2].max() <= double_thresh:
                continue
            for pred in pred_res:
                bbox = pred[:4]
                x1, y1, x2, y2 = bbox
                w = x2 -x1
                h = y2 - y1 
                x1 = img_w - x2
                score = pred[4]
                label = pred[5]
                results.append({
                    'image_id': img_id,
                    'img_name': img_name,
                    'bbox': [x1, y1, w, h],
                    'score': score,
                    'category_id': int(label)
                })

    with open(res_json_path, 'w') as fp:
        json.dump(results, fp, indent=4, separators=(',', ': '), cls=NpEncoder)
    iou_thr = 0.55
    final_json_file = "/home/megstudio/workspace/megengine-traffic-sign-det/detection/test_json/submit_test_v1.json"
    ensemble_wbf_full_img(res_json_path, val_anno_ori_file, iou_thr, final_json_file)
    return results


if __name__ == "__main__":
    infer_crop_imgs()
