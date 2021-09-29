#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time    :2021/09/10 21:09:51
'''
import collections.abc
import math
from typing import Sequence, Tuple
from albumentations.augmentations.geometric.functional import bbox_affine

import cv2
from megengine.data import transform
import numpy as np

from megengine.data.transform import Transform
from megengine.data.transform.vision import functional as F
from megengine.data.transform.vision import VisionTransform
import albumentations as A
from icecream import ic

__all__ = [
    "RandomSizedBBoxSafeCrop",
    "BBoxJitter"
]

class RandomSizedBBoxSafeCrop(VisionTransform):
    """Crop a random part of the input and rescale it to some size without loss of bboxes.

    Args:
        height (int): height after crop and resize.
        width (int): width after crop and resize.
        erosion_rate (float): erosion rate applied on input image height before crop.
        interpolation (OpenCV flag): flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
        p (float): probability of applying the transform. Default: 1.

    Targets:
        image, mask, bboxes

    Image types:
        uint8, float32
    """
    def __init__(self, height, width, p=0.5, erosion_rate=0.0, interpolation=1, always_apply=False, order=None):
        super().__init__(order=order)
        self.height = height
        self.width = width
        self.p = p 
        self.erosion_rate = erosion_rate
        self.interpolation = interpolation
        self.always_apply = always_apply
    
    def check_bbox(self, boxes, h, w):
        boxes[:, 0] = np.clip(boxes[:, 0], 0, w)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, h)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, w)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, h)
        return boxes

    def apply(self, input):
        self.transform = A.Compose([
            A.RandomSizedBBoxSafeCrop(
                height=self.height,
                width=self.width,
                erosion_rate=self.erosion_rate,
                interpolation=self.interpolation,
                always_apply=self.always_apply,
                p=self.p)], 
            bbox_params=A.BboxParams(
                format="pascal_voc", # x1, y1, x2, y2
                label_fields=['class_labels']))
        self.image = self._get_image(input)
        h, w, _ = self.image.shape
        self.bboxes = input[self.order.index("boxes")] # x1, y1, x2, y2
        self.bboxes = self.check_bbox(self.bboxes, h, w)
        self.boxes_category = input[self.order.index("boxes_category")]
        self.transformed = self.transform(
            image=self.image,
            bboxes=self.bboxes,
            class_labels=self.boxes_category)
        return super().apply(input)

    def _apply_image(self, image):
        transformed_image = self.transformed['image']
        return transformed_image

    def _apply_boxes(self, boxes):
        print(boxes)
        transformed_bboxes = self.transformed['bboxes'] # x1, y1, x2, y2
        transformed_bboxes = np.array(transformed_bboxes)
        return transformed_bboxes
    

class BBoxJitter(VisionTransform):
    """
    bbox jitter
    Args:
        min (int, optional): min scale
        max (int, optional): max scale
        ## origin w scale
    """

    def __init__(self, min=0, max=2, order=None):
        super().__init__(order=order)
        self.min_scale = min
        self.max_scale = max
        self.count = 0
        ic("USE BBOX_JITTER")
        ic(min, max)

    def bbox_jitter(self, bboxes, img_shape):
        """Flip bboxes horizontally.
        Args:
            bboxes(ndarray): shape (..., 4*k)
            img_shape(tuple): (height, width)
        """
        assert bboxes.shape[-1] % 4 == 0
        if len(bboxes) == 0:
            return bboxes
        jitter_bboxes = []
        for box in bboxes:
            w = box[2] - box[0]
            h = box[3] - box[1]
            center_x = (box[0] + box[2]) / 2
            center_y = (box[1] + box[3]) / 2
            scale = np.random.uniform(self.min_scale, self.max_scale)
            w = w * scale / 2.
            h = h * scale / 2.
            xmin = center_x - w
            ymin = center_y - h
            xmax = center_x + w
            ymax = center_y + h
            box2 = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
            jitter_bboxes.append(box2)
        jitter_bboxes = np.array(jitter_bboxes, dtype=np.float32)
        jitter_bboxes[:, 0::2] = np.clip(jitter_bboxes[:, 0::2], 0, img_shape[1] - 1)
        jitter_bboxes[:, 1::2] = np.clip(jitter_bboxes[:, 1::2], 0, img_shape[0] - 1)
        return jitter_bboxes

    def apply(self, input):
        self.image = self._get_image(input)
        self.img_h, self.img_w, _ = self.image.shape
        return super().apply(input)

    def _apply_boxes(self, boxes):
        # print('bbox:', boxes)
        transformed_bboxes = self.bbox_jitter(boxes, (self.img_h, self.img_w)) # x1, y1, x2, y2
        transformed_bboxes = np.array(transformed_bboxes)
        # print('transformed_bboxes:', transformed_bboxes)
        return transformed_bboxes
    
    def _apply_image(self, image):
        return image
