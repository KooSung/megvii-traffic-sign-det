# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
from megengine import hub

import models
from megengine.data import transform as T

@hub.pretrained(
    "https://data.megengine.org.cn/models/weights/"
    "faster_rcnn_res50_coco_3x_800size_40dot1_8682ff1a.pkl"
)
def faster_rcnn_res50_coco_3x_800size(**kwargs):
    r"""
    Faster-RCNN FPN trained from COCO dataset.
    `"Faster-RCNN" <https://arxiv.org/abs/1506.01497>`_
    `"FPN" <https://arxiv.org/abs/1612.03144>`_
    `"COCO" <https://arxiv.org/abs/1405.0312>`_
    """
    cfg = models.FasterRCNNConfig()
    cfg.backbone_pretrained = True
    return models.FasterRCNN(cfg, **kwargs)


class CustomerConfig(models.FasterRCNNConfig):
    def __init__(self):
        super().__init__()

        self.backbone = "resnet50"
        self.backbone_pretrained = True # must
        self.backbone_norm = "FrozenBN"
        self.backbone_freeze_at = 0 # <--------- 2
        self.fpn_norm = None
        self.fpn_in_features = ["res2", "res3", "res4", "res5"]
        self.fpn_in_strides = [4, 8, 16, 32]
        self.fpn_in_channels = [256, 512, 1024, 2048]
        self.fpn_out_channels = 256
        self.data_root = "/home/megstudio/workspace/data/train_cropped_800x800_min05_over02_debug"
        self.train_anno_path = "/home/megstudio/megengine-traffic-sign-det/annoations/coco_train_crop_800x800_min05_over02_debug.json"
        self.val_anno_path = "/home/megstudio/megengine-traffic-sign-det/annoations/coco_val_cropped_640x480_over02.json"
        self.test_anno_path = "/home/megstudio/megengine-traffic-sign-det/annoations/coco_val_cropped_640x480_over02.json"

        # ------------------------ data cfg -------------------------- #
        self.train_dataset = dict(
            name="trafficdet",
            root=self.data_root,
            ann_file=self.train_anno_path,
            remove_images_without_annotations=True,
        )
        self.test_dataset = dict(
            name="trafficdet",
            root=self.data_root,
            ann_file=self.val_anno_path,
            test_ann_file=self.test_anno_path,
            remove_images_without_annotations=False,
        )
        self.num_classes = 5 # <-------
        self.img_mean = [103.530, 116.280, 123.675]  # BGR
        self.img_std = [57.375, 57.120, 58.395]

        # ----------------------- rpn cfg ------------------------- #
        self.rpn_stride = [4, 8, 16, 32, 64]
        self.rpn_in_features = ["p2", "p3", "p4", "p5", "p6"]
        self.rpn_channel = 256
        self.rpn_reg_mean = [0.0, 0.0, 0.0, 0.0]
        self.rpn_reg_std = [1.0, 1.0, 1.0, 1.0]

        self.anchor_scales = [[x] for x in [32, 64, 128, 256, 512]] # [[32], [64], [128], [256], [512]]
        self.anchor_ratios = [[0.5, 1, 2]]
        self.anchor_offset = 0.5

        self.match_thresholds = [0.3, 0.7]
        self.match_labels = [0, -1, 1]
        self.match_allow_low_quality = True
        self.rpn_nms_threshold = 0.7
        self.num_sample_anchors = 256
        self.positive_anchor_ratio = 0.5

        # ----------------------- rcnn cfg ------------------------- #
        self.rcnn_stride = [4, 8, 16, 32]
        self.rcnn_in_features = ["p2", "p3", "p4", "p5"]
        self.rcnn_reg_mean = [0.0, 0.0, 0.0, 0.0]
        self.rcnn_reg_std = [0.1, 0.1, 0.2, 0.2]

        self.pooling_method = "roi_align"
        self.pooling_size = (7, 7)

        self.num_rois = 512
        self.fg_ratio = 0.5
        self.fg_threshold = 0.5
        self.bg_threshold_high = 0.5
        self.bg_threshold_low = 0.0
        self.class_aware_box = True

        # ------------------------ loss cfg -------------------------- #
        self.rpn_smooth_l1_beta = 0  # use L1 loss
        self.rcnn_smooth_l1_beta = 0  # use L1 loss
        self.num_losses = 5

        # ------------------------ training cfg ---------------------- #
        self.train_image_short_size = [800+64*x for x in range(15)]
        self.train_image_max_size = 2000
        self.train_prev_nms_top_n = 2000
        self.train_post_nms_top_n = 1000

        self.basic_lr = 0.02 / 16  # The basic learning rate for single-image
        self.momentum = 0.9
        self.weight_decay = 1e-4
        self.log_interval = 10
        self.nr_images_epoch = 7457 # <--------- 数据集图片数量
        self.max_epoch = 24 # <---------
        self.warm_iters = 100
        self.lr_decay_rate = 0.1
        self.lr_decay_stages = [16, 21] # <---------

        # ------------------------ testing cfg ----------------------- #
        self.test_image_short_size = 800
        self.test_image_max_size = 1600
        self.test_prev_nms_top_n = 1000
        self.test_post_nms_top_n = 1000
        self.test_max_boxes_per_image = 100
        self.test_vis_threshold = 0.3
        self.test_cls_threshold = 0.05
        self.test_nms = 0.5

        #------------------------- aug----------
        self.transforms = [
                T.ShortestEdgeResize(
                    self.train_image_short_size,
                    self.train_image_max_size,
                    sample_style="choice",
                ),
                # T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                T.RandomHorizontalFlip(),
                T.ToMode(),
            ]

Net = models.FasterRCNN
Cfg = CustomerConfig
