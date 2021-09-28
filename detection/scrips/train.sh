#!/usr/bin/env bash
export PYTHONPATH=megvii-traffic-sign-det/detection:$PYTHONPATH
# CONFIG=$1
CUDA_VISIBLE_DEVICES=0,1 python tools/train_bs.py -n 2 \
                      -b 2 \
                      -f /home/megstudio/workspace/megengine-traffic-sign-det/detection/configs/fr_r50_2x_bs_crop_800x800_min05_over02_freeze-0_recrop.py \
                      -d /home/megstudio/workspace/data/train_cropped_800x800_min05_over02_debug \
                      -w ../ckpt/faster_rcnn_res50_coco_3x_800size_40dot1_8682ff1a.pkl