#!/usr/bin/env bash
export PYTHONPATH=megvii-traffic-sign-det/detection:$PYTHONPATH
CUDA_VISIBLE_DEVICES=0 python tools/infer_final.py