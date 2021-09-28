### 整体方案

1. Faster R-CNN + ResNet-50
2. 切图训练，整图推理

### 安装依赖

主要依赖版本：

- python 3.8
- megengie 1.5.0

```bash
pip install -r requirements.txt
```

### 预训练模型

```bash
wget -P ckpt/ https://data.megengine.org.cn/models/weights/faster_rcnn_res50_coco_3x_800size_40dot1_8682ff1a.pkl
```

### 数据预处理

训练集以800x800的尺寸切图，重叠率为20%，舍弃IoU小于 50%的bbox。

```
cd detection
# 训练集切图
python tools/crop_train.py
# 转换为coco格式
python tools/coco_train.py
```

### 训练

```
# 将sh文件里面的export PYTHONPATH=megvii-traffic-sign-det/detection:$PYTHONPATH修改成自己的路径
sh scrips/train.sh
```

### 推理

权重文件 https://drive.google.com/file/d/1j2MVCafoGgYqHzrFoMbGB_rmbigFc6VM/view?usp=sharing

下载后放到`detection/workdirs/log-of-fr_r50_2x_bs_crop_800x800_min05_over02_freeze-0_recrop`中

```
# 将sh文件里面的export PYTHONPATH=megvii-traffic-sign-det/detection:$PYTHONPATH修改成自己的路径
sh scrips/infer.sh
```

