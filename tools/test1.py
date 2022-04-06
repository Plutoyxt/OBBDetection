# coding=utf-8

from mmdet.apis import init_detector
from mmdet.apis import inference_detector
from mmdet.apis import show_result

# 模型配置文件
config_file = '/content/mmdetection/OBBDetection/configs/obb/oriented_rcnn/faster_rcnn_orpn_r50_fpn_1x_ssdd.py'

# 预训练模型文件
checkpoint_file = '/content/RESULT1/epoch_12.pth'

# 通过模型配置文件与预训练文件构建模型
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# 测试单张图片并进行展示
img = '/content/SSDD-data/JPEGImages_test/000011'
result = inference_detector(model, img)
show_result(img, result, model.CLASSES)

