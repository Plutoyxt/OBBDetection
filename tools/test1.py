from mmdet.apis import init_detector, inference_detector
import mmcv
import os
config_file = '/content/mmdetection/BboxToolkit/OBBDetection/configs/obb/oriented_rcnn/faster_rcnn_orpn_r50_fpn_1x_ssdd.py'
checkpoint_file = '/content/RESULT1/latest.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')
# test a single image and show the results
filelist = os.listdir('/content/SSDD-data/JPEGImages_test/')#获取所有的img文件
for filename in filelist: 
  img = os.path.join('/content/SSDD-data/JPEGImages_test/'+filename)
#img = '/content/SSDD-data/JPEGImages_train/000031.jpg'  # or img = mmcv.imread(img), which will only load it once
  result = inference_detector(model, img)
# visualize the results in a new window
  model.show_result(img, result)
# or save the visualization results to image files
  model.show_result(img, result, out_file=os.path.join('/content/RESULT1/result/'+filename))
