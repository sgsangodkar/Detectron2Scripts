#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 09:58:36 2022

@author: sagar
"""

python3 detectron2/demo/demo.py \
--config-file detectron2/configs/COCO-Detection/faster_rcnn_R_101_C4_3x.yaml \
--video-input video-clip.mp4 --confidence-threshold 0.6 --output op_detection.mkv \
--opts MODEL.WEIGHTS detectron2://COCO-Detection/faster_rcnn_R_101_C4_3x/138204752/model_final_298dad.pkl \
MODEL.DEVICE cpu


python3 detectron2/demo/demo.py \
--config-file detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x.yaml \
--video-input video-clip.mp4 --confidence-threshold 0.6 --output op_instance.mkv \
--opts MODEL.WEIGHTS detectron2://COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x/138363239/model_final_a2914c.pkl \
MODEL.DEVICE cpu


python3 detectron2/demo/demo.py \
--config-file detectron2/configs/COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml \
--video-input video-clip.mp4 --confidence-threshold 0.6 --output op_panoptic.mkv \
--opts MODEL.WEIGHTS detectron2://COCO-PanopticSegmentation/panoptic_fpn_R_101_3x/139514519/model_final_cafdb1.pkl \
MODEL.DEVICE cpu


python3 detectron2/demo/demo.py \
--config-file detectron2/configs/COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml \
--video-input video-clip.mp4 --confidence-threshold 0.6 --output op_keypoint.mkv \
--opts MODEL.WEIGHTS detectron2://COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x/137849621/model_final_a6e10b.pkl \
MODEL.DEVICE cpu