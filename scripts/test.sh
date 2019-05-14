#!/usr/bin/env bash

# ---------voc2012----------
# FCN
python eval_segmentation.py \
     --model_name  fcn_resnet101_voc --dataset pascal_voc --split val --mode testval \
     --base-size 540 --crop-size 480 --jpu true --aux true --dilated false

# PSP
python eval_segmentation.py \
     --model_name  psp_resnet101_voc --dataset pascal_voc --split val --mode testval \
     --base-size 540 --crop-size 480 --jpu true --aux true --dilated false

# DeepLab
python eval_segmentation.py \
     --model_name  deeplab_resnet101_voc --dataset pascal_voc --split val --mode testval \
     --base-size 540 --crop-size 480 --jpu true --aux true --dilated false


# ---------cityscapes----------
python eval_segmentation.py \
     --model_name  fcn_resnet101_citys --dataset citys --split val --mode testval \
     --base-size 1024 --crop-size 768 --jpu true --aux true --dilated false

# PSP
python eval_segmentation.py \
     --model_name  psp_resnet101_citys --dataset citys --split val --mode testval \
     --base-size 1024 --crop-size 768 --jpu true --aux true --dilated false

# DeepLab
python eval_segmentation.py \
     --model_name  deeplab_resnet101_citys --dataset citys --split val --mode testval \
     --base-size 1024 --crop-size 768 --jpu true --aux true --dilated false