#!/usr/bin/env bash

# ---------voc2012----------
# FCN
export NGPUS=8
python -m torch.distributed.launch --nproc_per_node=$NGPUS train_segmentation.py \
     --model fcn --backbone resnet101 --dataset pascal_voc --batch-size 8 --test-batch-size 2 --lr -1 \
     --base-size 540 --crop-size 480 --jpu \
     --epochs 50 -j 16 --warmup-factor 0.01 --log-step 10 --eval-epochs -1 --save-epoch 10 --aux


# PSP
export NGPUS=8
python -m torch.distributed.launch --nproc_per_node=$NGPUS train_segmentation.py \
     --model psp --backbone resnet101 --dataset pascal_voc --batch-size 4 --test-batch-size 2 --lr -1 \
     --base-size 540 --crop-size 480 --dilated \
     --epochs 50 -j 16 --warmup-factor 0.01 --log-step 10 --eval-epochs -1 --save-epoch 10 --aux

# DeepLabv3
export NGPUS=8
python -m torch.distributed.launch --nproc_per_node=$NGPUS train_segmentation.py \
     --model deeplab --backbone resnet101 --dataset pascal_voc --batch-size 4 --test-batch-size 2 --lr -1 \
     --base-size 540 --crop-size 480 --jpu \
     --epochs 50 -j 16 --warmup-factor 0.01 --log-step 10 --eval-epochs -1 --save-epoch 10 --aux


# ---------cityscapes----------
# FCN
export NGPUS=8
python -m torch.distributed.launch --nproc_per_node=$NGPUS train_segmentation.py \
     --model fcn --backbone resnet101 --dataset citys --batch-size 4 --test-batch-size 2 --lr -1 \
     --base-size 1024 --crop-size 768 --jpu \
     --epochs -1 -j 16 --warmup-factor 0.01 --log-step 10 --eval-epochs -1 --save-epoch 20 --aux

# PSP
export NGPUS=8
python -m torch.distributed.launch --nproc_per_node=$NGPUS train_segmentation.py \
     --model psp --backbone resnet101 --dataset citys --batch-size 2 --test-batch-size 2 --lr -1 \
     --base-size 1024 --crop-size 768 --jpu \
     --epochs -1 -j 16 --warmup-factor 0.01 --log-step 10 --eval-epochs -1 --save-epoch 20 --aux

# DeepLab
export NGPUS=8
python -m torch.distributed.launch --nproc_per_node=$NGPUS train_segmentation.py \
     --model deeplab --backbone resnet101 --dataset citys --batch-size 2 --test-batch-size 2 --lr -1 \
     --base-size 1024 --crop-size 768 --jpu \
     --epochs -1 -j 16 --warmup-factor 0.01 --log-step 10 --eval-epochs -1 --save-epoch 20 --aux