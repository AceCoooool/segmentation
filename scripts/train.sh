#!/usr/bin/env bash

# ---------voc2012----------
# PSP
export NGPUS=8
python -m torch.distributed.launch --nproc_per_node=$NGPUS train_segmentation.py \
     --model psp --backbone resnet101 --dataset pascal_voc --batch-size 8 --test-batch-size 2 --lr -1 \
     --base-size 540 --crop-size 480 --jpu true --aux true --dilated false \
     --epochs 50 -j 16 --warmup-factor 0.01 --log-step 10 --eval-epochs -1 --save-epoch 10
