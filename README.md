# Semantic Segmentation

This is a sub-project of [pytorch-cv](https://github.com/AceCoooool/pytorch-cv)（for convenient）

**Support Models：**

- [x] FCN
- [x] PSPNet
- [x] DeepLabv3
- [x] DANet
- [x] OCNet

## Environment

- PyTorch 1.1

## Performance

#### Pascal VOC 2012

Here, we using train (10582), val (1449), test (1456) as most paper used. (More detail can reference [DeepLabv3](https://github.com/chenxi116/DeepLabv3.pytorch)) . And the performance is evaluated with single scale

- Base Size 540, Crop Size 480

|   Model    |   backbone    |   Paper    | OHEM | aux  | dilated | JPU  | Epoch |                          val (crop)                          |     val     |
| :--------: | :-----------: | :--------: | :--: | :--: | :-----: | :--: | :---: | :----------------------------------------------------------: | :---------: |
|    FCN     | ResNet101-v1s |     /      |  ✗   |  ✓   |    ✗    |  ✓   |  50   | [94.54/78.31](https://drive.google.com/open?id=1-FF5BUSB9hNCyldC1nV35LeWLSQFa9Jl) | 94.50/76.89 |
|   PSPNet   | ResNet101-v1s |     /      |  ✗   |  ✓   |    ✓    |  ✗   |  50   | [94.87/80.13](https://drive.google.com/open?id=1g40cVTJRCHLBKwVqjev4yQA5YflQwhrv) | 94.88/78.57 |
|   PSPNet   | ResNet101-v1s |     /      |  ✗   |  ✓   |    ✗    |  ✓   |  50   |                       [94.89/79.90](https://drive.google.com/open?id=1XRFiijt0tAbLgXhV5oVEot6qLccz9JyA)                        | 94.77/78.48 |
| DeepLabv3  | ResNet101-v1s | no / 77.02 |  ✗   |  ✓   |    ✗    |  ✓   |  50   | [95.17/81.00](https://drive.google.com/open?id=1R0C6qwCxOLztps4odVWLiZpe57n5TuDX) | 94.81/78.75 |
|   DANet    | ResNet101-v1s |     /      |  ✗   |  ✓   |    ✗    |  ✓   |  50   | [94.98/80.49](https://drive.google.com/open?id=1jS69enf_dyn8l27DfMDuzt_ulva7KSWt) | 94.85/78.72 |
| OCNet-Base | ResNet101-v1s |     /      |  ✗   |  ✓   |    ✗    |  ✓   |  50   | [94.91/80.33](https://drive.google.com/open?id=15gs_gzgAT_hciPgwm12G_MRMi0VZg9Gb) | 94.86/79.07 |
| OCNet-ASP  | ResNet101-v1s |     /      |  ✗   |  ✓   |    ✗    |  ✓   |  50   |                                                              |             |

> 1. the metric is `pixAcc/mIoU`
> 2. `aux_weight=0.5`

#### Cityscapes

Here, we only using fine train (2975), val (500) as most paper used. (More detail can reference [DeepLabv3](https://github.com/chenxi116/DeepLabv3.pytorch)) . And the performance is evaluated with single scale

- Base Size 1024, Crop Size 768

|   Model    |   backbone    | Paper(*) | OHEM | aux  | dilated | JPU  | Epoch | val (crop)  |                             val                              |
| :--------: | :-----------: | :------: | :--: | :--: | :-----: | :--: | :---: | :---------: | :----------------------------------------------------------: |
|    FCN     | ResNet101-v1s | no/75.96 |  ✗   |  ✓   |    ✗    |  ✓   |  120  | 96.29/73.60 | [96.18/78.61](https://drive.google.com/open?id=119bPxJwL6zAtEyvEJiHljK6US9f-d5Xr) |
|   PSPNet   | ResNet101-v1s | no/78.56 |  ✗   |  ✓   |    ✗    |  ✓   |  120  | 96.21/73.64 | [96.09/78.62](https://drive.google.com/open?id=1qJVXevkErgVvsa4x8mCvAdMlCIuVMJ1C) |
| DeepLabv3  | ResNet101-v1s | no/78.90 |  ✗   |  ✓   |    ✗    |  ✓   |  120  | 96.25/73.44 | [96.23/79.03](https://drive.google.com/open?id=1_XIIeIKEMbg4M2SO49Vq756d20zjeXxk) |
|   DANet    | ResNet101-v1s | no/78.83 |      |      |         |      |       |             |                                                              |
| OCNet-Base | ResNet101-v1s | no/79.67 |  ✗   |  ✓   |    ✗    |  ✓   |  120  | 96.30/74.18 | [TODO](https://drive.google.com/open?id=1q2h8d_mJeyHfmCRQellTSpralv8ubI5L) |
| OCNet-ASP  | ResNet101-v1s |          |      |      |         |      |       |             |                                                              |

> Note：
>
> 1. Paper(*) means results from: [openseg.pytorch](https://github.com/openseg-group/openseg.pytorch)（results with single scale without crop），there are a little different in the training strategy.  

## Demo

Demo of segmentation of a given image.  (Please download pre-trained model to `~/.torch/models` first. --- If you put pre-trained model to other folder, please change the `--root`)

```shell
$ python demo_segmentation_pil.py [--model fcn_resnet101_voc] [--input-pic <image>.jpg] [--cuda true] [--aux true] [--jpu true] [--dilated false]
```

> Note：
>
> 1. if not give `--input-pic`, using default image we provided. 
> 2. `aux, jpu, dilated` is depend on your model

## Evaluation

The default data root is `~/.torch/datasets` (You can download dataset and build a soft-link to it)

```shell
$ python eval_segmentation_pil.py [--model_name fcn_resnet101_voc] [--dataset pascal_paper] [--split val] [--mode testval|val] [--base-size 540] [--crop-size 480] [--aux true] [--jpu true] [--dilated false] [--cuda true]
```

> Note：
>
> 1. if you choose `mode=testval`，you can not set `base-size` and `crop-size`
> 2. `aux, jpu, dilated` is depend on your model 

## Train

Download pre-trained backbone and put it on `~/.torch/models`

Recommend to using distributed training.

```shell
$ export NGPUS=4
$ python -m torch.distributed.launch --nproc_per_node=$NGPUS train_segmentation_pil.py [--model fcn] [--backbone resnet101] [--dataset pascal_voc] [--batch-size 8] [--base-size 540] [--crop-size 480] [--aux true] [--jpu true] [--dilated false] [--log-step 10]
```

> Our training results' setting can see [train.sh](scripts/train.sh)

## Prepare data

#### VOC2012

```shell
mkdir data
cd data
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
tar -xf VOCtrainval_11-May-2012.tar
cd VOCdevkit/VOC2012/
wget http://cs.jhu.edu/~cxliu/data/SegmentationClassAug.zip
wget http://cs.jhu.edu/~cxliu/data/SegmentationClassAug_Visualization.zip
wget http://cs.jhu.edu/~cxliu/data/list.zip
unzip SegmentationClassAug.zip
unzip SegmentationClassAug_Visualization.zip
unzip list.zip
```

Your can make a soft link to `.torch/datasets/voc`

#### Cityscapes

```shell
unzip leftImg8bit_trainvaltest.zip
unzip gtFine_trainvaltest.zip
git clone https://github.com/mcordts/cityscapesScripts.git
mv cityscapesScripts/cityscapesscripts ./
```

Your can make a soft link to `.torch/datasets/citys`

## Download

#### Backbone

| resnet50-v1s                                                 | resnet101-v1s                                                |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [GoogleDrive](https://drive.google.com/open?id=1Mx_SIv1o1qjRz1tqEc-ggQ_MtZKQT3ET) | [GoogleDrive](https://drive.google.com/open?id=1pA_tN2MFi-7J5n1og10kDnV8raphM3V-) |
