# Semantic Segmentation

This is a sub-project of [pytorch-cv](https://github.com/AceCoooool/pytorch-cv)（for convenient）

**Support Models：**

- [x] FCN
- [x] PSPNet
- [x] DeepLabv3

## Performance

#### Pascal VOC 2012

Here, we using train (10582), val (1449), test (1456) as most paper used. (More detail can reference [DeepLabv3](https://github.com/chenxi116/DeepLabv3.pytorch)) . And the performance is evaluated with single scale

- Base Size 540, Crop Size 480

|   Model   |   backbone    |   Paper    | OHEM | aux  | dilated | JPU  | Epoch | val (crop)  |     val     |
| :-------: | :-----------: | :--------: | :--: | :--: | :-----: | :--: | :---: | :---------: | :---------: |
|    FCN    | ResNet101-v1s |     /      |  ✗   |  ✓   |    ✗    |  ✓   |  50   | 94.54/78.31 | 94.50/76.89 |
|  PSPNet   | ResNet101-v1s |     /      |  ✗   |  ✓   |    ✓    |  ✗   |  50   | 94.87/80.13 | 94.88/78.57 |
| DeepLabv3 | ResNet101-v1s | no / 77.02 |  ✗   |  ✓   |    ✗    |  ✓   |  50   | 95.17/81.00 | 94.81/78.75 |

> 1. the metric is `pixAcc/mIoU`
> 2. `aux_weight=0.5`

#### Cityscapes

Here, we only using fine train (2975), val (500) as most paper used. (More detail can reference [DeepLabv3](https://github.com/chenxi116/DeepLabv3.pytorch)) . And the performance is evaluated with single scale

- Base Size 1024, Crop Size 768

|   Model   |   backbone    | Paper(*) | OHEM | aux  | dilated | JPU  | Epoch | val (crop)  |     val     |
| :-------: | :-----------: | :------: | :--: | :--: | :-----: | :--: | :---: | :---------: | :---------: |
|    FCN    | ResNet101-v1s | no/75.96 |  ✗   |  ✓   |    ✗    |  ✓   |  120  | 96.29/73.60 | 96.18/78.61 |
|  PSPNet   | ResNet101-v1s | no/78.56 |      |      |         |      |       |             |             |
| DeepLabv3 | ResNet101-v1s | no/78.90 |  ✗   |  ✓   |    ✗    |  ✓   |  120  | 96.25/73.44 | 96.23/79.03 |

> Note：
>
> 1. Paper(*) means results from: [openseg.pytorch](https://github.com/openseg-group/openseg.pytorch)（results with single scale without crop），there are a little different in the training strategy.  

## Demo

Demo of segmentation of a given image.  (Please download pre-trained model to `~/.torch/models` first. --- If you put pre-trained model to other folder, please change the `--root`)

```shell
$ python demo_segmentation_pil.py [--model deeplab_resnet101_ade] [--input-pic <image>.jpg] [--cuda] [--aux] [--jpu]
```

> Note：
>
> 1. if not give `--input-pic`, using default image we provided. 
> 2. `aux, jpu, dilated` is depend on your model

## Evaluation

The default data root is `~/.torch/datasets` (You can download dataset and build a soft-link to it)

```shell
$ python eval_segmentation_pil.py [--model_name fcn_resnet101_voc] [--dataset pascal_paper] [--split val] [--mode testval|val] [--base-size 540] [--crop-size 480] [--aux] [--jpu] [--cuda]
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
$ python -m torch.distributed.launch --nproc_per_node=$NGPUS train_segmentation_pil.py [--model fcn] [--backbone resnet101] [--dataset pascal_voc] [--batch-size 8] [--base-size 540] [--crop-size 480] [--aux] [--jpu] [--log-step 10]
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

#### Trained Model

| fcn_resnet101_voc     | fcn_resnet101_citys                                          | psp_resnet101_voc | psp_resnet101_citys |
| --------------------- | ------------------------------------------------------------ | ----------------- | ------------------- |
|                       |                                                              |                   |                     |
| deeplab_resnet101_voc | deeplab_resnet101_citys                                      |                   |                     |
|                       | [GoogleDrive](https://drive.google.com/open?id=1MMxHSXDbSOLSR0MZAJ_OQNYkhRFmGjJg) |                   |                     |

