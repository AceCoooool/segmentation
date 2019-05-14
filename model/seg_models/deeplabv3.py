"""Pyramid Scene Parsing Network"""
import os
import math
import torch
from torch import nn
import torch.nn.functional as F

from model.seg_models.segbase import SegBaseModel
from model.module.basic import _FCNHead

__all__ = ['DeepLabV3', 'get_deeplab',
           'get_deeplab_resnet101_voc',
           'get_deeplab_resnet101_citys', ]


def _ASPPConv(in_channels, out_channels, atrous_rate):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=atrous_rate,
                                   dilation=atrous_rate, bias=False),
                         nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))


class _AsppPooling(nn.Module):
    def __init__(self, in_channels, out_channels, height=60, width=60):
        super(_AsppPooling, self).__init__()
        self.gap = list()
        self._up_kwargs = (height, width)
        self.gap.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.gap.append(nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False))
        self.gap.append(nn.BatchNorm2d(out_channels))
        self.gap.append(nn.ReLU(inplace=True))
        self.gap = nn.Sequential(*self.gap)

    def forward(self, x):
        pool = self.gap(x)
        return F.interpolate(pool, self._up_kwargs, mode='bilinear', align_corners=True)


class _ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, height=60, width=60):
        super(_ASPP, self).__init__()
        out_channels = 256
        self.b0 = list()
        self.b0.append(nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False))
        self.b0.append(nn.BatchNorm2d(out_channels))
        self.b0.append(nn.ReLU(inplace=True))
        self.b0 = nn.Sequential(*self.b0)

        rate1, rate2, rate3 = tuple(atrous_rates)
        self.b1 = _ASPPConv(in_channels, out_channels, rate1)
        self.b2 = _ASPPConv(in_channels, out_channels, rate2)
        self.b3 = _ASPPConv(in_channels, out_channels, rate3)
        self.b4 = _AsppPooling(in_channels, out_channels, height=height, width=width)

        self.project = nn.Sequential(nn.Conv2d(5 * out_channels, out_channels, kernel_size=1, bias=False),
                                     nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
                                     nn.Dropout(0.5))

    def forward(self, x):
        a0 = self.b0(x)
        a1 = self.b1(x)
        a2 = self.b2(x)
        a3 = self.b3(x)
        a4 = self.b4(x)
        return self.project(torch.cat([a0, a1, a2, a3, a4], 1))


class _DeepLabHead(nn.Module):
    def __init__(self, nclass, **kwargs):
        super(_DeepLabHead, self).__init__()
        self.aspp = _ASPP(2048, [12, 24, 36], **kwargs)
        self.block = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
                                   nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.Dropout(0.1),
                                   nn.Conv2d(256, nclass, kernel_size=1))

    def forward(self, x):
        x = self.aspp(x)
        return self.block(x)


class DeepLabV3(SegBaseModel):
    def __init__(self, nclass, backbone='resnet101', aux=True, dilated=True, jpu=False,
                 pretrained_base=True, base_size=520, crop_size=480, **kwargs):
        super(DeepLabV3, self).__init__(nclass, aux, backbone, base_size=base_size, crop_size=crop_size,
                                        dilated=dilated, jpu=jpu, pretrained_base=pretrained_base, **kwargs)
        self.head = _DeepLabHead(nclass, height=self._up_kwargs[0] // 8,
                                 width=self._up_kwargs[1] // 8, **kwargs)
        if self.aux:
            self.auxlayer = _FCNHead(1024, nclass, **kwargs)

        self.__setattr__('others', ['head', 'auxlayer'] if self.aux else ['head'])

    def forward(self, x):
        c3, c4 = self.base_forward(x)
        outputs = []
        if self.keep_shape:
            self.head.aspp.b4._up_kwargs = (math.ceil(self._up_kwargs[0] / 8), math.ceil(self._up_kwargs[1] / 8))
        x = self.head(c4)
        x = F.interpolate(x, self._up_kwargs, mode='bilinear', align_corners=True)
        outputs.append(x)

        if self.aux:
            auxout = self.auxlayer(c3)
            auxout = F.interpolate(auxout, self._up_kwargs, mode='bilinear', align_corners=True)
            outputs.append(auxout)
        return tuple(outputs)

    def demo(self, x):
        h, w = x.shape[2:]
        self._up_kwargs = (h, w)
        self.head.aspp.b4._up_kwargs = (math.ceil(h / 8), math.ceil(w / 8))
        pred = self.forward(x)
        if self.aux:
            pred = pred[0]
        return pred


def get_deeplab(dataset='pascal_voc', backbone='resnet101', pretrained=False, pretrained_base=True,
                root=os.path.expanduser('~/.torch/models'), **kwargs):
    acronyms = {
        'pascal_voc': 'voc',
        'citys': 'citys'
    }
    from data import datasets
    # infer number of classes
    model = DeepLabV3(datasets[dataset].NUM_CLASS, backbone=backbone, pretrained_base=pretrained_base, **kwargs)
    if pretrained:
        from model.model_store import get_model_file
        model.load_state_dict(torch.load(get_model_file('deeplab_%s_%s' % (backbone, acronyms[dataset]),
                                                        root=root)))
    return model


def get_deeplab_resnet101_voc(**kwargs):
    return get_deeplab('pascal_voc', 'resnet101', **kwargs)


def get_deeplab_resnet101_citys(**kwargs):
    return get_deeplab('citys', 'resnet101', **kwargs)


if __name__ == '__main__':
    deeplab = get_deeplab_resnet101_voc(dilated=False, jpu=True)
    deeplab.eval()
    # print(deeplab)
    a = torch.randn(1, 3, 480, 480)
    deeplab.eval()
    with torch.no_grad():
        out = deeplab(a)
