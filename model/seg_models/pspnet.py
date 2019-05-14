"""Pyramid Scene Parsing Network"""
import os
import torch
from torch import nn
import torch.nn.functional as F

from model.seg_models.segbase import SegBaseModel
from model.module.basic import _FCNHead

__all__ = ['PSPNet', 'get_psp',
           'get_psp_resnet101_voc',
           'get_psp_resnet101_citys']


# head
def _PSP1x1Conv(in_channels, out_channels):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                         nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))


class _PyramidPooling(nn.Module):
    def __init__(self, in_channels):
        super(_PyramidPooling, self).__init__()
        out_channels = in_channels // 4
        self.conv1 = _PSP1x1Conv(in_channels, out_channels)
        self.conv2 = _PSP1x1Conv(in_channels, out_channels)
        self.conv3 = _PSP1x1Conv(in_channels, out_channels)
        self.conv4 = _PSP1x1Conv(in_channels, out_channels)

    @staticmethod
    def pool(x, size):
        return F.adaptive_avg_pool2d(x, output_size=size)

    @staticmethod
    def upsample(x, h, w):
        return F.interpolate(x, (h, w), mode='bilinear', align_corners=True)

    def forward(self, x):
        _, _, h, w = x.shape
        feat1 = self.upsample(self.conv1(self.pool(x, 1)), h, w)
        feat2 = self.upsample(self.conv2(self.pool(x, 2)), h, w)
        feat3 = self.upsample(self.conv3(self.pool(x, 3)), h, w)
        feat4 = self.upsample(self.conv4(self.pool(x, 4)), h, w)
        return torch.cat([x, feat1, feat2, feat3, feat4], dim=1)


class _PSPHead(nn.Module):
    def __init__(self, nclass, **kwargs):
        super(_PSPHead, self).__init__(**kwargs)
        self.psp = _PyramidPooling(2048)
        self.block = list()
        self.block.append(nn.Conv2d(4096, 512, kernel_size=3, padding=1, bias=False))
        self.block.append(nn.BatchNorm2d(512))
        self.block.append(nn.ReLU(inplace=True))
        self.block.append(nn.Dropout(0.1))
        self.block.append(nn.Conv2d(512, nclass, kernel_size=1))
        self.block = nn.Sequential(*self.block)

    def forward(self, x):
        x = self.psp(x)
        return self.block(x)


class PSPNet(SegBaseModel):
    def __init__(self, nclass, backbone='resnet50', aux=True, dilated=True, jpu=False,
                 pretrained_base=True, base_size=520, crop_size=480, **kwargs):
        super(PSPNet, self).__init__(nclass, aux, backbone, base_size=base_size, dilated=dilated, jpu=jpu,
                                     crop_size=crop_size, pretrained_base=pretrained_base, **kwargs)
        self.head = _PSPHead(nclass, **kwargs)
        if self.aux:
            self.auxlayer = _FCNHead(1024, nclass, **kwargs)

        self.__setattr__('others', ['head', 'auxlayer'] if self.aux else ['head'])

    def forward(self, x):
        c3, c4 = self.base_forward(x)
        outputs = []
        x = self.head(c4)
        x = F.interpolate(x, self._up_kwargs, mode='bilinear', align_corners=True)
        outputs.append(x)

        if self.aux:
            auxout = self.auxlayer(c3)
            auxout = F.interpolate(auxout, self._up_kwargs, mode='bilinear', align_corners=True)
            outputs.append(auxout)
        return tuple(outputs)


def get_psp(dataset='pascal_voc', backbone='resnet101', pretrained=False, pretrained_base=True,
            root=os.path.expanduser('~/.torch/models'), **kwargs):
    acronyms = {
        'pascal_voc': 'voc',
        'citys': 'citys',
    }
    from data import datasets
    # infer number of classes
    model = PSPNet(datasets[dataset].NUM_CLASS, backbone=backbone,
                   pretrained_base=pretrained_base, **kwargs)
    if pretrained:
        from model.model_store import get_model_file
        model.load_state_dict(torch.load(get_model_file('psp_%s_%s' % (backbone, acronyms[dataset]),
                                                        root=root)))
    return model


def get_psp_resnet101_voc(**kwargs):
    return get_psp('pascal_voc', 'resnet101', **kwargs)


def get_psp_resnet101_citys(**kwargs):
    return get_psp('citys', 'resnet101', **kwargs)
