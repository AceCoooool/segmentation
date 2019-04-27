"""Fully Convolutional Network with Stride of 8"""
from __future__ import division

import os
import torch
import torch.nn.functional as F

from model.module.basic import _FCNHead
from model.seg_models.segbase import SegBaseModel

__all__ = ['FCN', 'get_fcn',
           'get_fcn_resnet50_voc',
           'get_fcn_resnet101_voc',
           'get_fcn_resnet101_citys']


class FCN(SegBaseModel):
    # pylint: disable=arguments-differ
    def __init__(self, nclass, backbone='resnet50', aux=True, dilated=True, jpu=False,
                 pretrained_base=True, base_size=520, crop_size=480, **kwargs):
        super(FCN, self).__init__(nclass, aux, backbone, base_size=base_size, crop_size=crop_size,
                                  dilated=dilated, jpu=jpu, pretrained_base=pretrained_base, **kwargs)
        self.head = _FCNHead(2048, nclass, **kwargs)
        if self.aux:
            self.auxlayer = _FCNHead(1024, nclass, **kwargs)
        self.__setattr__('others', ['head', 'auxlayer'] if aux else ['head'])

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


def get_fcn(dataset='pascal_voc', backbone='resnet50', pretrained=False,
            root=os.path.expanduser('~/.torch/models'), pretrained_base=True, **kwargs):
    acronyms = {
        'pascal_voc': 'voc',
        'citys': 'citys'
    }
    from data import datasets
    # infer number of classes
    model = FCN(datasets[dataset].NUM_CLASS, backbone=backbone, pretrained_base=pretrained_base,
                **kwargs)
    if pretrained:
        from model.model_store import get_model_file
        model.load_state_dict(torch.load(get_model_file(
            'fcn_%s_%s' % (backbone, acronyms[dataset]), root=root)))
    return model


def get_fcn_resnet50_voc(**kwargs):
    return get_fcn('pascal_voc', 'resnet50', **kwargs)


def get_fcn_resnet101_voc(**kwargs):
    return get_fcn('pascal_voc', 'resnet101', **kwargs)


def get_fcn_resnet101_citys(**kwargs):
    return get_fcn('citys', 'resnet101', **kwargs)
