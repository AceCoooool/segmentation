"""Pyramid Scene Parsing Network"""
import os
import torch
import torch.nn.functional as F

from model.seg_models.segbase import SegBaseModel
from model.module.basic import _PSPHead, _FCNHead

__all__ = ['PSPNet', 'get_psp',
           'get_psp_resnet101_voc',
           'get_psp_resnet101_citys']


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
