"""Pyramid Scene Parsing Network"""
import os
import math
import torch
import torch.nn.functional as F

from model.seg_models.segbase import SegBaseModel
from model.module.basic import _FCNHead, _DeepLabHead

__all__ = ['DeepLabV3', 'get_deeplab',
           'get_deeplab_resnet101_voc',
           'get_deeplab_resnet101_citys', ]


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
