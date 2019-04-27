from model.backbone.resnetv1b import *
from model.seg_models.fcn import *
from model.seg_models.pspnet import *
from model.seg_models.deeplabv3 import *

__all__ = ['get_model', 'get_model_list']

_models = {
    # backbone
    'resnet50_v1s': resnet50_v1s,
    'resnet101_v1s': resnet101_v1s,
    # fcn
    'fcn_resnet101_voc': get_fcn_resnet101_voc,
    'fcn_resnet101_citys': get_fcn_resnet101_citys,
    # psp
    'psp_resnet101_voc': get_psp_resnet101_voc,
    'psp_resnet101_citys': get_psp_resnet101_citys,
    # deeplab
    'deeplab_resnet101_voc': get_deeplab_resnet101_voc,
    'deeplab_resnet101_citys': get_deeplab_resnet101_citys,
}


def get_model(name, **kwargs):
    name = name.lower()
    if name not in _models:
        err_str = '"%s" is not among the following model list:\n\t' % name
        err_str += '%s' % ('\n\t'.join(sorted(_models.keys())))
        raise ValueError(err_str)
    net = _models[name](**kwargs)
    return net


def get_model_list():
    return _models.keys()
