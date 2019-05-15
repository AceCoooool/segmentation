import os
import torch
from torch import nn
import torch.nn.functional as F

from model.seg_models.segbase import SegBaseModel
from model.module.basic import _FCNHead

__all__ = ['get_danet', 'get_danet_resnet101_voc']


class _PAM_Module(nn.Module):
    """ Position attention module"""

    def __init__(self, in_dim):
        super(_PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        b, c, h, w = x.size()
        proj_query = self.query_conv(x).view(b, -1, w * h).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(b, -1, w * h)
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=1)
        proj_value = self.value_conv(x).view(b, -1, w * h)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(b, c, h, w)

        out = self.gamma * out + x
        return out


class _CAM_Module(nn.Module):
    """ Channel attention module"""

    def __init__(self, in_dim):
        super(_CAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        b, c, h, w = x.size()
        proj_query = x.view(b, c, -1)
        proj_key = x.view(b, c, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = F.softmax(energy_new, dim=-1)
        proj_value = x.view(b, c, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(b, c, h, w)

        out = self.gamma * out + x
        return out


class _DANetHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(_DANetHead, self).__init__()
        inter_channels = in_channels // 4
        self.conv5a = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(inter_channels), nn.ReLU(inplace=True))

        self.conv5c = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(inter_channels), nn.ReLU(inplace=True))

        self.sa = _PAM_Module(inter_channels)
        self.sc = _CAM_Module(inter_channels)
        self.conv51 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(inter_channels), nn.ReLU(inplace=True))
        self.conv52 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(inter_channels), nn.ReLU(inplace=True))

        # self.conv6 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(512, out_channels, 1))
        # self.conv7 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(512, out_channels, 1))
        self.conv8 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(512, out_channels, 1))

    def forward(self, x):
        feat1 = self.conv5a(x)
        sa_feat = self.sa(feat1)
        sa_conv = self.conv51(sa_feat)
        # sa_output = self.conv6(sa_conv)

        feat2 = self.conv5c(x)
        sc_feat = self.sc(feat2)
        sc_conv = self.conv52(sc_feat)
        # sc_output = self.conv7(sc_conv)

        feat_sum = sa_conv + sc_conv

        sasc_output = self.conv8(feat_sum)

        return sasc_output


class DANet(SegBaseModel):
    def __init__(self, nclass, backbone='resnet101', aux=False, dilated=True, jpu=False,
                 pretrained_base=True, base_size=520, crop_size=480, **kwargs):
        super(DANet, self).__init__(nclass, aux, backbone, dilated=dilated, jpu=jpu, base_size=base_size,
                                    crop_size=crop_size, pretrained_base=pretrained_base, **kwargs)
        self.head = _DANetHead(2048, nclass)
        if self.aux:
            self.auxlayer = _FCNHead(1024, nclass)

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


def get_danet(dataset='pascal_voc', backbone='resnet50', pretrained=False, jpu=False,
              root=os.path.expanduser('~/.torch/models'), pretrained_base=True, **kwargs):
    acronyms = {
        'pascal_voc': 'voc',
        'citys': 'citys'
    }
    from data import datasets
    # infer number of classes
    model = DANet(datasets[dataset].NUM_CLASS, backbone=backbone, pretrained_base=pretrained_base,
                  jpu=jpu, **kwargs)
    if pretrained:
        from model.model_store import get_model_file
        name = 'danet_%s_%s' % (backbone, acronyms[dataset])
        name = name + '_jpu' if jpu else name
        model.load_state_dict(torch.load(get_model_file(name, root=root)))
    return model


def get_danet_resnet101_voc(**kwargs):
    return get_danet('pascal_voc', 'resnet101', **kwargs)


if __name__ == '__main__':
    net = get_danet_resnet101_voc()
    net.eval()
    a = torch.randn(1, 3, 200, 200)
    with torch.no_grad():
        out = net(a)
    print(out)
