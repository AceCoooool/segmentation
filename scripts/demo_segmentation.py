import os
import sys
import argparse
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import torch
from torchvision import transforms

cur_path = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(cur_path, '..'))
from model.model_zoo import get_model
import utils as ptutil


def parse_args():
    parser = argparse.ArgumentParser(description='Demo for segmentation from a given image')
    parser.add_argument('--model', type=str, default='fcn_resnet101_voc',
                        help='name of the model to use')
    parser.add_argument('--saved-params', type=str, default='',
                        help='path to the saved model parameters')
    parser.add_argument('--cuda', type=ptutil.str2bool, default='true',
                        help='demo with GPU')
    parser.add_argument('--input-pic', type=str, default=None,
                        help='path to the input picture')
    parser.add_argument('--root', type=str, default=os.path.expanduser('~/.torch/models'),
                        help='Default pre-trained mdoel root.')
    # parser.add_argument('--root', type=str, default='/home/ace/cbb/own/pretrained/seg_jpu',
    #                     help='Default pre-trained mdoel root.')

    parser.add_argument('--aux', type=ptutil.str2bool, default='true',
                        help='whether using aux loss')
    parser.add_argument('--dilated', type=ptutil.str2bool, default='false',
                        help='whether using dilated in backbone')
    parser.add_argument('--jpu', type=ptutil.str2bool, default='true',
                        help='whether using JPU after backbone')
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = parse_args()
    device = torch.device('cpu')
    if opt.cuda:
        device = torch.device('cuda')
    # Load Model
    model_name = opt.model
    pretrained = True if opt.saved_params == '' else False
    model = get_model(model_name, pretrained=pretrained, pretrained_base=False,
                      aux=opt.aux, dilated=opt.dilated, jpu=opt.jpu, root=opt.root).to(device)
    model.eval()

    # Load Images
    if opt.input_pic is None:
        img_map = {'voc': 'voc_example.jpg', 'citys': 'city_example.jpg'}
        opt.input_pic = os.path.join(cur_path, 'png/' + img_map[model_name.split('_')[-1]])
    img = Image.open(opt.input_pic)

    # Transform
    transform_fn = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img = transform_fn(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model.demo(img)

    color_map = {'voc': 'pascal_voc', 'citys': 'citys'}
    predict = torch.argmax(output, 1).squeeze(0).cpu().numpy()
    mask = ptutil.get_color_pallete(predict, color_map[model_name.split('_')[-1]])
    mask.save(os.path.join(cur_path, 'png/output.png'))
    mmask = mpimg.imread(os.path.join(cur_path, 'png/output.png'))
    plt.imshow(mmask)
    plt.show()
