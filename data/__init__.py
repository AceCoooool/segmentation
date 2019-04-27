from .pascal_voc import VOCSegmentation
from .cityscapes import CitySegmentation

datasets = {
    'pascal_voc': VOCSegmentation,
    'citys': CitySegmentation,
}


def get_segmentation_dataset(name, **kwargs):
    """Segmentation Datasets"""
    return datasets[name.lower()](**kwargs)
