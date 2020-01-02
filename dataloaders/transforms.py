"""
Customized data transforms
"""
import random

from PIL import Image
from scipy import ndimage
import numpy as np
import torch
import torchvision.transforms.functional as tr_F


class RandomMirror(object):
    """
    Randomly filp the images/masks horizontally
    """
    def __call__(self, sample):
        img, label = sample['image'], sample['label']
        
        pre_label = sample['pre_label']
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if isinstance(label, dict):
                label = {catId: x.transpose(Image.FLIP_LEFT_RIGHT)
                         for catId, x in label.items()}
            else:
                label = label.transpose(Image.FLIP_LEFT_RIGHT)
        sample['image'] = img
        sample['label'] = label
        return sample


class Resize(object):
    """
    Resize images/masks to given size

    Args:
        size: output size
    """
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        img, label = sample['image'], sample['label']
        pre_label = sample['pre_label']
        label_t  = sample['label_t']
        
        img = tr_F.resize(img, self.size)
        if isinstance(label, dict):
            label = {catId: tr_F.resize(x, self.size, interpolation=Image.NEAREST)
                     for catId, x in label.items()}
        else:
            label = tr_F.resize(label, self.size, interpolation=Image.NEAREST)

        if isinstance(pre_label, dict):
            pre_label = {catId: tr_F.resize(x, self.size, interpolation=Image.NEAREST)
                     for catId, x in label.items()}
        else:
            pre_label = tr_F.resize(pre_label, self.size, interpolation=Image.NEAREST)

        if isinstance(label_t, dict):
            label_t = {catId: tr_F.resize(x, size=[480,854], interpolation=Image.NEAREST)
                     for catId, x in label_t.items()}
        else:
            label_t = tr_F.resize(label_t, self.size, interpolation=Image.NEAREST)

        sample['image'] = img
        sample['label'] = label
        sample['pre_label'] = pre_label
        sample['label_t'] = label_t
        return sample

class ToTensorNormalize(object):
    """
    Convert images/masks to torch.Tensor
    Scale images' pixel values to [0-1] and normalize with predefined statistics
    """
    def __call__(self, sample):
        img, label = sample['image'], sample['label']
        label_t = sample['label_t']
        pre_label = sample['pre_label']
        img = tr_F.to_tensor(img)
        img = tr_F.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        if isinstance(label, dict):
            label = {catId: torch.Tensor(np.array(x)).long()
                     for catId, x in label.items()}
        else:
            label = torch.Tensor(np.array(label)).long()

        if isinstance(pre_label, dict):
            pre_label = {catId: torch.Tensor(np.array(x)).long()
                     for catId, x in pre_label.items()}
        else:
            pre_label = torch.Tensor(np.array(pre_label)).long()

        if isinstance(label_t, dict):
            label_t = {catId: torch.Tensor(np.array(x)).long()
                     for catId, x in label_t.items()}
        else:
            label_t = torch.Tensor(np.array(label_t)).long()
        
        sample['image'] = img
        sample['label'] = label
        sample['label_t'] = label_t
        sample['pre_label'] = pre_label
        return sample
